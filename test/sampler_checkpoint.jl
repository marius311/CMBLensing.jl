using CMBLensing, Random, Test

@testset "sample_joint final checkpoint" begin
    (; ds) = load_sim(; Cℓ=CMBLensing._default_Cℓs(), pol=:I, T=Float32,
        θpix=4, Nside=8, storage=Array, L=LenseFlow(3), rng=MersenneTwister(7))
    seen = Float32[]
    function toy_transition!(state, _)
        ϕ = state[:ϕ]
        push!(seen, first(ϕ.arr))
        fill!(ϕ.arr, Float32(state[:step]))
        state[:Ω] = merge(state[:Ω], (;ϕ))
        state[:logpdf] = Float32(state[:step])
    end
    function toy_sample(n; filename=nothing, resume=false)
        sample_joint(ds; nsamps_per_chain=n, nchains=1, pmap=map,
            filename, resume, nfilewrite=4, nsavemaps=2, progress=false,
            ϕstart=0, θstart=(;), storage=Array, gibbs_samplers=[toy_transition!])
    end
    function read_chunks(filename)
        CMBLensing.jldopen(filename, "r") do io
            names = sort!(filter(k -> startswith(k, "chunks_"), collect(keys(io))); by=k -> parse(Int, k[8:end]))
            [read(io, name) for name in names]
        end
    end
    loaded(filename) = load_chains(filename; progress=false, unbatch=false)[1]

    mktempdir() do dir
        # A single initial sample must be saved, load through the default
        # unbatch path, and provide the step needed to resume.
        one_file = joinpath(dir, "one.jld2")
        one = toy_sample(1; filename=one_file)
        @test length(one[1]) == 1
        @test isfile(one_file)
        @test length(read_chunks(one_file)) == 1
        initial = only(loaded(one_file))
        @test initial[:step] == 1 && haskey(initial, :ϕ) && !haskey(initial, :logpdf)
        default_load = load_chains(one_file; progress=false)
        @test length(default_load) == 1 && length(default_load[1]) == 1
        @test default_load[1][1][:step] == 1
        empty!(seen)
        toy_sample(3; filename=one_file, resume=true)
        @test seen == Float32[0, 2]
        @test [sample[:step] for sample in loaded(one_file)] == [1, 2, 3]
        @test all(==(3f0), last(loaded(one_file))[:ϕ].arr)
        @test length(read_chunks(one_file)) == 2

        # Off-boundary final maps are included in files for resumption, while
        # the previous in-memory-only map cadence is retained.
        three_file = joinpath(dir, "three.jld2")
        three = toy_sample(3; filename=three_file)
        @test length(three[1]) == 3
        @test length(read_chunks(three_file)) == 1
        @test haskey(last(loaded(three_file)), :ϕ)
        memory_only = toy_sample(3)
        @test !haskey(last(memory_only[1]), :ϕ)

        # An exact periodic boundary is written once, with no extra chunk.
        four_file = joinpath(dir, "four.jld2")
        toy_sample(4; filename=four_file)
        @test length(read_chunks(four_file)) == 1
        @test [sample[:step] for sample in loaded(four_file)] == 1:4
        @test !haskey(loaded(four_file)[3], :ϕ)
        toy_sample(4; filename=four_file, resume=true)
        @test length(read_chunks(four_file)) == 1

        # The partial chunk follows the periodic one; a no-op resume does not
        # duplicate it, and the next transition starts from its saved field.
        five_file = joinpath(dir, "five.jld2")
        five = toy_sample(5; filename=five_file)
        @test length(five[1]) == 1 && only(five[1])[:step] == 5
        @test length(read_chunks(five_file)) == 2
        @test [sample[:step] for sample in loaded(five_file)] == 1:5
        @test all(==(5f0), last(loaded(five_file))[:ϕ].arr)
        toy_sample(5; filename=five_file, resume=true)
        @test length(read_chunks(five_file)) == 2
        empty!(seen)
        toy_sample(7; filename=five_file, resume=true)
        @test seen == Float32[5, 6]
        @test length(read_chunks(five_file)) == 3
        @test [sample[:step] for sample in loaded(five_file)] == 1:7
        @test all(==(7f0), last(loaded(five_file))[:ϕ].arr)

        # Simulate a pre-fix, normally written chunk whose first sample did
        # not include :step. Existing files remain readable and resumable.
        old_rundat, old_chunks = CMBLensing.jldopen(four_file, "r") do io
            (read(io, "rundat"), read(io, "chunks_1"))
        end
        delete!(old_chunks[1][1], :step)
        legacy_file = joinpath(dir, "legacy.jld2")
        CMBLensing.jldopen(legacy_file, "w") do io
            write(io, "rundat", old_rundat)
            write(io, "chunks_1", old_chunks)
        end
        @test length(loaded(legacy_file)) == 4
        @test !haskey(first(loaded(legacy_file)), :step)
        empty!(seen)
        toy_sample(5; filename=legacy_file, resume=true)
        @test seen == Float32[4]
        @test length(loaded(legacy_file)) == 5
        @test all(==(5f0), last(loaded(legacy_file))[:ϕ].arr)

        # An initial-only batch has no logpdf. Default loading must use the
        # saved potential to determine how many chains to unbatch.
        ϕ0 = zero(diag(ds.Cϕ))
        ϕ1 = copy(ϕ0)
        fill!(ϕ1.arr, 1f0)
        batch_field = CMBLensing.batch(ϕ0, ϕ1)
        batch_file = joinpath(dir, "initial-batched.jld2")
        CMBLensing.jldopen(batch_file, "w") do io
            write(io, "chunks_1", [Any[Dict(:step=>1, :ϕ=>batch_field)]])
        end
        batched = load_chains(batch_file; progress=false)
        @test length(batched) == 2
        @test all(iszero, batched[1][1][:ϕ].arr)
        @test all(==(1f0), batched[2][1][:ϕ].arr)

        # Also exercise actual Gibbs stages, not just the deterministic file
        # fixture, through a write and an off-boundary resume.
        actual_file = joinpath(dir, "actual.jld2")
        actual_options = (; nchains=1, pmap=map, filename=actual_file,
            nfilewrite=4, nsavemaps=2, progress=false, ϕstart=0, θstart=(;),
            storage=Array, symp_kwargs=[(; N=1, ϵ=0.01)],
            conjgrad_kwargs=(; tol=1e-2, nsteps=30))
        sample_joint(ds; nsamps_per_chain=2, resume=false, actual_options...)
        @test [sample[:step] for sample in loaded(actual_file)] == [1, 2]
        @test all(isfinite, last(loaded(actual_file))[:ϕ].arr)
        sample_joint(ds; nsamps_per_chain=3, resume=true, actual_options...)
        @test [sample[:step] for sample in loaded(actual_file)] == [1, 2, 3]
        @test all(isfinite, last(loaded(actual_file))[:ϕ].arr)
    end
end
