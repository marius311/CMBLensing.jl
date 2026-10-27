using CMBLensing, LinearAlgebra, Random, Test

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
        @test [first(sample[:ϕ].arr) for sample in loaded(three_file)] == Float32[0, 2, 3]
        memory_only = toy_sample(3)
        @test !haskey(last(memory_only[1]), :ϕ)
        @test [first(sample[:ϕ].arr) for sample in memory_only[1][1:2]] == Float32[0, 2]

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
        dropped = load_chains(batch_file; dropmaps=true, progress=false)
        @test length(dropped) == 2
        @test all(chain -> length(chain) == 1 && !haskey(chain[1], :ϕ), dropped)
        @test all(chain -> Set(keys(chain[1])) == Set([:step]), dropped)
        @test length(load_chains(batch_file; dropmaps=true, join=true, progress=false)) == 2

        # Thinning can select the initial no-logpdf sample even when later
        # samples have one. Preserve each selected sample's own batch width.
        batch3 = CMBLensing.batch(ϕ0, ϕ1, ϕ1)
        thin_file = joinpath(dir, "initial-batched-thin.jld2")
        CMBLensing.jldopen(thin_file, "w") do io
            write(io, "chunks_1", [
                Any[Dict(:step=>1, :ϕ=>batch_field),
                    Dict(:step=>2, :logpdf=>CMBLensing.batch(Float32[2, 2]))],
                Any[Dict(:step=>1, :ϕ=>batch3),
                    Dict(:step=>2, :logpdf=>CMBLensing.batch(Float32[2, 2, 2]))],
            ])
        end
        retained_thin = load_chains(thin_file; thin=2, progress=false)
        @test length(retained_thin) == 5
        @test all(chain -> length(chain) == 1 && haskey(chain[1], :ϕ), retained_thin)
        dropped_thin = load_chains(thin_file; thin=2, dropmaps=true, progress=false)
        @test length(dropped_thin) == 5
        @test all(chain -> length(chain) == 1 && Set(keys(chain[1])) == Set([:step]), dropped_thin)
        selected = load_chains(thin_file; thin=s -> s[:step] == 1,
            dropmaps=true, join=true, progress=false)
        @test length(selected) == 5 && all(s -> s[:step] == 1 && !haskey(s, :ϕ), selected)
        unexpanded = load_chains(thin_file; thin=2, dropmaps=true, unbatch=false, progress=false)
        @test length(unexpanded) == 2 && all(chain -> length(chain) == 1 && !haskey(chain[1], :ϕ), unexpanded)
        only_maps = load_chains(thin_file; thin=:hasmaps, dropmaps=true, progress=false)
        @test length(only_maps) == 5
        @test all(chain -> length(chain) == 1 && Set(keys(chain[1])) == Set([:step]), only_maps)
        only_maps_retained = load_chains(thin_file; thin=:hasmaps, progress=false)
        @test length(only_maps_retained) == 5
        @test all(chain -> length(chain) == 1 && haskey(chain[1], :ϕ), only_maps_retained)

        batched_ds = copy(ds)
        batched_ds.d = CMBLensing.batch(ds.d, ds.d)
        batch_actual_file = joinpath(dir, "actual-initial-batched.jld2")
        sample_joint(batched_ds; nsamps_per_chain=1, nchains=1, pmap=map,
            filename=batch_actual_file, resume=false, nfilewrite=4, nsavemaps=2,
            progress=false, ϕstart=0, θstart=(;), storage=Array)
        batched_actual = load_chains(batch_actual_file; progress=false)
        @test length(batched_actual) == 2
        @test all(chain -> length(chain) == 1 && chain[1][:step] == 1, batched_actual)

        # A custom initializer can save batched parameters without a map or
        # logpdf. Split batched components and retain ordinary components.
        plain_ds = CMBLensing.NoLensingDataSet(; d=FlatMap(zeros(2,2)), Cf=I, Cn=I)
        plain_batch_ds = CMBLensing.NoLensingDataSet(;
            d=CMBLensing.batch(plain_ds.d, plain_ds.d), Cf=I, Cn=I)
        initialize_parameters!(state, _) =
            (state[:θ] = (A=CMBLensing.batch(10.0, 20.0), fixed=1.0))
        parameters_file = joinpath(dir, "initial-parameters-batched.jld2")
        sample_joint(plain_batch_ds; nsamps_per_chain=1, nchains=1, pmap=map,
            filename=parameters_file, resume=false, storage=Array, progress=false,
            gibbs_initializers=[initialize_parameters!], gibbs_samplers=[])
        for dropmaps in (false, true)
            parameters = load_chains(parameters_file; dropmaps, thin=2, progress=false)
            @test length(parameters) == 2
            @test [chain[1][:θ].A for chain in parameters] == [10.0, 20.0]
            @test all(chain -> chain[1][:θ].fixed == 1.0 &&
                !haskey(chain[1], :ϕ) && !haskey(chain[1], :logpdf), parameters)
            @test length(load_chains(parameters_file; dropmaps, join=true, progress=false)) == 2
        end
        @test length(load_chains(parameters_file; unbatch=false, progress=false)) == 1

        # Reject differing numbers of persisted and requested chains before
        # merging their states; equal counts retain their individual histories.
        nextid = Ref(0)
        function initialize_counter!(state, _)
            if !haskey(state, :counter)
                nextid[] += 1
                state[:counter] = 10 * nextid[]
            end
        end
        count!(state, _) = (state[:counter] += state[:step])
        counter_options = (; pmap=map, storage=Array, progress=false,
            nfilewrite=4, nsavemaps=1, gibbs_initializers=[initialize_counter!],
            gibbs_samplers=[count!])
        for (saved_count, requested_count) in ((1, 2), (2, 1))
            path = joinpath(dir, "chain-count-$saved_count.jld2")
            nextid[] = 0
            sample_joint(plain_ds; counter_options..., filename=path, resume=false,
                nsamps_per_chain=3, nchains=saved_count)
            before = read(path)
            @test_throws ArgumentError sample_joint(plain_ds; counter_options...,
                filename=path, resume=true, nsamps_per_chain=4, nchains=requested_count)
            @test read(path) == before
            sample_joint(plain_ds; counter_options..., filename=path, resume=true,
                nsamps_per_chain=4, nchains=saved_count)
            counts = load_chains(path; progress=false, unbatch=false)
            @test [[s[:counter] for s in chain] for chain in counts] ==
                [10*i .+ [0, 2, 5, 9] for i in 1:saved_count]
        end

        # A malformed older file must not broadcast singleton chunks over
        # multiple chains. Unequal sample counts within two chains are fine.
        for (first_count, second_count) in ((1, 2), (2, 1))
            path = joinpath(dir, "bad-chunks-$first_count-$second_count.jld2")
            CMBLensing.jldopen(path, "w") do io
                write(io, "chunks_1", [Any[Dict(:step=>1)] for _ in 1:first_count])
                write(io, "chunks_2", [Any[Dict(:step=>2)] for _ in 1:second_count])
            end
            @test_throws ArgumentError load_chains(path; progress=false, unbatch=false)
        end
        unequal_file = joinpath(dir, "unequal-samples.jld2")
        CMBLensing.jldopen(unequal_file, "w") do io
            write(io, "chunks_1", [Any[Dict(:step=>1)], Any[Dict(:step=>1), Dict(:step=>2)]])
            write(io, "chunks_2", [Any[Dict(:step=>2)], Any[Dict(:step=>3)]])
        end
        unequal = load_chains(unequal_file; progress=false, unbatch=false)
        @test [[s[:step] for s in chain] for chain in unequal] == [[1,2], [1,2,3]]

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
