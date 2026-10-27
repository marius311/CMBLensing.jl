using CMBLensing
using CMBLensing: jldopen
using LinearAlgebra
using Test

@testset "load_chains" begin
    function save_chunks(path, chunks)
        jldopen(path, "w") do io
            write(io, "metadata", "not a chunk")
            for (key, value) in chunks
                write(io, key, value)
            end
        end
    end

    sample(i) = Dict{Symbol,Any}(:step => i, :logpdf => Float64(i))
    chunk(chain_steps...) = [Any[sample(i) for i in steps] for steps in chain_steps]

    mktempdir() do dir
        @testset "no chunks and empty saved chunk" begin
            no_chunks = joinpath(dir, "no_chunks.jld2")
            save_chunks(no_chunks, [])
            for unbatch in (false, true), join in (false, true)
                chains = load_chains(no_chunks; unbatch, join, progress=false)
                if join
                    @test chains isa CMBLensing.Chain
                    @test isempty(chains)
                else
                    @test chains isa CMBLensing.Chains
                    @test length(chains) == 1
                    @test isempty(chains[1])
                end
            end

            empty_chunk = joinpath(dir, "empty_chunk.jld2")
            save_chunks(empty_chunk, ["chunks_1" => [Any[], Any[]]])
            for unbatch in (false, true)
                chains = load_chains(empty_chunk; unbatch, progress=false)
                @test length(chains) == 2
                @test all(isempty, chains)
                @test isempty(load_chains(empty_chunk; unbatch, join=true, progress=false))
            end

            followed_by_data = joinpath(dir, "empty_then_data.jld2")
            save_chunks(followed_by_data, [
                "chunks_1" => [Any[], Any[]],
                "chunks_2" => chunk(1:2, 11:12),
            ])
            @test load_chains(followed_by_data; progress=false)[:step] == [[1, 2], [11, 12]]
        end

        @testset "one or more chunks" begin
            single = joinpath(dir, "single.jld2")
            save_chunks(single, ["chunks_1" => chunk(1:2, 11:12)])
            @test load_chains(single; progress=false)[:step] == [[1, 2], [11, 12]]
            @test load_chains(single; join=true, progress=false)[:step] == [1, 2, 11, 12]

            multiple = joinpath(dir, "multiple.jld2")
            # Write out of order to check that chunk suffixes are sorted numerically.
            save_chunks(multiple, [
                "chunks_10" => chunk(5:6, 15:16),
                "chunks_1"  => chunk(1:2, 11:12),
                "chunks_2"  => chunk(3:4, 13:14),
            ])
            @test load_chains(multiple; progress=false)[:step] == [collect(1:6), collect(11:16)]
            @test load_chains(multiple; progress=true)[:step] == [collect(1:6), collect(11:16)]
            @test load_chains(multiple; burnin_chunks=1, progress=false)[:step] == [collect(3:6), collect(13:16)]
            @test load_chains(multiple; burnin_chunks=-1, progress=false)[:step] == [[5, 6], [15, 16]]
            @test load_chains(multiple; burnin_chunks=-2, progress=false)[:step] == [collect(3:6), collect(13:16)]
            no_selected_chunks = load_chains(multiple; burnin_chunks=3, progress=false)
            @test length(no_selected_chunks) == 1 && isempty(no_selected_chunks[1])
            @test load_chains(multiple; burnin=1, thin=2, join=true, progress=false)[:step] == [2, 4, 6, 12, 14, 16]
            @test load_chains(multiple; burnin=-2, progress=false)[:step] == [[5, 6], [15, 16]]
        end

        @testset "dropping maps" begin
            mapped_sample(i) = merge(sample(i), Dict(:map => CMBLensing.FlatMap(fill(Float64(i), 2, 2))))
            mapped = joinpath(dir, "mapped.jld2")
            save_chunks(mapped, [
                "chunks_1" => [Any[mapped_sample(1)]],
                "chunks_2" => [Any[mapped_sample(2)]],
            ])
            retained = load_chains(mapped; progress=false)
            @test all(s -> s[:map] isa CMBLensing.Field, retained[1])
            dropped = load_chains(mapped; dropmaps=true, progress=false)
            @test dropped[:step] == [[1, 2]]
            @test all(s -> !haskey(s, :map), dropped[1])
        end

        @testset "batched samples" begin
            batched_sample(i, j) = Dict{Symbol,Any}(
                :step => batch([i, j]), :logpdf => batch([Float64(i), Float64(j)]),
            )
            batched = joinpath(dir, "batched.jld2")
            save_chunks(batched, [
                "chunks_1" => [Any[batched_sample(1, 11)], Any[batched_sample(101, 111)]],
                "chunks_2" => [Any[batched_sample(2, 12)], Any[batched_sample(102, 112)]],
            ])
            kept = load_chains(batched; unbatch=false, progress=false)
            @test length(kept) == 2
            @test [[batch_index(x, 1) for x in chain] for chain in kept[:step]] == [[1, 2], [101, 102]]
            @test [[batch_index(x, 2) for x in chain] for chain in kept[:step]] == [[11, 12], [111, 112]]
            expanded = load_chains(batched; unbatch=true, progress=false)
            @test expanded[:step] == [[1, 2], [11, 12], [101, 102], [111, 112]]
            @test load_chains(batched; unbatch=true, join=true, progress=false)[:step] == [1, 2, 11, 12, 101, 102, 111, 112]
            joined = load_chains(batched; unbatch=false, join=true, progress=false)
            @test batch_index.(joined[:step], 1) == [1, 2, 101, 102]
            @test batch_index.(joined[:step], 2) == [11, 12, 111, 112]
        end
    end
end

@testset "sample_joint resume" begin
    ds = CMBLensing.NoLensingDataSet(; d=FlatMap(zeros(2,2)), Cf=I, Cn=I)
    initialize! = (state, _) -> get!(state, :counter, 1)
    sample! = (state, _) -> (state[:counter] += 1)
    options = (; nsamps_per_chain=2, nchains=1, pmap=map, storage=Array,
        gibbs_initializers=[initialize!], gibbs_samplers=[sample!], progress=false)

    chain = sample_joint(ds; options...)
    @test [state[:counter] for state in chain[1]] == [1, 2]

    mktempdir() do dir
        path = joinpath(dir, "chain.jld2")
        jldopen(path, "w") do io
            write(io, "rundat", Dict())
            write(io, "chunks_1", [Any[Dict(:step => 1, :counter => 7)]])
        end
        @test_throws ErrorException sample_joint(ds; options..., filename=path)
        restarted = sample_joint(ds; options..., filename=path, resume=false)
        @test [state[:counter] for state in restarted[1]] == [1, 2]
        resumed = sample_joint(ds; options..., filename=path, resume=true)
        @test [state[:counter] for state in resumed[1]] == [8]
    end
end
