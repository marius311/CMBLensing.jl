using CMBLensing, LinearAlgebra, Random, Serialization, Test
using Distributed: RemoteChannel, myid

@testset "MAP_marg and dataset transport" begin
    sim = load_sim(; Cℓ=CMBLensing._default_Cℓs(), pol=:I, T=Float64,
        θpix=4, Nside=8, storage=Array, L=LenseFlow(3), rng=MersenneTwister(7))
    global map_marg_test_dataset = sim.ds()
    map_marg_test_dataset.G = 2I
    map_marg_test_dataset.L = precompute!!(LenseFlow(sim.ϕ, 3), sim.f)
    # This deliberately stateful, host-dependent operator must be evaluated
    # exactly once on the originating process per solve.
    calls = Ref(0)
    base_cf = map_marg_test_dataset.Cf
    map_marg_test_dataset.Cf = ParamDependentOp([:test_amp], function (; test_amp=1, _...)
        calls[] += 1
        (test_amp + 0.01 * (myid()-1)) * base_cf
    end)
    calls[] = 0
    original_lens = map_marg_test_dataset.L
    original_potential = original_lens.ϕ[]
    original_hash = hash(map_marg_test_dataset)
    CMBLensing.@distributed map_marg_test_dataset
    try
        # A new object may have the same hash as a registered object. It must
        # serialize by value rather than being confused with that registration.
        clone = copy(map_marg_test_dataset)
        @test clone !== map_marg_test_dataset
        @test hash(clone) == original_hash
        io = IOBuffer()
        serialize(io, clone)
        seekstart(io)
        @test hash(deserialize(io)) == original_hash

        ϕ, trace = MAP_marg((; test_amp=1.0), map_marg_test_dataset; rng=MersenneTwister(12),
            ϕstart=0.3sim.ϕ, nsteps=2, nsteps_with_meanfield_update=1,
            Nsims=1, progress=false, pmap=map, conjgrad_kwargs=(tol=1e-4, nsteps=100))
        @test length(trace) == 2
        @test all(isfinite, ϕ[:])
        @test all(t -> all(isfinite, t.g[:]), trace)
        @test hash(map_marg_test_dataset) == original_hash
        @test map_marg_test_dataset.G == 2I
        @test calls[] == 1
        @test map_marg_test_dataset.L === original_lens
        @test original_lens.ϕ[] === original_potential
        @test !original_lens.needs_precompute[]

        # Repeated solves have independent caches and deterministic results.
        ϕ_again, trace_again = MAP_marg((; test_amp=1.0), map_marg_test_dataset; rng=MersenneTwister(12),
            ϕstart=0.3sim.ϕ, nsteps=2, nsteps_with_meanfield_update=1,
            Nsims=1, progress=false, pmap=map, conjgrad_kwargs=(tol=1e-4, nsteps=100))
        @test isapprox(ϕ_again.arr, ϕ.arr; rtol=1e-10, atol=1e-12)
        @test all(isapprox(a.g.arr, b.g.arr; rtol=1e-10, atol=1e-12) for (a, b) in zip(trace_again, trace))
        @test calls[] == 2

        # Tasks sharing one worker must not use its mutable lens scratch at
        # the same time. Verify reuse, exclusion, and explicit release.
        registry = CMBLensing._map_marg_new_registry(map_marg_test_dataset)
        cache = CMBLensing._map_marg_worker_cache(registry)
        @test fetch(cache).ds !== map_marg_test_dataset
        @test fetch(cache).ds.L !== original_lens
        active = Ref(0)
        peak_active = Ref(0)
        lens_ids = UInt[]
        try
            @sync for _ in 1:3
                @async CMBLensing._map_marg_with_worker_cache(registry) do worker_ds
                    active[] += 1
                    peak_active[] = max(peak_active[], active[])
                    push!(lens_ids, objectid(worker_ds.L))
                    yield()
                    sleep(0.01)
                    active[] -= 1
                end
            end
            @test peak_active[] == 1
            @test length(unique(lens_ids)) == 1
            @test only(unique(lens_ids)) != objectid(original_lens)
            @test CMBLensing._map_marg_worker_cache(registry) == cache
        finally
            CMBLensing._map_marg_close_registry(registry)
        end
        @test !isopen(registry)
        @test !isopen(cache)

        # Unknown lens types take the conservative copy path even when the
        # cache is initialized away from the dataset's originating process.
        custom = copy(map_marg_test_dataset)
        custom.L = Ref([1.0])
        custom_cache = CMBLensing._map_marg_new_worker_cache(custom, myid()+1)
        try
            @test fetch(custom_cache).ds !== custom
            @test fetch(custom_cache).ds.L[] !== custom.L[]
        finally
            CMBLensing._map_marg_close_worker_cache(custom_cache)
        end

        # A mapper failure after one completed job must still release both
        # the owner registry and the already-created worker cache.
        released = Ref{Any}(nothing)
        function failing_map(f, indices, data, starts)
            f(first(indices), first(data), first(starts))
            refs = filter(x -> x isa RemoteChannel, [getfield(f, name) for name in fieldnames(typeof(f))])
            state = only(refs)
            released[] = (state, CMBLensing._map_marg_worker_cache(state))
            error("injected mapper failure")
        end
        err = try
            MAP_marg((; test_amp=1.0), map_marg_test_dataset; rng=MersenneTwister(12),
                ϕstart=0.3sim.ϕ, nsteps=1, Nsims=1, progress=false,
                pmap=failing_map, conjgrad_kwargs=(tol=1e-4, nsteps=100))
            nothing
        catch e
            e
        end
        @test err isa ErrorException && err.msg == "injected mapper failure"
        @test released[] !== nothing
        if released[] !== nothing
            @test !isopen(released[][1])
            @test !isopen(released[][2])
        end
        @test hash(map_marg_test_dataset) == original_hash
        @test original_lens.ϕ[] === original_potential

        # Cleanup must wait for a concurrent task that is using the cache,
        # without replacing the exception thrown by the user's mapper.
        active_failure = Ref{Any}(nothing)
        function failing_concurrent_map(f, indices, data, starts)
            state = only(filter(x -> x isa RemoteChannel,
                [getfield(f, name) for name in fieldnames(typeof(f))]))
            started, release = Channel{Bool}(1), Channel{Bool}(1)
            work = @async CMBLensing._map_marg_with_worker_cache(state) do worker_ds
                put!(started, true)
                take!(release)
                :finished
            end
            take!(started)
            cache = CMBLensing._map_marg_worker_cache(state)
            release_task = @async begin
                sleep(0.01)
                put!(release, true)
            end
            active_failure[] = (; state, cache, work, release_task)
            error("injected concurrent mapper failure")
        end
        outcome = @async try
            MAP_marg((; test_amp=1.0), map_marg_test_dataset; rng=MersenneTwister(12),
                ϕstart=0.3sim.ϕ, nsteps=1, Nsims=1, progress=false,
                pmap=failing_concurrent_map, conjgrad_kwargs=(tol=1e-4, nsteps=100))
            nothing
        catch e
            e
        end
        status = timedwait(() -> istaskdone(outcome), 180; pollint=0.1)
        @test status == :ok
        if status == :ok
            concurrent_error = fetch(outcome)
            @test concurrent_error isa ErrorException && concurrent_error.msg == "injected concurrent mapper failure"
            @test active_failure[] !== nothing
            if active_failure[] !== nothing
                @test fetch(active_failure[].work) == :finished
                @test istaskdone(active_failure[].release_task)
                @test !isopen(active_failure[].state)
                @test !isopen(active_failure[].cache)
            end
        end
    finally
        delete!(CMBLensing.distributed_datasets.name_hash, :map_marg_test_dataset)
        delete!(CMBLensing.distributed_datasets.objid_hash, objectid(map_marg_test_dataset))
    end
end
