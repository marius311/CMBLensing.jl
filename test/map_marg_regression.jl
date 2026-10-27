using CMBLensing, LinearAlgebra, Random, Serialization, Test
using Distributed: RemoteChannel, WorkerPool, addprocs, myid, pmap, remotecall_fetch, remotecall_wait, rmprocs

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
    for (updates, nsims, parameter) in ((0, 1, "nsteps_with_meanfield_update"), (1, 0, "Nsims"))
        err = try
            MAP_marg((; test_amp=1.0), map_marg_test_dataset; nsteps=1,
                nsteps_with_meanfield_update=updates, Nsims=nsims, progress=false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin(parameter, err.msg)
    end
    for (options, parameter) in (((; nsteps=0.5), "nsteps"),
                                 ((; nsteps_with_meanfield_update=0.5), "nsteps_with_meanfield_update"),
                                 ((; Nsims=0.5), "Nsims"),
                                 ((; nsteps=-1), "nsteps"),
                                 ((; nsteps_with_meanfield_update=-1), "nsteps_with_meanfield_update"),
                                 ((; Nsims=-1), "Nsims"))
        err = try
            MAP_marg((; test_amp=1.0), map_marg_test_dataset; options..., progress=false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin(parameter, err.msg)
    end
    zero_start = 0.3sim.ϕ
    zero_ϕ, zero_trace = MAP_marg((; test_amp=1.0), map_marg_test_dataset;
        ϕstart=zero_start, nsteps=0, nsteps_with_meanfield_update=0, Nsims=0, progress=false)
    @test zero_ϕ === zero_start && isempty(zero_trace)
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
            Nsims=1, progress=true, pmap=map, conjgrad_kwargs=(tol=1e-4, nsteps=100))
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

        # A mapper error after a real job is propagated with progress enabled.
        function failing_map(f, indices, data, starts)
            f(first(indices), first(data), first(starts))
            error("injected mapper failure")
        end
        err = try
            MAP_marg((; test_amp=1.0), map_marg_test_dataset; rng=MersenneTwister(12),
                ϕstart=0.3sim.ϕ, nsteps=1, Nsims=1, progress=true,
                pmap=failing_map, conjgrad_kwargs=(tol=1e-4, nsteps=100))
            nothing
        catch e
            e
        end
        @test err isa ErrorException && err.msg == "injected mapper failure"
        @test hash(map_marg_test_dataset) == original_hash
        @test original_lens.ϕ[] === original_potential

        # Hold an actual CG solve in its documented callback while the mapper
        # throws. Cleanup must wait for the task, including its progress update.
        started, release = Channel{Bool}(1), Channel{Bool}(1)
        entered = Ref(false)
        active_failure = Ref{Any}(nothing)
        function hold_first_iteration(i, x, res)
            if !entered[]
                entered[] = true
                put!(started, true)
                take!(release)
            end
        end
        function failing_concurrent_map(f, indices, data, starts)
            work = @async f(first(indices), first(data), first(starts))
            status = timedwait(() -> isready(started) || istaskdone(work), 180; pollint=0.1)
            status == :ok && isready(started) || error("solver task did not enter callback")
            take!(started)
            release_task = @async begin
                sleep(0.01)
                put!(release, true)
            end
            active_failure[] = (; work, release_task)
            error("injected concurrent mapper failure")
        end
        outcome = @async try
            MAP_marg((; test_amp=1.0), map_marg_test_dataset; rng=MersenneTwister(12),
                ϕstart=0.3sim.ϕ, nsteps=1, Nsims=1, progress=true,
                pmap=failing_concurrent_map,
                conjgrad_kwargs=(tol=1e-4, nsteps=100, callback=hold_first_iteration))
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
                @test fetch(active_failure[].work) isa NamedTuple
                @test istaskdone(active_failure[].release_task)
            end
        end

        @testset "actual worker transport and cleanup" begin
            workers = Int[]
            try
                # Use the active Pkg.test environment and avoid starting a
                # worker with the host's full default thread count.
                flags = Cmd(["--startup-file=no", "--threads=1", "--project=$(dirname(Base.active_project()))"])
                worker_env = ["JULIA_NUM_THREADS" => "1", "JULIA_NUM_PRECOMPILE_TASKS" => "1",
                              "OPENBLAS_NUM_THREADS" => "1", "FFTW_NUM_THREADS" => "1", "OMP_NUM_THREADS" => "1"]
                append!(workers, addprocs(1; exeflags=flags, env=worker_env))
                worker = only(workers)
                remotecall_wait(Core.eval, worker, Main, :(using CMBLensing, LinearAlgebra))

                # Exercise the full solver through an actual worker without
                # relying on the layout of its compiler-generated closure.
                function worker_map(f, indices, data, starts)
                    map(indices, data, starts) do i, d, start
                        remotecall_fetch(f, worker, i, d, start)
                    end
                end
                calls_before = calls[]
                worker_ϕ, worker_trace = MAP_marg((; test_amp=1.0), map_marg_test_dataset;
                    rng=MersenneTwister(12), ϕstart=0.3sim.ϕ, nsteps=1, Nsims=1,
                    progress=false, pmap=worker_map, conjgrad_kwargs=(tol=1e-4, nsteps=100))
                @test calls[] == calls_before + 1
                @test isapprox(worker_ϕ.arr, trace[1].ϕ.arr; rtol=1e-10, atol=1e-12)
                @test isapprox(worker_trace[1].g.arr, trace[1].g.arr; rtol=1e-10, atol=1e-12)

                # Check both the solver's default pool and a batched mapper;
                # the latter may run multiple tasks on the same worker cache.
                batched_pool = WorkerPool(workers)
                batched_map(f, args...) = pmap(f, batched_pool, args...; batch_size=2)
                for mapper in (nothing, batched_map)
                    calls_before = calls[]
                    mapper_kwargs = mapper === nothing ? (;) : (; pmap=mapper)
                    distributed_ϕ, distributed_trace = MAP_marg((; test_amp=1.0), map_marg_test_dataset;
                        rng=MersenneTwister(12), ϕstart=0.3sim.ϕ, nsteps=1, Nsims=1,
                        progress=false, conjgrad_kwargs=(tol=1e-4, nsteps=100), mapper_kwargs...)
                    @test calls[] == calls_before + 1
                    @test isapprox(distributed_ϕ.arr, trace[1].ϕ.arr; rtol=1e-10, atol=1e-12)
                    @test isapprox(distributed_trace[1].g.arr, trace[1].g.arr; rtol=1e-10, atol=1e-12)
                end

                function worker_fail_map(f, indices, data, starts)
                    remotecall_fetch(f, worker, first(indices), first(data), first(starts))
                    error("injected worker mapper failure")
                end
                calls_before = calls[]
                failure = try
                    MAP_marg((; test_amp=1.0), map_marg_test_dataset; rng=MersenneTwister(12),
                        ϕstart=0.3sim.ϕ, nsteps=1, Nsims=1, progress=false,
                        pmap=worker_fail_map, conjgrad_kwargs=(tol=1e-4, nsteps=100))
                    nothing
                catch e
                    e
                end
                @test failure isa ErrorException && failure.msg == "injected worker mapper failure"
                @test calls[] == calls_before + 1

                # Directly check that a caller-evaluated dataset is sent once
                # and reused, then both channels close, on the real worker.
                calls_before = calls[]
                evaluated = map_marg_test_dataset((; test_amp=1.0))
                @test calls[] == calls_before + 1
                registry = CMBLensing._map_marg_new_registry(evaluated)
                cache = nothing
                try
                    seen = map(1:2) do _
                        remotecall_fetch(worker, registry) do r
                            CMBLensing._map_marg_with_worker_cache(r) do ds
                                (; cf=sum(real.(diag(ds.Cf).arr)), lens=objectid(ds.L))
                            end
                        end
                    end
                    cache = fetch(registry).caches[worker]
                    @test seen[1].lens == seen[2].lens
                    @test all(x -> isapprox(x.cf, sum(real.(diag(base_cf).arr))), seen)
                    @test length(fetch(registry).caches) == 1
                finally
                    CMBLensing._map_marg_close_registry(registry)
                end
                @test !isopen(registry)
                @test cache !== nothing && !isopen(cache)

                # A failed cache must not prevent later workers' cleanup.
                registry = CMBLensing._map_marg_new_registry(evaluated)
                CMBLensing._map_marg_worker_cache(registry)
                remotecall_fetch(CMBLensing._map_marg_worker_cache, worker, registry)
                cache_dict = fetch(registry).caches
                cache_order = collect(keys(cache_dict))
                bad_cache, good_cache = cache_dict[first(cache_order)], cache_dict[last(cache_order)]
                take!(bad_cache)
                close(bad_cache)
                release_error = @test_logs (:warn, r"Could not release MAP_marg worker cache") begin
                    try
                        CMBLensing._map_marg_close_registry(registry)
                        nothing
                    catch e
                        e
                    end
                end
                @test release_error !== nothing
                @test !isopen(registry)
                @test !isopen(good_cache)

                # Removing an actual worker after its first job makes cache
                # cleanup fail. The original mapper error must still escape.
                function removed_worker_map(f, indices, data, starts)
                    remotecall_fetch(f, worker, first(indices), first(data), first(starts))
                    rmprocs(worker)
                    empty!(workers)
                    error("mapper error after worker removal")
                end
                removed_error = @test_logs (:warn, r"Could not release MAP_marg worker cache") (:warn, r"Could not release MAP_marg worker registry") begin
                    try
                        MAP_marg((; test_amp=1.0), map_marg_test_dataset; rng=MersenneTwister(12),
                            ϕstart=0.3sim.ϕ, nsteps=1, Nsims=1, progress=false,
                            pmap=removed_worker_map, conjgrad_kwargs=(tol=1e-4, nsteps=100))
                        nothing
                    catch e
                        e
                    end
                end
                @test removed_error isa ErrorException && removed_error.msg == "mapper error after worker removal"
            finally
                isempty(workers) || rmprocs(workers)
            end
        end
    finally
        delete!(CMBLensing.distributed_datasets.name_hash, :map_marg_test_dataset)
        delete!(CMBLensing.distributed_datasets.objid_hash, objectid(map_marg_test_dataset))
    end
end
