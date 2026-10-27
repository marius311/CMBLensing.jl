using CMBLensing, CUDA, Adapt, LinearAlgebra, Random, Serialization, Test, Zygote

function graph_test_fields(T, Ny, Nx, pol; batched=false)
    θpix = 60 * 180 / π
    make(fun) = T[fun(2π*(i-1)/Ny, 2π*(j-1)/Nx) for i=1:Ny, j=1:Nx]
    ϕ = FlatMap(make((x,y) -> 0.02sin(x+2y) + 0.01cos(2x-y)); θpix)
    a = make((x,y) -> sin(x+y) + 0.3cos(2x-y))
    b = make((x,y) -> cos(x-y) + 0.2sin(x+2y))
    f = pol == :T ? FlatMap(a; θpix) :
        pol == :P ? FlatQUMap(a, b; θpix) : FlatIQUMap(a, b, a+b; θpix)
    gpu(ϕ), gpu(batched ? batch(f, T(0.7)*f) : f)
end

@testset "CUDA fused spin contraction" begin
    for T in (Float32, Float64), pol in (:T, :P, :TP), batched in (false, true)
        ϕ, f = graph_test_fields(T, 9, 11, pol; batched)
        b = T(0.7)*f
        expected = CMBLensing.spin_adjoint(f) * b
        dst = similar(expected)
        mul!(dst, CMBLensing.spin_adjoint(f), b)
        @test Array(dst.arr) ≈ Array(expected.arr) rtol=5eps(T) atol=5eps(T)
    end
    # Broadcasted batches and noncontiguous arrays keep the generic semantics.
    a = CUDA.rand(Float64, 5, 7, 2, 1)
    b = CUDA.rand(Float64, 5, 7, 2, 3)
    dst = CUDA.zeros(Float64, 5, 7, 1, 3)
    CMBLensing._lambert_spin_mul!(dst, a, b)
    @test Array(dst) ≈ sum(Array(a).*Array(b); dims=3)
    av = view(a, 1:2:5, :, :, :)
    bv = view(b, 1:2:5, :, :, :)
    dv = similar(dst, 3, 7, 1, 3)
    CMBLensing._lambert_spin_mul!(dv, av, bv)
    @test Array(dv) ≈ sum(Array(av).*Array(bv); dims=3)
end

@testset "CUDA LenseFlow graph replay" begin
    ops = ((L,f) -> L*f, (L,f) -> L\f, (L,f) -> L'*f, (L,f) -> L'\f)
    @testset "$T, $Ny×$Nx, $pol, batch=$batched" for T in (Float32, Float64),
            (Ny,Nx) in ((8,10),(9,11)), pol in (:T,:P,:TP), batched in (false,true)
        ϕ, f = graph_test_fields(T, Ny, Nx, pol; batched)
        before = (Array(ϕ.arr), Array(f.arr))
        base = precompute!!(LenseFlow(ϕ, 3), f)
        graph = precompute!!(LenseFlow(ϕ, 3; gpu_graphs=true), f)
        tol = T == Float32 ? 2e-5 : 2e-12
        for op in ops
            first_result = op(graph, f)
            retained = Array(first_result.arr)
            @test retained ≈ Array(op(base, f).arr) rtol=tol atol=tol
            @test Array(op(graph, T(0.9)*f).arr) ≈ Array(op(base, T(0.9)*f).arr) rtol=tol atol=tol
            @test Array(first_result.arr) == retained
        end
        @test !isnothing(graph.odesolve.graphs.state)
        @test length(graph.odesolve.graphs.state.entries) == 4
        @test all(!isnothing(e) for e in values(graph.odesolve.graphs.state.entries))

        # A new potential updates coefficient VALUES in the retained buffers.
        graph(T(1.1)*ϕ); base(T(1.1)*ϕ)
        @test Array((graph*f).arr) ≈ Array((base*f).arr) rtol=tol atol=tol
        @test length(graph.odesolve.graphs.state.entries) == 4
        @test Array(ϕ.arr) == before[1]
        @test Array(f.arr) == before[2]

        # Exercise both directions of the joint (f, δf, δϕ) integration.
        # Independent numerical derivative checks below use more ODE steps.
        for inverse in (false,true)
            objective(L, ϕ, f) = sum(unbatch(dot(f, inverse ? L(ϕ)\f : L(ϕ)*f)))
            gb = Zygote.withgradient((ϕ,f) -> objective(base,ϕ,f), ϕ, f)
            gg = Zygote.withgradient((ϕ,f) -> objective(graph,ϕ,f), ϕ, f)
            @test gg.val ≈ gb.val rtol=tol atol=tol
            for (a,b) in zip(gg.grad, gb.grad)
                @test Array(a.arr) ≈ Array(b.arr) rtol=tol atol=tol
            end
        end
        @test length(graph.odesolve.graphs.state.entries) == 6
        @test all(!isnothing(e) for e in values(graph.odesolve.graphs.state.entries))
    end
end

@testset "CUDA graph cache ownership" begin
    ϕ, f = graph_test_fields(Float32, 9, 11, :P)
    base = precompute!!(LenseFlow(ϕ, 3), f)
    graph = precompute!!(LenseFlow(ϕ, 3; gpu_graphs=true), f)
    graph*f
    owner = graph.odesolve.graphs.state
    f64 = map(Float64, f)
    for α in (0.9f0,1.1f0,0.95f0)
        base(α*ϕ); graph(α*ϕ)
        @test Array((graph*f64).arr) ≈ Array((base*f64).arr) rtol=1e-11 atol=1e-11
        # A Float64 incoming tangent creates another temporary cache. It must
        # use that cache's coefficients, even when the primal was Float32.
        _, pb = Zygote.pullback((ϕ,f) -> base(ϕ)*f, α*ϕ, f)
        _, pg = Zygote.pullback((ϕ,f) -> graph(ϕ)*f, α*ϕ, f)
        for (a,b) in zip(pg(f64), pb(f64))
            @test Array(a.arr) ≈ Array(b.arr) rtol=1e-10 atol=1e-10
        end
    end
    @test graph.odesolve.graphs.state === owner
    @test length(owner.entries) == 1 # No accumulation of promoted caches.

    # A batch/spin change must build a different cache even at equal precision.
    fb = batch(f, 0.8f0*f)
    wider = precompute!!(graph, fb)
    @test wider !== graph
    @test isnothing(wider.odesolve.graphs.state)
    @test Array((wider*fb).arr) ≈ Array((LenseFlow(CMBLensing.getϕ(graph),3)*fb).arr)
    @test graph.odesolve.graphs.state === owner

    # Sequential stream/task transfers are ordered and recaptured. Concurrent
    # callers should have independent caches, as for ordinary CachedLenseFlow.
    CUDA.stream!(CUDA.CuStream()) do
        @test Array((graph*f).arr) ≈ Array((base*f).arr)
        @test graph.odesolve.graphs.state !== owner
    end
    @test Array((graph*f).arr) ≈ Array((base*f).arr)
    previous = graph.odesolve.graphs.state
    fetch(@async begin
        @test Array((graph*f).arr) ≈ Array((LenseFlow(CMBLensing.getϕ(graph),3)*f).arr)
        @test graph.odesolve.graphs.state !== previous
    end)

    CUDA.synchronize()
    for copied in (adapt(CuArray, graph), deepcopy(graph))
        @test isnothing(copied.odesolve.graphs.state)
        @test Array((copied*f).arr) ≈ Array((base*f).arr)
    end
    io = IOBuffer(); serialize(io, graph); seekstart(io)
    restored = deserialize(io)
    @test isnothing(restored.odesolve.graphs.state)
    @test Array((restored*f).arr) ≈ Array((base*f).arr)
    cpu_graph = cpu(graph)
    @test isnothing(cpu_graph.odesolve.graphs.state)
    @test (cpu_graph*cpu(f)).arr ≈ Array((base*f).arr)
end

@testset "CUDA graph directional derivatives" begin
    ϕ, f = graph_test_fields(Float64, 16, 16, :P)
    direction = 0.7ϕ
    L = precompute!!(LenseFlow(ϕ, 32; gpu_graphs=true), f)
    for inverse in (false,true)
        objective(α) = dot(f, inverse ? L(ϕ+α*direction)\f : L(ϕ+α*direction)*f)
        ad = gradient(objective, 0.0)[1]
        h = 1e-3
        fd = (objective(h)-objective(-h))/(2h)
        @test abs(fd) > 1e-5
        @test ad ≈ fd atol=2e-6 rtol=2e-5
    end
end

@testset "CUDA graph mixed likelihood and CPU parameters" begin
    for (T,θT) in ((Float32,Float32),(Float32,Float64),(Float64,Float64))
        N = 32
        sim = load_sim(; storage=CuArray,T,Nside=N,θpix=3,pol=:P,μKarcminT=1,beamFWHM=1,
            pixel_mask_kwargs=(edge_padding_deg=0.15*3N/60,apodization_deg=0,num_ptsrcs=0),
            seed=17,L=LenseFlow(3),Cℓ=CMBLensing._default_Cℓs())
        ds = sim.ds
        (;f°,ϕ°) = mix(ds;sim.f,sim.ϕ)
        θ = CMBLensing.ComponentVector(r=log(θT(0.2)),Aϕ=log(θT(1)))
        x = LenseBasis(FieldTuple(;f°,ϕ°,θ))
        before = copy(x)
        objective(x) = logpdf(Mixed(ds);x.f°,x.ϕ°,θ=exp.(x.θ))+sum(x.θ)
        base = ds.L
        graph = precompute!!(LenseFlow(CMBLensing.getϕ(base),3;gpu_graphs=true), f°)
        tol = T == Float32 ? 3e-5 : 1e-10
        retained = nothing
        for α in (T(1),0.98,1.02)
            xα = α*x
            ds.L = base
            vb = objective(xα)
            gb = Zygote.withgradient(objective,xα)
            ds.L = graph
            vg = objective(xα)
            gg = Zygote.withgradient(objective,xα)
            @test vg ≈ vb rtol=tol
            # The existing logdet adjoint has its own primal normalization;
            # compare each interface to the corresponding ordinary path.
            @test gg.val ≈ gb.val rtol=tol
            for k in (:f°,:ϕ°,:θ)
                a,b = getproperty(gg.grad[1],k), getproperty(gb.grad[1],k)
                @test norm(a-b) <= tol*max(norm(b),eps(T))
            end
            isnothing(retained) && (retained = (gg.grad[1],copy(gg.grad[1])))
        end
        @test norm(retained[1]-retained[2]) == 0
        @test norm(x-before) == 0
    end
end
