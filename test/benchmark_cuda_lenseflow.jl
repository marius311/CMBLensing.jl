# Run explicitly in a CUDA environment, e.g.:
#   julia --project=test test/benchmark_cuda_lenseflow.jl 128 256 512
# Compilation, simulation, first capture, and profiler overhead are excluded.
# Every row is an alternating, synchronized pair; normal GC remains enabled.
using CMBLensing, CUDA, LinearAlgebra, Statistics, Zygote

CUDA.allowscalar(false)
BLAS.set_num_threads(1)

function benchmark_fixture(N)
    sim = load_sim(;storage=CuArray,T=Float64,Nside=N,θpix=3,pol=:P,
        μKarcminT=1,beamFWHM=1,seed=17,L=LenseFlow(7),Cℓ=CMBLensing._default_Cℓs(),
        pixel_mask_kwargs=(edge_padding_deg=0.15*3N/60,apodization_deg=0,num_ptsrcs=0))
    ds = sim.ds
    (;f°,ϕ°) = mix(ds;sim.f,sim.ϕ)
    θ = CMBLensing.ComponentVector(r=log(0.2),Aϕ=0.0)
    x = LenseBasis(FieldTuple(;f°,ϕ°,θ))
    objective(x) = logpdf(Mixed(ds);x.f°,x.ϕ°,θ=exp.(x.θ))+sum(x.θ)
    ordinary = ds.L
    graph = precompute!!(LenseFlow(CMBLensing.getϕ(ordinary),7;gpu_graphs=true), f°)
    value(L) = (ds.L=L; objective(x))
    grad(L) = (ds.L=L; Zygote.withgradient(objective,x))
    (;value,grad,ordinary,graph)
end

function elapsed_ms(call)
    CUDA.synchronize()
    t = time_ns()
    call()
    CUDA.synchronize()
    (time_ns()-t)/1e6
end

function benchmark_pairs(round, N, mode, ordinary, graph; samples=21)
    for _ in 1:3
        ordinary(); graph()
    end
    CUDA.synchronize(); GC.gc()
    a,b = Float64[],Float64[]
    for sample in 1:samples
        if isodd(sample)
            ta=elapsed_ms(ordinary); tb=elapsed_ms(graph)
        else
            tb=elapsed_ms(graph); ta=elapsed_ms(ordinary)
        end
        push!(a,ta); push!(b,tb)
        println("SAMPLE,$round,$N,$mode,$sample,$ta,$tb")
    end
    println("MEDIAN,$round,$N,$mode,",median(a),",",median(b),",",median(a)/median(b))
    flush(stdout)
end

sizes = isempty(ARGS) ? [128,256,512] : parse.(Int,ARGS)
println("Julia ",VERSION,"; source ",pathof(CMBLensing))
CUDA.versioninfo()
fixtures = Dict(N => benchmark_fixture(N) for N in sizes)
for (round, order) in enumerate((sizes,reverse(sizes)))
    for N in order
        r = fixtures[N]
        for (mode,call) in ((:likelihood,r.value),(:value_gradient,r.grad))
            benchmark_pairs(round,N,mode,()->call(r.ordinary),()->call(r.graph))
        end
    end
end
