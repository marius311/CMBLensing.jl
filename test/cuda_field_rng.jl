using CMBLensing, CUDA, Adapt, GPUArrays, LinearAlgebra, Random, Test

@testset "CUDA.RNG field dispatch" begin
    proj=ProjLambert(;Ny=8,Nx=8,θpix=60*180/π,T=Float32)
    map_cpu()=FlatMap(zeros(Float32,8,8),proj)
    map_gpu()=adapt(CuArray,map_cpu())
    # Some older CUDA releases do not provide a CPU-array randn! fallback.
    # GPUArrays 12 defines it in GPUArrays instead of CUDA.
    cpu_fallback = which(randn!, (CUDA.RNG, Matrix{Float32})).module in (CUDA, GPUArrays)
    if !cpu_fallback
        @info "Skipping CPU-backed CUDA.RNG checks: CUDA has no CPU-array randn! fallback"
        @test_skip cpu_fallback
    end
    for T in (Float32,Float64), usegpu in (cpu_fallback ? (false,true) : (true,)), polarized in (false,true)
        p=ProjLambert(;Ny=8,Nx=8,θpix=60*180/π,T)
        host=polarized ? FlatQUMap(zeros(T,8,8),zeros(T,8,8),p) : FlatMap(zeros(T,8,8),p)
        f=usegpu ? adapt(CuArray,host) : host
        expected=CuArray{T}(undef,size(f.arr)); randn!(CUDA.RNG(102),expected)
        returned=randn!(CUDA.RNG(102),f)
        @test returned === f
        @test Array(f.arr) == Array(expected)
        @test all(isfinite,Array(f.arr))
    end
    # The existing Fourier-field rule returns a map-space draw.
    fourier=Fourier(map_gpu()); before=Array(fourier.arr)
    expected=similar(Map(fourier).arr); randn!(CUDA.RNG(103),expected)
    out=randn!(CUDA.RNG(103),fourier)
    @test out isa typeof(Map(fourier))
    @test Array(out.arr) == Array(expected)
    @test Array(fourier.arr) == before

    for heterogeneous in (cpu_fallback ? (false,true) : (false,))
        a,b=map_gpu(),heterogeneous ? map_cpu() : map_gpu(); ft=FieldTuple(;a,b)
        expected_a=similar(a.arr); expected_b=similar(b.arr)
        r=CUDA.RNG(104); randn!(r,expected_a); randn!(r,expected_b)
        out=randn!(CUDA.RNG(104),ft)
        @test out isa FieldTuple
        @test out.a === a && out.b === b
        @test Array(a.arr) == Array(expected_a)
        @test Array(b.arr) == Array(expected_b)
    end

    d=map_gpu(); C=Diagonal(one(d)); ds=BaseDataSet(;d,Cf=C,Cn=C,Cϕ=C,D=I,G=I,L=(ϕ->I),Nϕ=C)
    sim=simulate(CUDA.RNG(105),ds)
    reference=[similar(d.arr) for _ in 1:3]
    r=CUDA.RNG(105); foreach(x->randn!(r,x),reference)
    @test Array(Map(sim.f).arr) == Array(reference[1])
    @test Array(Map(sim.ϕ).arr) == Array(reference[2])
    @test Array(Map(sim.d).arr) ≈ Array(reference[1] .+ reference[3]) rtol=2e-6 atol=2e-6
    @test randn!(MersenneTwister(106),map_gpu()) isa BaseField
end
CUDA.synchronize()
