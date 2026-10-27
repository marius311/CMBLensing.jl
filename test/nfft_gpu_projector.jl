# Run in an environment with CUDA, NFFT and CMBLensing. For the NFFT 0.13
# path, also install CuNFFT and set CMBLENSING_TEST_CUNFFT=1.
using CUDA, GPUArrays, NFFT, CMBLensing, Zygote, Test, Adapt
const ForwardDiff = CMBLensing.ForwardDiff
struct TestCUDAAdaptor end
Adapt.adapt_storage(::TestCUDAAdaptor, x::Array) = CUDA.cu(x)

@assert CUDA.functional()
CUDA.allowscalar(false)
source = only(ARGS)
@assert realpath(pathof(CMBLensing)) == realpath(joinpath(source, "src", "CMBLensing.jl"))
println("source=", pathof(CMBLensing), " Julia=", VERSION,
        " CUDA=", CMBLensing.versionof(CUDA), " GPUArrays=", CMBLensing.versionof(GPUArrays), " NFFT=", CMBLensing.versionof(NFFT))

ny, nx = 8, 16
hp = ProjHealpix(2)
cp = ProjLambert(; Ny=ny, Nx=nx, θpix=600, T=Float32, rotator=(0,30,0), storage=Array)
gp = ProjLambert(; Ny=ny, Nx=nx, θpix=600, T=Float32, rotator=(0,30,0), storage=CuArray)

@testset "NFFT GPU backend availability" begin
    @test !CMBLensing.cu_nfft_loaded
    @test !CMBLensing.native_nfft_gpu_loaded(zeros(Float32, 2, 2))
    gpu_nodes = CuArray(zeros(Float32, 2, 2))
    if get(ENV, "CMBLENSING_TEST_CUNFFT", "0") == "1"
        @test !CMBLensing.native_nfft_gpu_loaded(gpu_nodes)
        @test_throws ErrorException CMBLensing.Projector(gp => hp; method=:fft)
        @eval using CuNFFT
        @test CMBLensing.cu_nfft_loaded
        println("CuNFFT=", CMBLensing.versionof(CuNFFT))
    else
        @test Base.find_package("CuNFFT") === nothing
        @test Base.get_extension(NFFT, :NFFTGPUArraysExt) !== nothing
        @test CMBLensing.native_nfft_gpu_loaded(gpu_nodes)
        old_backend = NFFT.AbstractNFFTs.active_backend()
        try
            NFFT.AbstractNFFTs.set_active_backend!(missing)
            @test !CMBLensing.native_nfft_gpu_loaded(gpu_nodes)
            @test_throws ErrorException CMBLensing.Projector(gp => hp; method=:fft)
            NFFT.activate!()
            @test CMBLensing.native_nfft_gpu_loaded(gpu_nodes)
        finally
            NFFT.AbstractNFFTs.set_active_backend!(old_backend)
        end
    end
end

pc = CMBLensing.Projector(cp => hp; method=:fft)
gpu_adaptor = TestCUDAAdaptor()
gp_adaptor = ProjLambert(; Ny=ny, Nx=nx, θpix=600, T=Float64, rotator=(0,30,0), storage=gpu_adaptor)
@test gp_adaptor.ℓmag isa CuArray{Float32}
grid = reshape(Float32.(sin.(collect(1:ny*nx) ./ 9)), ny, nx)
sphere = Float32.(cos.(collect(1:48) ./ 7))
gpu_grid, gpu_sphere = CuArray(grid), CuArray(sphere)
tocart(p, x, proj) = project(p, CMBLensing.HealpixMap(x, hp) => proj).arr
tohpx(p, x, proj) = project(p, CMBLensing.FlatMap(x, proj) => hp).arr

@testset "NFFT GPU projector and derivatives" begin
    for (storage_label, gpu_proj) in (("CuArray", gp), ("CUDA adaptor", gp_adaptor))
        @testset "$storage_label" begin
            pg = CMBLensing.Projector(gpu_proj => hp; method=:fft)
            println("$storage_label GPU plan types=", typeof(pg.nfft_plan), ", ", typeof(pg.nfft_plan_grid))
            cart_cpu, cart_gpu = tocart(pc, sphere, cp), tocart(pg, gpu_sphere, gpu_proj)
            hpx_cpu, hpx_gpu = tohpx(pc, grid, cp), tohpx(pg, gpu_grid, gpu_proj)
            @test cart_gpu isa CuArray
            @test hpx_gpu isa CuArray
            println("$storage_label healpix=>cart max cpu/gpu error=", maximum(abs.(Array(cart_gpu) .- cart_cpu)))
            println("$storage_label cart=>healpix max cpu/gpu error=", maximum(abs.(Array(hpx_gpu) .- hpx_cpu)))
            @test Array(cart_gpu) ≈ cart_cpu rtol=1e-3 atol=1e-4
            @test Array(hpx_gpu) ≈ hpx_cpu rtol=1e-3 atol=1e-4
            # Check the gradient of the Cartesian-to-Healpix path.
            fcpu(x) = sum(tohpx(pc, x, cp))
            fgpu(x) = sum(tohpx(pg, x, gpu_proj))
            grad_cpu = Zygote.gradient(fcpu, grid)[1]
            grad_gpu = Zygote.gradient(fgpu, gpu_grid)[1]
            @test grad_gpu isa CuArray
            println("$storage_label cart=>healpix reverse gradient max cpu/gpu error=", maximum(abs.(Array(grad_gpu) .- grad_cpu)))
            @test Array(grad_gpu) ≈ grad_cpu rtol=1e-3 atol=1e-4
            direction_cpu = ones(Float32, size(grid)); direction_gpu = CuArray(direction_cpu)
            fd_cpu = ForwardDiff.derivative(t -> fcpu(grid .+ t .* direction_cpu), 0.3f0)
            fd_gpu = ForwardDiff.derivative(t -> fgpu(gpu_grid .+ t .* direction_gpu), 0.3f0)
            println("$storage_label cart=>healpix forward derivative cpu=", fd_cpu, " gpu=", fd_gpu)
            @test fd_gpu ≈ fd_cpu rtol=1e-3 atol=1e-4
        end
    end
end
