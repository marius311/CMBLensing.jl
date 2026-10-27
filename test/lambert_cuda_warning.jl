using Logging

struct LambertWarningCUDAAdaptor end
Adapt.adapt_storage(::LambertWarningCUDAAdaptor, x::Array) = CuArray(x)

@testset "Lambert CUDA FFT warning" begin
    # A loaded CUDA extension must not warn for CPU maps, small GPU maps,
    # or exactly 1024² pixels.
    @test_logs min_level=Logging.Warn ProjLambert(; Ny=1024, Nx=1025, θpix=1.11, storage=Array)
    @test_logs min_level=Logging.Warn ProjLambert(; Ny=8, Nx=8, θpix=1.12, storage=CuArray)
    @test_logs min_level=Logging.Warn ProjLambert(; Ny=1024, Nx=1024, θpix=1.13, storage=CuArray)

    version = isdefined(CUDA, :runtime_version) ? CUDA.runtime_version() : CUDA.toolkit_version()
    for (projection_storage, θpix) in ((CuArray, 1.14), (LambertWarningCUDAAdaptor(), 1.15))
        if version >= v"11.5"
            @test_logs (:warn, r"In 2022, CMBLensing warned.*unknown; check FFT results") ProjLambert(; Ny=1024, Nx=1025, θpix, storage=projection_storage)
        else
            @test_logs min_level=Logging.Warn ProjLambert(; Ny=1024, Nx=1025, θpix, storage=projection_storage)
        end
    end
end
