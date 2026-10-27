# A storage adaptor can return ordinary Arrays without being the Array type.
struct NFFTCPUAdaptor end
Adapt.adapt_storage(::NFFTCPUAdaptor, x::Array) = x

@testset "NFFT CPU storage adaptor" begin
    hp = ProjHealpix(2)
    kwargs = (; Ny=8, Nx=16, θpix=600, T=Float64, rotator=(0, 30, 0))
    standard = ProjLambert(; kwargs..., storage=Array)
    adapted = ProjLambert(; kwargs..., storage=NFFTCPUAdaptor())
    standard_projector = CMBLensing.Projector(standard => hp; method=:fft)
    adapted_projector = CMBLensing.Projector(adapted => hp; method=:fft)
    grid = reshape(sin.(collect(1.0:128.0) ./ 9), 8, 16)
    sphere = cos.(collect(1.0:48.0) ./ 7)
    @test adapted.ℓmag isa Array
    @test project(adapted_projector, CMBLensing.FlatMap(grid, adapted) => hp).arr ≈
          project(standard_projector, CMBLensing.FlatMap(grid, standard) => hp).arr
    @test project(adapted_projector, CMBLensing.HealpixMap(sphere, hp) => adapted).arr ≈
          project(standard_projector, CMBLensing.HealpixMap(sphere, hp) => standard).arr
end
