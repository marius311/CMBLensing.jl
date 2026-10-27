
using CMBLensing
using CMBLensing: @SMatrix, @SVector, AbstractCℓs, basis, Basis,
    LinearInterpolation, Measurement, RK4Solver, ±, typealias, BatchedReal

##

if !haskey(ENV, "JULIA_CMBLENSING_TEST_CPU")
    using CUDA
end
if !haskey(ENV, "JULIA_CMBLENSING_TEST_CPU") && CUDA.functional()
    CUDA.allowscalar(false)
    maybegpu(x) = adapt(CuArray,x)
    storage = CuArray
    @info "Running tests on $(repr("text/plain", CUDA.device()))"
else
    maybegpu(x) = x
    storage = Array
    @info "Running tests on CPU"
end

##

using Adapt
using FileIO
using FFTW
using FiniteDifferences
using ForwardDiff
using LinearAlgebra
using NFFT
using OffsetArrays
using Random
using Requires
using Serialization
using SparseArrays
using StableRNGs
using Test
using Zygote

# Exercise storage adaptation without requiring a GPU. This also tests that
# fields' projection metadata is adapted together with their data.
struct TestFloat32Adaptor end
Adapt.adapt_storage(::TestFloat32Adaptor, x::Array{T}) where {T<:AbstractFloat} = Float32.(x)

CMBLensing.@fwdmodel function test_ordered_model()
    first_draw ← 1
    second_draw ← first_draw + 1
end

##

try
    push!(LOAD_PATH, "@v#.#") # assumes you have CirculantCov in your global environment
    using CirculantCov
catch
    @warn """CirculantCov.jl not found, not testing EquiRect fields.
    Run `pkg> add https://github.com/EthanAnderes/CirculantCov.jl` to add this package.
    """
end

##

macro test_real_gradient(f, x, tol=:(rtol=1e-3))
    esc(:(@test real(gradient($f,$x)[1]) ≈ central_fdm(5,1)($f,$x) $tol))
end

Nsides     = [(8,8), (4,8), (8,4)]
Nsides_big = [(128,128), (64,128), (128,64)]

rng = StableRNG(4)

has_batched_fft = (FFTW.fftw_provider != "mkl") || (storage != Array)

include("fft_view_preservation.jl")

println("Diagnostic: Julia=", VERSION, " CPU=", Sys.CPU_NAME,
        " threads=", Threads.nthreads(), " FFTW=", CMBLensing.versionof(FFTW),
        " NFFT=", CMBLensing.versionof(NFFT), " FFTW provider=", FFTW.fftw_provider)
@test CMBLensing.versionof(NFFT) == v"0.13.3"
@test CMBLensing.versionof(FFTW) == v"1.9.0"

@testset "Repeated exact constant-map projector" begin
    for repetition in 1:20
        ny, nx = 8, 16
        cart_proj = ProjLambert(; Ny=ny, Nx=nx, θpix=600, T=Float64, rotator=(0, 30, 0))
        hpx_proj = ProjHealpix(2)
        projector = CMBLensing.Projector(cart_proj => hpx_proj; method=:fft)
        cart = CMBLensing.FlatMap(ones(ny, nx), cart_proj)
        hpx = CMBLensing.HealpixMap(ones(48), hpx_proj)
        cart_to_hpx = project(projector, cart => hpx_proj).arr
        hpx_to_cart = project(projector, hpx => cart_proj).arr
        patch = projector.hpx_idxs_in_patch
        @test !isempty(patch)
        @test all(isfinite, hpx_to_cart)
        patch_values = cart_to_hpx[patch]
        grid_nodes = reduce(hcat, [[(i-ny÷2-1)/ny, (j-nx÷2-1)/nx] for i=1:ny, j=1:nx])
        patch_nodes = reduce(hcat, [[(projector.is[k]-ny÷2-1)/ny, (projector.js[k]-nx÷2-1)/nx] for k in patch])
        grid_dft = NFFT.NDFTPlan(grid_nodes, (ny, nx))
        patch_dft = NFFT.NDFTPlan(patch_nodes, (ny, nx))
        direct = real.(patch_dft * (adjoint(grid_dft) * complex.(cart.arr[:]))) ./ (ny*nx)
        println("iteration=", repetition, " patch=", repr(patch),
                " patch_values=", repr(patch_values), " direct=", repr(direct),
                " max_error=", maximum(abs.(patch_values .- direct)))
        @test all(isapprox.(patch_values, 1.0; atol=1e-5))
        @test patch_values ≈ direct rtol=1e-5 atol=1e-5
    end
end
