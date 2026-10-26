using Adapt
using CMBLensing
using Test

struct Float32CPUStorage end
const lenseflow_adapted_arrays = IdDict{Any,Nothing}()
const lenseflow_adapted_inputs = IdDict{Any,Nothing}()
record_lenseflow_adaptation(src, dst) = (lenseflow_adapted_inputs[src] = nothing; lenseflow_adapted_arrays[dst] = nothing; dst)
Adapt.adapt_storage(::Float32CPUStorage, x::Array{<:AbstractFloat}) = record_lenseflow_adaptation(x, Float32.(x))
Adapt.adapt_storage(::Float32CPUStorage, x::Array{<:Complex{<:AbstractFloat}}) = record_lenseflow_adaptation(x, ComplexF32.(x))

struct ScaleMapCPUStorage end
Adapt.adapt_storage(::ScaleMapCPUStorage, x::Array) = copy(x)
Adapt.adapt_storage(::ScaleMapCPUStorage, x::Matrix{Float64}) = size(x) == (8, 8) ? 2 .* x : copy(x)

struct ScaleCoefficientsCPUStorage
    potential :: Matrix{Float64}
end
Adapt.adapt_structure(::ScaleCoefficientsCPUStorage, proj::ProjLambert) = proj
Adapt.adapt_storage(storage::ScaleCoefficientsCPUStorage, x::Array) = x === storage.potential ? x : copy(x)
Adapt.adapt_storage(storage::ScaleCoefficientsCPUStorage, x::Matrix{Float64}) = x === storage.potential ? x : 2 .* x

struct IdentityCPUStorage end
Adapt.adapt_structure(::IdentityCPUStorage, proj::ProjLambert) = proj
Adapt.adapt_storage(::IdentityCPUStorage, x::Array) = x

@testset "LenseFlow storage adaptation" begin
    N = 8
    θpix = 60 * 180 / π
    ϕ = FlatMap([0.002sin(2π * (i - 1) / N) for i in 1:N, j in 1:N]; θpix)
    f = FlatMap([sin(2π * (i - 1) / N) + cos(2π * (j - 1) / N) for i in 1:N, j in 1:N]; θpix)
    L = LenseFlow(ϕ, 3)

    # Copy and convert both real and complex CPU arrays without requiring a GPU.
    storage = Float32CPUStorage()
    fcopy = adapt(storage, f)
    @testset "uncached" begin
        Ladapt = adapt(storage, L)
        @test Ladapt.ϕ.arr !== ϕ.arr
        @test Ladapt.ϕ.arr == Float32.(ϕ.arr)
        @test Ladapt.odesolve === L.odesolve
        @test Ladapt.odesolve.nsteps == 3
        @test (Ladapt.t₀, Ladapt.t₁) == (L.t₀, L.t₁)
        @test typeof(Ladapt.t₀) == Float32
        @test (Ladapt * fcopy).arr ≈ Float32.((L * f).arr) rtol=1e-5 atol=1e-6
    end

    @testset "cached" begin
        cached = precompute!!(L, f)
        cached_adapt = adapt(storage, cached)
        @test cached_adapt.ϕ[].arr !== ϕ.arr
        @test cached_adapt.ϕ[].arr == Float32.(ϕ.arr)
        @test cached_adapt.memÐf.arr !== cached.memÐf.arr
        @test eltype(cached_adapt.memÐf.arr) == ComplexF32
        # Reuse converted field buffers, but do not transfer stale coefficients.
        @test haskey(lenseflow_adapted_arrays, cached_adapt.ϕ[].arr)
        @test haskey(lenseflow_adapted_arrays, cached_adapt.memÐf.arr)
        @test !haskey(lenseflow_adapted_inputs, cached.p[Float16(0)][1].diag.arr)
        @test !haskey(lenseflow_adapted_inputs, cached.M⁻¹[Float16(0)][1,1].diag.arr)
        @test cached_adapt.odesolve === cached.odesolve
        @test cached_adapt.odesolve.nsteps == 3
        @test (cached_adapt.t₀, cached_adapt.t₁) == (cached.t₀, cached.t₁)
        @test typeof(cached_adapt.t₀) == typeof(cached_adapt.t₁) == Float32
        adapted_result = cached_adapt * fcopy
        fresh_adapt = precompute!!(adapt(storage, L), fcopy)
        @test adapted_result.arr ≈ (fresh_adapt * fcopy).arr rtol=1e-5 atol=1e-6
        @test adapted_result.arr ≈ Float32.((cached * f).arr) rtol=1e-5 atol=1e-6

        # Changing one cached potential must not invalidate or suppress
        # recomputation in the other independently adapted operator.
        ϕchanged = 2 * ϕ
        cached(ϕchanged)
        @test cached.needs_precompute[]
        @test !cached_adapt.needs_precompute[]
        dirty_adapt = adapt(storage, cached)
        @test dirty_adapt.needs_precompute[]
        fresh_changed = precompute!!(adapt(storage, LenseFlow(ϕchanged, 3)), fcopy)
        @test (dirty_adapt * fcopy).arr ≈ (fresh_changed * fcopy).arr rtol=1e-5 atol=1e-6
        @test cached.needs_precompute[]
        @test (cached_adapt * fcopy).arr ≈ adapted_result.arr rtol=1e-5 atol=1e-6
        @test (cached * f).arr ≈ (LenseFlow(ϕchanged, 3) * f).arr
    end

    @testset "cached, unchanged storage" begin
        cached = precompute!!(L, f)
        cached_adapt = adapt(Array, cached)
        @test !cached_adapt.needs_precompute[]
        @test cached_adapt.ϕ[].arr !== cached.ϕ[].arr
        @test cached_adapt.p[Float16(0)][1].diag.arr !== cached.p[Float16(0)][1].diag.arr
        @test cached_adapt.memÐf.arr !== cached.memÐf.arr

        ϕchanged = 2 * ϕ
        cached(ϕchanged)
        dirty_adapt = adapt(Array, cached)
        @test dirty_adapt.needs_precompute[]
        @test (dirty_adapt * f).arr ≈ (LenseFlow(ϕchanged, 3) * f).arr
        @test cached.needs_precompute[]
        @test (cached * f).arr ≈ (LenseFlow(ϕchanged, 3) * f).arr
        @test (cached_adapt * f).arr ≈ (L * f).arr

        ϕchanged_again = 3 * ϕ
        cached_adapt(ϕchanged_again)
        @test (cached_adapt * f).arr ≈ (LenseFlow(ϕchanged_again, 3) * f).arr
        @test (cached * f).arr ≈ (LenseFlow(ϕchanged, 3) * f).arr
    end

    @testset "lossy potential conversion" begin
        ϕlarge = FlatMap([1e6 + 0.002sin(2π * (i - 1) / N) for i in 1:N, j in 1:N]; θpix)
        Llarge = LenseFlow(ϕlarge, 3)
        cached = precompute!!(Llarge, f)
        cached_adapt = adapt(storage, cached)
        fresh_adapt = precompute!!(adapt(storage, Llarge), fcopy)
        @test all(==(1f6), cached_adapt.ϕ[].arr)
        @test cached_adapt.needs_precompute[]
        @test (cached_adapt * fcopy).arr ≈ (fresh_adapt * fcopy).arr rtol=1e-6 atol=1e-6
        @test !cached_adapt.needs_precompute[]
    end

    @testset "same-precision value conversion" begin
        storage64 = ScaleMapCPUStorage()
        f64 = adapt(storage64, f)
        cached = precompute!!(L, f)
        cached_adapt = adapt(storage64, cached)
        fresh_adapt = precompute!!(adapt(storage64, L), f64)
        @test eltype(cached_adapt.ϕ[].arr) == eltype(ϕ.arr)
        @test cached_adapt.ϕ[].arr == 2 .* ϕ.arr
        @test cached_adapt.needs_precompute[]
        @test (cached_adapt * f64).arr ≈ (fresh_adapt * f64).arr rtol=1e-10 atol=1e-10
    end

    @testset "identity and coefficient-only adapters" begin
        cached = precompute!!(L, f)
        identity_adapt = adapt(IdentityCPUStorage(), cached)
        @test !identity_adapt.needs_precompute[]
        @test (identity_adapt * f).arr ≈ (cached * f).arr

        storage64 = ScaleCoefficientsCPUStorage(ϕ.arr)
        f64 = adapt(storage64, f)
        cached_adapt = adapt(storage64, cached)
        fresh_adapt = precompute!!(adapt(storage64, L), f64)
        @test cached_adapt.ϕ[].arr == ϕ.arr
        @test cached_adapt.needs_precompute[]
        @test (cached_adapt * f64).arr ≈ (fresh_adapt * f64).arr rtol=1e-10 atol=1e-10
    end

    @testset "time-key conversion" begin
        # Construct a dirty cache around an endpoint just above a Float16
        # midpoint, where conversion to Float32 changes a Float16 time key.
        cached = precompute!!(L, f)
        t₀ = 0.5 + 1 / 4096 + 1e-10
        oldτs = Float16.(range(t₀, 1.0; length=7))
        newτs = Float16.(range(Float32(t₀), 1f0; length=7))
        @test Float16(t₀) != Float16(Float32(t₀))
        p = Dict(τ => first(values(cached.p)) for τ in oldτs)
        M⁻¹ = Dict(τ => first(values(cached.M⁻¹)) for τ in oldτs)
        custom = CachedLenseFlow(
            Ref{Any}(ϕ), Ref(true), cached.odesolve, t₀, 1.0, p, M⁻¹,
            cached.memŁf, cached.memÐf, cached.memŁvf, cached.memÐvf,
            cached.memŁϕ, cached.memÐϕ, cached.memŁvϕ, cached.memÐvϕ,
        )
        converted = adapt(storage, custom)
        @test converted.t₀ === Float32(t₀)
        @test Set(keys(converted.p)) == Set(newτs)
        @test Set(keys(converted.M⁻¹)) == Set(newτs)
        @test precompute!!(converted, fcopy) === converted
        @test all(isfinite, (converted * fcopy).arr)
    end
end
