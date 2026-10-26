using Adapt
using CMBLensing
using Test

struct Float32CPUStorage end
const lenseflow_adapted_arrays = IdDict{Any,Nothing}()
record_lenseflow_adaptation(x) = (lenseflow_adapted_arrays[x] = nothing; x)
Adapt.adapt_storage(::Float32CPUStorage, x::Array{<:AbstractFloat}) = record_lenseflow_adaptation(Float32.(x))
Adapt.adapt_storage(::Float32CPUStorage, x::Array{<:Complex{<:AbstractFloat}}) = record_lenseflow_adaptation(ComplexF32.(x))

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
        # Already converted arrays should not be copied a second time.
        @test haskey(lenseflow_adapted_arrays, cached_adapt.ϕ[].arr)
        @test haskey(lenseflow_adapted_arrays, cached_adapt.p[Float16(0)][1].diag.arr)
        @test haskey(lenseflow_adapted_arrays, cached_adapt.memÐf.arr)
        @test cached_adapt.odesolve === cached.odesolve
        @test cached_adapt.odesolve.nsteps == 3
        @test (cached_adapt.t₀, cached_adapt.t₁) == (cached.t₀, cached.t₁)
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
        @test cached_adapt.ϕ[].arr !== cached.ϕ[].arr
        @test cached_adapt.p[Float16(0)][1].diag.arr !== cached.p[Float16(0)][1].diag.arr
        @test cached_adapt.memÐf.arr !== cached.memÐf.arr

        ϕchanged = 2 * ϕ
        cached(ϕchanged)
        @test (cached * f).arr ≈ (LenseFlow(ϕchanged, 3) * f).arr
        @test (cached_adapt * f).arr ≈ (L * f).arr

        ϕchanged_again = 3 * ϕ
        cached_adapt(ϕchanged_again)
        @test (cached_adapt * f).arr ≈ (LenseFlow(ϕchanged_again, 3) * f).arr
        @test (cached * f).arr ≈ (LenseFlow(ϕchanged, 3) * f).arr
    end
end
