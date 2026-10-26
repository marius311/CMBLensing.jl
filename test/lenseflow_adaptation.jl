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

    @testset "standard interval results" begin
        cached = precompute!!(L, f)
        ops = (((L, x) -> L * x), ((L, x) -> L \ x), ((L, x) -> L' * x), ((L, x) -> L' \ x))
        expected = (
            [1.001233700237184, 0.7077233623262509, -0.2922766376737492, 0.0006171192539564704],
            [0.9987663007016729, 0.7064891231142287, -0.29351087688577127, -0.0006160437296268212],
            [0.998764778059493, 0.7071054359009138, -0.2937669204643614, -0.0006155038158639702],
            [1.00123370039366, 0.7071059740151333, -0.29202166795991186, 0.0006176574464821422],
        )
        for (op, reference) in zip(ops, expected)
            y = Map(op(cached, f)).arr
            @test [y[1,1], y[2,3], y[4,5], y[8,8]] ≈ reference rtol=1e-10 atol=1e-10
        end
    end

    @testset "RK4 stage-time keys" begin
        function custom_cache(ϕ, f, t₀)
            template = precompute!!(LenseFlow(ϕ, 3), f)
            p, M⁻¹ = CMBLensing._lenseflow_cache_dicts(ϕ, t₀, 1.0, 3)
            cache = CachedLenseFlow(
                Ref{Any}(ϕ), Ref(true), template.odesolve, t₀, 1.0, p, M⁻¹,
                template.memŁf, template.memÐf, template.memŁvf, template.memÐvf,
                template.memŁϕ, template.memÐϕ, template.memŁvϕ, template.memÐvϕ,
            )
            precompute!!(cache, f)
        end

        ops = (
            "forward" => ((L, x) -> L * x),
            "reverse" => ((L, x) -> L \ x),
            "adjoint" => ((L, x) -> L' * x),
            "inverse-adjoint" => ((L, x) -> L' \ x),
        )
        @testset "t₀ = $t₀32" for (t₀32, reverse_stage) in ((0.118017554f0, false), (0.709082f0, true))
            source = custom_cache(ϕ, f, Float64(t₀32))
            expected = [Float32.(Map(op(source, f)).arr) for (_, op) in ops]
            converted = adapt(storage, source)

            nominal = Float16.(range(t₀32, 1f0; length=7))
            if reverse_stage
                h = (t₀32 - 1f0) / 3
                t = range(1f0, t₀32; length=4)[3]
                stage_key = Float16(t + h / 2)
            else
                h = (1f0 - t₀32) / 3
                stage_key = Float16(t₀32 + h / 2)
            end
            nominal_key = nominal[2]
            @test stage_key ∉ nominal
            for ((name, op), reference) in zip(ops, expected)
                @testset "$name" begin
                    @test Map(op(converted, fcopy)).arr ≈ reference rtol=1e-5 atol=2e-6
                end
            end

            @test haskey(converted.p, stage_key)
            @test haskey(converted.M⁻¹, stage_key)
            if haskey(converted.p, stage_key) && haskey(converted.M⁻¹, stage_key)
                @test converted.p[stage_key][1].diag.arr === converted.p[nominal_key][1].diag.arr
                @test converted.M⁻¹[stage_key][1,1].diag.arr === converted.M⁻¹[nominal_key][1,1].diag.arr
            end

            copied = adapt(IdentityCPUStorage(), converted)
            @test !copied.needs_precompute[]
            @test haskey(copied.p, stage_key)
            @test haskey(copied.M⁻¹, stage_key)
            if haskey(copied.p, stage_key) && haskey(copied.M⁻¹, stage_key)
                @test copied.p[stage_key][1].diag.arr === copied.p[nominal_key][1].diag.arr
                @test copied.M⁻¹[stage_key][1,1].diag.arr === copied.M⁻¹[nominal_key][1,1].diag.arr
                @test copied.p[stage_key][1].diag.arr !== converted.p[stage_key][1].diag.arr
            end

            copied(2 * converted.ϕ[])
            source(2 * ϕ)
            @test copied.needs_precompute[]
            for (name, op) in ops
                @testset "refreshed $name" begin
                    @test Map(op(copied, fcopy)).arr ≈ Float32.(Map(op(source, f)).arr) rtol=1e-5 atol=2e-6
                end
            end
            if haskey(copied.p, stage_key) && haskey(copied.M⁻¹, stage_key)
                @test copied.p[stage_key][1].diag.arr === copied.p[nominal_key][1].diag.arr
                @test copied.M⁻¹[stage_key][1,1].diag.arr === copied.M⁻¹[nominal_key][1,1].diag.arr
            end
            @test Map(converted * fcopy).arr ≈ expected[1] rtol=1e-5 atol=2e-6

            converted(2 * converted.ϕ[])
            dirty_copy = adapt(IdentityCPUStorage(), converted)
            @test dirty_copy.needs_precompute[]
            @test Map(dirty_copy * fcopy).arr ≈ Float32.(Map(source * f).arr) rtol=1e-5 atol=2e-6
            @test converted.needs_precompute[]
            @test dirty_copy.p[stage_key][1].diag.arr === dirty_copy.p[nominal_key][1].diag.arr
            @test dirty_copy.M⁻¹[stage_key][1,1].diag.arr === dirty_copy.M⁻¹[nominal_key][1,1].diag.arr
        end
    end
end
