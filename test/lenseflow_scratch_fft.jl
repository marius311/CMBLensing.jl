using CMBLensing
using CMBLensing: @!, Ð!, Ł!
using FFTW
using LinearAlgebra
using Test
using Zygote

# The previous conversion preserves its Fourier input. Keep it as a reference
# for the same ODE, including inverse integration and batched fields.
function preserving_velocity(L, f₀)
    function v!(v, t, f)
        Ðf, Ð∇f, Ł∇f = L.memÐf, L.memÐvf, L.memŁvf
        p = L.p[CMBLensing.τ(t)]
        @! Ðf = Ð(f)
        @! Ð∇f = ∇ᵢ * Ðf
        @! Ł∇f = Ł(Ð∇f)
        @! v = p' * Ł∇f
    end
    (v!, L.memŁf .= Ł(f₀))
end

# A batched dot product returns a BatchedReal. Compare individual values so
# each assertion reduces to an ordinary Bool.
function all_batch_isapprox(a, b; kwargs...)
    n = CMBLensing.batch_length(a)
    n == CMBLensing.batch_length(b) || return false
    all(i -> isapprox(CMBLensing.batch_index(a, i), CMBLensing.batch_index(b, i); kwargs...), 1:n)
end

@testset "LenseFlow disposable inverse-FFT scratch" begin
    # Cover both precisions, odd/even sizes, polarization, IQU and batching
    # without repeating every combination in the normal test suite. MKL does
    # not support this package's batched CPU FFT path.
    nbat = FFTW.fftw_provider == "mkl" ? 1 : 2
    for (T, N, pol, nbat) in ((Float32, 8, :I, 1), (Float64, 9, :P, nbat), (Float32, 8, :IP, 1))
        θpix = 60 * 180 / π
        x = T[sin(2π*(i-1)/N) + 0.4cos(2π*(j-1)/N) for i in 1:N, j in 1:N]
        y = T[cos(2π*(i-1)/N + 2π*(j-1)/N) for i in 1:N, j in 1:N]
        z = T[sin(2π*(i-1)/N - 2π*(j-1)/N) for i in 1:N, j in 1:N]
        ϕ = FlatMap(T(0.03) .* x; θpix)
        bat(a) = nbat == 1 ? a : cat(a, T(0.7) .* a; dims=3)
        f = pol == :I ? FlatMap(bat(x); θpix) :
            pol == :P ? FlatQUMap(bat(x), bat(y); θpix) :
                        FlatIQUMap(bat(x), bat(y), bat(z); θpix)
        g = T(0.7) * f
        f_before, ϕ_before = copy(f.arr), copy(ϕ.arr)
        L = precompute!!(LenseFlow(ϕ, 4), f)

        # The source buffers cannot alias inputs, other scratch, or the
        # cached coefficients whose values are needed by later stages.
        other_fields = [f, ϕ, L.memŁf, L.memÐf, L.memŁϕ, L.memÐϕ,
                        L.memŁvf..., L.memŁvϕ..., L.memÐvϕ...]
        coefficients = [d.diag for v in values(L.p) for d in v]
        append!(coefficients, [d.diag for M in values(L.M⁻¹) for d in M])
        append!(other_fields, coefficients)
        coeff_before = [copy(c.arr) for c in coefficients]
        for (i, s) in enumerate(L.memÐvf)
            @test !any(q -> Base.mightalias(s.arr, q.arr), other_fields)
            @test all(j -> i == j || !Base.mightalias(s.arr, L.memÐvf[j].arr), 1:2)
        end

        reference_forward = L.odesolve(preserving_velocity(L, f)..., L.t₀ => L.t₁)
        result = copy(L * f)
        @test isapprox(result.arr, reference_forward.arr; rtol=50eps(T))
        @test isapprox((L * f).arr, result.arr; rtol=50eps(T))
        reference_inverse = L.odesolve(preserving_velocity(L, f)..., L.t₁ => L.t₀)
        @test isapprox((L \ f).arr, reference_inverse.arr; rtol=50eps(T))
        @test isapprox((L \ f).arr, reference_inverse.arr; rtol=50eps(T))

        @test all_batch_isapprox(dot(g, result), dot(L' * g, f); rtol=2e-4, atol=2e-4)
        @test all_batch_isapprox(dot(g, L \ f), dot(L' \ g, f); rtol=2e-4, atol=2e-4)
        @test f.arr == f_before
        @test ϕ.arr == ϕ_before
        @test all(c.arr == before for (c, before) in zip(coefficients, coeff_before))

        # A public basis conversion must still preserve its Fourier input.
        Fourier_f = Ð(f)
        Fourier_before = copy(Fourier_f.arr)
        Ł!(similar(f), Fourier_f)
        @test Fourier_f.arr == Fourier_before
    end

    @testset "non-Array fallback" begin
        N = 8
        f = FlatMap(reshape(Float64.(1:N^2), N, N); θpix=3)
        F = Fourier(f)
        Fview = CMBLensing.BaseField{Fourier}(view(F.arr, :, :), F.metadata)
        Fref = CMBLensing.BaseField{Fourier}(view(copy(F.arr), :, :), F.metadata)
        dst = similar(f)
        ref = similar(f)
        CMBLensing._lenseflow_scratch_to_map!(dst, Fview)
        Ł!(ref, Fref)
        @test Fview.arr == Fref.arr
        @test dst.arr == ref.arr
        @test dst ≈ f
    end

    @testset "potential gradient" begin
        N = 16
        θpix = 60 * 180 / π
        field(fun) = FlatMap([fun(2π*(i-1)/N, 2π*(j-1)/N) for i in 1:N, j in 1:N]; θpix)
        ϕ = field((x,y) -> 0.32sin(x+2y) + 0.23cos(2x-y) + 0.12sin(x-2y))
        f = field((x,y) -> sin(x+y) + 0.7cos(2x-y) + 0.5sin(x-2y))
        g = field((x,y) -> 0.8cos(x-y) + 0.35sin(x+2y) + 0.6cos(2x+y))
        δϕ = field((x,y) -> 0.3cos(x+y) + 0.2sin(2x-y) + 0.1cos(x-2y))
        L = precompute!!(LenseFlow(ϕ, 64), f)
        for mode in (:forward, :inverse)
            objective(α) = dot(g, mode == :forward ? L(ϕ + α*δϕ)*f : L(ϕ + α*δϕ)\f)
            automatic = gradient(objective, 0.0)[1]
            h = 1e-3
            finite = (objective(h) - objective(-h))/(2h)
            @test isapprox(automatic, finite; atol=2e-5, rtol=0)
        end
    end
end
