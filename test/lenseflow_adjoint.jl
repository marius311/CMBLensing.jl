using CMBLensing
using LinearAlgebra
using Test
using Zygote

@testset "LenseFlow potential adjoint" begin
    # Smooth, deterministic CPU fields with nonzero mixed Hessian components.
    N = 16
    θpix = 60 * 180 / π  # one radian per pixel
    field(fun) = FlatMap([fun(2π*(i-1)/N, 2π*(j-1)/N) for i=1:N, j=1:N]; θpix)
    ϕ = field((x,y) -> 0.32sin(x+2y) + 0.23cos(2x-y) + 0.12sin(x-2y))
    f = field((x,y) -> sin(x+y) + 0.7cos(2x-y) + 0.5sin(x-2y))
    g = field((x,y) -> 0.8cos(x-y) + 0.35sin(x+2y) + 0.6cos(2x+y))
    directions = (
        field((x,y) -> 0.3cos(x+y) + 0.2sin(2x-y) + 0.1cos(x-2y)),
        field((x,y) -> 0.2sin(x-y) + 0.35cos(x+2y) + 0.15sin(2x+y)),
    )

    # Construct the cached operator outside AD so that its requested step
    # count is used by both the forward computation and the adjoint.
    L = precompute!!(LenseFlow(ϕ, 64), f)
    M⁻¹ = L.M⁻¹[Float16(0.5)]
    @test norm(M⁻¹[1,2].diag) > 0.1
    @test norm(M⁻¹[1,1].diag - M⁻¹[2,2].diag) > 0.1

    for mode in (:forward, :inverse), (i, δϕ) in enumerate(directions)
        @testset "$mode, direction $i" begin
            objective(α) = dot(g, mode == :forward ? L(ϕ+α*δϕ)*f : L(ϕ+α*δϕ)\f)
            adjoint_derivative = gradient(objective, 0.0)[1]
            h = 1e-3
            finite_difference = (objective(h) - objective(-h)) / (2h)
            @test abs(finite_difference) > 0.01
            # The continuous-flow adjoint and the finite-step forward map need
            # not agree exactly; 64 steps make their difference small here.
            @test isapprox(adjoint_derivative, finite_difference; atol=2e-5, rtol=0)
        end
    end
end
