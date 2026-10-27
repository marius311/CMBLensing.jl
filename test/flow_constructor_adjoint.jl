using CMBLensing
using LinearAlgebra
using Test
using Zygote

@testset "FlowOp constructor preserves arguments during AD" begin
    N=16
    field(fn)=FlatMap([fn(2π*(i-1)/N,2π*(j-1)/N) for i=1:N,j=1:N]; θpix=60*180/π)
    ϕ=field((x,y)->0.32sin(x+2y)+0.23cos(2x-y)+0.12sin(x-2y))
    η=field((x,y)->0.30cos(x+y)+0.2sin(2x-y)+0.1cos(x-2y))
    f=field((x,y)->sin(x+y)+0.7cos(2x-y)+0.5sin(x-2y))
    g=field((x,y)->0.8cos(x-y)+0.35sin(x+2y)+0.6cos(2x+y))

    for n in (1, 2, 16, 64)
        traced, back = Zygote.pullback(p -> LenseFlow(p,n), ϕ)
        @test traced.odesolve.nsteps == n
        @test back(η)[1] ≈ η
        objective(a) = dot(g,LenseFlow(ϕ+a*η,n)*f)
        traced_value, _ = Zygote.pullback(objective,0.0)
        @test traced_value ≈ objective(0.0) atol=1e-12 rtol=1e-12
    end

    # The continuous-flow adjoint converges to the discrete forward derivative
    # as the RK4 discretization is refined.
    for mode in (:forward,:inverse)
        objective(a) = dot(g, mode == :forward ? LenseFlow(ϕ+a*η,64)*f : LenseFlow(ϕ+a*η,64)\f)
        reverse = gradient(objective,0.0)[1]
        h = 1e-3
        finite = (objective(h)-objective(-h))/(2h)
        @test isapprox(reverse,finite; atol=2e-5, rtol=0)
    end
end
