using CMBLensing
using Base.Test
import Base: ≈

≈(a::Field,b::Field) = pixstd(a-b)<1e-4

## calc Cℓs and store in Main since I reload CMBLensing alot during development
cls = isdefined(Main,:cls) ? Main.cls : @eval Main cls=$(class(lmax=6000,r=0.05))
## set up the types of maps
Θpix, nside, T = 3, 75, Float64
P = Flat{Θpix,nside}
## covariances 
Cf    = Cℓ_to_cov(T,P,S0,S2,cls[:ℓ],cls[:tt],   cls[:te],   cls[:ee],   cls[:bb])
Cϕ    = Cℓ_to_cov(T,P,S0,   cls[:ℓ],cls[:ϕϕ])
##
f,f1,f2 = @repeated(Ł(simulate(Cf)),3)
ϕ,ϕ1,ϕ2 = @repeated(Ł(simulate(Cϕ)),3)
δfϕ = δf,δϕ = Ł(FieldTuple(simulate(Cf),simulate(Cϕ)))
##
ϵ = 1e-7


## LenseFlow tests
L = LenseFlowOp

@testset "LenseFlow Jacobian" begin
    # Jacobian
    @test 1/(2ϵ)*(L(ϕ+ϵ*δϕ)*(f+ϵ*δf) - L(ϕ-ϵ*δϕ)*(f-ϵ*δf)) ≈ (δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f) * δfϕ)[1]
    # inverse Jacobian
    @test (δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f) \ (δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f) * δfϕ))[1] ≈ δfϕ[1]
    # Jacobian transpose
    @test (δfϕ ⋅ (δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f) * δfϕ)) ≈ ((δfϕ * δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f)) ⋅ δfϕ)
    # Jacobian inverse transpose
    @test (δfϕ ⋅ (δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f) \ δfϕ)) ≈ ((δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f)' \ δfϕ) ⋅ δfϕ)
end


## PowerLens tests
L = PowerLens

@testset "PowerLens Jacobian" begin
    # Jacobian
    @test 1/(2ϵ)*(L(ϕ+ϵ*δϕ)*(f+ϵ*δf) - L(ϕ-ϵ*δϕ)*(f-ϵ*δf)) ≈ (δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f) * δfϕ)[1]
    # Jacobian transpose
    @test (δfϕ ⋅ (δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f) * δfϕ)) ≈ ((δfϕ * δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f)) ⋅ δfϕ)
end
