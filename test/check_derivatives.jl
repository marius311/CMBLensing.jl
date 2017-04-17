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
Cf = Cℓ_to_cov(T,P,S0,S2,cls[:ℓ],cls[:tt],   cls[:te],   cls[:ee],   cls[:bb])
Cϕ = Cℓ_to_cov(T,P,S0,   cls[:ℓ],cls[:ϕϕ])
CN = FullDiagOp(FlatIQUMap{T,P}(repeated(fill(1 * deg2rad(Θpix/60)^2,(nside,nside)),3)...))
##
f = Ł(simulate(Cf))
ϕ = Ł(simulate(Cϕ))
δfϕ = δf,δϕ = Ł(FieldTuple(simulate(Cf),simulate(Cϕ)))
##
ϵ = 1e-7


## LenseFlow tests
L = LenseFlowOp
ds = DataSet(L(ϕ)*f + simulate(CN), CN, Cf, Cϕ, Squash, Squash);

@testset "LenseFlow Jacobian" begin
    # Jacobian
    @test 1/(2ϵ)*(L(ϕ+ϵ*δϕ)*(f+ϵ*δf) - L(ϕ-ϵ*δϕ)*(f-ϵ*δf)) ≈ (δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f) * δfϕ)[1]
    # inverse Jacobian
    @test (δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f) \ (δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f) * δfϕ))[1] ≈ δfϕ[1]
    # Jacobian transpose
    @test (δfϕ ⋅ (δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f) * δfϕ)) ≈ ((δfϕ * δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f)) ⋅ δfϕ)
    # Jacobian inverse transpose
    @test (δfϕ ⋅ (δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f) \ δfϕ)) ≈ ((δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f)' \ δfϕ) ⋅ δfϕ)
    # Likelihood gradient
    @test (lnP(0,f+ϵ*δf,ϕ+ϵ*δϕ,ds,L) - lnP(0,f-ϵ*δf,ϕ-ϵ*δϕ,ds,L))/(2ϵ) ≈ (δlnP_δfϕₜ(0,f,ϕ,ds,L)⋅FieldTuple(δf,δϕ)) rtol=1e-6
end


## PowerLens tests
L = PowerLens
ds = DataSet(L(ϕ)*f + simulate(CN), CN, Cf, Cϕ, Squash, Squash);

@testset "PowerLens Jacobian" begin
    # Jacobian
    @test 1/(2ϵ)*(L(ϕ+ϵ*δϕ)*(f+ϵ*δf) - L(ϕ-ϵ*δϕ)*(f-ϵ*δf)) ≈ (δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f) * δfϕ)[1]
    # Jacobian transpose
    @test (δfϕ ⋅ (δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f) * δfϕ)) ≈ ((δfϕ * δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f)) ⋅ δfϕ)
    # Likelihood gradient
    @test (lnP(0,f+ϵ*δf,ϕ+ϵ*δϕ,ds,L) - lnP(0,f-ϵ*δf,ϕ-ϵ*δϕ,ds,L))/(2ϵ) ≈ (δlnP_δfϕₜ(0,f,ϕ,ds,L)⋅FieldTuple(δf,δϕ)) rtol=1e-6
end
##
