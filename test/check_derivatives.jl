using CMBLensing
using Base.Test
import Base: ≈

≈(a::Field,b::Field) = pixstd(a-b)<1e-4

## calc Cℓs and store in Main since I reload CMBLensing alot during development
cls = isdefined(Main,:cls) ? Main.cls : @eval Main cls=$(class(lmax=6000,r=0.05))
## set up the types of maps
Θpix, nside, T = 3, 65, Float64
P = Flat{Θpix,nside}
## covariances
Cf = Cℓ_to_cov(T,P,S0,S2,cls[:ℓ],cls[:tt],   cls[:te],   cls[:ee],   cls[:bb])
Cf̃ = Cℓ_to_cov(T,P,S0,S2,cls[:ℓ],cls[:ln_tt],   cls[:ln_te],   cls[:ln_ee],   cls[:ln_bb])
Cϕ = Cℓ_to_cov(T,P,S0,   cls[:ℓ],cls[:ϕϕ])
μKarcminT = 1
Ωpix = deg2rad(Θpix/60)^2
CN = FullDiagOp(FlatIQUMap{T,P}(repeated(fill(μKarcminT^2 * Ωpix,(nside,nside)),3)...))
CN̂  = FullDiagOp(FieldTuple(
  FlatS0Fourier{T,P}(repeated(fill(μKarcminT^2 * Ωpix,(nside÷2+1,nside)),1)...),
FlatS2EBFourier{T,P}(repeated(fill(μKarcminT^2 * Ωpix,(nside÷2+1,nside)),2)...)
))
##
f = Ł(simulate(Cf))
ϕ = Ł(simulate(Cϕ))
δfϕ,δfϕ′ = (δf,δϕ),(δf′,δϕ′) = @repeated(Ł(FieldTuple(simulate(Cf),simulate(Cϕ))),2)
##
ϵ = 1e-7

## LenseFlow tests
L = LenseFlow
f̃ = L(ϕ)*f
# I'm unable to get a good t=1 likelihood derivative without some sort of masking,
ℓmax_mask, Δℓ_taper = 3000, 500
Ml = [ones(ℓmax_mask); (cos(linspace(0,π,Δℓ_taper))+1)/2]
Md = Cℓ_to_cov(T,P,S0,S2,1:(ℓmax_mask+Δℓ_taper),repeated(Ml,4)...) * Squash
ds = DataSet(L(ϕ)*f + simulate(CN), CN, Cf, Cϕ, Md, Md, Squash);
##
close("all")
plot(get_Cℓ((let δf̃ϕ_δfϕ = δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f)
    (δlnP_δfϕₜ(1,f̃+ϵ*δf,ϕ+ϵ*δϕ,ds,L) - δlnP_δfϕₜ(1,f̃-ϵ*δf,ϕ-ϵ*δϕ,ds,L))/(2ϵ)
end)[2])...)
plot(get_Cℓ((H_lnP(Val{1.},L(ϕ),L(ϕ)*f,ds) * δfϕ)[2])...)
yscale("log")
##

gfϕ = δlnP_δfϕₜ(1,f̃,ϕ,ds,L)

[(let δfϕ_δf̃ϕ = δfϕ_δf̃ϕ(L(ϕ,ode4{50}),f,L(ϕ)*f)
    δfϕ_δf̃ϕ' \ (δfϕ_δf̃ϕ' * δfϕ)
end)[2], δfϕ[2]] |> plot
##
[(let δfϕ_δf̃ϕ = δfϕ_δf̃ϕ(L(ϕ,ode4{50}),f,L(ϕ)*f)
    δfϕ_δf̃ϕ' \ gfϕ
end)[2],
(let δfϕ_δf̃ϕ = δfϕ_δf̃ϕ(L(ϕ,ode4{50}),f,L(ϕ)*f)
    δfϕ_δf̃ϕ' * (δfϕ_δf̃ϕ' \ gfϕ)
end)[2],
gfϕ[2]] |> plot
##
let δfϕ_δf̃ϕ = δfϕ_δf̃ϕ(L(ϕ),f,L(ϕ)*f)
    (δfϕ ⋅ (δfϕ_δf̃ϕ \ δfϕ′)) , ((δfϕ_δf̃ϕ' \ δfϕ) ⋅ δfϕ′)
end

##
@testset "LenseFlow Jacobian" begin
    let δf̃ϕ_δfϕ = δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f)
        # Jacobian
        @test (L(ϕ+ϵ*δϕ)*(f+ϵ*δf) - L(ϕ-ϵ*δϕ)*(f-ϵ*δf))/(2ϵ) ≈ (δf̃ϕ_δfϕ * δfϕ)[1]
        # inverse Jacobian
        @test (δf̃ϕ_δfϕ \ (δf̃ϕ_δfϕ * δfϕ)) ≈ δfϕ
        # Jacobian transpose
        @test (δfϕ ⋅ (δf̃ϕ_δfϕ * δfϕ′)) ≈ ((δfϕ * δf̃ϕ_δfϕ) ⋅ δfϕ′)
        # Jacobian inverse transpose
        @test (δfϕ ⋅ (δf̃ϕ_δfϕ \ δfϕ′)) ≈ ((δf̃ϕ_δfϕ' \ δfϕ) ⋅ δfϕ′)
        # Likelihood gradient at t=0
        @test (lnP(0,f+ϵ*δf,ϕ+ϵ*δϕ,ds,L) - lnP(0,f-ϵ*δf,ϕ-ϵ*δϕ,ds,L))/(2ϵ) ≈ (δlnP_δfϕₜ(0,f,ϕ,ds,L)⋅δfϕ) rtol=1e-5
        # Likelihood gradient at t=1
        @test (lnP(1,f̃+ϵ*δf,ϕ+ϵ*δϕ,ds,L) - lnP(1,f̃-ϵ*δf,ϕ-ϵ*δϕ,ds,L))/(2ϵ) ≈ (δlnP_δfϕₜ(1,f̃,ϕ,ds,L)⋅δfϕ) rtol=1e-4
    end
end

## PowerLens tests
L = PowerLens
ds = DataSet(L(ϕ)*f + simulate(CN), CN, Cf, Cϕ, Md, Md, Squash);

@testset "PowerLens Jacobian" begin
    let δf̃ϕ_δfϕ = δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f)
        # Jacobian
        @test (L(ϕ+ϵ*δϕ)*(f+ϵ*δf) - L(ϕ-ϵ*δϕ)*(f-ϵ*δf))/(2ϵ) ≈ (δf̃ϕ_δfϕ * δfϕ)[1]
        # Jacobian transpose
        @test (δfϕ ⋅ (δf̃ϕ_δfϕ * δfϕ)) ≈ ((δfϕ * δf̃ϕ_δfϕ) ⋅ δfϕ)
        # Likelihood gradient at t=0
        @test (lnP(0,f+ϵ*δf,ϕ+ϵ*δϕ,ds,L) - lnP(0,f-ϵ*δf,ϕ-ϵ*δϕ,ds,L))/(2ϵ) ≈ (δlnP_δfϕₜ(0,f,ϕ,ds,L)⋅FieldTuple(δf,δϕ)) rtol=1e-6
    end
end
##
close("all")
plot(get_Cℓ((let δf̃ϕ_δfϕ = δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f)
    (δlnP_δfϕₜ(0,f+ϵ*δf,ϕ+ϵ*δϕ,ds,L) - δlnP_δfϕₜ(0,f-ϵ*δf,ϕ-ϵ*δϕ,ds,L))/(2ϵ)
end)[1][1])...)
plot(get_Cℓ((H_lnP(Val{0.},L(ϕ),f,ds) * δfϕ)[1][1])...)
yscale("log")
##

let δf̃ϕ_δfϕ = δf̃ϕ_δfϕ(L(ϕ),L(ϕ)*f,f)
    (lnP(0,f+ϵ*δf,ϕ+ϵ*δϕ,ds,L) - lnP(0,f-ϵ*δf,ϕ-ϵ*δϕ,ds,L))/(2ϵ) , (δlnP_δfϕₜ(0,f,ϕ,ds,L)⋅δfϕ)
end

ℕ = FullDiagOp(Field2Tuple(CN̂.f,0Cϕ.f))
𝕊 = FullDiagOp(Field2Tuple(Cf.f,Cϕ.f))
approxℍ⁻¹ = FullDiagOp(FieldTuple(Mf*(@. (CN̂^-1 + Cf^-1)^-1).f, Cϕ.f))



##
fcur,ϕcur = fϕcur = FieldTuple(𝕎(Cf,CN̂)*ds.d,0ϕ)
gfϕ = δlnP_δfϕₜ(1,fϕcur...,ds,L)
##

[(approxℍ⁻¹ * gfϕ)[2], ϕ]|> plot

x,hist = nothing,nothing
let δfϕ_δf̃ϕ = δfϕ_δf̃ϕ(L(ϕcur),fcur,fcur)
    P = sqrt.(approxℍ⁻¹)
    A = Squash * ℕ^-1 + δfϕ_δf̃ϕ' * (Squash * 𝕊^-1 * δfϕ_δf̃ϕ)
    x,hist = mypcg(A,gfϕ,P; nsteps=500)
end
let δfϕ_δf̃ϕ = δfϕ_δf̃ϕ(L(ϕcur),fcur,fcur)
    A = Squash * ℕ^-1 + δfϕ_δf̃ϕ' * (Squash * 𝕊^-1 * δfϕ_δf̃ϕ)
    plot([(A*x)[2], gfϕ[2]])
end
loglog(hist)
[x[2], (approxℍ⁻¹ * gfϕ)[2], ϕ] |> plot
[x[1], (approxℍ⁻¹ * gfϕ)[1], Mf*(f-fcur)] |> plot



semilogy(hist)


[Cϕ * gfϕ[2], ((ℕ + 𝕊) * gfϕ)[2], ϕ] |> plot

##
semilogy(get_Cℓ((let δfϕ_δf̃ϕ = δfϕ_δf̃ϕ(L(ϕcur),fcur,fcur)
    (Squash * ℕ^-1 + δfϕ_δf̃ϕ' * (Squash * 𝕊^-1 * δfϕ_δf̃ϕ)) * gfϕ
end)[2])...)
semilogy(get_Cℓ(((Squash * ℕ^-1 + Squash * 𝕊^-1) / Ωpix * gfϕ)[2])...)
semilogy(get_Cℓ(ϕ)...)
##
##
semilogy(get_Cℓ((let δfϕ_δf̃ϕ = δfϕ_δf̃ϕ(L(ϕcur),fcur,fcur)
    (Squash * ℕ^-1 + δfϕ_δf̃ϕ' * (Squash * 𝕊^-1 * δfϕ_δf̃ϕ)) * gfϕ
end)[1][1])...)
semilogy(get_Cℓ(((Squash * ℕ^-1 + Squash * 𝕊^-1) / Ωpix * gfϕ)[1][1])...)
##


##
semilogy(get_Cℓ((let δfϕ_δf̃ϕ = δfϕ_δf̃ϕ(L(ϕcur),fcur,fcur)
    (ℕ + δfϕ_δf̃ϕ' * (𝕊 * δfϕ_δf̃ϕ)) * gfϕ
end)[2])...)
semilogy(get_Cℓ(((ℕ + 𝕊) * gfϕ)[2])...)
##


[((@. nan2zero((ℕ + 𝕊)^1)) * gfϕ)[2],ϕ] |> plot

(ℕ + δf̃ϕ_δfϕ' * (𝕊 * δf̃ϕ_δfϕ))


using IterativeSolvers

cg(eye(4),ones(4))
