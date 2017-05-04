push!(LOAD_PATH, pwd()*"/src")
using CMBLensing
using BayesLensSPTpol: class


cls = class()

nside = 64
T = Float64
P = Flat{1,nside}

CT = Cℓ_to_cov(P,S0,cls[:ell],cls[:tt])
Cϕ = Cℓ_to_cov(P,S0,cls[:ell],cls[:ϕϕ])

f = simulate(CT) |> LenseBasis
ϕ = simulate(Cϕ) |> LenseBasis

# test inverse lensing is exact
rtol = 1e-3
L = LenseFlow(ϕ, ode45{rtol,rtol})
ρ = ((L\(L*f))/f)[:Tx]
@assert all(isapprox.(ρ,1; rtol=rtol))

# test propagation of perturbations δf and δϕ
ϵ = 1e-5
δf = ϵ*simulate(CT) |> LenseBasis
δϕ = ϵ*simulate(Cϕ) |> LenseBasis
ρ = (δlense_flow(L,f,δf,δϕ,[0,1])[2] / (LenseFlow(ϕ+δϕ)*(f+δf)-LenseFlow(ϕ)*f))[:Tx]
maximum(ρ), minimum(ρ)
@assert all(isapprox.(ρ, 1; rtol=1e-2))
