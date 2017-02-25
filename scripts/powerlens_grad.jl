push!(LOAD_PATH, pwd()*"/src")
using CMBFields

using BayesLensSPTpol
cls = class();

##
broadcast{F<:Field}(f,x::F) = F((broadcast(f,d) for d=data(x))...)

##

using PyPlot
cls = Main.cls
nside = 64
T = Float64
Θpix = 1
P = Flat{Θpix,nside}
g = FFTgrid(T,P)

clipcov(op::LinearDiagOp) = (for d in data(op.f) d[d.==0] = minimum(abs(d[d.!=0])/1e4) end; op)
CEB = Cℓ_to_cov(P,S2,cls[:ell],cls[:ee],1e-4*cls[:bb])  |> clipcov
CT  = Cℓ_to_cov(P,S0,cls[:ell],cls[:tt])                |> clipcov
Cϕ  = Cℓ_to_cov(P,S0,cls[:ell],cls[:ϕϕ])                |> clipcov

μKarcminT = 0.1
Ωpix = deg2rad(Θpix/60)^2

##

# Cf = CT
# CN  = LinearDiagOp(FlatS0Map{T,P}(fill(μKarcminT^2 * Ωpix,(nside,nside))))
# Cmask = LinearDiagOp(FlatS0Fourier{T,P}(cls_to_cXXk(1:10000,[l<5000 ? 1 : 0 for l=1:10000],g.r)[1:g.nside÷2+1,:]))

Cmask = LinearDiagOp(FlatS2EBFourier{T,P}((cls_to_cXXk(1:10000,[l<5000 ? 1 : 0 for l=1:10000],g.r)[1:g.nside÷2+1,:] for i=1:2)...))
CN  = LinearDiagOp(FlatS2QUMap{T,P}((fill(μKarcminT^2 * Ωpix,(nside,nside)) for i=1:2)...))
Cf = CEB

##

ϕ₀ = simulate(Cϕ)  |> LenseBasis
f₀ = simulate(Cf) |> LenseBasis
L = PowerLens(ϕ₀)
df̃ = L*Ł(f₀) + simulate(CN)
ϵ = 1e-7
δϕ = simulate(Cϕ)
δf = simulate(Cf)

##

function lnL(f,ϕ) 
    Δf̃ = df̃ - PowerLens(ϕ)*Ł(f)
    (Δf̃⋅(Cmask*(CN^-1*Δf̃)) + f⋅(Cmask*(Cf^-1*f)) + ϕ⋅(Cϕ^-1*ϕ))/2
end

function dlnL_dfϕ(f,ϕ)
    L = PowerLens(ϕ)
    Δf̃ = df̃ - L*Ł(f)
    [-df̃dfᵀ(L, Ł(Cmask*(CN^-1*Δf̃))) + Cmask*(Cf^-1*f),
     -df̃dϕᵀ(L, f, Ł(Cmask*(CN^-1*Δf̃))) + Cϕ^-1*ϕ]
end

##

# check accuracy of derviative against numerical
dlnL = dlnL_dfϕ(f₀,ϕ₀)
dlnL[1]⋅δf + dlnL[2]⋅δϕ
(lnL(f₀+ϵ*δf,ϕ₀+ϵ*δϕ) - lnL(f₀-ϵ*δf,ϕ₀-ϵ*δϕ))/(2ϵ)

##

using Optim

res = optimize(
    x->lnL(x[~(f₀,ϕ₀)]...), 
    (x,∇f)->(∇f .= dlnL_dfϕ(x[~(f₀,ϕ₀)]...)[:]),
    [f₀,ϕ₀][:],
    GradientDescent(),
    Optim.Options(time_limit = 30.0))


res = optimize(
    x->lnL(x[~(f₀,ϕ₀)]...), 
    [f₀,ϕ₀][:],
    Optim.Options(iterations=2))
##
