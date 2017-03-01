push!(LOAD_PATH, pwd()*"/src")
using CMBFields

using BayesLensSPTpol
cls = class();

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

μKarcminT = 5
Ωpix = deg2rad(Θpix/60)^2

##

Cf = CT
CN  = LinearDiagOp(FlatS0Map{T,P}(fill(μKarcminT^2 * Ωpix,(nside,nside))))
Cmask = 1#LinearDiagOp(FlatS0Fourier{T,P}(cls_to_cXXk(1:10000,[l<3000 ? 1 : 0 for l=1:10000],g.r)[1:g.nside÷2+1,:]))

# Cmask = LinearDiagOp(FlatS2EBFourier{T,P}((cls_to_cXXk(1:10000,[l<5000 ? 1 : 0 for l=1:10000],g.r)[1:g.nside÷2+1,:] for i=1:2)...))
# CN  = LinearDiagOp(FlatS2QUMap{T,P}((fill(μKarcminT^2 * Ωpix,(nside,nside)) for i=1:2)...))
# Cf = CEB

##

ϕ₀ = simulate(Cϕ)  |> LenseBasis
f₀ = simulate(Cf) |> LenseBasis
L = LenseFlowOp(ϕ₀)
df̃ = L*Ł(f₀) + simulate(CN)
ϵ = 1e-7
δϕ = simulate(Cϕ)
δf = simulate(Cf)

##
function lnL(f,ϕ,df̃,LenseOp) 
    Δf̃ = df̃ - LenseOp(ϕ)*Ł(f)
    (Δf̃⋅(Cmask*(CN^-1*Δf̃)) + f⋅(Cmask*(Cf^-1*f)) + ϕ⋅(Cϕ^-1*ϕ))/2
end

function dlnL_dfϕ(f,ϕ,df̃,LenseOp)
    L = LenseOp(ϕ)
    Δf̃ = df̃ - L*Ł(f)
    df̃dfᵀ,df̃dϕᵀ = dLdf̃_df̃dfϕ(L,Ł(f),Ł(Cmask*(CN^-1*Δf̃)))
    [-df̃dfᵀ + Cmask*(Cf^-1*f), -df̃dϕᵀ + Cϕ^-1*ϕ]
end

##
dlnL = dlnL_dfϕ(f₀,ϕ₀,df̃,LenseFlowOp)
dlnL[1]⋅δf + dlnL[2]⋅δϕ , (lnL(f₀+ϵ*δf,ϕ₀+ϵ*δϕ,df̃,LenseFlowOp) - lnL(f₀-ϵ*δf,ϕ₀-ϵ*δϕ,df̃,LenseFlowOp))/(2ϵ)
##
dlnL = dlnL_dfϕ(f₀,ϕ₀,df̃,PowerLens)
dlnL[1]⋅δf + dlnL[2]⋅δϕ , (lnL(f₀+ϵ*δf,ϕ₀+ϵ*δϕ,df̃,PowerLens) - lnL(f₀-ϵ*δf,ϕ₀-ϵ*δϕ,df̃,PowerLens))/(2ϵ)
##
f = 1.2f₀
Δf̃ = df̃ - L*Ł(f)
dLdf̃ = Ł(Cmask*(CN^-1*Δf̃))
##
[dLdf̃_df̃dfϕ(LenseFlowOp(ϕ₀),f,dLdf̃) dLdf̃_df̃dfϕ(PowerLens(ϕ₀),f,dLdf̃)] |> plot
