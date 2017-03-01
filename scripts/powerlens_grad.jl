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

# ϕ₀ = simulate(Cϕ)  |> LenseBasis
ϕ₀ = zero(FlatS0Map{T,P})
f₀ = simulate(Cf) |> LenseBasis
L = PowerLens(ϕ₀)
df̃ = L*Ł(f₀) + simulate(CN)
ϵ = 1e-7
δϕ = simulate(Cϕ)
δf = simulate(Cf)

##
function lnL(f,ϕ,df̃) 
    Δf̃ = df̃ - PowerLens(ϕ)*Ł(f)
    (Δf̃⋅(Cmask*(CN^-1*Δf̃)) + f⋅(Cmask*(Cf^-1*f)) + ϕ⋅(Cϕ^-1*ϕ))/2
end

function dlnL_dfϕ(f,ϕ,df̃)
    L = PowerLens(ϕ)
    Δf̃ = df̃ - L*Ł(f)
    [-df̃dfᵀ(L, Ł(Cmask*(CN^-1*Δf̃))) + Cmask*(Cf^-1*f),
     -df̃dϕᵀ(L, f, Ł(Cmask*(CN^-1*Δf̃))) + Cϕ^-1*ϕ]
end

##

# check accuracy of derviative against numerical
dlnL = dlnL_dfϕ(f₀,ϕ₀,df̃)
dlnL[1]⋅δf + dlnL[2]⋅δϕ
(lnL(f₀+ϵ*δf,ϕ₀+ϵ*δϕ,df̃) - lnL(f₀-ϵ*δf,ϕ₀-ϵ*δϕ,df̃))/(2ϵ)

0.002 / 1147 * 100
##


using Optim

##

# solve for f given ϕ=0

μKarcminT = 0.1
CN  = LinearDiagOp(FlatS0Map{T,P}(fill(μKarcminT^2 * Ωpix,(nside,nside))))
Cmask = 1

f₀ = Ł(simulate(Cf))
ϕ₀ = zero(FlatS0Map{T,P})
L = PowerLens(ϕ₀)
df̃ = L*f₀ + simulate(CN)
fs = Ł(simulate(Cf))

[-dlnL_dfϕ(fs,ϕ₀,df̃)[1],f₀] |> plot
##

γ = 1e-10

res = optimize(
    x->γ*lnL(x[~f₀],ϕ₀,df̃), 
    (x,∇f)->(∇f .= γ*dlnL_dfϕ(x[~f₀],ϕ₀,df̃)[1][:]),
    fs[:],
    GradientDescent(),
    Optim.Options(time_limit = 10.0, x_tol=1e-1, f_tol=1e-3, g_tol=1e-1))
    
[fs,res.minimizer[~f₀],f₀] |> plot

lnL(fs,ϕ₀), lnL(res.minimizer[~f₀],ϕ₀)

##

# solve for f given some fixed ϕ!=0

f₀ = Ł(simulate(Cf))
ϕ₀ = Ł(simulate(Cϕ))
L = PowerLens(ϕ₀; order=1)
df̃ = L*f₀ + simulate(CN)
fs = Ł(simulate(Cf))

γ = 1e-10

res = optimize(
    x->γ*lnL(x[~f₀],ϕ₀,df̃), 
    (x,∇f)->(∇f .= γ*dlnL_dfϕ(x[~f₀],ϕ₀,df̃)[1][:]),
    fs[:],
    GradientDescent(),
    Optim.Options(time_limit = 30.0))
    
[fs,res.minimizer[~f₀],f₀] |> plot
res
##


# solve for ϕ given some fixed f

f₀ = Ł(simulate(Cf))
ϕ₀ = Ł(simulate(Cϕ))
L = PowerLens(ϕ₀)
df̃ = L*f₀ + simulate(CN)
ϕs = Ł(simulate(Cϕ))

##
γ = 1e-18

res = optimize(
    x->γ*lnL(f₀,x[~ϕ₀],df̃), 
    (x,∇f)->(∇f .= γ*dlnL_dfϕ(f₀,x[~ϕ₀],df̃)[2][:]),
    ϕs[:],
    GradientDescent(),
    Optim.Options(time_limit = 300.0))
    
[ϕs,res.minimizer[~ϕ₀],ϕ₀] |> plot

res

df̃-L*f₀ |> plot
