push!(LOAD_PATH, pwd()*"/src")
using CMBFields

using BayesLensSPTpol
cls = class();


broadcast{F<:Field}(f,x::F) = F((broadcast(f,d) for d=data(x))...)

# technically dotting can happen in any basis and I should not have to hard code
# these basis conversions, but until bugs worked just do it in this hacky way...
import Base.LinAlg: dot
dot(a::FlatS2,b::FlatS2) = (EBMap(a)[:] ⋅ EBMap(b)[:]) * FFTgrid(typeof(a).parameters...).Δx^2
dot(a::FlatS0,b::FlatS0) = (Map(a)[:] ⋅ Map(b)[:]) * FFTgrid(typeof(a).parameters...).Δx^2


using PyPlot
cls = Main.cls
nside = 32
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

# Cf = CT
# CN  = LinearDiagOp(FlatS0Map{T,P}(fill(μKarcminT^2 * Ωpix,(nside,nside))))
# Cmask = LinearDiagOp(FlatS0Fourier{T,P}(cls_to_cXXk(1:10000,[l<5000 ? 1 : 0 for l=1:10000],g.r)[1:g.nside÷2+1,:]))

Cmask = LinearDiagOp(FlatS2EBFourier{T,P}((cls_to_cXXk(1:10000,[l<5000 ? 1 : 0 for l=1:10000],g.r)[1:g.nside÷2+1,:] for i=1:2)...))
CN  = LinearDiagOp(FlatS2QUMap{T,P}((fill(μKarcminT^2 * Ωpix,(nside,nside)) for i=1:2)...))
Cf = CEB



ϕ₀ = simulate(Cϕ)  |> LenseBasis
f₀ = simulate(Cf) |> LenseBasis
L = PowerLens(ϕ₀)
df̃ = L*Ł(f₀) + simulate(CN)
ϵ = 1e-7
δϕ = simulate(Cϕ)
δf = simulate(Cf)


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


# check accuracy of derviative against numerical
dlnL = dlnL_dfϕ(f₀,ϕ₀)
dlnL[1]⋅δf + dlnL[2]⋅δϕ
(lnL(f₀+ϵ*δf,ϕ₀+ϵ*δϕ) - lnL(f₀-ϵ*δf,ϕ₀-ϵ*δϕ))/(2ϵ)



using Optim
optimize(x->norm(x), (x,dx)->(dx.=x), [1.,2], GradientDescent())

fstart = L*f₀
res = optimize(
    x->lnL(x[~(f₀,ϕ₀)]...), 
    (x,∇x)->(∇x .= dlnL_dfϕ(x[~(f₀,ϕ₀)]...)[:]),
    [fstart,ϕ₀][:],
    GradientDescent())

f = simulate(Cf)

# TODO: why is this not giving zero???
(Ł(f)[:Bx] - f[:Bx]) |> matshow
