push!(LOAD_PATH, pwd()*"/src")
using CMBFields

using BayesLensSPTpol
cls = class();
##

using PyPlot
cls = Main.cls
nside = 512
T = Float64
Θpix = 1
P = Flat{Θpix,nside}
g = FFTgrid(T,P)

clipcov(op::LinDiagOp) = (for d in fieldvalues(op.f) d[d.==0] = minimum(abs(d[d.!=0])/1e4) end; op)
CEB = Cℓ_to_cov(P,S2,cls[:ell],cls[:ee],1e-4*cls[:bb])  |> clipcov
CT  = Cℓ_to_cov(P,S0,cls[:ell],cls[:tt])                |> clipcov
Cϕ  = Cℓ_to_cov(P,S0,cls[:ell],cls[:ϕϕ])                |> clipcov

μKarcminT = 0.001
Ωpix = deg2rad(Θpix/60)^2
##
Cf = CT
CN  = FullDiagOp(FlatS0Map{T,P}(fill(μKarcminT^2 * Ωpix,(nside,nside))))
CÑ = FullDiagOp(FlatS0Fourier{T,P}(map(x->complex(x)[1:size(x,1)÷2+1,:],fieldvalues(CN.f))...))
Cmask = FullDiagOp(FlatS0Fourier{T,P}(cls_to_cXXk(1:10000,[l<4000 ? 1 : 0 for l=1:10000],g.r)[1:g.nside÷2+1,:]))
##
Cf = CEB
CN  = FullDiagOp(FlatS2QUMap{T,P}((fill(μKarcminT^2 * Ωpix,(nside,nside)) for i=1:2)...))
CÑ = FullDiagOp(FlatS2EBFourier{T,P}(map(x->complex(x)[1:size(x,1)÷2+1,:],fieldvalues(CN.f))...))
Cmask = FullDiagOp(FlatS2EBFourier{T,P}((cls_to_cXXk(1:10000,[l<6000 ? 1 : 0 for l=1:10000],g.r)[1:g.nside÷2+1,:] for i=1:2)...))

##
ϕ₀ = simulate(Cϕ) |> LenseBasis
f₀ = simulate(Cf) |> LenseBasis
n₀ = simulate(CN) |> LenseBasis
# df̃_lf = LenseFlowOp(ϕ₀)*Ł(f₀) + n₀
df̃_pl = PowerLens(ϕ₀)*f₀ + n₀
ϵ = 1e-7
δϕ = simulate(Cϕ)
δf = simulate(Cf)
##
L = PowerLens(ϕ₀)
@time L*f₀

slowlens(L,f₀)

##
using ProfileView
Profile.clear()  # in case we have any previous profiling data
@profile L*f₀
ProfileView.view()
##
@benchmark L.∇ϕ'*inv(eye(2)+0.3*L.Jϕ)*Ł(∇*f₀)
@benchmark L.∇ϕ'*L.Jϕ*L.∇ϕ
@benchmark inv(eye(2)+0.3*L.Jϕ)
@benchmark Ł(∇*f₀)
@benchmark L.∇ϕ'*(L.Jϕ)*Ł(∇*f₀)
@benchmark Fourier(f₀)
@benchmark inv(L.Jϕ)
@benchmark velocity(L,f₀,0.5)
f̃,ϕ
50 * 40
##
close("all")
L = LenseFlowOp(ϕ₀,ode45{1e-3,0,20})
f₀′ = L\(L*f₀)
f₀′ - f₀ |> plot
# plt[:hist]((f₀′ / f₀ - 1)[:],range=(-0.5,0.5))
# plt[:hist]((f₀′ / f₀ - 1)[:],bins=50,range=(-0.001,0.001))
##
function lnL(f,ϕ,df̃,LenseOp) 
    Δf̃ = df̃ - LenseOp(ϕ)*Ł(f)
    (Δf̃⋅(Cmask*(CN\Δf̃)) + f⋅(Cmask*(Cf\f)) + ϕ⋅(Cϕ\ϕ))/2
end

function dlnL_dfϕ(f,ϕ,df̃,LenseOp)
    L = LenseOp(ϕ)
    Δf̃ = df̃ - L*Ł(f)
    df̃dfᵀ,df̃dϕᵀ = dLdf̃_df̃dfϕ(L,Ł(f),Ł(Cmask*(CN\Δf̃)))
    [-df̃dfᵀ + Cmask*(Cf\f), -df̃dϕᵀ + Cϕ\ϕ]
end
#
function lnL̃(f̃,ϕ,df̃,LenseOp) 
    Δf̃ = df̃ - f̃
    f = LenseOp(ϕ)\f̃
    (Δf̃⋅(Cmask*(CN\Δf̃)) + f⋅(Cmask*(Cf\f)) + ϕ⋅(Cϕ\ϕ))/2
end
function dlnL̃_df̃ϕ(f̃,ϕ,df̃,LenseOp)
    L = LenseOp(ϕ)
    f = L\f̃
    Δf̃ = df̃ - f̃
    dfdf̃ᵀ,dfdϕᵀ = dLdf_dfdf̃ϕ(L,Ł(f),Ł(Cmask*(Cf\f)))
    [-Ł(Cmask*(CN\Δf̃)) + dfdf̃ᵀ, dfdϕᵀ + Cϕ\ϕ]
end
##
dlnL = dlnL_dfϕ(f₀,ϕ₀,df̃_lf,LenseFlowOp)
dlnL[1]⋅δf + dlnL[2]⋅δϕ , (lnL(f₀+ϵ*δf,ϕ₀+ϵ*δϕ,df̃_lf,LenseFlowOp) - lnL(f₀-ϵ*δf,ϕ₀-ϵ*δϕ,df̃_lf,LenseFlowOp))/(2ϵ)
##
f̃₀ = LenseFlowOp(ϕ₀)*f₀
dlnL = dlnL̃_df̃ϕ(f̃₀,ϕ₀,df̃_lf,LenseFlowOp)
dlnL[1]⋅δf + dlnL[2]⋅δϕ , (lnL̃(f̃₀+ϵ*δf,ϕ₀+ϵ*δϕ,df̃_lf,LenseFlowOp) - lnL̃(f̃₀-ϵ*δf,ϕ₀-ϵ*δϕ,df̃_lf,LenseFlowOp))/(2ϵ)
##

dlnL = dlnL_dfϕ(f₀,ϕ₀,df̃_pl,PowerLens)
dlnL[1]⋅δf + dlnL[2]⋅δϕ , (lnL(f₀+ϵ*δf,ϕ₀+ϵ*δϕ,df̃_pl,PowerLens) - lnL(f₀-ϵ*δf,ϕ₀-ϵ*δϕ,df̃_pl,PowerLens))/(2ϵ)
##
using Optim
f₀ = simulate(Cf) |> LenseBasis
ϕ₀ = simulate(Cϕ) |> LenseBasis
L = PowerLens(ϕ₀)
df̃ = L*Ł(f₀) + simulate(CN)
##
fstart = [Ł(𝕎(Cf,CÑ)*df̃), zero(FlatS0Map{T,P})]
[f₀,fstart[1],df̃] |> plot
##
import Base.LinAlg.A_ldiv_B!
struct foo end
A_ldiv_B!(s,::foo,q) = ((f,ϕ) = q[~(f₀,ϕ₀)]; s.=[Ł(CÑ*f),Ł(Cϕ*ϕ)][:])
##
res = optimize(
    x->lnL̃(x[~(f₀,ϕ₀)]...,df̃,LenseFlowOp),
    (x,∇f)->(∇f .= dlnL̃_df̃ϕ(x[~(f₀,ϕ₀)]...,df̃,LenseFlowOp)[:]),
    fstart[:],
    LBFGS(P=foo()),
    Optim.Options(time_limit = 600.0, store_trace=true, show_trace=true))
##
fstart = res.minimizer[~(f₀,ϕ₀)]
fstart[1] = LenseFlowOp(fstart[2])\fstart[1]
##
res2 = optimize(
    x->lnL(x[~(f₀,ϕ₀)]...,df̃,LenseFlowOp),
    (x,∇f)->(∇f .= dlnL_dfϕ(x[~(f₀,ϕ₀)]...,df̃,LenseFlowOp)[:]),
    fstart[:],
    LBFGS(P=foo()),
    Optim.Options(time_limit = 60.0, store_trace=true, show_trace=true))
##
[f₀,fstart[1]-f₀,res.minimizer[~(f₀,ϕ₀)][1]-f₀] |> plot
[ϕ₀,fstart[2]-ϕ₀,res.minimizer[~(f₀,ϕ₀)][2]-ϕ₀] |> plot
[fstart[2] res.minimizer[~(f₀,ϕ₀)][2] ϕ₀] |> plot
plot([res.minimizer[~(f₀,ϕ₀)][2] ϕ₀]; vmin=-6e-6, vmax=6e-6)
norm(f₀[:],1)
maximum(abs(f₀[:]))
##
[f₀,dlnL_dfϕ(fstart...,df̃,LenseFlowOp)[1]] |> plot
[f₀, dlnL_dfϕ(f₀,ϕ₀,df̃,LenseFlowOp)[1]] |> plot
[-Cϕ*dlnL_dfϕ(𝕎(Cf,CÑ)*df̃,0ϕ₀,df̃,LenseFlowOp)[2] -Cϕ*dlnL_dfϕ(𝕎(Cf,CÑ)*df̃,0.9ϕ₀,df̃,LenseFlowOp)[2] ϕ₀;
 -Cϕ*dlnL_dfϕ(𝕎(Cf,CÑ)*df̃,0ϕ₀,df̃,PowerLens)[2] -Cϕ*dlnL_dfϕ(𝕎(Cf,CÑ)*df̃,0.9ϕ₀,df̃,PowerLens)[2] ϕ₀] |> plot
plot([Cϕ*dlnL_dfϕ(𝕎(Cf,CÑ)*df̃,4ϕ₀,df̃,LenseFlowOp)[2] Cϕ*dlnL_dfϕ(𝕎(Cf,CÑ)*df̃,4ϕ₀,df̃,PowerLens)[2]])#; vmin=-2e16, vmax=2e16)

[-dlnL_dfϕ(0𝕎(Cf,CÑ)*df̃,0ϕ₀,df̃,PowerLens)[1] f₀] |> plot
##
fstart = [Ł(𝕎(Cf,CÑ)*df̃), zero(FlatS0Map{T,P})]
∇L = dlnL_dfϕ(fstart...,df̃,PowerLens)
iP_∇L = [(CÑ^-1+Cf^-1)^-1*∇L[1], Cϕ*∇L[2]]
l = lnL(fstart...,df̃,PowerLens)
close("all")
α=logspace(-10,-12,100)
loglog(α,[-(lnL((fstart - α*iP_∇L)...,df̃,PowerLens)-l) for α=α])
# yscale("symlog")
##
fstart = [Ł(𝕎(Cf,CÑ)*df̃), zero(FlatS0Map{T,P})]
##
∇L = dlnL̃_df̃ϕ(fstart...,df̃,LenseFlowOp)
iP_∇L = [CÑ*∇L[1], Cϕ*∇L[2]]
l = lnL̃(fstart...,df̃,LenseFlowOp)
close("all")
α=logspace(log10(0.4),-3,100)
semilogx(α,[(l-lnL̃((fstart - α*iP_∇L)...,df̃,LenseFlowOp)) for α=α])
##
ylim(-1000,1000)
lnL̃((fstart - 0*iP_∇L)...,df̃,LenseFlowOp)
lnL̃((fstart - 0.17*iP_∇L)...,df̃,LenseFlowOp)

fstart = (fstart - 0.17*iP_∇L)


[ϕ₀ fstart[2]] |> plot
