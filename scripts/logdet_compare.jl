push!(LOAD_PATH, pwd()*"/src")

using CMBLensing
using BayesLensSPTpol: class
cls = class(lmax=4000);


"""
Gets a matrix representation of an operator in the T->T basis
TODO: needs some tweaks to work generally then move into main source
"""
function matrix{F<:Field}(::Type{F}, L::LinOp)
    hcat(((F(L*(x=zeros(length(F)); x[i]=1; x)[Tuple{F}]))[:] for i=1:length(F))...);
end


nside = 32

for nside=[8,16,32,64]
    T = Float64
    P = Flat{1,nside}
    ϕ = simulate(Cℓ_to_cov(P, S0, T.(cls[:ell]), T.(cls[:ϕϕ])))
    # @show logdet(matrix(FlatS0Map{T,P},LenseFlow(ϕ,CMBLensing.ode45{0,nside/100,100,false})))
    @show logdet(matrix(FlatS0Map{T,P},LenseFlow(ϕ,CMBLensing.ode45{1e-3,1e-3,100,false})))
end


# @show logdet(matrix(FlatS0Map{T,P},PowerLens(ϕ)))
