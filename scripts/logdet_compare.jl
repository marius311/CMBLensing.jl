push!(LOAD_PATH, pwd()*"/src")

using CMBFields
using BayesLensSPTpol: class
cls = class(lmax=4000);


"""
Gets a matrix representation of an operator in the T->T basis
TODO: needs some tweaks to work generally then move into main source
"""
function matrix{F<:Field}(::Type{F}, L::LinOp)
    hcat(((F(L*F((x=zeros(nside,nside); x[i]=1; x))))[:] for i=1:nside^2)...);
end

nside = 32

for nside=[8,16,32,64]
    T = Float64
    P = Flat{1,nside}
    ϕ = simulate(Cℓ_to_cov(P, S0, T.(cls[:ell]), T.(cls[:ϕϕ])))
    # @show logdet(matrix(FlatS0Map{T,P},LenseFlowOp(ϕ,CMBFields.ode45{0,nside/100,100,false})))
    @show logdet(matrix(FlatS0Map{T,P},LenseFlowOp(ϕ,CMBFields.ode45{1e-3,1e-3,100,false})))
end


# @show logdet(matrix(FlatS0Map{T,P},PowerLens(ϕ)))
