push!(LOAD_PATH, pwd()*"/src")

using CMBFields
using BayesLensSPTpol: class
cls = class();

nside = 16
T = Float64
P = Flat{1,nside}

ϕ = simulate(Cℓ_to_cov(P, S0, cls[:ell], cls[:ϕϕ]))

"""
Gets a matrix representation of an operator in the T->T basis
TODO: needs some tweaks to work generally then move into main source
"""
function matrix{T<:Field}(::Type{T}, L::LinOp)
    hcat(((convert(T,L*T((x=zeros(nside,nside); x[i]=1; x))))[:] for i=1:nside^2)...);
end

L_taylens = PowerLens(ϕ)
L_lenseflow = LenseFlowOp(ϕ)

@show logdet(matrix(FlatS0Map{T,P},L_lenseflow))
@show logdet(matrix(FlatS0Map{T,P},L_taylens))
