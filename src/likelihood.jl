using NamedTuples

"""
Stores variables needed to construct the likelihood
"""
const DataSet=@NT(d,CN,Cf,Cϕ,Cmask)

"""
The log posterior probability, lnP, s.t. 

-2lnP(f,ϕ) = (d - f̃)ᵀ*CN⁻¹*(d - f̃) + fᵀ*Cf⁻¹*f + ϕᵀ*Cϕ⁻¹*ϕ

# Arguments:
* f,ϕ : the T and/or EB field and lensing potential at time t
* ds : the DataSet (includes the data and signal/noise covariances)
* t : the time at which f,ϕ are specified, e.g. if t=0, f is the unlensed field

"""
lnP(f,ϕ,ds,t::Real,::Type{L}=LenseFlowOp) where {L<:LenseOp} = lnP(f,ϕ,ds,Val{float(t)},L)
lnP(Δ,f,ϕ,ds::DataSet) = -(Δ⋅(ds.Cmask*(ds.CN\Δ)) + f⋅(ds.Cmask*(ds.Cf\f)) + ϕ⋅(ds.Cϕ\ϕ))/2
lnP(f,ϕ,ds,::Type{Val{0.}},::Type{L}) where {L<:LenseOp} = lnP(ds.d-L(ϕ)*f,f,ϕ,ds)
lnP(f̃,ϕ,ds,::Type{Val{1.}},::Type{L}) where {L<:LenseFlowOp} = lnP(ds.d-f̃,L(ϕ)\f̃,ϕ,ds)

"""
Gradient of the log posterior probability with
respect to the field f and lensing potential ϕ. See `lnP` for definition of
arguments. 

Returns : 
"""
δlnP_δfϕ(f,ϕ,ds,t::Real,::Type{L}=LenseFlowOp) where {L<:LenseOp} = δlnP_δfϕ(f,ϕ,ds,Val{float(t)},L)

function δlnP_δfϕ(f,ϕ,ds,::Type{Val{0.}},::Type{L}) where {L<:LenseOp}
    Lϕ = L(ϕ)
    Δ =  ds.d - Lϕ*f
    δlnL_δf, δlnL_δϕ = (δf̃_δfϕᵀ(Lϕ,f)*Ł(ds.Cmask*(ds.CN\Δ))) # derivatives of the likelihood term
    (δlnL_δf - ds.Cmask*(ds.Cf\f), δlnL_δϕ - ds.Cϕ\ϕ)
end



#=
function dlnL_dfϕ(f,ϕ,df̃,LenseOp)
    L = LenseOp(ϕ)
    Δf̃ = df̃ - L*Ł(f)
    df̃dfᵀ,df̃dϕᵀ = dLdf̃_df̃dfϕ(L,Ł(f),Ł(Cmask*(CN\Δf̃)))
    [-df̃dfᵀ + Cmask*(Cf\f), -df̃dϕᵀ + Cϕ\ϕ]
end
function dlnL̃_df̃ϕ(f̃,ϕ,df̃,LenseOp)
    L = LenseOp(ϕ)
    f = L\f̃
    Δf̃ = df̃ - f̃
    dfdf̃ᵀ,dfdϕᵀ = dLdf_dfdf̃ϕ(L,Ł(f),Ł(Cmask*(Cf\f)))
    [-Ł(Cmask*(CN\Δf̃)) + dfdf̃ᵀ, dfdϕᵀ + Cϕ\ϕ]
end
=#
