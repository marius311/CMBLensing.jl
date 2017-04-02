export DataSet, lnP, δlnP_δfϕ


"""
Stores variables needed to construct the likelihood
"""
const DataSet=@NT(d,CN,Cf,Cϕ,Mf,Mϕ)

"""
The log posterior probability, lnP, s.t. 

-2lnP(f,ϕ) = (d - f̃)ᵀ*CN⁻¹*(d - f̃) + fᵀ*Cf⁻¹*f + ϕᵀ*Cϕ⁻¹*ϕ

# Arguments:
* f : the T/E/B field at time t
* t : the time at which f is specified (i.e. t=0 means f is the unlensed field, t=1 means f is the lensed field)
* ϕ : the lensing potential
* ds : the DataSet (includes the data and signal/noise covariances)
* L : the Lensing operator to use
"""
lnP(fₜ,t::Real,ϕ,ds,::Type{L}=LenseFlow) where {L} = lnP(fₜ,Val{t},ϕ,ds,L(ϕ))
lnP(fₜ,::Type{Val{t}},ϕ,ds,L::LenseOp) where {t} = lnP(ds.d-L[t→1]*fₜ, L[t→0]*fₜ, ϕ, ds) 
lnP(Δ,f,ϕ,ds) = -(Δ⋅(ds.Mf*(ds.CN\Δ)) + f⋅(ds.Mf*(ds.Cf\f)) + ϕ⋅(ds.Mϕ*(ds.Cϕ\ϕ)))/2

"""
Gradient of the log posterior probability with
respect to the field f and lensing potential ϕ. See `lnP` for definition of
arguments. 

Returns : 
"""
δlnP_δfₜϕ(fₜ,t::Real,ϕ,ds,::Type{L}=LenseFlowOp) where {L} = δlnP_δfₜϕ(fₜ,Val{float(t)},ϕ,ds,L(ϕ))
function δlnP_δfₜϕ(fₜ,::Type{Val{t}},ϕ,ds,L::LenseOp) where {t}
    f̃ =  L[t→1]*fₜ
    f =  L[t→0]*fₜ
    
    (   δlnL_δf̃(f̃,ds) * δf̃_δfₜϕ(L,f̃,fₜ,Val{t})
     .+ δlnΠ_δf(f,ds) * δf_δfₜϕ(L,f,fₜ,Val{t})
     .+ δlnΠᵩ_δfϕ(ϕ,ds))
end

# derivatives of the three posterior probability terms at the times at which
# they're easy to take
δlnL_δf̃(f̃,ds) = (Δ=ds.d-f̃; ds.Mf*(ds.CN\Δ))
δlnΠ_δf(f,ds) = -ds.Mf*(ds.Cf\f)
δlnΠᵩ_δfϕ(ϕ,ds) = (0, -ds.Mϕ*(ds.Cϕ\ϕ))
