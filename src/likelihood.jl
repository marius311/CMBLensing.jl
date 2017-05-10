export DataSet, lnP, δlnP_δfϕₜ, ℕ, 𝕊


"""
Stores variables needed to construct the likelihood
"""
const DataSet=@NT(d,CN,Cf,Cϕ,Md,Mf,Mϕ)
ℕ(ds) = FuncOp(op   = fϕ->FieldTuple(ds.Md*(ds.CN*fϕ[1]),0fϕ[2]), 
               op⁻¹ = fϕ->FieldTuple(ds.Md*(ds.CN\fϕ[1]),0fϕ[2]), symmetric=true)
𝕊(ds) = FuncOp(op   = fϕ->FieldTuple(ds.Mf*(ds.Cf*fϕ[1]),ds.Mϕ*(ds.Cϕ*fϕ[2])),
               op⁻¹ = fϕ->FieldTuple(ds.Mf*(ds.Cf\fϕ[1]),ds.Mϕ*(ds.Cϕ\fϕ[2])), symmetric=true)

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
lnP(t::Real,fₜ,ϕ,ds,::Type{L}=LenseFlow) where {L} = lnP(Val{t},fₜ,ϕ,ds,L(ϕ))
lnP(t::Real,fₜ,ϕ,ds,L::LenseOp) = lnP(Val{t},fₜ,ϕ,ds,L)
lnP(::Type{Val{t}},fₜ,ϕ,ds,L::LenseOp) where {t} = lnP(ds.d-L[t→1]*fₜ, L[t→0]*fₜ, ϕ, ds)
lnP(Δ,f,ϕ,ds) = -(Δ⋅(ds.Md*(ds.CN\Δ)) + f⋅(ds.Mf*(ds.Cf\f)) + ϕ⋅(ds.Mϕ*(ds.Cϕ\ϕ)))/2

"""
Gradient of the log posterior probability with
respect to the field f and lensing potential ϕ. See `lnP` for definition of
arguments.

Returns :
"""
δlnP_δfϕₜ(t::Real,fₜ,ϕ,ds,::Type{L}=LenseFlow) where {L} = δlnP_δfϕₜ(Val{float(t)},fₜ,ϕ,ds,L(ϕ))
δlnP_δfϕₜ(t::Real,fₜ,ϕ,ds,L::LenseOp) = δlnP_δfϕₜ(Val{float(t)},fₜ,ϕ,ds,L)
function δlnP_δfϕₜ(::Type{Val{t}},fₜ,ϕ,ds,L::LenseOp) where {t}
    f̃ =  L[t→1]*fₜ
    f =  L[t→0]*fₜ

    (    δlnL_δf̃ϕ(f̃,ϕ,ds) * δf̃ϕ_δfϕₜ(L,f̃,fₜ,Val{t})
      + δlnΠᶠ_δfϕ(f,ϕ,ds) * δfϕ_δfϕₜ(L,f,fₜ,Val{t})
      + δlnΠᶲ_δfϕ(f,ϕ,ds))
end

# derivatives of the three posterior probability terms at the times at which
# they're easy to take
δlnL_δf̃ϕ(f̃,ϕ::Φ,ds) where {Φ}  = FieldTuple(ds.Md*(ds.CN\(ds.d-f̃)), zero(Φ)         )
δlnΠᶠ_δfϕ(f,ϕ::Φ,ds) where {Φ} = FieldTuple(-ds.Mf*(ds.Cf\f)      , zero(Φ)         )
δlnΠᶲ_δfϕ(f::F,ϕ,ds) where {F} = FieldTuple(zero(F)               , -ds.Mϕ*(ds.Cϕ\ϕ))


## Hessian

HlnP(t,fₜ,ϕ,ds,::Type{L}=LenseFlow) where {L} = HlnP(Val{float(t)},fₜ,ϕ,ds,L(ϕ)) 
HlnP(t,fₜ,ϕ,ds,L::LenseOp) = HlnP(Val{float(t)},fₜ,ϕ,ds,L) 
HlnP(::Type{Val{1.}},f̃,ϕ,ds,L::LenseOp) = let δfϕ_δf̃ϕ = δfϕ_δf̃ϕ(L,L\f̃,f̃)
    ℕ(ds)^-1 + δfϕ_δf̃ϕ' * (𝕊(ds)^-1 * δfϕ_δf̃ϕ)
end
HlnP(::Type{Val{0.}},f,ϕ,ds,L::LenseOp) = let δf̃ϕ_δfϕ = δf̃ϕ_δfϕ(L,L*f,f)
    δf̃ϕ_δfϕ' * (ℕ(ds)^-1 * δf̃ϕ_δfϕ) + 𝕊(ds)^-1
end
