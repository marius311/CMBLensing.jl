export DataSet, lnP, δlnP_δfϕₜ, HlnP, ℕ, 𝕊


# 
# This file contains function which compute things dealing with the posterior
# probability of f and ϕ given data, d. 
# 
# By definition, we take as our data model
# 
#     `d = M * B * L * f + n`
#
# where M, B, and L are the mask, beam/instrumental transfer functions, and
# lensing operators. Note this means that the noise n is defined as being
# unbeamed, and also is unmasked. If we're using simulated data, its easy to not
# mask the noise. For runs with real data, the noise outside the mask should be
# filled in with a realization of the noise. 
#
# Under this data model, the posterior probability is, 
# 
#     `-2 ln P(f,ϕ|d) = (d - M*B*L*f̃)ᴴ*Cn⁻¹*(d - M*B*L*f̃) + fᴴ*Cf⁻¹*f + ϕᴴ*Cϕ⁻¹*ϕ`
#
# The various covariances and M, B, and d are stored in a `DataSet` structure. 
#
# Below are also functions to compute derivatives of this likelihood, as well as
# a Wiener filter of the data (since that's `argmax_f P(f|ϕ,d)`).
#




# Stores variables needed to construct the likelihood
struct DataSet{Td,TCn,TCf,TCϕ,TM,TB}
    d  :: Td      # data
    Cn :: TCn     # noise covariance
    Cf :: TCf     # (unlensed) field covariance
    Cϕ :: TCϕ     # ϕ covariance
    M  :: TM      # user mask
    B  :: TB      # beam and instrumental transfer functions
end


## likelihood 


"""
    lnP(t::Real,fₜ,ϕ,ds,::Type{L}=LenseFlow)
    lnP(t::Real,fₜ,ϕ,ds,L::LenseOp) 

Compute the log posterior probability as a function of the field, fₜ, at time t,
and ϕ. The log posterior is defined such that, 

     `-2 ln P(f,ϕ|d) = (d - M*B*L*f̃)ᴴ*Cn⁻¹*(d - M*B*L*f̃) + fᴴ*Cf⁻¹*f + ϕᴴ*Cϕ⁻¹*ϕ`

The argument `ds` should be a `DataSet` and store the masks, data, and
covariances needed. `L` can be a type of lensing like `PowerLens` or
`LenseFlow`, or an already constructed LenseOp.
"""
lnP(t::Real,fₜ,ϕ,ds,::Type{L}=LenseFlow) where {L} = lnP(Val{t},fₜ,ϕ,ds,L(ϕ))
lnP(t::Real,fₜ,ϕ,ds,L::LenseOp) = lnP(Val{t},fₜ,ϕ,ds,L)
function lnP(::Type{Val{t}},fₜ,ϕ,ds,L::LenseOp) where {t}
    @unpack Cn,Cf,Cϕ,M,B,d = ds
    Δ = d-M*B*L[t→1]*fₜ
    f = L[t→0]*fₜ
    -(Δ⋅(Cn\Δ) + f⋅(Cf\f) + ϕ⋅(Cϕ\ϕ))/2
end


## likelihood gradients

"""

    δlnP_δfϕₜ(t::Real,fₜ,ϕ,ds,::Type{L}=LenseFlow)
    δlnP_δfϕₜ(t::Real,fₜ,ϕ,ds,L::LenseOp)

Compute a gradient of the log posterior probability with respect to the field f
and at some time t and the lensing potential ϕ. See `lnP` for definition of
arguments.
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
δlnL_δf̃ϕ{Φ}(f̃,ϕ::Φ,ds)  = (@unpack M,B,Cn,d=ds; FieldTuple(M'*B'*(Cn\(d-M*B*f̃)), zero(Φ)))
δlnΠᶠ_δfϕ{Φ}(f,ϕ::Φ,ds) = (@unpack Cf=ds;       FieldTuple(-Cf\f               , zero(Φ)))
δlnΠᶲ_δfϕ{F}(f::F,ϕ,ds) = (@unpack Cϕ=ds;       FieldTuple(zero(F)             , -Cϕ\ϕ))



## wiener filter


doc"""
    wf(ds, L; kwargs...)

Computes the Wiener filter of data $d$ at a fixed $\phi$, defined as, 

```math
{\rm argmax}_f \mathcal{P}(f\,|\,\phi,d)
```

The data model assumed is, 

```math
d = \mathcal{M} \mathcal{B} \mathcal{L} \, f + n
```

Note that the noise is defined as un-debeamed and also unmasked (so it needs to
be filled in outside the mask if using real data). The mask, $\mathcal{M}$, can
be any composition of real and/or fourier space diagonal operators.
    
The argument `ds::DataSet` stores the mask, $\mathcal{M}$, the beam/instrumental
transfer functions, $\mathcal{B}$, as well as the various covariances which are
needed.

The Wiener filter is performed in the most optimal form we've found (so far).

"""
function lensing_wiener_filter(ds::DataSet, L; kwargs...)
    
    @unpack d, Cn, Cf, M, B = ds
    
    pcg2(
        (Cf^-1) + (Cn^-1),
        (Cf^-1) + L'*B'*M'*(Cn^-1)*M*B*L,
        L'*B'*M'*(Cn^-1)*d;
        kwargs...
    )
    
end
