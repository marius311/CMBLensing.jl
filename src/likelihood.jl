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


# mixing matrix for mixed parametrization
D_mix(Cf::FullDiagOp; σ²len=deg2rad(5/60)^2) = @. nan2zero(sqrt((Cf+σ²len)/Cf))


# Stores variables needed to construct the likelihood
@with_kw struct DataSet{Td,TCn,TCf,TCf̃,TCϕ,TCn̂,TB̂,TM,TB,TD}
    d  :: Td                 # data
    Cn :: TCn                # noise covariance
    Cϕ :: TCϕ                # ϕ covariance
    Cf :: TCf                # unlensed field covariance
    Cf̃ :: TCf̃  = nothing     # lensed field covariance (not always needed)
    Cn̂ :: TCn̂  = Cn          # approximate noise covariance, diagonal in same basis as Cf
    B̂  :: TB̂   = B           # approximate beam and instrumental transfer functions, diagonal in same basis as Cf
    M  :: TM   = 1           # user mask
    B  :: TB   = 1           # beam and instrumental transfer functions
    D  :: TD   = D_mix(Cf)   # mixing matrix for mixed parametrization
end


## likelihood 


@doc doc"""
    lnP(t, fₜ, ϕ, ds, ::Type{L}=LenseFlow)
    lnP(t, fₜ, ϕ, ds, L::LenseOp) 

Compute the log posterior probability as a function of the field, $f_t$, and the
lensing potential, $ϕ$. The subscript $t$ can refer to either a "time", e.g.
$t=0$ corresponds to the unlensed parametrization and $t=1$ to the lensed one,
or can be `:mix` correpsonding to the mixed parametrization. In all cases, the
argument `fₜ` should then be $f$ in that particular parametrization.

The log posterior is defined such that, 

```math
-2 \ln \mathcal{P}(f,ϕ\,|\,d) = (d - \mathcal{M}\mathcal{B}\mathcal{L}{\tilde f})^{\dagger} \mathcal{C_n}^{-1} (d - \mathcal{M}\mathcal{B}\mathcal{L}{\tilde f}) \
                                + f^\dagger \mathcal{C_f}^{-1} f + \phi^\dagger \mathcal{C_\phi}^{-1} \mathcal{\phi}
```

The argument `ds` should be a `DataSet` and stores the masks, data, mixing
matrix, and covariances needed. `L` can be a type of lensing like `PowerLens` or
`LenseFlow`, or an already constructed `LenseOp`.
"""
lnP(t,fₜ,ϕ,ds,::Type{L}=LenseFlow) where {L} = lnP(Val{t},fₜ,ϕ,ds,L(ϕ))
lnP(t,fₜ,ϕ,ds,L::LenseOp) = lnP(Val{t},fₜ,ϕ,ds,L)

# log posterior in the unlensed or lensed parametrization
function lnP(::Type{Val{t}},fₜ,ϕ,ds,L::LenseOp) where {t}
    @unpack Cn,Cf,Cϕ,M,B,d = ds
    Δ = d-M*B*L[t→1]*fₜ
    f = L[t→0]*fₜ
    -(Δ⋅(Cn\Δ) + f⋅(Cf\f) + ϕ⋅(Cϕ\ϕ))/2
end
# log posterior in the mixed parametrization
lnP(::Type{Val{:mix}},f̆,ϕ,ds,L::LenseOp) = (@unpack D = ds; lnP(0, D\(L\f̆), ϕ, ds, L))



## likelihood gradients

@doc doc"""

    δlnP_δfϕₜ(t, fₜ, ϕ, ds, ::Type{L}=LenseFlow)
    δlnP_δfϕₜ(t, fₜ, ϕ, ds, L::LenseOp)

Compute a gradient of the log posterior probability. See `lnP` for definition of
arguments of this function. 

The return type is a `FieldTuple` corresponding to the $(f_t,\phi)$ derivative.
"""
δlnP_δfϕₜ(t,fₜ,ϕ,ds,::Type{L}=LenseFlow) where {L} = δlnP_δfϕₜ(Val{t},fₜ,ϕ,ds,L(ϕ))
δlnP_δfϕₜ(t,fₜ,ϕ,ds,L::LenseOp) = δlnP_δfϕₜ(Val{t},fₜ,ϕ,ds,L)

# derivatives of the three posterior probability terms at the times at which
# they're easy to take (used below)
 δlnL_δf̃ϕ(f̃,    ϕ::Φ, ds) where {Φ} = (@unpack M,B,Cn,d=ds; FieldTuple(M'*B'*(Cn\(d-M*B*f̃)), zero(Φ)))
δlnΠᶠ_δfϕ(f,    ϕ::Φ, ds) where {Φ} = (@unpack Cf=ds;       FieldTuple(-Cf\f               , zero(Φ)))
δlnΠᶲ_δfϕ(f::F, ϕ,    ds) where {F} = (@unpack Cϕ=ds;       FieldTuple(zero(F)             , -Cϕ\ϕ))


# log posterior gradient in the lensed or unlensed parametrization
function δlnP_δfϕₜ(::Type{Val{t}},fₜ,ϕ,ds,L::LenseOp) where {t}
    f̃ =  L[t→1]*fₜ
    f =  L[t→0]*fₜ

    (    δlnL_δf̃ϕ(f̃,ϕ,ds) * δf̃ϕ_δfϕₜ(L,f̃,fₜ,Val{t})
      + δlnΠᶠ_δfϕ(f,ϕ,ds) * δfϕ_δfϕₜ(L,f,fₜ,Val{t})
      + δlnΠᶲ_δfϕ(f,ϕ,ds))
end
# log posterior gradient in the mixed parametrization
function δlnP_δfϕₜ(::Type{Val{:mix}},f̆,ϕ,ds,L::LenseOp)

    @unpack D = ds
    L⁻¹f̆ = L \ f̆
    f = D \ L⁻¹f̆

    # gradient w.r.t. (f,ϕ)
    δlnP_δf, δlnP_δϕ = δlnP_δfϕₜ(0, f, ϕ, ds, L)
    
    # chain rule
    FieldTuple(δlnP_δf * D^-1, δlnP_δϕ) * δfϕ_δf̃ϕ(L, L⁻¹f̆, f̆)
end




## wiener filter


@doc doc"""
    lensing_wiener_filter(ds::DataSet, L, which=:wf)

Computes the Wiener filter at fixed $\phi$, i.e. the best-fit of
$\mathcal{P}(f\,|\,\phi,d)$, or a sample from this posterior. 

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

The `which` parameter controls which operation to do and can be one of three
things:

* `:wf` - Compute the Wiener filter
* `:sample` - Compute a sample from the posterior
* `:fluctuation` - Compute a fluctuation around the mean (i.e. a sample minus the Wiener filter)

"""
function lensing_wiener_filter(ds::DataSet{F}, L, which=:wf; guess=nothing, kwargs...) where F
    
    @unpack d, Cn, Cn̂, Cf, M, B, B̂ = ds
    
    b = 0
    if (which in (:wf, :sample))
        b += L'*B'*M'*(Cn^-1)*d
    end
    if (which in (:fluctuation, :sample))
        b += sqrt(Cf)\white_noise(F) + L'*B'*M'*(sqrt(Cn)\white_noise(F))
    end
    
    pcg2(
        (Cf^-1) + B̂'*(Cn̂^-1)*B̂,
        (Cf^-1) + L'*B'*M'*(Cn^-1)*M*B*L,
        b,
        guess==nothing ? 0d : guess;
        kwargs...
    )
    
end
