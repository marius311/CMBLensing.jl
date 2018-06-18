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


doc"""
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

doc"""

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
δlnL_δf̃ϕ{Φ}(f̃,ϕ::Φ,ds)  = (@unpack M,B,Cn,d=ds; FieldTuple(M'*B'*(Cn\(d-M*B*f̃)), zero(Φ)))
δlnΠᶠ_δfϕ{Φ}(f,ϕ::Φ,ds) = (@unpack Cf=ds;       FieldTuple(-Cf\f               , zero(Φ)))
δlnΠᶲ_δfϕ{F}(f::F,ϕ,ds) = (@unpack Cϕ=ds;       FieldTuple(zero(F)             , -Cϕ\ϕ))


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


doc"""
    lensing_wiener_filter(ds::DataSet, L, which=:wf)

Computes either, 
* the Wiener filter at fixed $\phi$, i.e. the best-fit of
$\mathcal{P}(f\,|\,\phi,d)$
* a sample from $\mathcal{P}(f\,|\,\phi,d)$

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
        b += sqrtm(Cf)\white_noise(F) + L'*B'*M'*(sqrtm(Cn)\white_noise(F))
    end
    
    pcg2(
        (Cf^-1) + B̂'*(Cn̂^-1)*B̂,
        (Cf^-1) + L'*B'*M'*(Cn^-1)*M*B*L,
        b,
        guess==nothing ? 0d : guess;
        kwargs...
    )
    
end


doc"""

    max_lnP_joint(ds::DataSet, L::Type{<:LenseOp}, nsteps=5, Nϕ=nothing, Ncg=500, cgtol=1e-1, αtol=1e-5, αmax=1, progress=false)

Compute the maximum of the joint posterior, or a quasi-sample from the joint posterior. 

The `ds` argument stores the data and other relevant objects for the dataset
being considered. `L` gives which type of lensing operator to use. 

`Nϕ` can optionally specify an estimate of the ϕ effective noise, and if
provided is used to estimate a Hessian which is used in the ϕ
quasi-Newton-Rhapson step. `Nϕ=:qe` automatically uses the quadratic estimator
noise. 

`quasi_sample` can be set to an integer seed to compute a quasi-sample from the
posterior rather than the maximum. 

The following arguments control the maximiation procedure, and can generally be
left at their defaults:

* `nsteps` - The number of iteration steps to do (each iteration updates f then updates ϕ)
* `Ncg` - Maximum number of conjugate gradient steps during the f update
* `cgtol` - Conjugrate gradient tolerance (will stop at cgtol or Ncg, whichever is first)
* `αtol` - Tolerance for the linesearch in the ϕ quasi-Newton-Rhapson step, `x′ = x - α*H⁻¹*g`
* `αmax` - Maximum value for α in the linesearch
* `progress` - Whether to print out conjugate gradient progress.

"""
function max_lnP_joint(
    ds;
    L = LenseFlow,
    nsteps = 10, 
    Nϕ = nothing,
    Ncg = 500,
    cgtol = 1e-1,
    αtol = 1e-5,
    αmax = 1.,
    quasi_sample = nothing, 
    progress = false)
    
    @unpack d, D, Cϕ, Cf, Cf̃, Cn = ds
    
    fcur, f̊cur = nothing, nothing
    ϕcur = zero(Ł(d)'Ł(d)) # fix needing to get zero(Φ) this way
    tr = []
    hist = nothing
    
    # compute approximate inverse ϕ Hessian used in gradient descent, possibly
    # from quadratic estimate
    if (Nϕ == :qe); Nϕ = ϕqe(d, Cf, Cf̃, Cn)[2]; end
    Hϕ⁻¹ = (Nϕ == nothing) ? Cϕ : (Cϕ^-1 + Nϕ^-1)^-1
    
    
    for i=1:nsteps

        # f step
        let L = (i==1 ? IdentityOp : cache(L(ϕcur)))
            
            # if we're doing a quasi_sample, set the random seed here, which controls the
            # sample from the posterior we get from inside `lensing_wiener_filter`
            if (quasi_sample != nothing); srand(quasi_sample); end
            
            fcur,hist = lensing_wiener_filter(ds, L, 
                (quasi_sample==nothing) ? :wf : :sample, # if doing a quasi-sample, we get a sample instead of the WF
                guess=(i==1 ? nothing : fcur),           # after first iteration, use the previous f as starting point
                tol=cgtol, nsteps=Ncg, hist=(:i,:res), progress=progress)
                
            f̊cur = L * D * fcur
        end
        
        # ϕ step
        if i!=nsteps
            ϕnew = Hϕ⁻¹*(δlnP_δfϕₜ(:mix,f̊cur,ϕcur,ds,L))[2]
            res = optimize(α->(-lnP(:mix,f̊cur,ϕcur+α*ϕnew,ds,L)), 0., αmax, abs_tol=αtol)
            α = res.minimizer
            ϕcur = ϕcur+α*ϕnew
            lnPcur = -res.minimum
            if progress; @show i,lnPcur,length(hist),α; end
            push!(tr,@dictpack(i,lnPcur,hist,α,ϕnew,ϕcur,fcur))
        end

    end

    return f̊cur, fcur, ϕcur, tr
    
end



doc"""
    load_sim_dataset
    
Create a `DataSet` object with some simulated data. 

"""
function load_sim_dataset(;
    θpix = throw(UndefVarError(:θpix)),
    Nside = throw(UndefVarError(:Nside)),
    use = throw(UndefVarError(:use)),
    T = Float32,
    μKarcminT = 3,
    ℓknee = 100,
    ℓmax_data = 3000,
    beamFWHM = 0,
    Cℓf = throw(UndefVarError(:Cℓf)),
    Cℓf̃ = throw(UndefVarError(:Cℓf̃)),
    seed = nothing,
    M = nothing,
    B = nothing,
    D = nothing,
    mask_kwargs = nothing,
    L = LenseFlow
    )
    
    # Cℓs
    Cℓn = noisecls(μKarcminT, beamFWHM=0, ℓknee=ℓknee)
    
    # types which depend on whether T/E/B
    SS,ks = Dict(:TEB=>((S0,S2),(:TT,:EE,:BB,:TE)), :EB=>((S2,),(:EE,:BB)), :T=>((S0,),(:TT,)))[use]
    F,F̂,nF = Dict(:TEB=>(FlatIQUMap,FlatTEBFourier,3), :EB=>(FlatS2QUMap,FlatS2EBFourier,2), :T=>(FlatS0Map,FlatS0Fourier,1))[use]
    
    # covariances
    P = Flat{θpix,Nside}
    Cϕ = Cℓ_to_cov(T,P,S0, Cℓf[:ℓ], Cℓf[:ϕϕ])
    Cf,Cf̃,Cn = (Cℓ_to_cov(T,P,SS..., Cℓx[:ℓ], (Cℓx[k] for k=ks)...) for Cℓx in (Cℓf,Cℓf̃,Cℓn))
    
    # data mask
    if (M == nothing) && (mask_kwargs != nothing)
        M = FullDiagOp(F{T,P}(repeated(T.(sptlike_mask(Nside,θpix; mask_kwargs...)),nF)...)) * LP(ℓmax_data)
    else
        M = LP(ℓmax_data)
    end
    
    # beam
    if (B == nothing)
        B = let ℓ=0:10000; Cℓ_to_cov(T,P,SS..., ℓ, ((k==:TE ? 0.*ℓ : @.(exp(-ℓ^2*deg2rad(beamFWHM/60)^2/(8*log(2))/2))) for k=ks)...); end;
    end
    
    # mixing matrix
    if (D == nothing); D = D_mix(Cf); end
    
    # simulate data
    if (seed != nothing); srand(seed); end
    f = simulate(Cf)
    ϕ = simulate(Cϕ)
    f̃ = L(ϕ)*f
    n = simulate(Cn)
    d = M*B*f̃ + n
    
    # put everything in DataSet
    ds = DataSet(;(@dictpack d Cn Cf Cf̃ Cϕ M B D)...)
    
    return @dictpack f f̃ ϕ n ds T P
    
end
