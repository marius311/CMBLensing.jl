export DataSet, lnP, δlnP_δfϕₜ, HlnP, ℕ, 𝕊


# 
# This file contains function which compute things dealing with the posterior
# probability of f and ϕ given data, d. 
# 
# By definition, we take as our data model
# 
#     `d = P * M * B * L * f + n`
#
# where M, B, and L are the mask, beam/instrumental transfer functions, and
# lensing operators, and P is a pixelization operator. Since we track P, 
# it means we can estimate the fields on a higher resolution than the data. 
# Note also that this form means that the noise n is defined as being
# unbeamed, and also is unmasked. If we're using simulated data, its easy to not
# mask the noise. For runs with real data, the noise outside the mask should be
# filled in with a realization of the noise. 
#
# Under this data model, the posterior probability is, 
# 
#     `-2 ln P(f,ϕ|d) = (d - P*M*B*L*f̃)ᴴ*Cn⁻¹*(d - P*M*B*L*f̃) + fᴴ*Cf⁻¹*f + ϕᴴ*Cϕ⁻¹*ϕ`
#
# The various covariances and M, B, and d are stored in a `DataSet` structure. 
#
# Below are also functions to compute derivatives of this likelihood, as well as
# a Wiener filter of the data (since that's `argmax_f P(f|ϕ,d)`).
#


# mixing matrix for mixed parametrization
D_mix(Cf::LinOp; σ²len=deg2rad(5/60)^2) = @. nan2zero(sqrt(($Diagonal(Cf)+σ²len)/$Diagonal(Cf)))


# Stores variables needed to construct the likelihood
@with_kw struct DataSet{Td,TCn,TCf,TCf̃,TCϕ,TCn̂,TB̂,TM,TB,TD,TP}
    d  :: Td                 # data
    Cn :: TCn                # noise covariance
    Cϕ :: TCϕ                # ϕ covariance
    Cf :: TCf                # unlensed field covariance
    Cf̃ :: TCf̃  = nothing     # lensed field covariance (not always needed)
    Cn̂ :: TCn̂  = Cn          # approximate noise covariance, diagonal in same basis as Cf
    M  :: TM   = 1           # user mask
    B  :: TB   = 1           # beam and instrumental transfer functions
    B̂  :: TB̂   = B           # approximate beam and instrumental transfer functions, diagonal in same basis as Cf
    D  :: TD   = D_mix(Cf)   # mixing matrix for mixed parametrization
    P  :: TP   = 1           # pixelization operator to estimate field on higher res than data
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
lnP(t,fₜ,ϕ,ds,::Type{L}=LenseFlow) where {L} = lnP(Val{t},fₜ,ϕ,ds,cache(L(ϕ),fₜ))
lnP(t,fₜ,ϕ,ds,L::LenseOp) = lnP(Val{t},fₜ,ϕ,ds,L)

# log posterior in the unlensed or lensed parametrization
function lnP(::Type{Val{t}},fₜ,ϕ,ds,L::LenseOp) where {t}
    @unpack Cn,Cf,Cϕ,M,P,B,d = ds
    Δ = d-M*P*B*L[t→1]*fₜ
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
δlnP_δfϕₜ(t,fₜ,ϕ,ds,::Type{L}=LenseFlow) where {L} = δlnP_δfϕₜ(Val{t},fₜ,ϕ,ds,cache(L(ϕ),fₜ))
δlnP_δfϕₜ(t,fₜ,ϕ,ds,L::LenseOp) = δlnP_δfϕₜ(Val{t},fₜ,ϕ,ds,L)

# derivatives of the three posterior probability terms at the times at which
# they're easy to take (used below)
δlnL_δf̃ϕ(f̃,ϕ::Φ,ds) where {Φ} = (@unpack P,M,B,Cn,d=ds; FieldTuple(B'*P'*M'*(Cn\(d-M*P*B*f̃)), zero(Φ)))
δlnΠᶠ_δfϕ(f,ϕ::Φ,ds) where {Φ} = (@unpack Cf=ds;         FieldTuple(-Cf\f                    , zero(Φ)))
δlnΠᶲ_δfϕ(f::F,ϕ,ds) where {F} = (@unpack Cϕ=ds;         FieldTuple(zero(F)                  , -Cϕ\ϕ))


# log posterior gradient in the lensed or unlensed parametrization
function δlnP_δfϕₜ(::Type{Val{t}},fₜ,ϕ,ds,L::LenseOp) where {t}
    f̃ =  L[t→1]*fₜ
    f =  L[t→0]*fₜ

    (   δf̃ϕ_δfϕₜ(L,f̃,fₜ,Val{t})' * δlnL_δf̃ϕ(f̃,ϕ,ds)
      + δfϕ_δfϕₜ(L,f,fₜ,Val{t})' * δlnΠᶠ_δfϕ(f,ϕ,ds)
                                 + δlnΠᶲ_δfϕ(f,ϕ,ds)  )
end
# log posterior gradient in the mixed parametrization
function δlnP_δfϕₜ(::Type{Val{:mix}},f̆,ϕ,ds,L::LenseOp)

    D = ds.D
    L⁻¹f̆ = L \ f̆
    f = D \ L⁻¹f̆

    # gradient w.r.t. (f,ϕ)
    δlnP_δf, δlnP_δϕ = δlnP_δfϕₜ(0, f, ϕ, ds, L)
    
    # chain rule
    δfϕ_δf̃ϕ(L, L⁻¹f̆, f̆)' * FieldTuple(D^-1 * δlnP_δf, δlnP_δϕ)
end




## wiener filter


@doc doc"""
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
    
    @unpack d, Cn, Cn̂, Cf, M, B, P, B̂ = ds
    
    b = 0
    if (which in (:wf, :sample))
        b += L'*B'*P'*M'*(Cn^-1)*d
    end
    if (which in (:fluctuation, :sample))
        b += sqrt(Cf)\white_noise(F) + L'*B'*P'*M'*(sqrt(Cn)\white_noise(F))
    end
    
    pcg2(
        (Cf^-1) + B̂'*(Cn̂^-1)*B̂,
        (Cf^-1) + L'*B'*P'*M'*(Cn^-1)*M*P*B*L,
        b,
        guess==nothing ? 0*b : guess;
        kwargs...
    )
    
end


@doc doc"""

    max_lnP_joint(ds::DataSet; L=LenseFlow, Nϕ=nothing, quasi_sample=nothing, nsteps=10, Ncg=500, cgtol=1e-1, αtol=1e-5, αmax=0.5, progress=false)

Compute the maximum of the joint posterior, or a quasi-sample from the joint posterior. 

The `ds` argument stores the data and other relevant objects for the dataset
being considered. `L` gives which type of lensing operator to use. 

`ϕstart` can be used to specify the starting point of the minimizer, but this is
not necessary and otherwise it will start at ϕ=0. 

`Nϕ` can optionally specify an estimate of the ϕ effective noise, and if
provided is used to estimate a Hessian which is used in the ϕ
quasi-Newton-Rhapson step. `Nϕ=:qe` automatically uses the quadratic estimator
noise. 

This function can also be used to draw quasi-samples, wherein for the f step, we
draw a sample from  P(f|ϕ) instead of maximizing it (ie instead of computing
Wiener filter). `quasi_sample` can be set to an integer seed, in which case each
time in the `f` step we draw a same-seeded sample. If `quasi_sample` is instead
just `true`, then each iteration in the algorithm draws a different sample so
the solution bounces around rather than asymptoting to a maximum. 

The following arguments control the maximiation procedure, and can generally be
left at their defaults:

* `nsteps` - The number of iteration steps to do (each iteration updates f then updates ϕ)
* `Ncg` - Maximum number of conjugate gradient steps during the f update
* `cgtol` - Conjugrate gradient tolerance (will stop at cgtol or Ncg, whichever is first)
* `αtol` - Tolerance for the linesearch in the ϕ quasi-Newton-Rhapson step, `x′ = x - α*H⁻¹*g`
* `αmax` - Maximum value for α in the linesearch
* `progress` - Whether to print out conjugate gradient progress.

Returns a tuple `(f̊, f, ϕ, tr)` where `f̊` and `f` are the best-fit (or
quasi-sample) field in the mixed and unlensed parametrization, respectively, `ϕ`
is the lensing potential, and `tr` contains info about the run. 

"""
function max_lnP_joint(
    ds;
    ϕstart = nothing,
    L = LenseFlow,
    Nϕ = nothing,
    quasi_sample = false, 
    nsteps = 10, 
    Ncg = 500,
    cgtol = 1e-1,
    αtol = 1e-5,
    αmax = 0.5,
    cache_function = (L->cache(L,ds.d)),
    callback = nothing,
    progress = false)
    
    if !(isa(quasi_sample,Bool) || isa(quasi_sample,Int))
        throw(ArgumentError("quasi_sample should be true, false, or an Int."))
    end
    
    @unpack d, D, Cϕ, Cf, Cf̃, Cn, Cn̂ = ds
    
    fcur, f̊cur = nothing, nothing
    ϕcur = (ϕstart != nothing) ? ϕstart : ϕcur = zero(simulate(Cϕ)) # fix needing to get zero(Φ) this way
    tr = []
    hist = nothing
    
    # compute approximate inverse ϕ Hessian used in gradient descent, possibly
    # from quadratic estimate
    if (Nϕ == :qe); Nϕ = ϕqe(zero(simulate(Cf)), Cf, Cf̃, Cn̂)[2]; end
    Hϕ⁻¹ = (Nϕ == nothing) ? Cϕ : (Cϕ^-1 + Nϕ^-1)^-1
    
    
    for i=1:nsteps

        # f step
        let L = ((i==1 && ϕstart==nothing) ? IdentityOp : cache_function(L(ϕcur)))
            
            # if we're doing a fixed quasi_sample, set the random seed here,
            # which controls the sample from the posterior we get from inside
            # `lensing_wiener_filter`
            if isa(quasi_sample,Int); seed!(quasi_sample); end
            
            fcur,hist = lensing_wiener_filter(ds, L, 
                (quasi_sample==false) ? :wf : :sample, # if doing a quasi-sample, we get a sample instead of the WF
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
        
        if callback != nothing
            callback(f̊cur, fcur, ϕcur, tr)
        end

    end

    return f̊cur, fcur, ϕcur, tr
    
end



@doc doc"""
    load_sim_dataset
    
Create a `DataSet` object with some simulated data. 

"""
function load_sim_dataset(;
    θpix = throw(UndefVarError(:θpix)),
    θpix_data = θpix,
    Nside = throw(UndefVarError(:Nside)),
    use = throw(UndefVarError(:use)),
    T = Float32,
    μKarcminT = 3,
    ℓknee = 100,
    αknee = 3,
    ℓmax_data = 3000,
    beamFWHM = 0,
    Cℓf = throw(UndefVarError(:Cℓf)),
    Cℓf̃ = throw(UndefVarError(:Cℓf̃)),
    Cℓn = nothing,
    seed = nothing,
    M = nothing,
    B = nothing,
    D = nothing,
    mask_kwargs = nothing,
    L = LenseFlow,
    ∂mode = fourier∂
    )
    
    # the biggest ℓ on the 2D fourier grid
    ℓmax = round(Int,ceil(√2*FFTgrid(T,Flat{θpix,Nside}).nyq))
    
    # Cℓs
    if (Cℓn == nothing)
        Cℓn = noisecls(μKarcminT, beamFWHM=0, ℓknee=ℓknee, αknee=αknee, ℓmax=ℓmax)
    end
    
    # types which depend on whether T/E/B
    SS,ks = Dict(:TEB=>((S0,S2),(:TT,:EE,:BB,:TE)), :EB=>((S2,),(:EE,:BB)), :T=>((S0,),(:TT,)))[use]
    F,F̂,nF = Dict(:TEB=>(FlatIQUMap,FlatTEBFourier,3), :EB=>(FlatS2QUMap,FlatS2EBFourier,2), :T=>(FlatS0Map,FlatS0Fourier,1))[use]
    
    # pixelization
    P = (θpix_data == θpix) ? 1 : FuncOp(
        op  = f -> ud_grade(f, θpix_data, deconv_pixwin=false, anti_aliasing=false),
        opᴴ = f -> ud_grade(f, θpix,      deconv_pixwin=false, anti_aliasing=false)
    )
    Pix      = Flat{θpix,Nside,∂mode}
    Pix_data = Flat{θpix_data,Nside÷(θpix_data÷θpix),∂mode}
    
    # covariances
    Cϕ       =  Cℓ_to_cov(T,Pix,     S0,    Cℓf[:ℓ], Cℓf[:ϕϕ])
    Cf,Cf̃,Cn̂ = (Cℓ_to_cov(T,Pix,     SS..., Cℓx[:ℓ], (Cℓx[k] for k=ks)...) for Cℓx in (Cℓf,Cℓf̃,Cℓn))
    Cn       =  Cℓ_to_cov(T,Pix_data,SS..., Cℓn[:ℓ], (Cℓn[k] for k=ks)...)
    
    # data mask
    if (M == nothing) && (mask_kwargs != nothing)
        M = LP(ℓmax_data) * FullDiagOp(F{T,Pix_data}(repeated(T.(sptlike_mask(Nside÷(θpix_data÷θpix),θpix_data; mask_kwargs...)),nF)...))
    elseif (M == nothing)
        M = LP(ℓmax_data)
    end
    
    # beam
    if (B == nothing)
        B = let ℓ=0:ℓmax; Cℓ_to_cov(T,Pix,SS..., ℓ, ((k==:TE ? 0 .* ℓ : @.(exp(-ℓ^2*deg2rad(beamFWHM/60)^2/(8*log(2))/2))) for k=ks)...); end;
    end
    
    # mixing matrix
    if (D == nothing); D = D_mix(Cf); end
    
    # simulate data
    if (seed != nothing); seed!(seed); end
    ϕ = simulate(Cϕ)
    f = simulate(Cf)
    f̃ = cache(L(ϕ),f)*f
    n = simulate(Cn)
    d = M*P*B*f̃ + n
    
    # put everything in DataSet
    ds = DataSet(;(@ntpack d Cn Cn̂ Cf Cf̃ Cϕ M B D P)...)
    
    return @ntpack f f̃ ϕ n ds T P
    
end
