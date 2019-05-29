export DataSet, lnP, δlnP_δfϕₜ


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
D_mix(Cf::LinOp; rfid=0.1, σ²len=deg2rad(5/60)^2) =
     ParamDependentOp((;r=rfid, _...)->(nan2zero.(sqrt.((Diagonal(evaluate(Cf,r=rfid))+σ²len) ./ Diagonal(evaluate(Cf,r=r))))))

# Stores variables needed to construct the likelihood
@with_kw struct DataSet{Td,TCn,TCf,TCf̃,TCϕ,TCn̂,TB̂,TM,TB,TD,TG,TP}
    d  :: Td                # data
    Cn :: TCn               # noise covariance
    Cϕ :: TCϕ               # ϕ covariance
    Cf :: TCf               # unlensed field covariance
    Cf̃ :: TCf̃ = nothing     # lensed field covariance (not always needed)
    Cn̂ :: TCn̂ = Cn          # approximate noise covariance, diagonal in same basis as Cf
    M  :: TM  = 1           # user mask
    B  :: TB  = 1           # beam and instrumental transfer functions
    B̂  :: TB̂  = B           # approximate beam and instrumental transfer functions, diagonal in same basis as Cf
    D  :: TD  = IdentityOp  # mixing matrix for mixed parametrization
    G  :: TG  = IdentityOp  # reparametrization for ϕ
    P  :: TP  = 1           # pixelization operator to estimate field on higher res than data
end

function subblock(ds::DataSet, block)
    DataSet(map(collect(pairs(fields(ds)))) do (k,v)
        @match (k,v) begin
            ((:Cϕ || :G), v)                    => v
            (_, L::Union{Nothing,FuncOp,Real})  => L
            (_, L)                              => getproperty(L,block)
        end
    end...)
end

function (ds::DataSet)(;θ...)
    DataSet(map(fieldvalues(ds)) do v
        (v isa ParamDependentOp) ? v(;θ...) : v
    end...)
end

    
@doc doc"""
    resimulate(ds::DataSet; f=..., ϕ=...)
    
Resimulate the data in a given dataset, potentially at a fixed f and/or ϕ (both
are resimulated if not provided)
"""
function resimulate(ds::DataSet; f=simulate(ds.Cf), ϕ=simulate(ds.Cϕ), n=simulate(ds.Cn), f̃=LenseFlow(ϕ)*f)
    @unpack M,P,B = ds
    DataSet(ds, d = M*P*B*f̃ + n)
end


## likelihood 


@doc doc"""
    lnP(t, fₜ, ϕₜ,                ds::DataSet, L=LenseFlow)
    lnP(t, fₜ, ϕₜ, θ::NamedTuple, ds::DataSet, L=LenseFlow)

Compute the log posterior probability in the joint parameterization as a
function of the field, $f_t$, the lensing potential, $\phi_t$, and possibly some
cosmological parameters, $\theta$. The subscript $t$ can refer to either a
"time", e.g. passing `t=0` corresponds to the unlensed parametrization and `t=1`
to the lensed one, or can be `:mix` correpsonding to the mixed parametrization.
In all cases, the arguments `fₜ` and `ϕₜ` should then be $f$ and $\phi$ in
that particular parametrization.

If any parameters $\theta$ are passed, we also include the three determinant
terms to properly normalize the posterior,

```math
+ \log\det\mathcal{C}_n(\theta) + \log\det\mathcal{C}_f(\theta) + \log\det\mathcal{C}_ϕ(\theta)
```

The argument `ds` should be a `DataSet` and stores the masks, data, mixing
matrix, and covariances needed. `L` can be a type of lensing like `PowerLens` or
`LenseFlow`, or an already constructed `LenseOp`.
"""
lnP(t, fₜ, ϕₜ,                ds::DataSet, L=LenseFlow) = lnP(Val(t), fₜ, ϕₜ, NamedTuple(), ds, L)
lnP(t, fₜ, ϕₜ, θ::NamedTuple, ds::DataSet, L=LenseFlow) = lnP(Val(t), fₜ, ϕₜ, θ,            ds, L)

function lnP(::Val{t}, fₜ, ϕₜ, θ::NamedTuple, ds::DataSet, L) where {t}
    dsθ = ds(;θ...)
    ϕ = (t==:mix) ? dsθ.G\ϕₜ : ϕₜ
    Lϕ = (L isa LenseOp) ? cache!(L,ϕ) : cache(L(ϕ),fₜ)
    lnP(Val(t), fₜ, ϕₜ, ϕ, θ, ds, dsθ, Lϕ)
end

# we now have ϕ (unmixed) computed, Lϕ is now a cached lensing operator
# evaluated at ϕ, so now we can evaluate the posterior in either the mixed
# parametrization or at some time, t
function lnP(::Val{:mix}, f°, ϕ°, ϕ, θ, ds, dsθ, Lϕ::LenseOp)
    
    @unpack D,G = dsθ
    
    # unmix f° and evaluate at t=0, as well as adding necessary Jacobian
    # determiant terms from the mixing
    (lnP(Val(0), D\(Lϕ\f°), ϕ°, ϕ, θ, ds, dsθ, Lϕ)
     - (depends_on(ds.D, θ) ? logdet(D) : 0)
     - (depends_on(ds.G, θ) ? logdet(G) : 0))
     
end
function lnP(::Val{t}, fₜ, ϕₜ, ϕ, θ, ds, dsθ, Lϕ::LenseOp) where {t}
    
    @unpack Cn,Cf,Cϕ,M,P,B,d = dsθ
    
    # the unnormalized part of the posterior
    Δ = d-M*P*B*Lϕ[t→1]*fₜ
    f = Lϕ[t→0]*fₜ
    lnP = -(Δ⋅(Cn\Δ) + f⋅(Cf\f) + ϕ⋅(Cϕ\ϕ))/2
    
    # add the normalization (the logdet terms), offset by their values at
    # fiducial parameters (to avoid roundoff errors, since its otherwise a large
    # number). note: only the terms which depend on parameters that were passed
    # in via `θ... ` will be computed. 
    lnP += lnP_logdet_terms(ds,ds(),dsθ; θ...)

    lnP
    
end

# logdet terms in the posterior given the covariances in `dsθ` which is the
# dataset evaluated at parameters θ.  `ds` is used to check which covariances
# were param-dependent prior to evaluation, and these are not calculated
function lnP_logdet_terms(ds, ds₀, dsθ; θ...)
    -(  (depends_on(ds.Cn, θ) ? logdet(inv(ds₀.Cn)*dsθ.Cn) : 0) 
      + (depends_on(ds.Cf, θ) ? logdet(inv(ds₀.Cf)*dsθ.Cf) : 0)
      + (depends_on(ds.Cϕ, θ) ? logdet(inv(ds₀.Cϕ)*dsθ.Cϕ) : 0))/2
end



## joint posterior gradients

@doc doc"""

    δlnP_δfϕₜ(t, fₜ, ϕ, ds, ::Type{L}=LenseFlow)
    δlnP_δfϕₜ(t, fₜ, ϕ, ds, L::LenseOp)

Compute a gradient of the log posterior probability. See `lnP` for definition of
arguments of this function. 

The return type is a `FieldTuple` corresponding to the $(f_t,\phi)$ derivative.
"""
δlnP_δfϕₜ(t, fₜ, ϕ,                ds, L=LenseFlow) = δlnP_δfϕₜ(Val(t), fₜ, ϕ, NamedTuple(), ds, L)
δlnP_δfϕₜ(t, fₜ, ϕ, θ::NamedTuple, ds, L=LenseFlow) = δlnP_δfϕₜ(Val(t), fₜ, ϕ, θ,            ds, L)

function δlnP_δfϕₜ(::Val{t}, fₜ, ϕₜ, θ::NamedTuple, ds::DataSet, L) where {t}
    dsθ = ds(;θ...)
    ϕ = (t==:mix) ? dsθ.G\ϕₜ : ϕₜ
    Lϕ = (L isa LenseOp) ? cache!(L,ϕ) : cache(L(ϕ),fₜ)
    δlnP_δfϕₜ(Val(t), fₜ, ϕₜ, ϕ, θ, ds, dsθ, Lϕ)
end

# in the lensed or unlensed parametrization
function δlnP_δfϕₜ(::Val{t}, fₜ, ϕₜ, ϕ, θ, ds, dsθ, Lϕ::LenseOp) where {t}
    f̃ = Lϕ[t→1]*fₜ
    f = Lϕ[t→0]*fₜ

    (   δf̃ϕ_δfϕₜ(Lϕ,f̃,fₜ,Val(t))' *  δlnL_δf̃ϕ(f̃,ϕ,dsθ)
      + δfϕ_δfϕₜ(Lϕ,f,fₜ,Val(t))' * δlnΠᶠ_δfϕ(f,ϕ,dsθ)
                                  + δlnΠᶲ_δfϕ(f,ϕ,dsθ) )
end
# in the mixed parametrization
function δlnP_δfϕₜ(::Val{:mix}, f°, ϕ°, ϕ, θ, ds, dsθ, Lϕ::LenseOp)
    
    @unpack D,G = ds
    Lϕ⁻¹f° = Lϕ \ f°
    f = D \ Lϕ⁻¹f°

    # gradient w.r.t. (f,ϕ)
    δlnP_δf, δlnP_δϕ = δlnP_δfϕₜ(Val(0), f, ϕ, ϕ, θ, ds, dsθ, Lϕ)
    
    # chain rule
    (δlnP_δf°, δlnP_δϕ°) = δfϕ_δf̃ϕ(Lϕ, Lϕ⁻¹f°, f°)' * FieldTuple(D \ δlnP_δf, δlnP_δϕ)
    FieldTuple(δlnP_δf°, G \ δlnP_δϕ°)

end
# derivatives of the three posterior probability terms at the times at which
# they're easy to take (used above)
δlnL_δf̃ϕ(f̃,ϕ::ɸ,dsθ)  where {ɸ} = (@unpack P,M,B,Cn,Cf,Cϕ,d=dsθ; FieldTuple(B'*P'*M'*(Cn\(d-M*P*B*f̃)), zero(Cϕ)))
δlnΠᶠ_δfϕ(f,ϕ::ɸ,dsθ) where {ɸ} = (@unpack Cf,Cϕ=dsθ;            FieldTuple(-(Cf\f)                  , zero(Cϕ)))
δlnΠᶲ_δfϕ(f::F,ϕ,dsθ) where {F} = (@unpack Cf,Cϕ=dsθ;            FieldTuple(zero(Cf)                 , -(Cϕ\ϕ)))



## marginal posterior gradients

δlnP_δϕ(ϕ, ds, ::Type{L}=LenseFlow; kwargs...) where {L} = δlnP_δϕ(L(ϕ), ds; kwargs...)

function δlnP_δϕ(L::LenseOp, ds; Nmc_det=100, progress=false, return_sims=false)
    
    @unpack d,P,M,B,Cn,Cf,Cn̂,G = ds
    
    if G!=IdentityOp; @warn "δlnP_δϕ does not currently handle the G mixing matrix"; end

    function gQD(L, ds)
        y = B' * M' * P' * (Σ(L, ds) \ ds.d)
        y * δLf_δϕ(Cf*(L'*y), L)
    end

    det_sims = @showprogress pmap(1:Nmc_det) do i gQD(L, resimulate(ds, f̃=L*simulate(ds.Cf))) end

    g = gQD(L, ds) - mean(det_sims)

    return_sims ? (g, det_sims) : g 

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
        b += Cf\simulate(Cf) + L'*B'*P'*M'*(Cn\simulate(Cn))
    end
    
    pcg2(
        (Cf^-1) + B̂'*(Cn̂^-1)*B̂,
        (Cf^-1) + L'*B'*P'*M'*(Cn^-1)*M*P*B*L,
        b,
        guess==nothing ? 0*b : guess;
        kwargs...
    )
    
end


# todo: figure out if this and `lensing_wiener_filter` above are the same and
# can be combined
@doc doc"""
    Σ(ϕ, ds, ::Type{L}=LenseFlow) where {L}
    Σ(L::LenseOp, ds) 
    
Operator for the data covariance, Cn + P*M*B*L*Cf*L'*B'*M'*P', which can applied
and inverted.
"""
Σ(ϕ, ds, ::Type{L}=LenseFlow) where {L} = Σ(L(ϕ),ds)
Σ(L::LenseOp, ds) = begin

    @unpack d,P,M,B,Cn,Cf,Cn̂, B̂ = ds

    SymmetricFuncOp(
        op   = x -> (Cn + P*M*B*L*Cf*L'*B'*M'*P')*x,
        op⁻¹ = x -> pcg2((Cn̂ .+ B̂*Cf*B̂'), Σ(L, ds), x, nsteps=100, tol=1e-1)
    )

end



@doc doc"""

    MAP_joint(ds::DataSet; L=LenseFlow, Nϕ=nothing, quasi_sample=nothing, nsteps=10, Ncg=500, cgtol=1e-1, αtol=1e-5, αmax=0.5, progress=false)

Compute the maximum a posteri estimate (MAP) from the joint posterior (can also
do a quasi-sample). 

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

Returns a tuple `(f, ϕ, tr)` where `f` is the best-fit (or quasi-sample) field,
`ϕ` is the lensing potential, and `tr` contains info about the run. 

"""
function MAP_joint(
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
    cache_function = nothing,
    callback = nothing,
    interruptable = false,
    progress = false)
    
    @assert progress in [false,:summary,:verbose]
    if !(isa(quasi_sample,Bool) || isa(quasi_sample,Int))
        throw(ArgumentError("quasi_sample should be true, false, or an Int."))
    end
    
    # since MAP estimate is done at fixed θ, we don't need to reparametrize to
    # ϕₘ = G(θ)*ϕ, so set G to constant here to avoid wasted computation
    @set! ds.G = IdentityOp
    @unpack d, D, Cϕ, Cf, Cf̃, Cn, Cn̂ = ds
    
    f, f° = nothing, nothing
    ϕ = (ϕstart==nothing) ? zero(Cϕ) : ϕstart
    Lϕ = cache(L(ϕ),d)
    α = 0
    tr = []
    hist = nothing
    
    
    # compute approximate inverse ϕ Hessian used in gradient descent, possibly
    # from quadratic estimate
    if (Nϕ == :qe); Nϕ = ϕqe(ds,false)[2]/2; end
    Hϕ⁻¹ = (Nϕ == nothing) ? Cϕ : (Cϕ^-1 + Nϕ^-1)^-1
    
    try
        @showprogress (progress==:summary ? 1 : Inf) "MAP_joint: " for i=1:nsteps

            # ==== f step ====
                
                # if we're doing a fixed quasi_sample, set the random seed here,
                # which controls the sample from the posterior we get from inside
                # `lensing_wiener_filter`
                if isa(quasi_sample,Int); seed!(quasi_sample); end
                
            # recache Lϕ for new ϕ
            if i!=1; cache!(Lϕ,ϕ); end
            
            # run wiener filter
            (f, hist) = lensing_wiener_filter(ds, ((i==1 && ϕstart==nothing) ? IdentityOp : Lϕ), 
                    (quasi_sample==false) ? :wf : :sample,   # if doing a quasi-sample, we get a sample instead of the WF
                    guess=(i==1 ? nothing : f),              # after first iteration, use the previous f as starting point
                    tol=cgtol, nsteps=Ncg, hist=(:i,:res), progress=(progress==:verbose))
                    
            f° = Lϕ * D * f
            lnPcur = lnP(:mix,f°,ϕ,ds,Lϕ)
            
            # ==== show progress ====
            if (progress==:verbose)
                @printf("(step=%i, χ²=%.2f, Ncg=%i%s)\n", i, -2lnPcur, length(hist), (α==0 ? "" : @sprintf(", α=%.6f",α)))
            end
            push!(tr,@dictpack(i,lnPcur,hist,ϕ,f))
            if callback != nothing
                callback(f, ϕ, tr)
            end
            
            # ==== ϕ step =====
            if (i!=nsteps)
                ϕnew = Hϕ⁻¹*(δlnP_δfϕₜ(:mix,f°,ϕ,ds,Lϕ))[2]
                res = optimize(α->(-lnP(:mix,f°,ϕ+α*ϕnew,ds,Lϕ)), 0., αmax, abs_tol=αtol)
                α = res.minimizer
                ϕ = ϕ+α*ϕnew
            end

        end
    catch err
        if (err isa InterruptException) && interruptable
            println()
            @warn("Maximization interrupted. Returning current progress.")
        else
            rethrow(err)
        end
    end

    return f, ϕ, tr
    
end


@doc doc"""

    MAP_marg( ds; kwargs...)

Compute the maximum a posteri estimate (MAP) of the marginl posterior.
"""
function MAP_marg(
    ds;
    ϕstart = nothing,
    L = LenseFlow,
    Nϕ = nothing,
    nsteps = 10, 
    Ncg = 500,
    cgtol = 1e-1,
    α = 0.02,
    Nmc_det = 50,
    )
    
    @unpack Cf, Cϕ, Cf̃, Cn̂ = ds
    
    # compute approximate inverse ϕ Hessian used in gradient descent, possibly
    # from quadratic estimate
    if (Nϕ == :qe); Nϕ = ϕqe(zero(Cf), Cf, Cf̃, Cn̂)[2]; end
    Hϕ⁻¹ = (Nϕ == nothing) ? Cϕ : (Cϕ^-1 + Nϕ^-1)^-1

    ϕ = (ϕstart != nothing) ? ϕstart : ϕ = zero(Cϕ) # fix needing to get zero(ɸ) this way
    tr = []

    for i=1:nsteps
        g, det_sims = δlnP_δϕ(ϕ, ds, progress=true, Nmc_det=Nmc_det, return_sims=true)
        ϕ += α * Hϕ⁻¹ * g
        push!(tr,@dictpack(i,g,det_sims,ϕ))
    end
    
    return ϕ, tr

end



@doc doc"""
    load_sim_dataset
    
Create a `DataSet` object with some simulated data. 

"""
function load_sim_dataset(;
    θpix,
    θpix_data = θpix,
    Nside,
    use,
    T = Float32,
    μKarcminT = 3,
    ℓknee = 100,
    αknee = 3,
    ℓmax_data = 3000,
    beamFWHM = 0,
    rfid = 0.05,
    Cℓ = camb(r=rfid),
    Cℓn = nothing,
    Cn = nothing,
    seed = nothing,
    M = nothing,
    B = nothing, B̂ = nothing,
    D = nothing,
    G = nothing,
    ϕ=nothing, f=nothing, f̃=nothing, Bf̃=nothing, n=nothing, d=nothing, # override any of these simulated fields
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
    Cℓf, Cℓf̃ = Cℓ[:f], Cℓ[:f̃]
    
    # types which depend on whether T/E/B
    use = Symbol(use)
    if (use == :EB)
        @warn("switch to use=:P")
        use = :P
    elseif (use == :TEB)
        @warn("switch to use=:TP")
        use = :TP
    end
    SS,ks = Dict(:TP=>((S0,S2),(:TT,:EE,:BB,:TE)), :P=>((S2,),(:EE,:BB)), :T=>((S0,),(:TT,)))[use]
    F,F̂,nF = Dict(:TP=>(FlatIQUMap,FlatTEBFourier,3), :P=>(FlatS2QUMap,FlatS2EBFourier,2), :T=>(FlatS0Map,FlatS0Fourier,1))[use]
    
    # pixelization
    P = (θpix_data == θpix) ? 1 : FuncOp(
        op  = f -> ud_grade(f, θpix_data, deconv_pixwin=false, anti_aliasing=false),
        opᴴ = f -> ud_grade(f, θpix,      deconv_pixwin=false, anti_aliasing=false)
    )
    Pix      = Flat{θpix,Nside,∂mode}
    Pix_data = Flat{θpix_data,Nside÷(θpix_data÷θpix),∂mode}
    
    # covariances
    Cϕ₀            =  Cℓ_to_cov(T,Pix,     S0,    Cℓf[:ϕϕ])
    Cfs,Cft,Cf̃,Cn̂  = (Cℓ_to_cov(T,Pix,     SS..., (Cℓx[k] for k=ks)...) for Cℓx in (Cℓ[:fs],Cℓ[:ft],Cℓf̃,Cℓn))
    if (Cn == nothing)
        Cn         =  Cℓ_to_cov(T,Pix_data,SS..., (Cℓn[k] for k=ks)...)
    end
    Cf = ParamDependentOp((;r=rfid, _...)->(@. Cfs + (r/rfid)*Cft))
    Cϕ = ParamDependentOp((;Aϕ=1,   _...)->(@. Aϕ*Cϕ₀))
    
    # data mask
    if (M == nothing) && (mask_kwargs != nothing)
        M = LowPass(ℓmax_data) * FullDiagOp(F{T,Pix_data}(repeated(T.(sptlike_mask(Nside÷(θpix_data÷θpix),θpix_data; mask_kwargs...)),nF)...))
    elseif (M == nothing)
        M = LowPass(ℓmax_data)
    end
    
    # beam
    if (B == nothing)
        B = let ℓ=0:ℓmax; Cℓ_to_cov(T,Pix,SS..., (InterpolatedCℓs(ℓ, (k==:TE ? zero(ℓ) : @.(exp(-ℓ^2*deg2rad(beamFWHM/60)^2/(8*log(2))/2)))) for k=ks)...); end;
    end
    if (B̂ == nothing)
        B̂ = B
    end
    
    # mixing matrices
    if (D == nothing); D = D_mix(Cf,rfid=rfid); end
    if (G == nothing); G = IdentityOp; end
    
    # simulate data
    if (seed != nothing); seed!(seed); end
    if (ϕ  == nothing); ϕ  = simulate(Cϕ); end
    if (f  == nothing); f  = simulate(Cf); end
    if (n  == nothing); n  = simulate(Cn); end
    if (f̃  == nothing); f̃  = L(ϕ)*f;       end
    if (Bf̃ == nothing); Bf̃ = B*f̃;          end
    if (d  == nothing); d  = M*P*Bf̃ + n;   end
    
    # put everything in DataSet
    ds = DataSet(;(@ntpack d Cn Cn̂ Cf Cf̃ Cϕ M B B̂ D G P)...)
    
    return @ntpack f f̃ ϕ n ds ds₀=>ds() T P=>Pix Cℓ
    
end


###


function load_healpix_sim_dataset(;
    Nside,
    use,
    gradient_cache,
    T = Float32,
    μKarcminT = 3,
    ℓknee = 100,
    αknee = 3,
    ℓmax_ops = 2Nside,
    ℓmax_data = 3000,
    beamFWHM = 0,
    rfid = 0.05,
    Cℓ = camb(r=rfid, ℓmax=ℓmax_ops),
    Cℓn = nothing,
    Cn = nothing,
    seed = nothing,
    M = nothing,
    B = nothing,
    D = nothing,
    G = nothing,
    ϕ = nothing,
    f = nothing,
    mask_kwargs = nothing,
    L = LenseFlow)
    
    @assert use==:T 
    
    # Cℓs
    if (Cℓn == nothing)
        Cℓn = noisecls(μKarcminT, beamFWHM=0, ℓknee=ℓknee, αknee=αknee, ℓmax=ℓmax_ops)
    end
    Cℓf, Cℓf̃ = Cℓ[:f], Cℓ[:f̃]

    Cf = IsotropicHarmonicCov(T.(nan2zero.(Cℓf[:TT][0:ℓmax_ops])), gradient_cache)
    Cf̃ = IsotropicHarmonicCov(T.(nan2zero.(Cℓf̃[:TT][0:ℓmax_ops])), gradient_cache)
    Cn = IsotropicHarmonicCov(T.(nan2zero.(Cℓn[:TT][0:ℓmax_ops])), gradient_cache)
    Cϕ = IsotropicHarmonicCov(T.(nan2zero.(Cℓf[:ϕϕ][0:ℓmax_ops])), gradient_cache)
    
    P=B=1 #for now
    
    if (seed != nothing); seed!(seed); end
    if (ϕ==nothing); ϕ = simulate(Cϕ); end
    if (f==nothing); f = simulate(Cf); end
    f̃ = L(ϕ)*f
    n = simulate(Cn)
    d = M*P*B*f̃ + n

    
    # put everything in DataSet
    ds = DataSet(;(@ntpack d Cn Cf Cf̃ Cϕ M)...)

    
    return @ntpack f f̃ ϕ n ds ds₀=>ds()


end
