
"""
    mix(f, ϕ,                ds::DataSet)
    mix(f, ϕ, θ::NamedTuple, ds::DataSet)
    
Compute the mixed `(f°, ϕ°)` from the unlensed field `f` and lensing potential `ϕ`. 
"""
mix(f, ϕ, ds::DataSet) = mix(f,ϕ,NamedTuple(),ds)
function mix(f, ϕ, θ::NamedTuple, ds::DataSet)
    @unpack D,G,L = ds(;θ...)
    L(ϕ)*D*f, G*ϕ
end


"""
    unmix(f°, ϕ°,                ds::DataSet)
    unmix(f°, ϕ°, θ::NamedTuple, ds::DataSet)

Compute the unmixed/unlensed `(f, ϕ)` from the mixed field `f°` and mixed lensing potential `ϕ°`. 
"""
unmix(f°, ϕ°, ds::DataSet) = unmix(f°,ϕ°,NamedTuple(),ds)
function unmix(f°, ϕ°, θ::NamedTuple, ds::DataSet)
    @unpack D,G,L = ds(;θ...)
    ϕ = G\ϕ°
    D\(L(ϕ)\f°), ϕ
end


@doc doc"""
    lnP(t, fₜ, ϕₜ,                ds::DataSet, Lϕ=nothing)
    lnP(t, fₜ, ϕₜ, θ::NamedTuple, ds::DataSet, Lϕ=nothing)

Compute the log posterior probability in the joint parameterization as a
function of the field, $f_t$, the lensing potential, $\phi_t$, and possibly some
cosmological parameters, $\theta$. The subscript $t$ can refer to either a
"time", e.g. passing `t=0` corresponds to the unlensed parametrization and `t=1`
to the lensed one, or can be `:mix` correpsonding to the mixed parametrization.
In all cases, the arguments `fₜ` and `ϕₜ` should then be $f$ and $\phi$ in
that particular parametrization.

If any parameters $\theta$ are passed, we also include the three determinant
terms to properly normalize the posterior.

The argument `ds` should be a `DataSet` and stores the masks, data, etc...
needed to construct the posterior. If `Lϕ` is provided, it will be used as
memory to recache the lensing operator at the specified ϕ.
"""
lnP(t, fₜ, ϕₜ,                ds::DataSet, Lϕ=nothing) = lnP(Val(t), fₜ, ϕₜ, NamedTuple(), ds, Lϕ)
lnP(t, fₜ, ϕₜ, θ::NamedTuple, ds::DataSet, Lϕ=nothing) = lnP(Val(t), fₜ, ϕₜ, θ,            ds, Lϕ)

function lnP(::Val{t}, fₜ, ϕₜ, θ::NamedTuple, ds::DataSet, Lϕ) where {t}
    dsθ = ds(;θ...)
    ϕ = (t==:mix) ? dsθ.G\ϕₜ : ϕₜ
    Lϕ = (Lϕ == nothing) ? cache(ds.L(ϕ),fₜ) : cache!(Lϕ,ϕ)
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
    lnP = -(Δ'*pinv(Cn)*Δ + f'*pinv(Cf)*f + ϕ'*pinv(Cϕ)*ϕ)/2
    
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
    -(  (depends_on(ds.Cn, θ) ? logdet(ds₀.Cn\dsθ.Cn) : 0) 
      + (depends_on(ds.Cf, θ) ? logdet(ds₀.Cf\dsθ.Cf) : 0)
      + (depends_on(ds.Cϕ, θ) ? logdet(ds₀.Cϕ\dsθ.Cϕ) : 0))/2f0
end



## joint posterior gradients

@doc doc"""

    δlnP_δfϕₜ(t, fₜ, ϕ,                ds, Lϕ=nothing)
    δlnP_δfϕₜ(t, fₜ, ϕ, θ::NamedTuple, ds, Lϕ=nothing)

Compute a gradient of the log posterior probability. See `lnP` for definition of
arguments of this function. 

The return type is a `FieldTuple` corresponding to the $(f_t,\phi)$ derivative.
"""
δlnP_δfϕₜ(t, fₜ, ϕ,                ds, Lϕ=nothing) = δlnP_δfϕₜ(Val(t), fₜ, ϕ, NamedTuple(), ds, Lϕ)
δlnP_δfϕₜ(t, fₜ, ϕ, θ::NamedTuple, ds, Lϕ=nothing) = δlnP_δfϕₜ(Val(t), fₜ, ϕ, θ,            ds, Lϕ)

function δlnP_δfϕₜ(::Val{t}, fₜ, ϕₜ, θ::NamedTuple, ds::DataSet, Lϕ) where {t}
    dsθ = ds(;θ...)
    ϕ = (t==:mix) ? dsθ.G\ϕₜ : ϕₜ
    Lϕ = (Lϕ == nothing) ? cache(ds.L(ϕ),fₜ) : cache!(Lϕ,ϕ)
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
    (δlnP_δf°, δlnP_δϕ°) = δfϕ_δf̃ϕ(Lϕ, Lϕ⁻¹f°, f°)' * FΦTuple(D \ δlnP_δf, δlnP_δϕ)
    FΦTuple(δlnP_δf°, G \ δlnP_δϕ°)

end
# derivatives of the three posterior probability terms at the times at which
# they're easy to take (used above)
δlnL_δf̃ϕ(f̃,ϕ::ɸ,dsθ)  where {ɸ} = (@unpack P,M,B,Cn,Cf,Cϕ,d=dsθ; FΦTuple(B'*P'*M'*(Cn\(d-M*P*B*f̃)), zero(Cϕ.diag)))
δlnΠᶠ_δfϕ(f,ϕ::ɸ,dsθ) where {ɸ} = (@unpack Cf,Cϕ=dsθ;            FΦTuple(-(Cf\f)                  , zero(Cϕ.diag)))
δlnΠᶲ_δfϕ(f::F,ϕ,dsθ) where {F} = (@unpack Cf,Cϕ=dsθ;            FΦTuple(zero(Cf.diag)            , -(Cϕ\ϕ)))



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
    argmaxf_lnP(ϕ,                ds::DataSet; kwargs...)
    argmaxf_lnP(ϕ, θ::NamedTuple, ds::DataSet; kwargs...)
    argmaxf_lnP(Lϕ::LenseOp,      ds::DataSet; kwargs...)
    
Computes either the Wiener filter at fixed $\phi$, or a sample from this slice
along the posterior.

Keyword arguments: 

* `which` — `:wf`, `:sample`, or `fluctuation` to compute 1) the Wiener filter,
  i.e. the best-fit of $\mathcal{P}(f\,|\,\phi,d)$, 2) a sample from
  $\mathcal{P}(f\,|\,\phi,d)$, or 3) a sample minus the Wiener filter, i.e. the
  fluctuation on top of the mean.
* `guess` — starting guess for `f` for the conjugate gradient solver
* `kwargs...` — all other arguments are passed to [`conjugate_gradient`](@ref)

"""
argmaxf_lnP(ϕ::Field,                ds; kwargs...) = argmaxf_lnP(cache(ds.L(ϕ),ds.d), ds();      kwargs...)
argmaxf_lnP(ϕ::Field, θ::NamedTuple, ds; kwargs...) = argmaxf_lnP(cache(ds.L(ϕ),ds.d), ds(;θ...); kwargs...)

function argmaxf_lnP(Lϕ::LenseOp, ds::DataSet; which=:wf, guess=nothing, kwargs...)
    
    check_hat_operators(ds)
    @unpack d, Cn, Cn̂, Cf, M, M̂, B, B̂, P = ds
    
    b = 0
    if (which in (:wf, :sample))
        b += Lϕ'*B'*P'*M'*(Cn\d)
    end
    if (which in (:fluctuation, :sample))
        b += Cf\simulate(Cf) + Lϕ'*B'*P'*M'*(Cn\simulate(Cn))
    end
    
    conjugate_gradient(
        pinv(Cf) + M̂'*B̂'*pinv(Cn̂)*B̂*M̂,
        pinv(Cf) + Lϕ'*B'*P'*M'*pinv(Cn)*M*P*B*Lϕ,
        b,
        guess==nothing ? 0*b : guess;
        kwargs...
    )
    
end


@doc doc"""
    Σ(ϕ, ds, ::Type{L}=LenseFlow) where {L}
    Σ(L::LenseOp, ds) 
    
An operator for the data covariance, Cn + P*M*B*L*Cf*L'*B'*M'*P', which can
applied and inverted.
"""
Σ(ϕ, ds) = Σ(ds.L(ϕ),ds)
Σ(Lϕ::LenseOp, ds) = begin

    @unpack d,P,M,B,Cn,Cf,Cn̂,B̂ = ds

    SymmetricFuncOp(
        op   = x -> (Cn + P*M*B*Lϕ*Cf*Lϕ'*B'*M'*P')*x,
        op⁻¹ = x -> conjugate_gradient((Cn̂ .+ B̂*Cf*B̂'), Σ(Lϕ, ds), x, nsteps=100, tol=1e-1)
    )

end



@doc doc"""

    MAP_joint(ds::DataSet; kwargs...)
    
Compute the maximum a posteriori (i.e. "MAP") estimate of the joint posterior,
$\mathcal{P}(f,\phi,\theta\,|\,d)$, or compute a quasi-sample. 


Keyword arguments:

* `ϕstart` — Starting point of the maximizer *(default:* $\phi=0$*)*
* `Nϕ` — Noise to use in the approximate hessian matrix. Can also give `Nϕ=:qe` 
         to use the EB quadratic estimate noise *(default:* `:qe`*)*
* `quasi_sample` — `true` to iterate quasi-samples, or an integer to compute
                   a specific quasi-sample.
* `nsteps` — The number of iterations for the maximizer
* `Ncg` — Maximum number of conjugate gradient steps during the $f$ update
* `cgtol` — Conjugrate gradient tolerance (will stop at `cgtol` or `Ncg`, whichever is first)
* `αtol` — Absolute tolerance on $\alpha$ in the linesearch in the $\phi$ quasi-Newton-Rhapson step, $x^\prime = x - \alpha H^{-1} g$
* `αmax` — Maximum value for $\alpha$ in the linesearch
* `progress` — `false`, `:summary`, or `:verbose`, to control progress output

Returns a tuple `(f, ϕ, tr)` where `f` is the best-fit (or quasi-sample) field,
`ϕ` is the lensing potential, and `tr` contains info about the run. 

"""
function MAP_joint(
    ds;
    ϕstart = nothing,
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
    @unpack d, D, Cϕ, Cf, Cf̃, Cn, Cn̂, L = ds
    
    f, f° = nothing, nothing
    ϕ = (ϕstart==nothing) ? zero(Cϕ.diag) : ϕstart
    Lϕ = cache(L(ϕ),d)
    T = real(eltype(d))
    α = 0
    tr = []
    hist = nothing
    
    
    # compute approximate inverse ϕ Hessian used in gradient descent, possibly
    # from quadratic estimate
    if (Nϕ == :qe); Nϕ = quadratic_estimate(ds).Nϕ/2; end
    Hϕ⁻¹ = (Nϕ == nothing) ? Cϕ : pinv(pinv(Cϕ) + pinv(Nϕ))
    
    try
        @showprogress (progress==:summary ? 1 : Inf) "MAP_joint: " for i=1:nsteps

            # ==== f step ====
                
            # if we're doing a fixed quasi_sample, set the random seed here,
            # which controls the sample from the posterior we get from inside
            # `argmaxf_lnP`
            if isa(quasi_sample,Int); seed!(quasi_sample); end
                
            # recache Lϕ for new ϕ
            if i!=1; cache!(Lϕ,ϕ); end
            
            # run wiener filter
            (f, hist) = argmaxf_lnP(((i==1 && ϕstart==nothing) ? NoLensing() : Lϕ), ds, 
                    which = (quasi_sample==false) ? :wf : :sample, # if doing a quasi-sample, we get a sample instead of the WF
                    guess = (i==1 ? nothing : f), # after first iteration, use the previous f as starting point
                    tol=cgtol, nsteps=Ncg, hist=(:i,:res), progress=(progress==:verbose))
                    
            f° = Lϕ * D * f
            lnPcur = lnP(:mix,f°,ϕ,ds,Lϕ)
            
            # ==== show progress ====
            if (progress==:verbose)
                @printf("(step=%i, χ²=%.2f, Ncg=%i%s)\n", i, -2lnPcur, length(hist), (α==0 ? "" : @sprintf(", α=%.6f",α)))
            end
            push!(tr,@namedtuple(i,lnPcur,hist,ϕ,f,α))
            if callback != nothing
                callback(f, ϕ, tr)
            end
            
            # ==== ϕ step =====
            if (i!=nsteps)
                ϕnew = Hϕ⁻¹*(δlnP_δfϕₜ(:mix,f°,ϕ,ds,Lϕ))[2]
                res = @ondemand(Optim.optimize)(α->(-lnP(:mix,f°,ϕ+α*ϕnew,ds,Lϕ)), T(0), T(αmax), abs_tol=αtol)
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

    return @namedtuple(f, ϕ, tr)
    
end


@doc doc"""

    MAP_marg(ds; kwargs...)

Compute the maximum a posteriori (i.e. "MAP") estimate of the marginal posterior,
$\mathcal{P}(\phi,\theta\,|\,d)$.

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
    if (Nϕ == :qe); Nϕ = quadratic_estimate(ds).Nϕ/2; end
    Hϕ⁻¹ = (Nϕ == nothing) ? Cϕ : pinv(pinv(Cϕ) + pinv(Nϕ))

    ϕ = (ϕstart != nothing) ? ϕstart : ϕ = zero(Cϕ.diag) # fix needing to get zero(ɸ) this way
    tr = []

    for i=1:nsteps
        g, det_sims = δlnP_δϕ(ϕ, ds, progress=true, Nmc_det=Nmc_det, return_sims=true)
        ϕ += α * Hϕ⁻¹ * g
        push!(tr,@dictpack(i,g,det_sims,ϕ))
    end
    
    return ϕ, tr

end
