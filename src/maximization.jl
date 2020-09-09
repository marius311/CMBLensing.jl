
## wiener filter

@doc doc"""
    argmaxf_lnP(ϕ,                ds::DataSet; kwargs...)
    argmaxf_lnP(ϕ, θ::NamedTuple, ds::DataSet; kwargs...)
    argmaxf_lnP(Lϕ,               ds::DataSet; kwargs...)
    
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
argmaxf_lnP(ϕ::Field,                ds::DataSet; kwargs...) = argmaxf_lnP(cache(ds.L(ϕ),ds.d), NamedTuple(), ds; kwargs...)
argmaxf_lnP(ϕ::Field, θ::NamedTuple, ds::DataSet; kwargs...) = argmaxf_lnP(cache(ds.L(ϕ),ds.d), θ,            ds; kwargs...)

function argmaxf_lnP(
    Lϕ, 
    θ::NamedTuple,
    ds::DataSet; 
    which = :wf, 
    guess = nothing, 
    preconditioner = :diag, 
    conjgrad_kwargs = (tol=1e-1,nsteps=500)
)
    
    check_hat_operators(ds)
    @unpack d, Cn, Cn̂, Cf, M, M̂, B, B̂, P = ds(θ)
    D = batchsize(d)
    
    Δ = d - nonCMB_data_components(θ,ds)
    b = 0
    if (which in (:wf, :sample))
        b += Lϕ'*B'*P'*M'*(Cn\Δ)
    end
    if (which in (:fluctuation, :sample))
        b += Cf\simulate(batch(Cf,D)) + Lϕ'*B'*P'*M'*(Cn\simulate(batch(Cn,D)))
    end
    
    A_diag  = pinv(Cf) +     B̂' *  M̂'*pinv(Cn̂)*M̂ * B̂
    A_zeroϕ = pinv(Cf) +     B'*P'*M'*pinv(Cn)*M*P*B
    A       = pinv(Cf) + Lϕ'*B'*P'*M'*pinv(Cn)*M*P*B*Lϕ
    
    A_preconditioner = @match preconditioner begin
        :diag  => A_diag
        :zeroϕ => FuncOp(op⁻¹ = (b -> conjugate_gradient(A_diag, A_zeroϕ, b, 0*b, tol=1e-1)))
        _      => error("Unrecognized preconditioner='$preconditioner'")
    end
    
    conjugate_gradient(A_preconditioner, A, b, (guess==nothing ? 0*b : guess); conjgrad_kwargs...)
    
end


@doc doc"""
    Σ(ϕ::Field,  ds; [conjgrad_kwargs])
    Σ(Lϕ,        ds; [conjgrad_kwargs])
    
An operator for the data covariance, Cn + P*M*B*L*Cf*L'*B'*M'*P', which can
applied and inverted. `conjgrad_kwargs` are passed to the underlying call to
`conjugate_gradient`.
"""
Σ(ϕ::Field, ds; kwargs...) = Σ(ds.L(ϕ), ds; kwargs...)
function Σ(Lϕ, ds; conjgrad_kwargs=(tol=1e-1,nsteps=500))
    @unpack d,P,M,B,Cn,Cf,Cn̂,B̂,M̂ = ds
    SymmetricFuncOp(
        op   = x -> (Cn + P*M*B*Lϕ*Cf*Lϕ'*B'*M'*P')*x,
        op⁻¹ = x -> conjugate_gradient((Cn̂ .+ M̂*B̂*Cf*B̂'*M̂'), Σ(Lϕ, ds), x; conjgrad_kwargs...)
    )
end



@doc doc"""

    MAP_joint(ds::DataSet; kwargs...)
    
Compute the maximum a posteriori (i.e. "MAP") estimate of the joint posterior,
$\mathcal{P}(f,\phi,\theta\,|\,d)$, or compute a quasi-sample. 


Keyword arguments:

* `ϕstart` — Starting point of the maximizer *(default:* $\phi=0$*)*
* `Nϕ` — Noise to use in the approximate hessian matrix. Can also give
    `Nϕ=:qe` to use the EB quadratic estimate noise *(default:* `:qe`*)*
* `quasi_sample` — `true` to iterate quasi-samples, or an integer to compute
    a specific quasi-sample.
* `nsteps` — The number of iterations for the maximizer
* `Ncg` — Maximum number of conjugate gradient steps during the $f$ update
* `cgtol` — Conjugrate gradient tolerance (will stop at `cgtol` or `Ncg`,
    whichever is first)
* `αtol` — Absolute tolerance on $\alpha$ in the linesearch in the $\phi$
    quasi-Newton-Rhapson step, $x^\prime = x - \alpha H^{-1} g$.  
* `αmax` — Maximum value for $\alpha$ in the linesearch
* `αiters` — Number of Brent's method steps actually used in the $\alpha$ linesearch 
* `progress` — whether to show progress bar

Returns a tuple `(f, ϕ, tr)` where `f` is the best-fit (or quasi-sample)
field, `ϕ` is the lensing potential, and `tr` contains info about the run. 

"""
MAP_joint(ds::DataSet; kwargs...) = MAP_joint(NamedTuple(), ds; kwargs...)
function MAP_joint(
    θ :: NamedTuple, 
    ds :: DataSet;
    ϕstart = nothing,
    Nϕ = :qe,
    quasi_sample = false, 
    nsteps = 10, 
    conjgrad_kwargs = (tol=1e-1,nsteps=500),
    preconditioner = :diag,
    αtol = 1e-5,
    αmax = 0.5,
    αiters = Int(floor(log(αmax/αtol))),
    cache_function = nothing,
    callback = nothing,
    interruptable::Bool = false,
    progress::Bool = true,
    aggressive_gc = fieldinfo(ds.d).Nside>=1024
)
    
    if !(isa(quasi_sample,Bool) || isa(quasi_sample,Int))
        throw(ArgumentError("quasi_sample should be true, false, or an Int."))
    end
    
    # since MAP estimate is done at fixed θ, we don't need to reparametrize to
    # ϕ° = G(θ)*ϕ, so set G to constant here to avoid wasted computation
    ds = copy(ds)
    ds.G = 1
    @unpack d, D, Cϕ, Cf, Cf̃, Cn, Cn̂, L = ds
    
    f, f° = nothing, nothing
    D = batchsize(d)
    ϕ = (ϕstart==nothing) ? zero(identity.(batch(diag(Cϕ),D))) : ϕstart
    ϕstep = nothing
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
        pbar = Progress(nsteps, (progress ? 0 : Inf), "MAP_joint: ")
        
        for i=1:nsteps

            # ==== f step ====
                
            # if we're doing a fixed quasi_sample, set the random seed here,
            # which controls the sample from the posterior we get from inside
            # `argmaxf_lnP`
            if isa(quasi_sample,Int); seed!(quasi_sample); end
                
            # recache Lϕ for new ϕ
            if i!=1; cache!(Lϕ,ϕ); end
            
            # run wiener filter
            (f, hist) = argmaxf_lnP(
                (i==1 && ϕstart==nothing) ? 1 : Lϕ, 
                θ,
                ds, 
                which = (quasi_sample==false) ? :wf : :sample, # if doing a quasi-sample, we get a sample instead of the WF
                guess = (i==1 ? nothing : f), # after first iteration, use the previous f as starting point
                conjgrad_kwargs=(hist=(:i,:res), progress=(progress==:verbose), conjgrad_kwargs...),
                preconditioner=preconditioner
            )
            aggressive_gc && GC.gc(true)

            f°, = mix(f,ϕ,ds)
            lnPcur = lnP(:mix,f°,ϕ,ds)
            
            # ==== show progress ====
            next!(pbar, showvalues=[("step",i), ("χ²",-2lnPcur), ("Ncg",length(hist)), ("α",α)])
            push!(tr,@namedtuple(i,lnPcur,hist,ϕ,f,α,ϕstep))
            if callback != nothing
                callback(f, ϕ, tr)
            end
            
            # ==== ϕ step =====
            if (i!=nsteps)
                ϕstep = Hϕ⁻¹ * gradient(ϕ->lnP(:mix,f°,ϕ,ds), ϕ)[1]
                neg_lnP(α) = -lnP(:mix, f°, ϕ + α*ϕstep, ds)
                α = optimize(batch(neg_lnP,D), T(0), T(αmax), abs_tol=0, rel_tol=0, iterations=αiters)
                ϕ = ϕ + α*ϕstep
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
MAP_marg(ds::DataSet; kwargs...) = MAP_marg(NamedTuple(), ds; kwargs...)
function MAP_marg(
    θ,
    ds :: DataSet;
    ϕstart = nothing,
    Nϕ = :qe,
    nsteps = 10, 
    nsteps_with_meanfield_update = 4,
    conjgrad_kwargs = (tol=1e-1,nsteps=500),
    α = 0.2,
    weights = :unlensed, 
    Nsims = 50,
    Nbatch = 1,
    progress::Bool = true,
    aggressive_gc = fieldinfo(ds.d).Nside>=512
)
    
    ds = (@set ds.G = 1)
    @unpack Cf, Cϕ, Cf̃, Cn̂ = ds
    T = eltype(Cf)
    
    # compute approximate inverse ϕ Hessian used in gradient descent, possibly
    # from quadratic estimate
    if (Nϕ == :qe); Nϕ = quadratic_estimate(ds).Nϕ/2; end
    Hϕ⁻¹ = (Nϕ == nothing) ? Cϕ : pinv(pinv(Cϕ) + pinv(Nϕ))

    ϕ = (ϕstart != nothing) ? ϕstart : ϕ = zero(diag(Cϕ))
    tr = []
    state = nothing
    pbar = Progress(nsteps, (progress ? 0 : Inf), "MAP_marg: ")
    
    for i=1:nsteps
        aggressive_gc && GC.gc(true)
        g, state = δlnP_δϕ(
            ϕ, θ, ds,
            use_previous_MF = i>nsteps_with_meanfield_update,
            Nsims=Nsims, Nbatch=Nbatch, weights=weights,
            progress=false, return_state=true, previous_state=state,
            conjgrad_kwargs=conjgrad_kwargs
        )
        ϕ += T(α) * Hϕ⁻¹ * g
        push!(tr, @dict(i,g,ϕ))
        next!(pbar, showvalues=[
            ("step",i), 
            ("Ncg", length(state.gQD.hist)), 
            ("Ncg_sims", i<=nsteps_with_meanfield_update ? length(state.gQD_sims[1].hist) : "(MF not updated)"),
            ("α",α)
        ])
    end
    
    return ϕ, tr

end


"""
    optimize(f::Function, args...; kwargs...)
    optimize(f::BatchedFunction, args...; kwargs...)

Tiny wrapper around `Optim.optimize` but if the target function is a batched
function, takes care of making the call asynchronous and batching the result.
E.g.:

    optimize(batch(x -> x.^[2,4,6], 3), -1, 1)

simultaneously optimizes `x^2`, `x^4`, and `x^6`, but results in only a single
call to `x -> x.^[2,4,6]` per iteration of the Optim algorithm. Note: unlike
Optim.optimize, this just returns the minimizer point. 
"""
optimize(f::Function, args...; kwargs...) = @ondemand(Optim.optimize)(f, args...; kwargs...).minimizer
function optimize(f::BatchedFunction, args...; kwargs...)
    results = asyncmap(1:batchsize(f)) do _
        optimize(f.f, args...; kwargs...)
    end
    batch(results...)
end