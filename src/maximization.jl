
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
argmaxf_lnP(ϕ::Field,                ds::DataSet; kwargs...) = argmaxf_lnP(cache(ds.L(ϕ),ds.d), ds();      kwargs...)
argmaxf_lnP(ϕ::Field, θ::NamedTuple, ds::DataSet; kwargs...) = argmaxf_lnP(cache(ds.L(ϕ),ds.d), ds(;θ...); kwargs...)

function argmaxf_lnP(Lϕ, ds::DataSet; which=:wf, guess=nothing, preconditioner=:diag, conjgrad_kwargs=())
    
    check_hat_operators(ds)
    @unpack d, Cn, Cn̂, Cf, M, M̂, B, B̂, P = ds
    
    b = 0
    if (which in (:wf, :sample))
        b += Lϕ'*B'*P'*M'*(Cn\d)
    end
    if (which in (:fluctuation, :sample))
        b += Cf\simulate(Cf) + Lϕ'*B'*P'*M'*(Cn\simulate(Cn))
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
    Σ(ϕ::Field,  ds; conjgrad_kwargs=())
    Σ(Lϕ,        ds; conjgrad_kwargs=())
    
An operator for the data covariance, Cn + P*M*B*L*Cf*L'*B'*M'*P', which can
applied and inverted. `conjgrad_kwargs` are passed to the underlying call to
`conjugate_gradient`.
"""
Σ(ϕ::Field, ds; kwargs...) = Σ(ds.L(ϕ), ds; kwargs...)
Σ(Lϕ,       ds; conjgrad_kwargs=()) = begin

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
    ϕ = (ϕstart==nothing) ? zero(diag(Cϕ)) : ϕstart
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
            (f, hist) = argmaxf_lnP(((i==1 && ϕstart==nothing) ? IdentityOp : Lϕ), ds, 
                    which = (quasi_sample==false) ? :wf : :sample, # if doing a quasi-sample, we get a sample instead of the WF
                    guess = (i==1 ? nothing : f), # after first iteration, use the previous f as starting point
                    conjgrad_kwargs=(tol=cgtol, nsteps=Ncg, hist=(:i,:res), progress=(progress==:verbose)))
                    
            f°, = mix(f,ϕ,ds)
            lnPcur = lnP(:mix,f°,ϕ,ds)
            
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
                ϕnew = Hϕ⁻¹*(gradient(ϕ->lnP(:mix,f°,ϕ,ds), ϕ)[1])
                res = @ondemand(Optim.optimize)(α->(-lnP(:mix,f°,ϕ+α*ϕnew,ds)), T(0), T(αmax), abs_tol=αtol)
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
    Nϕ = nothing,
    nsteps = 10, 
    conjgrad_kwargs = (nsteps=500, tol=1e-1),
    α = 0.2,
    weights = :unlensed, 
    Nsims = 50,
    progress = :summary,
    )
    
    @unpack Cf, Cϕ, Cf̃, Cn̂ = ds
    T = eltype(Cf)
    
    # compute approximate inverse ϕ Hessian used in gradient descent, possibly
    # from quadratic estimate
    if (Nϕ == :qe); Nϕ = quadratic_estimate(ds).Nϕ/2; end
    Hϕ⁻¹ = (Nϕ == nothing) ? Cϕ : pinv(pinv(Cϕ) + pinv(Nϕ))

    ϕ = (ϕstart != nothing) ? ϕstart : ϕ = zero(diag(Cϕ))
    tr = []

    state = nothing
    
    @showprogress (progress==:summary ? 1 : Inf) "MAP_marg: " for i=1:nsteps
        g, state = δlnP_δϕ(
            ϕ, ds, Nsims=Nsims, weights=weights,
            progress=(progress==:verbose), return_state=true, previous_state=state,
            conjgrad_kwargs=conjgrad_kwargs
        )
        ϕ += T(α) * Hϕ⁻¹ * g
        push!(tr,@dictpack(i,g,state,ϕ))
    end
    
    return ϕ, tr

end
