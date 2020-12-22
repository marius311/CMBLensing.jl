
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
* `conjgrad_kwargs` — Passed to the inner call to [`conjugate_gradient`][@ref]

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
    A_zeroϕ = pinv(Cf) +     B'*P'*M'*pinv(Cn̂)*M*P*B
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
* `nsteps` — The number of iterations for the maximizer.
* `lbfgs_rank` — The maximum rank of the LBFGS approximation to the
   Hessian *(default: 5)*
* `conjgrad_kwargs` — Passed to the inner call to [`conjugate_gradient`][@ref]
* `progress` — whether to show progress bar
* `Nϕ` — Noise to use in the initial approximation to the Hessian. Can also
   give `Nϕ=:qe` to use the quadratic estimate noise *(default:* `:qe`*)*
* `quasi_sample` — `false` *(default)* to compute the MAP, `true` to iterate
   quasi-samples, or an integer to compute a fixed-seed quasi-sample.

Returns a tuple `(f, ϕ, tr)` where `f` is the best-fit (or quasi-sample) field,
`ϕ` is the lensing potential, and `tr` contains info about the run. 

"""
MAP_joint(ds::DataSet; kwargs...) = MAP_joint(NamedTuple(), ds; kwargs...)
function MAP_joint(
    θ :: NamedTuple, 
    ds :: DataSet; 
    nsteps = 20,
    lbfgs_rank = 5, 
    Nϕ = :qe,
    ϕstart = nothing,
    progress::Bool = true,
    verbosity = (0,0),
    conjgrad_kwargs = (tol=1e-1,nsteps=500),
    quasi_sample = false,
    preconditioner = :diag,
    aggressive_gc = fieldinfo(ds.d).Nx>=1024 & fieldinfo(ds.d).Ny>=1024,
    αtol = nothing,
    αmax = nothing,
)

    if αtol!=nothing || αmax!=nothing
        @warn "αtol and αmax are deprecated and will be removed in the future. This is now handled automatically."
    end

    dsθ = copy(ds(θ))
    dsθ.G = 1 # MAP estimate is invariant to G so avoid wasted computation

    ϕ = Map(ϕstart==nothing ? zero(diag(ds.Cϕ)) : ϕstart)
    
    # compute approximate inverse ϕ Hessian used in gradient descent, possibly
    # from quadratic estimate
    if (Nϕ == :qe)
        Nϕ = quadratic_estimate(dsθ).Nϕ/2
    end
    Hϕ⁻¹ = (Nϕ == nothing) ? dsθ.Cϕ : pinv(pinv(dsθ.Cϕ) + pinv(Nϕ))

    tr = []
    argmaxf_lnP_kwargs = (
        which = (quasi_sample==false) ? :wf : :sample,
        preconditioner = preconditioner,
        conjgrad_kwargs = (hist=(:i,:res), progress=false, conjgrad_kwargs...),
    )
    pbar = Progress(nsteps, (progress ? 0 : Inf), "MAP_joint: ")

    f, = argmaxf_lnP((ϕstart==nothing ? 1 : ϕ), θ, dsθ; argmaxf_lnP_kwargs...)
    f°, = mix(f, ϕ, dsθ)
    ϕ, = optimize(
        ϕ -> (
            sum(unbatch(-2lnP(:mix,f°,ϕ,dsθ))), 
            gradient(ϕ->-2lnP(:mix,f°,ϕ,dsθ), ϕ)[1]
        ),
        Map(ϕ),
        OptimKit.LBFGS(
            lbfgs_rank; 
            maxiter = nsteps, 
            verbosity = verbosity[1], 
            linesearch = OptimKit.HagerZhangLineSearch(verbosity=verbosity[2], maxiter=5)
        ), 
        finalize! = function (ϕ,χ²,g,i)
            if isa(quasi_sample,Int) 
                seed!(global_rng_for(f),quasi_sample)
            end
            (f, hist) = argmaxf_lnP(
                ϕ, θ, dsθ;
                guess = f, 
                argmaxf_lnP_kwargs...
            )
            aggressive_gc && gc()
            f°, = mix(f, ϕ, dsθ)
            g .= gradient(ϕ->-2lnP(:mix,f°,ϕ,dsθ), ϕ)[1]
            χ²s = -2lnP(:mix,f°,ϕ,dsθ)
            χ² = sum(unbatch(χ²s))
            next!(pbar, showvalues=[("step",i), ("χ²",χ²s), ("Ncg",length(hist))])
            push!(tr, @namedtuple(f,ϕ,lnPcur=-χ²/2))
            ϕ, χ², g
        end;
        inner = (_,ξ1,ξ2)->sum(unbatch(dot(ξ1,ξ2))),
        precondition = (_,η)->Map(Hϕ⁻¹*η)
    )

    f, = argmaxf_lnP(ϕ, θ, dsθ; guess=f, argmaxf_lnP_kwargs...)

    f,ϕ,tr

end

OptimKit._scale!(η::Field, β) = η .*= β
OptimKit._add!(η::Field, ξ::Field, β) = η .+= β .* ξ




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
    aggressive_gc = fieldinfo(ds.d).Nx >=512 & fieldinfo(ds.d).Ny >=512
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
        aggressive_gc && gc()
        g, state = δlnP_δϕ(
            ϕ, θ, ds,
            use_previous_MF = i>nsteps_with_meanfield_update,
            Nsims=Nsims, Nbatch=Nbatch, weights=weights,
            progress=false, return_state=true, previous_state=state,
            conjgrad_kwargs=conjgrad_kwargs, aggressive_gc=aggressive_gc
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
