
## wiener filter

@doc doc"""
    argmaxf_logpdf(ds::DataSet, Ω::NamedTuple, [d = ds.d]; kwargs...)

Maximize the `logpdf` for `ds` over `f`, given all the other arguments
are held fixed at `Ω`. E.g.: `argmaxf_logpdf(ds, (; ϕ, θ=(Aϕ=1.1,))`.

Keyword arguments: 

* `fstart` — starting guess for `f` for the conjugate gradient solver
* `conjgrad_kwargs` — Passed to the inner call to
  [`conjugate_gradient`](@ref)

"""
function argmaxf_logpdf(
    ds :: DataSet,
    Ω :: NamedTuple, 
    d = ds.d;
    fstart = nothing, 
    preconditioner = :diag, 
    conjgrad_kwargs = (tol=1e-1,nsteps=500)
)
    
    @unpack Cf, B̂, M̂, Cn̂ = ds

    # TODO: generalize this to something like: A_preconditioner = preconditioner(ds)[:f]
    A_preconditioner = pinv(Cf) + B̂'*M̂'*pinv(Cn̂)*M̂*B̂

    zero_f = zero(diag(Cf))

    # brittle (but working) performance hack until we switch to Diffractor (see also flowops.jl)
    task_local_storage(:AD_constants, keys(Ω)) do 

        # the following will give the argmax for any model with Gaussian P(f,d|z...)
        b  = -gradientf_logpdf(ds; f=zero_f, d=d,       Ω...)
        a₀ =  gradientf_logpdf(ds; f=zero_f, d=zero(d), Ω...)
        A = FuncOp(f -> (gradientf_logpdf(ds; f, d=zero(d), Ω...) - a₀))
        conjugate_gradient(A_preconditioner, A, b, (isnothing(fstart) ? zero_f : fstart); conjgrad_kwargs...)

    end

end

@doc doc"""
    sample_f([rng::AbstractRNG], ds::DataSet, Ω::NamedTuple, [d = ds.d]; kwargs...)

Draw a posterior sample of `f` from the `logpdf` for `ds`, given all the other arguments are
held fixed at `Ω`. E.g.: `sample_f(ds, (; ϕ, θ=(Aϕ=1.1,))`.

Keyword arguments: 

* `fstart` — starting guess for `f` for the conjugate gradient solver
* `conjgrad_kwargs` — Passed to the inner call to [`conjugate_gradient`](@ref)

"""
function sample_f(rng::AbstractRNG, ds::DataSet, Ω, d=ds.d; kwargs...)
    # the following will give a sapmle for any model with Gaussian P(f,d|z...)
    sim = simulate(rng, ds; Ω...)
    sim.f + argmaxf_logpdf(ds, Ω, d - sim.d; kwargs...)[1]
end
sample_f(ds::DataSet, args...; kwargs...) = sample_f(Random.default_rng(), ds, args...; kwargs...)


# allows specific DataSets to override this as a performance
# optimization, since Zygote is ~50% slower than the old hand-written
# code even after the above hack. shouldn't need this once we have
# Diffractor. the following is the fallback which just uses Zygote:
gradientf_logpdf(ds::DataSet; f, Ω...) = gradient(f -> logpdf(ds; f, Ω...), f)[1]



@doc doc"""

    MAP_joint(ds::DataSet; kwargs...)
 
Compute the maximum a posteriori (i.e. "MAP") estimate of the joint
posterior, $\mathcal{P}(f,\phi,\theta\,|\,d)$, or compute a
quasi-sample. 


Keyword arguments:

* `nsteps` — The maximum number of iterations for the maximizer.
* `ϕstart = 0` — Starting point of the maximizer.
* `ϕtol = nothing` — If given, stop when `ϕ` updates reach this
  tolerance. `ϕtol` is roughly the relative per-pixel standard
  deviation between changes to `ϕ` and draws from the `ϕ` prior.
  Values in the range $10^{-2}-10^{-4}$ are reasonable. 
* `nburnin_update_hessian = 10` — How many steps to wait before
  starting to do diagonal updates to the Hessian
* `conjgrad_kwargs = (;)` — Passed to the inner call to
  [`conjugate_gradient`](@ref).
* `progress = true` — Whether to show the progress bar.
* `Nϕ = :qe` — Noise to use in the initial approximation to the
   Hessian. Can give `:qe` to use the quadratic estimate noise.
* `quasi_sample = false` — `false` to compute the MAP, `true` to
   iterate quasi-samples, or an integer to compute a fixed-seed
   quasi-sample.
* `history_keys` — What quantities to include in the returned
  `history`. Can be any subset of `(:f, :f°, :ϕ, :∇ϕ_lnP, :χ², :lnP)`.

Returns a tuple `(f, ϕ, history)` where `f` is the best-fit (or
quasi-sample) field, `ϕ` is the lensing potential, and `history`
contains the history of steps during the run. 

"""
MAP_joint(ds::DataSet; kwargs...) = MAP_joint((;), ds; kwargs...)
function MAP_joint(
    θ, 
    ds :: DataSet;
    nsteps = 20,
    Nϕ = :qe,
    ϕstart = nothing,
    fstart = nothing,
    ϕtol = nothing,
    αtol = 1e-4,
    nburnin_update_hessian = Inf,
    progress::Bool = true,
    conjgrad_kwargs = (tol=1e-1, nsteps=500),
    quasi_sample = false,
    history_keys = (:logpdf,),
    aggressive_gc = false
)

    dsθ = copy(ds(θ))
    dsθ.G = I # MAP estimate is invariant to G so avoid wasted computation

    ϕ = Map(isnothing(ϕstart) ? zero(diag(ds.Cϕ)) : ϕstart)
    T = eltype(ϕ)
    
    # compute approximate inverse ϕ Hessian used in gradient descent, possibly
    # from quadratic estimate
    if (Nϕ == :qe)
        Nϕ = quadratic_estimate(dsθ).Nϕ/2
    end
    Hϕ⁻¹ = (Nϕ == nothing) ? dsθ.Cϕ : pinv(pinv(dsθ.Cϕ) + pinv(Nϕ))

    history = []
    pbar = Progress(nsteps, (progress ? 0 : Inf), "MAP_joint: ")
    ProgressMeter.update!(pbar)

    f = prevf = prevϕ = prev_∇ϕ_logpdf = nothing
 
    for step = 1:nsteps

        # f step
        isa(quasi_sample,Int) && seed!(global_rng_for(ϕ), quasi_sample)
        sample_or_argmax_f = quasi_sample ? sample_f : argmaxf_logpdf
        (f, argmaxf_logpdf_history) = @⌛ sample_or_argmax_f(
            dsθ, 
            (;ϕ, θ);
            fstart = prevf, 
            conjgrad_kwargs = (history_keys=(:i,:res), progress=false, conjgrad_kwargs...)
        )
        aggressive_gc && cuda_gc()

        # ϕ step
        f°, = mix(dsθ; f, ϕ)
        ∇ϕ_logpdf, = @⌛ gradient(ϕ->logpdf(Mixed(dsθ); f°, ϕ°=ϕ), ϕ)
        s = (Hϕ⁻¹ * ∇ϕ_logpdf)
        αmax = 0.5 * get_max_lensing_step(ϕ, s)
        soln = @ondemand(Optim.optimize)(0, T(αmax), @ondemand(Optim.Brent)(); abs_tol=αtol) do α
            total_logpdf = @⌛(sum(unbatch(-logpdf(Mixed(dsθ); f°, ϕ°=ϕ+α*s, dsθ.d))))
            isnan(total_logpdf) ? T(α/αmax) * prevfloat(T(Inf)) : total_logpdf # workaround for https://github.com/JuliaNLSolvers/Optim.jl/issues/828
        end
        α = T(soln.minimizer)
        ϕ += α * s
        
        # finalize
        _logpdf = @⌛ logpdf(Mixed(dsθ); f°, ϕ°=ϕ, dsθ.d)
        total_logpdf = sum(unbatch(_logpdf))
        next!(pbar, showvalues = [
            ("step",       step), 
            ("logpdf",     *(map(x->@sprintf("%.2f, ",x), unbatch(_logpdf))...)[1:end-2]),
            ("α",          α), 
            ("CG",         "$(length(argmaxf_logpdf_history)) iterations"), 
            ("Linesearch", "$(soln.iterations) bisections")
        ])
        push!(history, select((;f,f°,ϕ,∇ϕ_logpdf,total_logpdf,α,logpdf=_logpdf,Hϕ⁻¹,argmaxf_logpdf_history), history_keys))
        if (
            !isnothing(ϕtol) &&
            !isnothing(prevϕ) &&
            sum(unbatch(norm(LowPass(1000) * (sqrt(ds.Cϕ) \ (ϕ - prevϕ))) / sqrt(2length(ϕ)))) < ϕtol
        )
            break
        else
            if step > nburnin_update_hessian
                Hϕ⁻¹_unsmooth = Diagonal(abs.(Fourier(ϕ - prevϕ) ./ Fourier(∇ϕ_logpdf - prev_∇ϕ_logpdf)))
                Hϕ⁻¹ = Cℓ_to_Cov(:I, f.proj, smooth(ℓ⁴*cov_to_Cℓ(Hϕ⁻¹_unsmooth), xscale=:log, yscale=:log, smoothing=0.05)/ℓ⁴)
            end
            prevf, prevϕ, prev_∇ϕ_logpdf = f, ϕ, ∇ϕ_logpdf
        end

    end

    ProgressMeter.finish!(pbar)

    f, ϕ, history

end

MAP_joint(θ, ds::NoLensingDataSet; kwargs...) = (argmaxf_lnP(I, θ, ds; kwargs...), nothing)


@doc doc"""

    MAP_marg(ds; kwargs...)

Compute the maximum a posteriori (i.e. "MAP") estimate of the marginal posterior,
$\mathcal{P}(\phi,\theta\,|\,d)$.

"""
MAP_marg(ds::DataSet; kwargs...) = MAP_marg((;), ds; kwargs...)
function MAP_marg(
    θ,
    ds :: DataSet;
    rng = Random.default_rng(),
    ϕstart = nothing,
    nsteps = 10, 
    nsteps_with_meanfield_update = 4,
    conjgrad_kwargs = (tol=1e-1,nsteps=500),
    α = 0.2,
    weights = :unlensed, 
    Nsims = 50,
    Nbatch = 1,
    progress::Bool = true,
    pmap = (myid() in workers() ? map : (f,args...) -> pmap(f, default_worker_pool(), args...)),
    aggressive_gc = false
)
    
    (mod(Nsims+1,nworkers()) == 0) || @warn "MAP_marg is most efficient when Nsims+1 is divisible by the number of workers." maxlog=1
    (ds.d.Nbatch == Nbatch == 1) || error("MAP_marg for batched fields not implemented")

    dsθ = ds(θ)
    dsθ.G = I # MAP_marg is invariant to G so avoid wasted computation
    set_distributed_dataset(dsθ)
    @unpack Cϕ, Nϕ, d, Cf, Cn, L = dsθ
    Hϕ⁻¹ = pinv(pinv(Cϕ) + pinv(Nϕ))
    T = real(eltype(d))
    
    ϕ = (ϕstart != nothing) ? ϕstart : ϕ = zero(diag(Cϕ))
    f_wf_sims_prev = fill(nothing, Nsims÷Nbatch)
    f_wf_prev = nothing
    ḡ = nothing

    tr = []

    if progress
        pbar = DistributedProgress(nsteps_with_meanfield_update*Nsims + nsteps, 0.1, "MAP_marg: ")
        ProgressMeter.update!(pbar)
    end

    for step = 1:nsteps

        aggressive_gc && cuda_gc()

        # generate simulated data for the current ϕ
        Lϕ = L(ϕ)
        _rng = copy(rng)
        d_sims = map(1:Nsims) do _
            simulate(_rng, dsθ; ϕ).d
        end

        # gradient of data and mean-field sims

        function gMAP(Lϕ, dsθ, f_wf_prev, i)
            f_wf, history = argmaxf_logpdf(
                dsθ, 
                (;ϕ, θ);
                fstart = (isnothing(f_wf_prev) ? 0d : f_wf_prev),
                conjgrad_kwargs = (history_keys=(:i,:res), conjgrad_kwargs...)
            )
            aggressive_gc && cuda_gc()
            g, = gradient(ϕ -> logpdf(dsθ; f=f_wf, ϕ, dsθ.d), getϕ(Lϕ))
            progress && next!(pbar, showvalues=[
                ("step", step),
                ("α",    α),
                ("CG ($(i==0 ? "data" : "sim $i"))", "$(length(history)) iterations"),
            ])
            (;g, f_wf, history)
        end

        if step > nsteps_with_meanfield_update
            gMAP_data = gMAP(Lϕ, dsθ, f_wf_prev, 0)
            f_wf_prev = gMAP_data.f_wf
        else
            gMAP_data, gMAP_sims = peel(pmap(0:Nsims, [dsθ.d, d_sims...], [f_wf_prev, f_wf_sims_prev...]) do i, d, f_wf_prev
                gMAP(Lϕ, @set(get_distributed_dataset().d=d), f_wf_prev, i)
            end)
            ḡ = mean(map(gMAP_sims) do gMAP
                mean(unbatch(gMAP.g))
            end)
            f_wf_sims_prev = getindex.(gMAP_sims,:f_wf)
        end
    
        # final total posterior gradient, including gradient of the prior
        g = gMAP_data.g - ḡ - Cϕ\ϕ

        # take step
        ϕ += T(α) * Hϕ⁻¹ * g

        push!(tr, (;step, g, ϕ))
        
    end

    set_distributed_dataset(nothing) # free memory
    
    return ϕ, tr

end
