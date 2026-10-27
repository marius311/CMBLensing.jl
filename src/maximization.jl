
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
    conjgrad_kwargs = (tol=1e-1,nsteps=500),
    offset = false,
)
    
    Hess_preconditioner = Hessian_logpdf_preconditioner(:f, ds)
    zero_f = zero(diag(ds.Cf))

    # brittle (but working) performance hack until we switch to Diffractor (see also flowops.jl)
    task_local_storage(:AD_constants, keys(Ω)) do 

        # the following will give the argmax for any model with Gaussian P(f,d|z...)
        b  = -gradientf_logpdf(ds; f=zero_f, d=d,       Ω...)
        a₀ =  gradientf_logpdf(ds; f=zero_f, d=zero(d), Ω...)
        offset && (b += a₀)
        Hess = FuncOp(f -> (gradientf_logpdf(ds; f, d=zero(d), Ω...) - a₀))
        conjugate_gradient(Hess_preconditioner, Hess, b, (isnothing(fstart) ? zero_f : fstart); conjgrad_kwargs...)

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
    # the following will give a sample for any model with Gaussian P(f,d|z...)
    sim = simulate(rng, ds; Ω...)
    Δf, history = argmaxf_logpdf(ds, Ω, d - sim.d; kwargs..., offset=true)
    sim.f + Δf, history
end
sample_f(ds::DataSet, args...; kwargs...) = sample_f(Random.default_rng(), ds, args...; kwargs...)


# allows specific DataSets to override this as a performance
# optimization, since Zygote is ~50% slower than the old hand-written
# code even after the above hack. shouldn't need this once we have
# Diffractor. the following is the fallback which just uses Zygote:
gradientf_logpdf(ds::DataSet; f, Ω...) = gradient(f -> logpdf(ds; f, Ω...), f)[1]



@doc doc"""

    MAP_joint([θ], ds::DataSet, [Ωstart=(ϕ=0,)]; kwargs...)
 
Compute the maximum a posteriori (i.e. "MAP") estimate of the joint
posterior, $\mathcal{P}(f,\phi,\theta\,|\,d)$, or compute a
quasi-sample. 

Positional arguments:

* `[θ]` — Optional θ at which to do maximization.
* `ds::DataSet` — The DataSet which defines the posterior
* `[Ωstart=(ϕ=0,)]` — Optional starting point for the non-Gaussian
  fields to optimize over. The maximizer does a coordinate descent
  which alternates between updating `f` which the posterior is assumed
  to be Gaussian in, and updating the fields in `Ωstart` (which by
  default is just `ϕ`).

Keyword arguments:

* `nsteps` — The maximum number of iterations for the maximizer.
* `ϕtol = nothing` — If given, stop when `ϕ` updates reach this
  tolerance. `ϕtol` is roughly the relative per-pixel standard
  deviation between changes to `ϕ` and draws from the `ϕ` prior.
  Values in the range $10^{-2}-10^{-4}$ are reasonable. 
* `nburnin_update_hessian = Inf` — How many steps to wait before
  starting to do diagonal updates to the Hessian
* `conjgrad_kwargs = (;)` — Passed to the inner call to
  [`conjugate_gradient`](@ref).
* `progress = true` — Whether to show the progress bar.
* `quasi_sample = false` — `false` to compute the MAP, `true` to
   iterate quasi-samples, or an integer to compute a fixed-seed
   quasi-sample.
* `history_keys` — What quantities to include in the returned
  `history`. Can be any subset of `(:f, :f°, :ϕ, :∇ϕ_logpdf, :χ²,
  :logpdf)`.

Returns a tuple `(f, ϕ, history)` where `f` is the best-fit (or
quasi-sample) field, `ϕ` is the lensing potential, and `history`
contains the history of steps during the run. 

"""
MAP_joint(ds::DataSet, args...; kwargs...) = MAP_joint((;), ds, args...; kwargs...)
function MAP_joint(
    θ, 
    ds :: DataSet,
    Ωstart = FieldTuple(ϕ=Map(zero(diag(ds.Cϕ))));
    nsteps = 20,
    minsteps = 0,
    fstart = nothing,
    αtol = 1e-4,
    gradtol = 0,
    αmax = nothing,
    prior_deprojection_factor = 0,
    nburnin_update_hessian = Inf,
    progress::Bool = true,
    conjgrad_kwargs = (tol=1e-1, nsteps=500),
    quasi_sample = false,
    history_keys = (:logpdf,),
    aggressive_gc = false,
)

    if isfinite(nburnin_update_hessian)
        keys((;Ωstart...,)) == (:ϕ,) || error("nburnin_update_hessian only implemented for (f,ϕ)-only maximization.")
    end

    sample_or_argmax_f = 
        quasi_sample == false ? argmaxf_logpdf :
        quasi_sample == true ? sample_f : 
        quasi_sample isa AbstractRNG ? (args...; kwargs...) -> sample_f(copy(quasi_sample), args...; kwargs...) : 
        error("`quasi_sample` should be true, false, or an AbstractRNG")

    dsθ = copy(ds(θ))
    dsθ.G = I # MAP estimate is invariant to G so avoid wasted computation

    
    history = []
    pbar = Progress(nsteps, (progress ? 0 : Inf), "MAP_joint: ")
    ProgressMeter.update!(pbar)
    
    prevΩ = prevΩ° = prev_∇Ω°_logpdf = HΩ° = showvalues = nothing
    Ω = Ωstart
    f = prevf = fstart
    α = 1
    αmax_initial = αmax
    t_f_total = t_ϕ_total = 0

    for step = 1:nsteps

        ## f step
        t_f = @elapsed begin
            (f, argmaxf_logpdf_history) = @⌛ sample_or_argmax_f(
                dsθ, 
                (;Ω..., θ);
                fstart = prevf, 
                conjgrad_kwargs = (history_keys=(:i,:res), progress=false, conjgrad_kwargs...)
            )
            aggressive_gc && cuda_gc()
        end

        # gradient
        t_ϕ = @elapsed begin
            ## ϕ step
            @unpack f° = (Ω° = mix(dsθ; f, Ω..., θ))
            Ω° = FieldTuple(delete(Ω°, (:f°, :θ)))
            ∇Ω°_logpdf, = @⌛ gradient(Ω°->logpdf(Mixed(dsθ); f°, Ω°..., θ), Ω°)
            # Hessian
            if step > nburnin_update_hessian
                HΩ°⁻¹_unsmooth = Diagonal(abs.(Fourier(Ω°.ϕ° - prevΩ°.ϕ°) ./ Fourier(∇Ω°_logpdf.ϕ° - prev_∇Ω°_logpdf.ϕ°)))
                HΩ°⁻¹_smooth = Cℓ_to_Cov(:I, f.proj, smooth(ℓ⁴*cov_to_Cℓ(HΩ°⁻¹_unsmooth), xscale=:log, yscale=:log, smoothing=0.05)/ℓ⁴)
                HΩ° = Diagonal(FieldTuple(ϕ°=diag(pinv(HΩ°⁻¹_smooth))))
            elseif HΩ° == nothing
                HΩ° = Hessian_logpdf_preconditioner(keys((;Ω°...,)), dsθ)
            end
            # line search
            ΔΩ° = pinv(HΩ°) * ∇Ω°_logpdf
            T = real(eltype(f))
            if prior_deprojection_factor != 0
                ΔΩ°_perp = pinv(HΩ°) * gradient(ΔΩ° -> logprior(dsθ; unmix(dsθ; f°, ΔΩ°...)...), ΔΩ°)[1]
                ΔΩ° .-= T(prior_deprojection_factor * dot(ΔΩ°,ΔΩ°_perp) * pinv(dot(ΔΩ°_perp,ΔΩ°_perp))) .* ΔΩ°_perp
            end
            αmax = @something(αmax_initial, 2α)
            soln = @ondemand(Optim.optimize)(T(0), T(αmax), @ondemand(Optim.Brent)(); abs_tol=T(αtol)) do α
                Ω°′ = Ω° + T(α) * ΔΩ°
                total_logpdf = @⌛(sum(unbatch(-(logpdf(Mixed(dsθ); f°, Ω°′..., θ)))))
                isnan(total_logpdf) ? T(α/αmax) * prevfloat(T(Inf)) : total_logpdf # workaround for https://github.com/JuliaNLSolvers/Optim.jl/issues/828
            end
            α = T(soln.minimizer)
            Ω° += α * ΔΩ°
        end
        
        ## finalize
        _logpdf = @⌛ logpdf(Mixed(dsθ); f°, Ω°..., θ)
        Ω = delete(unmix(dsθ; f°, Ω°..., θ), (:f, :θ))
        ΔΩ°_norm = norm(ΔΩ°)
        total_logpdf = sum(unbatch(_logpdf))
        showvalues = [
            ("step",       step), 
            ("logpdf",     join(map(x->@sprintf("%.2f",x), [unbatch(_logpdf)...]), ", ")),
            ("α",          α),
            ("ΔΩ°_norm",   @sprintf("%.2g", ΔΩ°_norm)),
            ("CG",         "$(length(argmaxf_logpdf_history)) iterations ($(@sprintf("%.2f",t_f)) sec)"), 
            ("Linesearch", "$(soln.iterations) bisections ($(@sprintf("%.2f",t_ϕ)) sec)")
        ]
        next!(pbar; showvalues)
        push!(history, select((;f°,f,Ω°...,Ω...,∇Ω°_logpdf,total_logpdf,α,αmax,ΔΩ°,ΔΩ°_norm,logpdf=_logpdf,HΩ°,argmaxf_logpdf_history), history_keys))
        
        # early stop based on tolerance
        if (step > minsteps) && (norm(ΔΩ°) < gradtol)
            break
        end
        prevf, prevΩ, prevΩ°, prev_∇Ω°_logpdf = f, Ω, Ω°, ∇Ω°_logpdf

    end

    ProgressMeter.finish!(pbar)
    ProgressMeter.updateProgress!(pbar; showvalues)

    (;f, Ω..., history)

end

MAP_joint(θ, ds::NoLensingDataSet; kwargs...) = (argmaxf_logpdf(I, θ, ds; kwargs...), nothing)

# A solve owns one small registry on its originating process. A worker gets a
# private dataset only when it is first used; later tasks send a channel
# reference rather than serializing its potentially large lens cache again.
function _map_marg_new_registry(evaluated_ds)
    registry = RemoteChannel(() -> Channel{Any}(1), myid())
    # Keep a fresh, unregistered dataset object for by-value worker transport.
    put!(registry, (; ds=copy(evaluated_ds), caches=Dict{Int,RemoteChannel}()))
    registry
end

function _map_marg_builtin_cpu_lens(ds)
    ds isa BaseDataSet || return false
    L = ds.L
    L isa CachedLenseFlow || return false
    fields = (L.ϕ[], L.memŁf, L.memÐf, L.memŁϕ, L.memÐϕ,
              L.memŁvf..., L.memÐvf..., L.memŁvϕ..., L.memÐvϕ...)
    all(f -> f isa BaseField && f.arr isa Array, fields)
end

function _map_marg_new_worker_cache(evaluated_ds, origin)
    # A deserialized built-in CPU lens is already private on a remote worker.
    # Keep the conservative copy for local, custom, or non-Array storage.
    local_ds = if myid() == origin || !_map_marg_builtin_cpu_lens(evaluated_ds)
        local_copy = copy(evaluated_ds)
        local_copy.L = deepcopy(local_copy.L)
        local_copy
    else
        evaluated_ds
    end
    cache = RemoteChannel(() -> Channel{Any}(1), myid())
    put!(cache, (; ds=local_ds, guard=ReentrantLock(), active=Ref(true)))
    cache
end

function _map_marg_worker_cache(registry)
    remotecall_fetch(registry.where, registry, myid()) do registry, worker
        state = take!(registry)
        try
            get!(state.caches, worker) do
                remotecall_fetch(_map_marg_new_worker_cache, worker, state.ds, registry.where)
            end
        finally
            put!(registry, state)
        end
    end
end

function _map_marg_with_worker_cache(f, registry)
    cache = _map_marg_worker_cache(registry)
    entry = fetch(cache)
    lock(entry.guard) do
        entry.active[] || error("MAP_marg worker cache is closed")
        f(entry.ds)
    end
end

function _map_marg_close_worker_cache(cache)
    entry = fetch(cache)
    lock(entry.guard) do
        entry.active[] = false
        take!(cache)
        close(cache)
    end
    nothing
end

function _map_marg_close_registry(registry)
    state = take!(registry)
    close(registry)
    cleanup_error = nothing
    for (worker, cache) in state.caches
        try
            remotecall_wait(_map_marg_close_worker_cache, worker, cache)
        catch err
            # Continue releasing the other workers even if one fails.
            @warn "Could not release MAP_marg worker cache" worker exception=(err, catch_backtrace())
            isnothing(cleanup_error) && (cleanup_error = err)
        end
    end
    cleanup_error === nothing || throw(cleanup_error)
    nothing
end


@doc doc"""

    MAP_marg(ds; kwargs...)

Compute the maximum a posteriori (i.e. "MAP") estimate of the marginal posterior,
$\mathcal{P}(\phi,\theta\,|\,d)$.

A solve with positive `nsteps` requires positive `Nsims` and
`nsteps_with_meanfield_update`, because no initial mean field is supplied.

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
    if nsteps > 0
        nsteps_with_meanfield_update > 0 || throw(ArgumentError("nsteps_with_meanfield_update must be positive when nsteps is positive"))
        Nsims > 0 || throw(ArgumentError("Nsims must be positive when nsteps is positive"))
    end
    
    (mod(Nsims+1,nworkers()) == 0) || @warn "MAP_marg is most efficient when Nsims+1 is divisible by the number of workers." maxlog=1
    (ds.d.Nbatch == Nbatch == 1) || error("MAP_marg for batched fields not implemented")

    dsθ = copy(ds(θ))
    dsθ.G = I # MAP_marg is invariant to G so avoid wasted computation
    # The lensing operator may own mutable caches. Keep the caller's dataset
    # (including any @distributed registration) unchanged during this solve.
    dsθ.L = deepcopy(dsθ.L)
    @unpack Cϕ, Nϕ, d, Cf, Cn, L = dsθ
    Hϕ⁻¹ = pinv(pinv(Cϕ) + pinv(Nϕ))
    T = real(eltype(d))
    
    ϕ = (ϕstart != nothing) ? ϕstart : ϕ = zero(diag(Cϕ))
    f_wf_sims_prev = fill(nothing, Nsims÷Nbatch)
    f_wf_prev = nothing
    ḡ = nothing

    tr = []

    registry = _map_marg_new_registry(dsθ)
    pbar = nothing
    solve_failed = false
    try
        if progress
            pbar = DistributedProgress(nsteps_with_meanfield_update*Nsims + nsteps, 0.1, "MAP_marg: ")
            ProgressMeter.update!(pbar)
        end
        for step = 1:nsteps

            aggressive_gc && cuda_gc()

            # generate simulated data for the current ϕ
            _rng = copy(rng)
            d_sims = map(1:Nsims) do _
                simulate(_rng, dsθ; ϕ).d
            end

            # gradient of data and mean-field sims

            function gMAP(dsθ, f_wf_prev, i)
                f_wf, history = argmaxf_logpdf(
                    dsθ,
                    (;ϕ, θ);
                    fstart = (isnothing(f_wf_prev) ? 0 * dsθ.d : f_wf_prev),
                    conjgrad_kwargs = (history_keys=(:i,:res), conjgrad_kwargs...)
                )
                aggressive_gc && cuda_gc()
                g, = gradient(ϕ -> logpdf(dsθ; f=f_wf, ϕ, dsθ.d), ϕ)
                progress && next!(pbar, showvalues=[
                    ("step", step),
                    ("α",    α),
                    ("CG ($(i==0 ? "data" : "sim $i"))", "$(length(history)) iterations"),
                ])
                (;g, f_wf, history)
            end

            if step > nsteps_with_meanfield_update
                gMAP_data = gMAP(dsθ, f_wf_prev, 0)
                f_wf_prev = gMAP_data.f_wf
            else
                gMAP_data, gMAP_sims = peel(pmap(0:Nsims, [dsθ.d, d_sims...], [f_wf_prev, f_wf_sims_prev...]) do i, d, f_wf_prev
                    _map_marg_with_worker_cache(registry) do worker_ds
                        gMAP(@set(worker_ds.d=d), f_wf_prev, i)
                    end
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

        return ϕ, tr
    catch
        solve_failed = true
        rethrow()
    finally
        # Workers may still be reporting progress while their cache drains.
        # Release those workers before closing the progress channel.
        cleanup_error = nothing
        try
            _map_marg_close_registry(registry)
        catch err
            @warn "Could not release MAP_marg worker registry" exception=(err, catch_backtrace())
            cleanup_error = err
        end
        if pbar !== nothing
            try
                ProgressMeter.finish!(pbar)
            catch err
                @warn "Could not finish MAP_marg progress" exception=(err, catch_backtrace())
                isnothing(cleanup_error) && (cleanup_error = err)
            end
        end
        # Preserve an exception from the solve, but surface cleanup failures
        # when the solve itself succeeded.
        !solve_failed && cleanup_error !== nothing && throw(cleanup_error)
    end

end
