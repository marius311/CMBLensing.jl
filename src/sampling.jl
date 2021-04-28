
"""
    symplectic_integrate(x₀, p₀, Λ, U, δUδx, N=50, ϵ=0.1, progress=false)

Do a symplectic integration of the potential energy `U` (with gradient
`δUδx`) starting from point `x₀` with momentum `p₀` and mass matrix
`Λ`. The number of steps is `N` and the step size `ϵ`. 

Returns `ΔH, xᵢ, pᵢ` corresponding to change in Hamiltonian, and final
position and momenta. If `history_keys` is specified a history of
requested variables throughout each step is also returned. 

"""
function symplectic_integrate(
    x₀::AbstractVector{T}, p₀, Λ, U, δUδx=x->gradient(U,x)[1]; 
    N=50, ϵ=T(0.1), progress=false, history_keys=nothing
) where {T}
    
    xᵢ, pᵢ = x₀, p₀
    δUδxᵢ = δUδx(xᵢ)
    H(x,p) = U(x) - p⋅(Λ\p)/2
    
    history = []

    @showprogress (progress ? 1 : Inf) "Symplectic Integration: " for i=1:N
        xᵢ₊₁    = xᵢ - T(ϵ) * (Λ \ (pᵢ - T(ϵ)/2 * δUδxᵢ))
        δUδxᵢ₊₁ = δUδx(xᵢ₊₁)
        pᵢ₊₁    = pᵢ - T(ϵ)/2 * (δUδxᵢ₊₁ + δUδxᵢ)
        xᵢ, pᵢ, δUδxᵢ = xᵢ₊₁, pᵢ₊₁, δUδxᵢ₊₁
        
        if !isnothing(history_keys)
            historyᵢ = (;i, x=xᵢ, p=pᵢ, δUδx=δUδxᵢ₊₁, H=(haskey(history_keys,:H) ? H(xᵢ,pᵢ) : nothing))
            push!(history, select(historyᵢ, history_keys))
        end

    end

    ΔH = H(xᵢ,pᵢ) - H(x₀,p₀)
    
    if isnothing(history)
        return ΔH, xᵢ, pᵢ
    else
        return ΔH, xᵢ, pᵢ, history
    end
    
end



@doc doc"""
    grid_and_sample(lnP::Function; range::NamedTuple; progress=false, nsamples=1)

Interpolate the log pdf `lnP` with support on `range`, and return  the
integrated log pdf as well `nsamples` samples (drawn via inverse transform
sampling)

`lnP` should either accept a NamedTuple argument and `range` should be a
NamedTuple mapping those same names to `range` objects specifying where to
evaluate `lnP`, e.g.:

```julia
grid_and_sample(nt->-(nt.x^2+nt.y^2)/2, (x=range(-3,3,length=100),y=range(-3,3,length=100)))
```

or `lnP` should accept a single scalar argument and `range` should be directly
the range for this variable:

```julia
grid_and_sample(x->-x^2/2, range(-3,3,length=100))
```

The return value is `(lnP, samples, Px)` where `lnP` is an interpolated/smoothed
log PDF which can be evaluated anywhere within the original range, `Px` are
sampled points of the original PDF, and `samples` is a NamedTuple giving the
Monte-Carlo samples of each of the parameters.

(Note: only 1D sampling is currently implemented, but 2D like in the example
above is planned)
"""
function grid_and_sample(lnP::Function, range::AbstractVector; progress=false, kwargs...)
    lnPs = @showprogress (progress ? 1 : Inf) "Grid Sample: " map(lnP, range)
    grid_and_sample(lnPs, range; progress=progress, kwargs...)
end

function grid_and_sample(lnPs::Vector{<:BatchedReal}, xs::AbstractVector; kwargs...)
    batches = [grid_and_sample(batch_index.(lnPs,i), xs; kwargs...) for i=1:batch_length(lnPs[1])]
    ((batch(getindex.(batches,i)) for i=1:3)...,)
end

function grid_and_sample(lnPs::Vector, xs::AbstractVector; progress=false, nsamples=1, span=0.25, require_convex=false)
    
    # trim leading/trailing zero-probability regions
    support = findnext(isfinite,lnPs,1):findprev(isfinite,lnPs,length(lnPs))
    xs = xs[support]
    lnPs = lnPs[support]
    
    if require_convex
        support = longest_run_of_trues(finite_second_derivative(lnPs) .< 0)
        xs = xs[support]
        lnPs = lnPs[support]
    end

    # interpolate PDF
    xmin, xmax = first(xs), last(xs)
    lnPs = lnPs .- maximum(lnPs)
    ilnP = loess(xs, lnPs, span=span)
    
    # normalize the PDF. note the smoothing is done of the log PDF.
    cdf(x) = quadgk(nan2zero∘exp∘ilnP,xmin,x,rtol=1e-3)[1]
    logA = nan2zero(log(cdf(xmax)))
    lnPs = (ilnP.ys .-= logA)
    ilnP.bs[:,1] .-= logA
    
    # draw samples via inverse transform sampling
    θsamples = @showprogress (progress ? 1 : Inf) map(1:nsamples) do i
        r = rand()
        if (cdf(xmin)-r)*(cdf(xmax)-r) >= 0
            first(lnPs) > last(lnPs) ? xmin : xmax
        else
            fzero(x->cdf(x)-r, xmin, xmax, xatol=(xmax-xmin)*1e-3)
        end
    end
    
    (nsamples==1 ? θsamples[1] : θsamples), ilnP, lnPs
    
end

function grid_and_sample(lnP::Function, range::NamedTuple{S,<:NTuple{1}}; kwargs...) where {S}
    NamedTuple{S}.(Ref.(grid_and_sample(x -> lnP(NamedTuple{S}(x)), first(range); kwargs...)))
end

# allow more convenient evaluation of Loess-interpolated functions
(m::Loess.LoessModel)(x) = Loess.predict(m,x)




@doc doc"""
    sample_joint(ds::DataSet; kwargs...)

Sample the joint posterior, $\mathcal{P}(f,\phi,\theta\,|\,d)$. 

Keyword arguments: 

* `nsamps_per_chain` — The number of samples per chain.
* `nchains = 1` — Number of chains in parallel.
* `nchunk = 1` — Number of steps between parallel chain communication.
* `nsavemaps = 1` — Number of steps in between saving maps into chain.
* `nburnin_always_accept = 0` — Number of steps at the beginning of
  the chain to always accept HMC steps regardless of integration
  error.
* `nburnin_fixθ = 0` — Number of steps at the beginning of the chain
  before starting to sample `θ`.
* `Nϕ = :qe` — Noise to use in the initial approximation to the
  Hessian. Can give `:qe` to use the quadratic estimate noise.
* `chains = nothing` — `nothing` to start a new chain; the return
  value from a previous call to `sample_joint` to resume those chains;
  `:resume` to resume chains from a file given by `filename`
* `θrange` — Range and density to grid sample parameters as a
  NamedTuple, e.g. `(Aϕ=range(0.7,1.3,length=20),)`. 
* `θstart` — Starting values of parameters as a NamedTuple, e.g.
  `(Aϕ=1.2,)`, or nothing to randomly sample from θrange
* `ϕstart` — Starting `ϕ`, either a `Field` object, `:quasi_sample`,
  or `:best_fit`
* `metadata` — Does nothing, but is saved into the chain file
* `nhmc = 1` — Number of HMC passes per `ϕ` Gibbs step.
* `symp_kwargs = fill((N=25, ϵ=0.01), nhmc)` — an array of NamedTupe
  kwargs to pass to [`symplectic_integrate`](@ref). E.g.
  `[(N=50,ϵ=0.1),(N=25,ϵ=0.01)]` would do 50 large steps then 25
  smaller steps per each Gibbs pass. If specified, `nhmc` is ignored.
* `wf_kwargs` — Keyword arguments to pass to [`argmaxf_lnP`](@ref) in
  the Wiener Filter Gibbs step.
* `MAP_kwargs` — Keyword arguments to pass to [`MAP_joint`](@ref) when
  computing the starting point.
"""
function sample_joint(
    ds :: DataSet;
    gibbs_initializers = [
        gibbs_initialize_θ!,
        gibbs_initialize_ϕ!,
        gibbs_initialize_f!
    ],
    gibbs_samplers = [
        gibbs_sample_f!,
        gibbs_mix!,
        gibbs_sample_ϕ!,
        gibbs_unmix!,
        gibbs_postprocess!
    ],
    nsamps_per_chain,
    nchains = nworkers(),
    nfilewrite = 5,
    nsavemaps = 1,
    filename = nothing,
    ϕstart = :prior,
    θstart = :prior,
    θrange = NamedTuple(),
    pmap = (myid() in workers() ? map : pmap),
    pool = default_worker_pool(),
    conjgrad_kwargs = (tol=1e-1, nsteps=500),
    preconditioner = :diag,
    nhmc = 1,
    symp_kwargs = fill((N=25, ϵ=0.01), nhmc),
    MAP_kwargs = (nsteps=40,),
    metadata = nothing,
    progress = false,
    storage = basetype(fieldinfo(ds.d).M),
    grid_and_sample_kwargs = (;),
    kwargs...
)

    # rundat is a Dict with all the args and kwargs minus a few removed ones
    rundat = merge!(foldl(delete!, (:ds, :gibbs_initializers, :gibbs_samplers, :kwargs, :pmap, :pool), init=Base.@locals()), kwargs)
    rundat[:Nbatch] = batch_length(ds.d)

    # dont adapt things passed in kwargs when we adapt the state dict
    _adapt(storage, state) = Dict(k => (haskey(rundat,k) ? v : adapt(storage, v)) for (k,v) in state)

    function filter_for_saving(state, step)
        Dict(k=>v for (k,v) in state if (!(haskey(rundat,k)) && !(k in (:pbar_dict, :timer)) && (step == 1 || (step % nsavemaps) == 0 || !isa(v,Field))))
    end

    # validate arguments
    if !(progress in [false,:summary,:verbose])
        error("`progress` should be one of [false, :summary, :verbose]")
    end
    if (filename!=nothing && splitext(filename)[2]!=".jld2")
        error("Chain filename '$filename' should have '.jld2' extension.")
    end
    
    # seed
    @everywhere @eval CMBLensing seed!(global_rng_for($storage))

    # distributed dataset object to workers once
    set_distributed_dataset(ds)

    # initialize chains
    if (filename != nothing) && isfile(filename)

        @info "Resuming chain at $filename"
        local chunks_index, prev_chunks
        jldopen(filename,"r") do io
            chunks_index = maximum([parse(Int,k[8:end]) for k in keys(io) if startswith(k,"chunks_")]) + 1
            states = last.(read(io, "chunks_$(chunks_index-1)"))
            merge!.(states, Ref(rundat))
            @unpack step = states[1]
            chain_chunks = map(copy, repeated([], nchains))
        end

    else

        chunks_index = step = 1

        states = pmap(map(copy,repeated(rundat,nchains))) do state
            state = _adapt(storage, state)
            for gibbs_initialize! in gibbs_initializers
                gibbs_initialize!(state, get_distributed_dataset())
            end
            @pack! state = step
            _adapt(Array, state)
        end
        chain_chunks = [Any[filter_for_saving(state, step)] for state in states]
        
        if filename != nothing
            save(filename, "rundat", cpu(rundat))
        end

    end
    
    # setup progressbar
    setindex!.(states, copy.(Ref(OrderedDict{String,Any}("step"=>step))), :pbar_dict)
    if progress == :summary
        pbar = Progress(nsamps_per_chain-step, dt=0, desc="Gibbs chain: ")
        ProgressMeter.update!(pbar, showvalues=[("step", step)])
        @everywhere $reset_timer!()
    end
    
    # start sampling
    for step in (step+1):nsamps_per_chain
        
        setindex!.(states, step, :step)

        state₁, = states = pmap(pool, states) do state
            
            state = @⌛ _adapt(storage, state)
            @unpack step, pbar_dict = state
                
            timing = @⌛ "Gibbs passes" map(gibbs_samplers) do gibbs_sample!
                @elapsed gibbs_sample!(state, get_distributed_dataset())
            end
            
            state = @⌛ _adapt(Array, state)

            timer = get_defaulttimer()
            @pack! state = timing, timer

            state

        end

        push!.(chain_chunks, filter_for_saving.(states, step))

        if (filename != nothing) && ((step % nfilewrite) == 0)
            @⌛ jldopen(filename,"a+") do io
                wsession = JLDWriteSession()
                write(io, "chunks_$chunks_index", chain_chunks, wsession)
            end
            chunks_index += 1
            empty!.(chain_chunks)
        end

        if progress == :summary
            print("\033[2J")
            next!(pbar, showvalues = [("step",step), ("timing",state₁[:timing]), collect(state₁[:pbar_dict])...])
            merge!(state₁[:timer].inner_timers, get_defaulttimer().inner_timers)
            print(state₁[:timer])
            flush(stdout)
        end

    end

    Chains(chain_chunks)

end


lnP_arg_names(p::Union{Int,Symbol}, ds::DataSet) = lnP_arg_names(Val(p), ds)
lnP_arg_names(::Val{0},    ds::DataSet) = (:f,  :ϕ,  :θ)
lnP_arg_names(::Val{:mix}, ds::DataSet) = (:f°, :ϕ°, :θ)

## initialization

function gibbs_initialize_θ!(state, ds::DataSet)
    @unpack θstart, θrange, nchains, Nbatch = state
    θ = @match θstart begin
        :prior => map(range->batch((first(range) .+ rand(Nbatch) .* (last(range) - first(range)))...), θrange)
        (_::NamedTuple) => θstart
        _ => throw(ArgumentError(θstart))
    end
    lnPθ = map(_->missing, θrange)
    @pack! state = θ, lnPθ
end

function gibbs_initialize_f!(state, ds::DataSet)
    f = missing
    @pack! state = f
end

function gibbs_initialize_ϕ!(state, ds::DataSet)
    @unpack ϕstart, θ, nchains, Nbatch = state
    ϕ = @match ϕstart begin
        :prior     => simulate(ds.Cϕ(θ); Nbatch)
        0          => zero(diag(ds.Cϕ)) * batch(ones(Int,Nbatch)...)
        (_::Field) => ϕstart
        (:quasi_sample|:best_fit) => MAP_joint(
            adapt(storage, ds(θstart)), 
            progress = (progress==:verbose ? :summary : false), 
            Nϕ = adapt(storage,Nϕ),
            quasi_sample = (ϕstart==:quasi_sample); MAP_kwargs...
        ).ϕ
        _ => throw(ArgumentError(ϕstart))
    end
    @pack! state = ϕ
end


## gibbs passes

@⌛ function gibbs_sample_f!(state, ds::DataSet)
    @unpack f, progress, preconditioner, conjgrad_kwargs = state
    z_other = delete(select(state, lnP_arg_names(0, ds)), :f)
    f = argmaxf_lnP(
        z_other..., ds;
        which = :sample, 
        guess = ismissing(f) ? nothing : f, 
        preconditioner = preconditioner, 
        conjgrad_kwargs = (progress=(progress==:verbose), conjgrad_kwargs...)
    )
    @pack! state = f
end

@⌛ function gibbs_sample_ϕ!(state, ds::DataSet)
    @unpack f°, ϕ°, θ, symp_kwargs, progress, step, nburnin_always_accept = state
    U = ϕ° -> lnP(:mix, f°, ϕ°, θ, ds)
    ϕ°, ΔH, accept = hmc_step(U, ϕ°, mass_matrix_ϕ(θ,ds); symp_kwargs, progress, always_accept=(step<nburnin_always_accept))
    @pack! state = ϕ°, ΔH, accept
end

function hmc_step(U::Function, x, Λ, δUδx=x->gradient(U, x)[1]; symp_kwargs, progress, always_accept)
    local ΔH, accept
    for kwargs in symp_kwargs
        p = simulate(Λ)
        (ΔH, xtest) = symplectic_integrate(
            x, p, Λ, U, δUδx;
            progress = (progress==:verbose),
            kwargs...
        )
        accept = batch(@. always_accept | (log(rand()) < $unbatch(ΔH)))
        @. x = accept * xtest + (1 - accept) * x
    end
    x, ΔH, accept
end

@⌛ function mass_matrix_ϕ(θ, ds)
    @unpack G, Cϕ, Nϕ = ds(θ)
    pinv(G)^2 * (pinv(Cϕ) + pinv(Nϕ))
end

function gibbs_sample_slice_θ!(k::Symbol)
    @⌛ function gibbs_sample_slice_θ!(state, ds::DataSet)
        @unpack θ, θrange, lnPθ, pbar_dict, progress, grid_and_sample_kwargs = state
        z_other = delete(select(state, lnP_arg_names(:mix, ds)), :θ)
        θₖ, lnPθₖ = grid_and_sample(θₖ -> lnP(:mix, z_other..., @set(θ[k]=θₖ), ds), cpu(θrange[k]); progress=(progress==:verbose), grid_and_sample_kwargs...)
        @set! θ[k] = θₖ
        @set! lnPθ[k] = lnPθₖ
        pbar_dict[string(k)] = string_trunc(θₖ)
        @pack! state = θ, lnPθ
    end
end

@⌛ function gibbs_mix!(state, ds::DataSet)
    @unpack f, ϕ, θ = state
    f°, ϕ° = mix(f, ϕ, θ, ds)
    @pack! state = f°, ϕ°
end

@⌛ function gibbs_unmix!(state, ds::DataSet)
    @unpack f°, ϕ°, θ = state
    f, ϕ = unmix(f°, ϕ°, θ, ds)
    @pack! state = f, ϕ
end



## postprocessing

@⌛ function gibbs_postprocess!(state, ds::DataSet)
    @unpack f, ϕ, θ, pbar_dict, ΔH = state
    lnP = pbar_dict["lnP"] = CMBLensing.lnP(0, select(state, lnP_arg_names(0, ds))..., ds)
    pbar_dict["ΔH"] = ΔH
    f̃ = ds.L(ϕ) * f
    @pack! state = f̃, lnP
end


## util

function once_every(n)
    function (gibbs_sample!)
        function (state, ds)
            if iszero(state[:step] % n)
                gibbs_sample!(state, ds)
            end
        end
    end
end

function start_after_burnin(n)
    function (gibbs_sample!)
        function (state, ds)
            if state[:step] > n
                gibbs_sample!(state, ds)
            end
        end
    end
end

