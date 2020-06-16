"""
    symplectic_integrate(x₀, p₀, Λ, U, δUδx, N=50, ϵ=0.1, progress=false)
    
Do a symplectic integration of the potential energy `U` (with gradient `δUδx`)
starting from point `x₀` with momentum `p₀` and mass matrix `Λ`. The number of
steps is `N` and the step size `ϵ`. 

Returns `ΔH, xᵢ, pᵢ` corresponding to change in Hamiltonian, and final position
and momenta. If `hist` is specified a trace of requested variables throughout
each step is also returned. 

"""
function symplectic_integrate(
    x₀::AbstractVector{T}, p₀, Λ, U, δUδx=x->gradient(U,x)[1]; 
    N=50, ϵ=T(0.1), progress=false, hist=nothing) where {T}
    
    xᵢ, pᵢ = x₀, p₀
    δUδxᵢ = δUδx(xᵢ)
    H(x,p) = U(x) - p⋅(Λ\p)/2
    
    _hist = []

    @showprogress (progress ? 1 : Inf) "Symplectic Integration: " for i=1:N
        xᵢ₊₁    = xᵢ - T(ϵ) * (Λ \ (pᵢ - T(ϵ)/2 * δUδxᵢ))
        δUδxᵢ₊₁ = δUδx(xᵢ₊₁)
        pᵢ₊₁    = pᵢ - T(ϵ)/2 * (δUδxᵢ₊₁ + δUδxᵢ)
        xᵢ, pᵢ, δUδxᵢ = xᵢ₊₁, pᵢ₊₁, δUδxᵢ₊₁
        
        if hist!=nothing
            histᵢ = (i=i, x=xᵢ, p=pᵢ, δUδx=δUδxᵢ₊₁, H=(((:H in hist) ? H(xᵢ,pᵢ) : nothing)))
            push!(_hist, getfield.(Ref(histᵢ),hist))
        end

    end

    ΔH = H(xᵢ,pᵢ) - H(x₀,p₀)
    
    if hist == nothing
        return ΔH, xᵢ, pᵢ
    else
        return ΔH, xᵢ, pᵢ, _hist
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
    batches = [grid_and_sample(batchindex.(lnPs,i), xs; kwargs...) for i=1:batchsize(lnPs[1])]
    ((batch(getindex.(batches,i)) for i=1:3)...,)
end

function grid_and_sample(lnPs::Vector, xs::AbstractVector; progress=false, nsamples=1, span=0.25, rtol=1e-5)
    
    xmin, xmax = first(xs), last(xs)
    lnPs = lnPs .- maximum(lnPs)
    ilnP = loess(xs, lnPs, span=span)
    
    # normalize the PDF. note the smoothing is done of the log PDF.
    A = @ondemand(QuadGK.quadgk)(exp∘ilnP, xmin, xmax)[1]
    lnPs .-= log(A)
    ilnP = loess(xs, lnPs, span=span)
    
    # draw samples via inverse transform sampling
    # (the `+ eps()`` is a workaround since Loess.predict seems to NaN sometimes when
    # evaluated right at the lower bound)
    θsamples = @showprogress (progress ? 1 : Inf) map(1:nsamples) do i
        r = rand()
        fzero((x->@ondemand(QuadGK.quadgk)(exp∘ilnP,xmin+sqrt(eps()),x,rtol=rtol)[1]-r),xmin+sqrt(eps()),xmax,rtol=rtol)
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

* `nsamps_per_chain` — *(required)* The number of samples per chain
* `nchains` — Run `nchains` chains in parallel *(default: 1)*
* `nchunk` — Do `nchunk` steps between parallel chain communication *(default: 1)*
* `nsavemaps` — Save maps into chain every `nsavemaps` steps *(default: 1)*
* `nburnin_always_accept` — The first `nburnin_always_accept` steps, always accept
                            HMC steps independent of integration error *(default: 0)*
* `nburnin_fixθ` — For the first `nburnin_fixθ` steps, fix θ at its starting point *(default: 0)*
* `Nϕ` — Noise to use in the HMC mass matrix. can also give `Nϕ=:qe` to use the 
         EB quadratic estimate noise *(default: `:qe`)*
* `chains` — `nothing` to start a new chain; the return value from a previous call to
             `sample_joint` to resume those chains; `:resume` to resume chains
             from a file given by `filename`
* `θrange` — Range and density to grid sample parameters as a NamedTuple, 
             e.g. `(Aϕ=range(0.7,1.3,length=20),)`. 
* `θstart` — Starting values of parameters as a NamedTuple, e.g. `(Aϕ=1.2,)`, 
             or nothing to randomly sample from θrange
* `ϕstart` — Starting ϕ, either a `Field` object, `:quasi_sample`, or `:best_fit`
* `metadata` — Does nothing, but is saved into the chain file
* `nhmc` — The number of HMC passes per ϕ Gibbs step *(default: 1)*
* `symp_kwargs` — an array of NamedTupe kwargs to pass to [`symplectic_integrate`](@ref). 
                  E.g. `[(N=50,ϵ=0.1),(N=25,ϵ=0.01)]` would do 50 large steps then 25 
                  smaller steps per each Gibbs pass. If specified, `nhmc` is ignored.
* `wf_kwargs` — Keyword arguments to pass to [`argmaxf_lnP`](@ref) in the Wiener Filter Gibbs step.
* `MAP_kwargs` — Keyword arguments to pass to [`MAP_joint`](@ref) when computing the starting point.
"""
function sample_joint(
    ds :: DataSet;
    nsamps_per_chain,
    nchains = nworkers(),
    nchunk = 1,
    nsavemaps = 1,
    nburnin_always_accept = 0,
    nburnin_fixθ = 0,
    Nϕ = :qe,
    filename = nothing,
    ϕstart = :prior,
    θstart = :prior,
    θrange = NamedTuple(),
    Nϕ_fac = 2,
    pmap = (myid() in workers() ? map : pmap),
    conjgrad_kwargs = (tol=1e-1, nsteps=500),
    preconditioner = :diag,
    nhmc = 1,
    symp_kwargs = fill((N=25, ϵ=0.01), nhmc),
    MAP_kwargs = (αmax=0.3, nsteps=40),
    metadata = nothing,
    progress = false,
    interruptable = false,
    gibbs_pass_θ::Union{Function,Nothing} = nothing,
    storage = basetype(fieldinfo(ds.d).M)
    )
    
    ds = cpu(ds)
    
    # save input configuration to later write to chain file
    rundat = Base.@locals
    pop!.(Ref(rundat), (:metadata, :ds)) # saved separately
    
    # validate arguments
    if (length(θrange)>1 && gibbs_pass_θ==nothing)
        error("Can only currently sample one parameter at a time, otherwise must pass custom `gibbs_pass_θ`")
    end
    if !(progress in [false,:summary,:verbose])
        error("`progress` should be one of [false,:summary,:verbose]")
    end
    if (filename!=nothing && splitext(filename)[2]!=".jld2")
        error("Chain filename '$filename' should have '.jld2' extension.")
    end
    if mod(nchunk,nsavemaps) != 0
        error("`nsavemaps` should divide evenly into `nchunk`")
    end
    
    # seed
    @everywhere seed_for_storage!((Array,$storage))

    # initialize chains
    if (filename != nothing) && isfile(filename)
        @info "Resuming chain at $filename"
        local chunks_index, last_chunks
        jldopen(filename,"r") do io
            chunks_index = maximum([parse(Int,k[8:end]) for k in keys(io) if startswith(k,"chunks_")])
            last_chunks = read(io, "chunks_$(chunks_index)")
        end
    else

        D = batchsize(ds.d)
        
        θstarts = if θstart == :prior
            [map(range->batch((first(range) .+ rand(D) .* (last(range) - first(range)))...), θrange) for i=1:nchains]
        elseif (θstart isa NamedTuple)
            fill(θstart, nchains)
        else
            error("`θstart` should be either `nothing` to randomly sample the starting value or a NamedTuple giving the starting point.")
        end
        
        ϕstarts = if ϕstart == :prior
            pmap(θstarts) do θstart
                simulate(batch(ds(;θstart...).Cϕ, D))
            end
        elseif ϕstart == 0 
            fill(batch(zero(diag(ds().Cϕ)), D), nchains)
        elseif ϕstart isa Field
            fill(ϕstart, nchains)
        elseif ϕstart in [:quasi_sample, :best_fit]
            pmap(θstarts) do θstart
                MAP_joint(adapt(storage,ds(;θstart...)), progress=(progress==:verbose ? :summary : false), Nϕ=adapt(storage,Nϕ), quasi_sample=(ϕstart==:quasi_sample); MAP_kwargs...).ϕ
            end
        else
            error("`ϕstart` should be 0, :quasi_sample, :best_fit, or a Field.")
        end
        
        last_chunks = pmap(θstarts,ϕstarts) do θstart,ϕstart
            [@dict i=>1 f=>nothing ϕ°=>cpu(ds(;θstart...).G*ϕstart) θ=>θstart]
        end
        chunks_index = 1
        if filename != nothing
            save(
                filename, 
                "rundat",   cpu(rundat),
                "ds",       cpu(ds),
                "ds₀",      cpu(ds()), # save separately incase θ-dependent has trouble loading
                "metadata", cpu(metadata),
                "chunks_1", cpu(last_chunks)
            )
        end
    end
    
    
    @unpack L, Cϕ = ds
    if (Nϕ == :qe)
        Nϕ = quadratic_estimate(ds()).Nϕ / Nϕ_fac
    end
    dsₐ,Nϕₐ = ds,Nϕ
    t_write = 0

    # start chains
    try
        
        if progress==:summary
            @spawnat first(workers()) global pbar = Progress(nsamps_per_chain, 0, "Gibbs chain: ")
        end

        for chunks_index = (chunks_index+1):(chunks_index+nsamps_per_chain÷nchunk)
            
            last_chunks = pmap(last.(last_chunks)) do state
                
                @unpack i,ϕ°,f,θ = state
                f,ϕ°,ds,Nϕ = (adapt(storage, x) for x in (f,ϕ°,dsₐ,Nϕₐ))
                dsθ = ds(θ)
                ϕ = dsθ.G\ϕ°
                pϕ°, ΔH, accept = nothing, nothing, nothing
                L = ds.L
                lnPθ = nothing
                chain_chunk = []
                
                for (i, savemaps) in zip( (i+1):(i+nchunk), cycle([fill(false,nsavemaps-1); true]) )
                    
                    # ==== gibbs P(f°|ϕ°,θ) ====
                    t_f = @elapsed begin
                        f = argmaxf_lnP(
                            ϕ, θ, dsθ;
                            which=:sample, 
                            guess=f, 
                            preconditioner=preconditioner, 
                            conjgrad_kwargs=(progress=(progress==:verbose), conjgrad_kwargs...)
                        )
                        f°, = mix(f,ϕ,dsθ)
                    end
                    
                    
                    # ==== gibbs P(ϕ°|f°,θ) ==== 
                    t_ϕ = @elapsed begin
                        
                        Λm = (Nϕ == nothing) ? pinv(dsθ.Cϕ) : (pinv(dsθ.Cϕ) + pinv(Nϕ))
                        
                        for kwargs in symp_kwargs
                        
                            pϕ° = simulate(Λm)
                            (ΔH, ϕtest°) = symplectic_integrate(
                                ϕ°, pϕ°, Λm, 
                                ϕ°->lnP(:mix, f°, ϕ°, θ, dsθ);
                                progress=(progress==:verbose),
                                kwargs...
                            )
                            
                            accept = batch(@. (i < nburnin_always_accept) | (log(rand()) < $unbatch(ΔH)))
                            ϕ° = @. accept * ϕtest° + (1 - accept) * ϕ°
                            
                        end
                        
                    end
                    
                    
                    # ==== gibbs P(θ|f°,ϕ°) ====
                    t_θ = @elapsed begin
                        if (i > nburnin_fixθ && length(θrange)>0)
                            if gibbs_pass_θ == nothing
                                θ, lnPθ = grid_and_sample(θ->lnP(:mix,f°,ϕ°,θ,ds), θrange, progress=(progress==:verbose))
                            else
                                θ, lnPθ = gibbs_pass_θ(;(Base.@locals)...)
                            end
                            dsθ = ds(θ)
                        end
                    end

                    
                    # compute un-mixed maps
                    f, ϕ = unmix(f°,ϕ°,θ,dsθ)
                    f̃ = L(ϕ)*f

                    
                    # save state to chain and print progress
                    timing = (f=t_f, θ=t_θ, ϕ=t_ϕ)
                    state = @dict i θ lnPθ ΔH accept lnP=>lnP(0,f,ϕ,θ,dsθ) timing
                    if savemaps
                        merge!(state, @dict f f° f̃ ϕ ϕ° pϕ°)
                    end
                    push!(chain_chunk, cpu(state))
                    
                    if @isdefined pbar
                        string_trunc(x) = Base._truncate_at_width_or_chars(string(x), displaysize(stdout)[2]-14)
                        next!(pbar, showvalues = [
                            ("step",i), 
                            tuple.(keys(θ), string_trunc.(values(θ)))..., 
                            ("ΔH", string_trunc(ΔH)), 
                            ("accept", string_trunc(accept)), 
                            ("timing", timing)
                        ])
                    end

                    
                end

                return chain_chunk
                
            end
            
            if filename != nothing
                last_chunks[1][end][:t_write] = t_write
                t_write = @elapsed jldopen(filename,"a+") do io
                    wsession = JLDWriteSession()
                    write(io, "chunks_$chunks_index", last_chunks, wsession)
                end
            end
            
        end
        
    catch err
        if interruptable && (err isa InterruptException)
            println()
            @warn("Chain interrupted. Returning current progress.")
        else
            rethrow(err)
        end
    end
    
    
end
