"""
    symplectic_integrate(x₀, p₀, Λ, U, δUδx, N=50, ϵ=0.1, progress=false)
    
Do a symplectic integration of the potential energy `U` (with gradient `δUδx`)
starting from point `x₀` with momentum `p₀` and mass matrix `Λ`. The number of
steps is `N` and the step size `ϵ`. 

Returns `ΔH, xᵢ, pᵢ` corresponding to change in Hamiltonian, and final position
and momenta. If `hist` is specified a trace of requested variables throughout
each step is also returned. 

"""
function symplectic_integrate(x₀::AbstractVector{T}, p₀, Λ, U, δUδx; N=50, ϵ=T(0.1), progress=false, hist=nothing) where {T}
    
    xᵢ, pᵢ = x₀, p₀
    δUδxᵢ = δUδx(xᵢ)
    H(x,p) = U(x) - p⋅(Λ\p)/2
    
    _hist = []

    @showprogress (progress ? 1 : Inf) "Symplectic Integration: " for i=1:N
        xᵢ₊₁    = xᵢ - T(ϵ) * (Λ \ (pᵢ - T(ϵ)/2 * δUδxᵢ))
        δUδx₊₁  = δUδx(xᵢ₊₁)
        pᵢ₊₁    = pᵢ - T(ϵ)/2 * (δUδx₊₁ + δUδxᵢ)
        xᵢ, pᵢ, δUδxᵢ = xᵢ₊₁, pᵢ₊₁, δUδx₊₁
        
        if hist!=nothing
            histᵢ = (i=i, x=xᵢ, p=pᵢ, δUδx=δUδx₊₁, H=(((:H in hist) ? H(xᵢ,pᵢ) : nothing)))
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

`lnP` should accept a NamedTuple argument and `range` should be a NamedTuple mapping
those same names to `range` objects specifying where to evaluate `lnP`, e.g.:

```julia
grid_and_sample(nt->-(nt.x^2+nt.y^2)/2, (x=range(-3,3,length=100),y=range(-3,3,length=100)))
```

The return value is `(lnP, samples, Px)` where `lnP` is an interpolated/smoothed
log PDF which can be evaluated anywhere within the original range, `Px` are
sampled points of the original PDF, and `samples` is a NamedTuple giving the
Monte-Carlo samples of each of the parameters.

(Note: only 1D sampling is currently implemented, but 2D like in the example
above is planned)
"""
function grid_and_sample(lnP::Function, range::NamedTuple{S, <:NTuple{1}}; progress=false, nsamples=1, span=0.25, rtol=1e-5) where {S}
    
    xs = first(range)
    xmin,xmax = first(xs),last(xs)
    
    # probe the pdf along the grid and interpolate
    lnPs = @showprogress (progress ? 1 : Inf) "Grid Sample: " map(x->Float64(lnP(NamedTuple{S}(x))), xs)
    lnPs .-= maximum(lnPs)
    ilnP = loess(xs,lnPs,span=span)
    
    # normalize the PDF. note the smoothing is done of the log PDF.
    A = @ondemand(QuadGK.quadgk)(exp∘ilnP, xmin, xmax)[1]
    lnPs .-= log(A)
    ilnP = loess(xs,lnPs,span=span)
    
    # draw samples via inverse transform sampling
    # (the `+ eps()`` is a workaround since Loess.predict seems to NaN sometimes when
    # evaluated right at the lower bound)
    θsamples = NamedTuple{S}(@showprogress (progress ? 1 : Inf) map(1:nsamples) do i
        r = rand()
        fzero((x->@ondemand(QuadGK.quadgk)(exp∘ilnP,xmin+sqrt(eps()),x,rtol=rtol)[1]-r),xmin+sqrt(eps()),xmax,rtol=rtol)
    end)
    
    if nsamples==1
        ilnP, map(first, θsamples), lnPs
    else
        ilnP, θsamples, lnPs
    end
    
end
grid_and_sample(lnP::Function, range, progress=false) = error("Can only currently sample from 1D distributions.")
# allow more conveniently evaluation of Loess-interpolated functions
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
    ds :: DataSet{<:FlatField{T,P}};
    nsamps_per_chain,
    nchains = nworkers(),
    nchunk = 1,
    nsavemaps = 1,
    nburnin_always_accept = 0,
    nburnin_fixθ = 0,
    Nϕ = :qe,
    chains = nothing,
    ϕstart = 0,
    θrange = (),
    θstart = nothing,
    pmap = (myid() in workers() ? map : pmap),
    wf_kwargs = (tol=1e-1, nsteps=500),
    nhmc = 1,
    symp_kwargs = fill((N=25, ϵ=0.01), nhmc),
    MAP_kwargs = (αmax=0.3, nsteps=40),
    metadata = nothing,
    progress = false,
    interruptable = false,
    filename = nothing) where {T,P}
    
    # save input configuration to chain
    rundat = Base.@locals
    
    @assert length(θrange) in [0,1] "Can only currently sample one parameter at a time."
    @assert progress in [false,:summary,:verbose]

    @unpack d, Cϕ, Cn, M, B, L = ds

    if (chains == nothing)
        if (θstart == nothing)
            θstarts = [map(range->(first(range) + rand() * (last(range) - first(range))), θrange) for i=1:nchains]
        else 
            @assert θstart isa NamedTuple "θstart should be either `nothing` to randomly sample the starting value or a NamedTuple giving the starting point."
            θstarts = fill(θstart, nchains)
        end
        if (ϕstart == 0); ϕstart = zero(Cϕ().diag); end
        @assert ϕstart isa Field || ϕstart in [:quasi_sample, :best_fit]
        if ϕstart isa Field
            ϕstarts = fill(ϕstart, nchains)
        else
            ϕstarts = pmap(θstarts) do θstart
                Random.seed!()
                MAP_joint(ds(;θstart...), progress=(progress==:verbose ? :summary : false), Nϕ=Nϕ, quasi_sample=(ϕstart==:quasi_sample); MAP_kwargs...)[2]
            end
        end
        chains = pmap(θstarts,ϕstarts) do θstart,ϕstart
            Any[@dictpack i=>1 ϕ°=>ds(;θstart...).G*ϕstart θ=>θstart seed=>deepcopy(Random.seed!())]
        end
    elseif chains == :resume
        chains = @ondemand(FileIO.load)(filename,"chains")
    elseif chains isa String
        chains = @ondemand(FileIO.load)(chains,"chains")
    end
    
    if (Nϕ == :qe); Nϕ = quadratic_estimate(ds()).Nϕ/2; end
    Λm = (Nϕ == nothing) ? pinv(Cϕ) : (pinv(Cϕ) + pinv(Nϕ))
    
    swap_filename = (filename == nothing) ? nothing : joinpath(dirname(filename), ".swap.$(basename(filename))")

    # start chains
    try
        @showprogress (progress==:summary ? 1 : Inf) "Gibbs chain: " for _=1:nsamps_per_chain÷nchunk
            
            append!.(chains, pmap(last.(chains)) do state
                
                local f°, f̃, ΔH, accept
                @unpack i,ϕ°,θ,seed = state
                copy!(Random.GLOBAL_RNG, seed)
                f = nothing
                dsθ = ds(;θ...)
                ϕ = dsθ.G\ϕ°
                Lϕ = cache(L(ϕ), ds.d)
                lnPθ = nothing
                chain = []
                
                for i=(i+1):(i+nchunk)
                    
                    # ==== gibbs P(f°|ϕ°,θ) ====
                    t_f = @elapsed begin
                        f° = Lϕ * dsθ.D * argmaxf_lnP(Lϕ, dsθ; which=:sample, guess=f, progress=(progress==:verbose), wf_kwargs...)
                    end
                    
                    # ==== gibbs P(θ|f°,ϕ°) ====
                    t_θ = @elapsed begin
                        if (i > nburnin_fixθ)
                            lnPθ, θ = grid_and_sample(θ->lnP(:mix,f°,ϕ°,θ,ds,Lϕ), θrange, progress=(progress==:verbose))
                            dsθ = ds(;θ...)
                        end
                    end
                    
                    # ==== gibbs P(ϕ°|f°,θ) ==== 
                    t_ϕ = @elapsed begin
                        
                        for kwargs in symp_kwargs
                        
                            (ΔH, ϕtest°) = symplectic_integrate(
                                ϕ°, simulate(Λm), Λm, 
                                ϕ°->      lnP(:mix, f°, ϕ°, θ, dsθ, Lϕ), 
                                ϕ°->δlnP_δfϕₜ(:mix, f°, ϕ°, θ, dsθ, Lϕ)[2];
                                progress=(progress==:verbose),
                                kwargs...
                            )

                            if (i < nburnin_always_accept) || (log(rand()) < ΔH)
                                ϕ° = ϕtest°
                                accept = true
                            else
                                accept = false
                            end
                            
                        end
                        
                        # compute un-mixed maps
                        ϕ = dsθ.G\ϕ°
                        cache!(Lϕ,ϕ)
                        f = dsθ.D\(Lϕ\f°)
                        f̃ = Lϕ*f
                        
                    end
                    
                    # save quantities to chain and print progress
                    timing = (f=t_f, θ=t_θ, ϕ=t_ϕ)
                    push!(chain, @dictpack i f f° f̃ ϕ ϕ° θ lnPθ ΔH accept lnP=>lnP(0,f,ϕ,θ,dsθ) seed=>deepcopy(Random.GLOBAL_RNG) timing)
                    if (progress==:verbose)
                        @show i, accept, ΔH, θ
                    end
                    
                end

                chain
                
            end)
            
            # only keep maps every `nsavemaps` samples, as well as the last
            # sample (so we can continue the chain)
            for chain in chains
                for sample in chain[1:end-1]
                    if mod(sample[:i]-1,nsavemaps)!=0
                        filter!(kv->!(kv.second isa Field), sample)
                    end
                end
            end
            
            if filename != nothing
                @ondemand(FileIO.save)(swap_filename, "chains", chains, "rundat", rundat)
                mv(swap_filename, filename, force=true)
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
    
    @namedtuple(rundat, chains)
    
end
