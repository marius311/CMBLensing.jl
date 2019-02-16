"""
    symplectic_integrate(x₀, p₀, Λ, U, δUδx, N=50, ϵ=0.1, progress=false)
    
Do a symplectic integration of the potential energy `U` (with gradient `δUδx`)
starting from point `x₀` with momentum `p₀` and mass matrix `Λ`. The number of
steps is `N` and the step size `ϵ`. 

Returns `ΔH, xᵢ, pᵢ` corresponding to change in Hamiltonian, and final position
and momenta. If `hist` is specified a trace of requested variables throughout
each step is also returned. 

"""
function symplectic_integrate(x₀, p₀, Λ, U, δUδx; N=50, ϵ=0.1, progress=false, hist=nothing)
    
    xᵢ, pᵢ = x₀, p₀
    δUδxᵢ = δUδx(xᵢ)
    H(x,p) = U(x) - p⋅(Λ\p)/2
    
    _hist = []

    @showprogress (progress ? 1 : Inf) "Symplectic Integration: " for i=1:N
        xᵢ₊₁    = xᵢ - ϵ * (Λ \ (pᵢ - ϵ/2 * δUδxᵢ))
        δUδx₊₁  = δUδx(xᵢ₊₁)
        pᵢ₊₁    = pᵢ - ϵ/2 * (δUδx₊₁ + δUδxᵢ)
        xᵢ, pᵢ, δUδxᵢ = xᵢ₊₁, pᵢ₊₁, δUδx₊₁
        
        if hist!=nothing
            histᵢ = (i=i, x=xᵢ, p=pᵢ, δUδx=δUδx₊₁, H=(((hist != nothing) && (:H in hist)) ? H(xᵢ,pᵢ) : nothing))
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



"""
    function grid_and_sample(lnP::Function; range::NamedTuple; progress=false, nsamples=1)

Interpolate the log pdf `lnP` with support on `range`, and return  the
integrated pdf as well `nsamples` samples (drawn via inverse transform
sampling)

`lnP` should accept keyword arguments and `range` should be a NamedTuple mapping
those same names to `range` objects specifying where to evaluate `lnP`, e.g.:

```
    grid_and_sample((;x,y)->-(x^2+y^2)/2, (x=range(-3,3,length=100),y=range(-3,3,length=100)))
```

The return value is `(P, samples, Px)` where `P` is an interpolated/smoothed PDF
which can be evaluated anywhere, `Px` are sampled points of the original PDF,
and `samples` is a NamedTuple giving the Monte-Carlo samples of each of the
parameters.

(Note: only 1D sampling is currently implemented, but 2D like in the example
above is planned)
"""
function grid_and_sample(lnP::Function, range::NamedTuple{S, <:NTuple{1}}; progress=false, nsamples=1) where {S}
    
    xs = first(range)
    xmin,xmax = first(xs),last(xs)
    
    # sample the pdf
    lnPs = @showprogress (progress ? 1 : Inf) "Grid Sample: " map(x->lnP(;Dict(first(keys(range))=>x)...), xs)
    lnPs .-= maximum(lnPs)
    ilnP = loess(xs,lnPs)
    
    # do the smoothing in lnP (and normalize with A in a type-stable way hence the let-block)
    # also return the sampled P(x) so we can check the smoothing
    (iP, Px) = let A = quadgk(x->exp(Loess.predict(ilnP,x)),xmin,xmax)[1]
        (x->exp(Loess.predict(ilnP,x))/A,
         @. exp(lnPs)/A)
    end
    
    # draw samples via inverse transform sampling
    # (the `+ eps()`` is a workaround since Loess.predict seems to NaN sometimes when
    # evaluated right at the lower bound)
    θsamples = NamedTuple{S}(((@showprogress (progress ? 1 : Inf) [(r=rand(); fzero((x->quadgk(iP,xmin+eps(),x)[1]-r),xmin+eps(),xmax)) for i=1:nsamples]),))
    
    if nsamples==1
        iP, map(first, θsamples), Px
    else
        iP, θsamples, Px
    end
    
end
grid_and_sample(lnP::Function, range, progress=false) = error("Can only currently sample from 1D distributions.")


"""
    sample_joint(ds::DataSet; kwargs...)
    
Sample from the joint PDF of P(f,ϕ,r). Runs `nworkers()` chains in parallel
using `pmap`. 
    
Possible keyword arguments: 

* `nsamps_per_chain` - the number of samples per chain
* `nchunk` - do `nchunk` steps in between parallel chain communication
* `nthin` - only save once every `nthin` steps
* `chains` - resume these existing chain (starts a new one if nothing)
* `ϕstart`/`θstart` - starting values of ϕ and r 

"""
function sample_joint(
    ds :: DataSet{<:FlatField{T,P}};
    L = LenseFlow,
    Cℓ,
    nsamps_per_chain,
    Nϕ = :qe,
    nchains = nworkers(),
    nchunk = 1,
    nthin = 1,
    chains = nothing,
    ϕstart = 0,
    θrange = (),
    θstart = (),
    wf_kwargs = (tol=1e-1, nsteps=500),
    symp_kwargs = (N=100, ϵ=0.01),
    MAP_kwargs = (αmax=0.3, nsteps=40),
    progress = false,
    filename = nothing) where {T,P}
    
    @assert length(θrange) == 1 "Can only currently sample one parameter at a time."
    @assert progress in [false,:summary,:verbose]

    @unpack d, Cϕ, Cn, M, B = ds

    if (chains==nothing)
        if (ϕstart==0); ϕstart=zero(Cϕ); end
        @assert ϕstart isa Field || ϕstart in [:quasi_sample, :best_fit]
        if ϕstart isa Field
            chains = [Any[@dictpack ϕ°=>ds(;θstart...).G*ϕstart θ=>θstart] for i=1:nchains]
        elseif ϕstart in [:quasi_sample, :best_fit]
            chains = pmap(1:nchains) do i
                f, ϕ = MAP_joint(ds(;θstart...), progress=progress, Nϕ=Nϕ, quasi_sample=(ϕstart==:quasi_sample); MAP_kwargs...)
                Any[@dictpack ϕ°=>ds(;θstart...).G*ϕ θ=>θstart]
            end
        end
    elseif chains isa String
        chains = load(chains,"chains")
    end
    
    if (Nϕ == :qe); Nϕ = ϕqe(ds(;θstart...))[2]/2; end
    Λm = nan2zero.((Nϕ == nothing) ? Cϕ^-1 : (Cϕ^-1 + Nϕ^-1))
    
    swap_filename = (filename == nothing) ? nothing : joinpath(dirname(filename), ".swap.$(basename(filename))")

    # start chains
    try
        @showprogress (progress==:summary ? 1 : Inf) "Gibbs chain: " for i=1:nsamps_per_chain÷nchunk
            
            append!.(chains, pmap(last.(chains)) do state
                
                local f°, f̃, Pθ
                @unpack ϕ°,θ = state
                f = nothing
                ϕ = ds(;θ...).G\ϕ°
                chain = []
                
                for i=1:nchunk
                        
                    # ==== gibbs P(f°|ϕ°,θ) ====
                    let L=cache(L(ϕ),ds.d)
                        let ds=ds(;θ...)
                            f  = lensing_wiener_filter(ds, L, :sample; guess=f, progress=(progress==:verbose), wf_kwargs...)
                            f̃  = L*f
                            f° = L*ds.D*f
                        end
                    end
                    
                    # ==== gibbs P(θ|f°,ϕ°) ====
                    # todo: if not sampling Aϕ, could cache L(ϕ) here...
                    Pθ, θ = grid_and_sample((;θ...)->lnP(:mix,f°,ϕ°,ds,L; θ...), θrange, progress=(progress==:verbose))
                        
                    # ==== gibbs P(ϕ°|f°,θ) ==== 
                    let ds=ds(;θ...)
                            
                        (ΔH, ϕtest°) = symplectic_integrate(
                            ϕ°, simulate(Λm), Λm, 
                            ϕ°->      lnP(:mix, f°, ϕ°, ds, L), 
                            ϕ°->δlnP_δfϕₜ(:mix, f°, ϕ°, ds, L)[2];
                            progress=(progress==:verbose),
                            symp_kwargs...
                        )

                        if log(rand()) < ΔH
                            ϕ° = ϕtest°
                            ϕ = ds.G\ϕ°
                            accept = true
                        else
                            accept = false
                        end
                        
                        if (progress==:verbose)
                            @show accept, ΔH, θ
                        end

                        push!(chain, @dictpack f f° f̃ ϕ ϕ° θ ΔH accept lnP=>lnP(0,f,ϕ,ds(;θ...)) Pθ)
                    end

                end
                    
                chain[1:nthin:end]
            
            end)
            
            if filename != nothing
                save(swap_filename, "chains", chains)
                mv(swap_filename, filename, force=true)
            end
            
        end
    catch err
        if err isa InterruptException
            println()
            @warn("Chain interrupted. Returning current progress.")
        else
            rethrow(err)
        end
    end
    
    chains
    
end
