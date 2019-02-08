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
    function grid_and_sample(lnP::Function; range=(1e-3,0.2), ngrid=20, s=0)

Interpolate the log pdf `lnP` with support on `range`, and return 
the integrated pdf as well a sample (drawn via inverse transform sampling)
"""
function grid_and_sample(lnP::Function, range::NamedTuple{S, <:NTuple{1}}; progress=false, nsamples=1) where {S}
    xs = first(range)
    xmin,xmax = first(xs),last(xs)
    lnPs = @showprogress (progress ? 1 : Inf) "Grid Sample: " map(x->lnP(;Dict(first(keys(range))=>x)...), xs)
    Ps = exp.(lnPs .- maximum(lnPs))
    
    iP = CubicSplineInterpolation(xs,Ps,extrapolation_bc=0)
    
    A = quadgk(iP,xmin,xmax)[1]
    
    θsamples = NamedTuple{S}(((@showprogress (progress ? 1 : Inf) [(r=rand(); fzero((x->quadgk(iP,xmin,x)[1]/A-r),xmin,xmax)) for i=1:nsamples]),))
    
    if nsamples==1
        iP, map(first, θsamples)
    else
        iP, θsamples
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
    progress = 1,
    filename = nothing) where {T,P}
    
    @assert length(θrange) == 1 "Can only currently sample one parameter at a time."

    @unpack d, Cϕ, Cn, M, B = ds

    if (chains==nothing)
        @assert ϕstart in [0, :quasi_sample, :best_fit]
        if ϕstart==0
            chains = [Any[@dictpack ϕcur=>zero(Cϕ) θcur=>θstart] for i=1:nchains]
        elseif ϕstart in [:quasi_sample, :best_fit]
            chains = pmap(1:nchains) do i
                fcur, ϕcur = MAP_joint(ds(;θstart...), progress=(progress==2), Nϕ=Nϕ, quasi_sample=(ϕstart==:quasi_sample); MAP_kwargs...)
                Any[@dictpack ϕcur fcur θcur=>θstart]
            end
        end
    elseif chains isa String
        chains = load(filename,"chains")
    end
    
    if (Nϕ == :qe); Nϕ = ϕqe(ds(;θstart...))[2]; end
    Λm = nan2zero.((Nϕ == nothing) ? Cϕ^-1 : (Cϕ^-1 + Nϕ^-1))
    
    swap_filename = (filename == nothing) ? nothing : joinpath(dirname(filename), ".swap.$(basename(filename))")

    # start chains
    dt = (progress==1 ? 1 : Inf)
    
    try
        @showprogress dt "Gibbs chain: " for i=1:nsamps_per_chain÷nchunk
            
            append!.(chains, map(last.(chains)) do state
                
                local fcur, f̊cur, f̃cur, Pθ
                @unpack ϕcur,θcur = state
                chain = []
                
                for i=1:nchunk
                        
                    # ==== gibbs P(f|ϕ,θ) ====
                    let L=cache(L(ϕcur),ds.d)
                        let ds=ds(;θcur...)
                            fcur = lensing_wiener_filter(ds, L, :sample; progress=(progress==2), wf_kwargs...)
                            f̃cur = L*fcur
                            f̊cur = L*ds.D*fcur
                        end
                            
                    # ==== gibbs P(θ|f,ϕ) ====
                        Pθ, θcur = grid_and_sample((;θ...)->lnP(:mix,f̊cur,ϕcur,ds,L; θ...), θrange, progress=(progress==2))
                    end
                        
                    # ==== gibbs P(ϕ|f,θ) ==== 
                    let ds=ds(;θcur...)
                            
                        (ΔH, ϕtest) = symplectic_integrate(
                            ϕcur, simulate(Λm), Λm, 
                            ϕ->      lnP(:mix, f̊cur, ϕ, ds, L), 
                            ϕ->δlnP_δfϕₜ(:mix, f̊cur, ϕ, ds, L)[2];
                            progress=(progress==2),
                            symp_kwargs...
                        )

                        if log(rand()) < ΔH
                            ϕcur = ϕtest
                            accept = true
                        else
                            accept = false
                        end
                        
                        if (progress==2)
                            @show accept, ΔH, θcur
                        end

                        push!(chain, @dictpack fcur f̊cur f̃cur ϕcur θcur ΔH accept lnP=>lnP(0,fcur,ϕcur,ds(;θcur...)) Pθ)
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
