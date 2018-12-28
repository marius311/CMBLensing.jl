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
    function grid_and_sample_1D(lnP::Function; range=(1e-3,0.2), ngrid=20, s=0)

Interpolate the log pdf `lnP` with support on `range`, and return 
the integrated pdf as well a sample (drawn via inverse transform sampling)
"""
function grid_and_sample_1D(lnP::Function; range=(1e-3,0.2), ngrid=20, s=0, progress=false)
    
    xmin,xmax = range
    xs = Base.range(xmin, stop=xmax, length=ngrid)
    lnPs = @showprogress (progress ? 1 : Inf) "Grid Sample: " map(lnP, xs)
    Ps = exp.(lnPs .- maximum(lnPs))
    
    iP = CubicSplineInterpolation(xs,Ps,extrapolation_bc=0)
    
    A = quadgk(iP,range...)[1]
    
    r = rand()
    iP, fzero((x->quadgk(iP,xmin,x)[1]/A-r),range...)

end

"""
    sample_joint(ds::DataSet; kwargs...)
    
Sample from the joint PDF of P(f,ϕ,r). Runs `nworkers()` chains in parallel
using `pmap`. 
    
Possible keyword arguments: 

* `nsamps_per_chain` - the number of samples per chain
* `nchunk` - do `nchunk` steps in between parallel chain communication
* `nthin` - only save once every `nthin` steps
* `chains` - resume these existing chain (starts a new one if nothing)
* `ϕstart`/`rstart` - starting values of ϕ and r 

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
    ϕstart = :zero,
    rstart = 0.1,
    r_grid_range = nothing,
    r_grid_kwargs = (range=r_grid_range, ngrid=32),
    wf_kwargs = (tol=1e-1, nsteps=500),
    symp_kwargs = (N=100, ϵ=0.01),
    MAP_kwargs = (αmax=0.3, nsteps=40),
    progress = 1,
    filename = nothing) where {T,P}

    @unpack d, Cϕ, Cn, M, B = ds

    if (chains==nothing)
        @assert ϕstart in [:zero, :quasi_sample, :best_fit]
        if ϕstart==:zero
            chains = [Any[@dictpack ϕcur=>zero(simulate(Cϕ)) rcur=>rstart] for i=1:nchains]
        elseif ϕstart in [:quasi_sample, :best_fit]
            chains = pmap(1:nchains) do i
                fcur, ϕcur = MAP_joint(ds(r=rstart), progress=(progress==2), Nϕ=Nϕ, quasi_sample=(ϕstart==:quasi_sample); MAP_kwargs...)
                Any[@dictpack ϕcur fcur rcur=>rstart]
            end
        end
    elseif chains isa String
        chains = load(filename,"chains")
    end
    
    if (Nϕ == :qe); Nϕ = ϕqe(ds(r=rstart))[2]; end
    Λm = nan2zero.((Nϕ == nothing) ? Cϕ^-1 : (Cϕ^-1 + Nϕ^-1))
    
    swap_filename = (filename == nothing) ? nothing : joinpath(dirname(filename), ".swap.$(basename(filename))")

    # start chains
    dt = (progress==1 ? 1 : Inf)
    
    try
        @showprogress dt "Gibbs chain: " for i=1:nsamps_per_chain÷nchunk
            
            append!.(chains, pmap(last.(chains)) do state
                
                local fcur, f̊cur, f̃cur, Pr
                @unpack ϕcur,rcur = state
                chain = []
                
                for i=1:nchunk
                        
                    # ==== gibbs P(f|ϕ,r) ====
                    let L=L(ϕcur), ds=ds(r=rcur)
                        fcur = lensing_wiener_filter(ds, L, :sample; progress=(progress==2), wf_kwargs...)
                        f̃cur = L*fcur
                        f̊cur = L*ds.D*fcur
                    end
                            
                    # ==== gibbs P(r|f,ϕ) ====
                    let L=L(ϕcur)
                        Pr, rcur = grid_and_sample_1D(r->lnP(:mix,f̊cur,ϕcur,ds(r=r),L); progress=(progress==2), r_grid_kwargs...)
                    end
                        
                    # ==== gibbs P(ϕ|f,r) ==== 
                    let ds=ds(r=rcur)
                            
                        (ΔH, ϕtest) = symplectic_integrate(
                            ϕcur, simulate(Λm), Λm, 
                            ϕ->      lnP(:mix, f̊cur, ϕ, ds), 
                            ϕ->δlnP_δfϕₜ(:mix, f̊cur, ϕ, ds)[2];
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
                            @show accept, ΔH, rcur
                        end

                        push!(chain, @dictpack fcur f̊cur f̃cur ϕcur rcur ΔH accept lnP=>lnP(0,fcur,ϕcur,ds(r=rcur)) Pr)
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
