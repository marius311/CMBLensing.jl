using JLD
using CMBLensing
using CMBLensing: @dictpack, jrk4, δlnΠᶠ_δfϕ, cache, LP, D_mix
using Optim: optimize
using Parameters
using PyCall
using Base.Iterators: repeated


"""

The joint maximization algorithm via coordinate descent in the mixed
parametrization. 

"""
function run3(;
    Θpix = 3,
    nside = 64,
    T = Float32,
    L = LenseFlow{jrk4{7}},
    outfile = nothing,
    seed = nothing,
    mask = nothing,
    θ = Dict(:r=>0.05),
    θ_data = Dict(:r=>0.05),
    Cℓ = camb(lmax=6900; θ...),
    Cℓ_data = (θ==θ_data ? Cℓ : camb(lmax=6900; θ_data...)),
    use = :TEB,
    ℓmax_data = 3000,
    μKarcminT = 1,
    beamFWHM = 3,
    ℓknee = 100,
    Ncg = 1000,
    cgtol = 1e-3,
    progress = 10,
    αtol = 1e-6,
    αmax = 0.3,
    nsteps = 10,
    resume = nothing,
    quiet = false
    )
    
    # Cℓs
    Cℓf, Cℓf̃, Cℓf_data = Cℓ[:f], Cℓ[:f̃], Cℓ_data[:f]
    Cℓn = noisecls(μKarcminT, beamFWHM=0, ℓknee=ℓknee)
    
    # types which depend on whether T/E/B
    SS,ks = Dict(:TEB=>((S0,S2),(:TT,:EE,:BB,:TE)), :EB=>((S2,),(:EE,:BB)), :T=>((S0,),(:TT,)))[use]
    F,F̂,nF = Dict(:TEB=>(FlatIQUMap,FlatTEBFourier,3), :EB=>(FlatS2QUMap,FlatS2EBFourier,2), :T=>(FlatS0Map,FlatS0Fourier,1))[use]
    
    # covariances
    P = Flat{Θpix,nside}
    Cϕ = Cℓ_to_cov(T,P,S0, Cℓf[:ℓ], Cℓf[:ϕϕ])
    Cf,Cf̃,Cn,Cf_data = (Cℓ_to_cov(T,P,SS..., Cℓx[:ℓ], (Cℓx[k] for k=ks)...) for Cℓx in (Cℓf,Cℓf̃,Cℓn,Cℓf_data))
    
    # data mask
    if mask!=nothing
        M = FullDiagOp(F{T,P}(repeated(T.(sptlike_mask(nside,Θpix; mask...)),nF)...)) * LP(ℓmax_data)
    else
        M = LP(ℓmax_data)
    end
    
    # beam
    B = let ℓ=0:10000; Cℓ_to_cov(T,P,SS..., ℓ, ((k==:TE ? 0.*ℓ : @.(exp(-ℓ^2*deg2rad(beamFWHM/60)^2/(8*log(2))/2))) for k=ks)...); end;
    
    # simulate data
    seed!=nothing && srand(seed)
    f = simulate(Cf_data)
    ϕ = simulate(Cϕ)
    f̃ = L(ϕ)*f
    d = M*B*f̃ + simulate(Cn)
    
    # put everything in DataSet
    D = D_mix(Cf)
    ds = DataSet(;(@dictpack d Cn Cf Cf̃ Cϕ M B D)...)

    target_lnP = -nF * nside^2 ÷ 2
    !quiet && @show target_lnP
    rundat = @dictpack Θpix nside T θ θ_data μKarcminT ℓknee beamFWHM target_lnP Cℓ Cℓ_data Cℓn f f̃ ϕ ds

    local hist, fcur, f̆cur
    if resume==nothing
        trace = []
        ϕcur = 0ϕ
        i₀ = 0
    else
        for (k,v) in resume["rundat"]
            if isa(v,Number) && !(k in [:target_lnP]) && v!=rundat[k]
                warn("Parameter '$k' differs between current run ($(rundat[k])) and the one being resumed ($v)")
            end
        end
        trace = resume["trace"]
        @unpack ϕcur, fcur, f̆cur = trace[end]
        i₀ = length(trace)
    end
    
    outfile!=nothing && mkpath(dirname(outfile))

    # Start the run...
    for i=1:nsteps
        
        # f step
        let L = (i==1 ? IdentityOp : cache(L(ϕcur)))
            fcur,hist = lensing_wiener_filter(ds, L, guess=(i==1 ? nothing : fcur), tol=cgtol, nsteps=Ncg, hist=(:i,:res), progress=true)
            f̆cur = L * D * fcur
        end

        # ϕ step
        if i!=nsteps
            ϕnew = Cϕ*(δlnP_δfϕₜ(:mix,f̆cur,ϕcur,ds,L))[2]
            res = optimize(α->(-lnP(:mix,f̆cur,(1-α)*ϕcur+α*ϕnew,ds,L)), T(0), T(αmax), abs_tol=αtol)
            α = res.minimizer
            ϕcur = (1-α)*ϕcur+α*ϕnew
            lnPcur = -res.minimum
            push!(trace, @dictpack f̆cur fcur ϕcur ϕnew lnPcur α hist)
            !quiet && @printf("%i %.4f %i %.4f\n",i,lnPcur,length(hist),α)
        else
            lnPcur = lnP(0,fcur,ϕcur,ds,L)
            push!(trace, @dictpack f̆cur fcur ϕcur lnPcur hist)
            !quiet && @printf("%i %.4f %i\n",i,lnPcur,length(hist))
        end
        
        outfile!=nothing && save(outfile,"rundat",rundat,"trace",trace)
            
    end
    
    !quiet && @printf("%.1fσ from expected\n",(target_lnP - trace[end][:lnPcur])/sqrt(-target_lnP))
    
    f̆cur, fcur, ϕcur, trace, rundat
    
end
