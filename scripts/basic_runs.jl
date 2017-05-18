using JLD
using CMBLensing
using CMBLensing: @dictpack, ode4
using CMBLensing.Minimize
using Base.Iterators: repeated

function run1(;
    Θpix = 3,
    nside = 64,
    T = Float32,
    r = 0.05,
    Nt1 = 15,   # number of t=1 branch steps
    Nt0 = 15,   # number of t=0 branch steps
    Ncg1₀ = 5,  # initial Ncg for t=1 steps
    Ncg0₀ = 80, # initial Ncg for t=0 steps
    seed = nothing, # random seed
    L = LenseFlow{ode4{7}},
    LJ = LenseFlow{ode4{2}},
    outfile = nothing,
    cls = nothing)
    
    seed!=nothing && srand(seed)
    
    cls==nothing && (cls = class(lmax=8000,r=r))
    
    ## set up the types of maps
    P = Flat{Θpix,nside}
    ## covariances
    Cf = Cℓ_to_cov(T,P,S2,cls[:ℓ], cls[:ee],    cls[:bb])
    Cf̃ = Cℓ_to_cov(T,P,S2,cls[:ℓ], cls[:ln_ee], cls[:ln_bb])
    Cϕ = Cℓ_to_cov(T,P,S0,cls[:ℓ], cls[:ϕϕ])
    μKarcminT = 1
    Ωpix = deg2rad(Θpix/60)^2
    CN = FullDiagOp(FlatS2QUMap{T,P}(repeated(fill(μKarcminT^2 * Ωpix,(nside,nside)),2)...))
    CN̂  = FullDiagOp(FlatS2EBFourier{T,P}(repeated(fill(μKarcminT^2 * Ωpix,(nside÷2+1,nside)),2)...))
    ##
    f = simulate(Cf)
    ϕ = simulate(Cϕ)
    f̃ = L(ϕ)*f
    
    # data mask
    ℓmax_mask, Δℓ_taper = 3000, 0
    Ml = [ones(ℓmax_mask); (cos(linspace(0,π,Δℓ_taper))+1)/2]
    Md = Cℓ_to_cov(T,P,S2,1:(ℓmax_mask+Δℓ_taper),repeated(Ml,2)...) * Squash
    
    # field prior mask
    # ℓmax_mask, Δℓ_taper = 3500, 0
    # Ml = [ones(ℓmax_mask); (cos(linspace(0,π,Δℓ_taper))+1)/2]
    # Mf = Cℓ_to_cov(T,P,S2,1:(ℓmax_mask+Δℓ_taper),repeated(Ml,2)...) * Squash
    Ml = ones(Complex{T},nside÷2+1,nside)
    i = indexin([-FFTgrid(T,P).nyq],FFTgrid(T,P).k)[1]
    Ml[:,i]=Ml[i,:]=0
    Mf = FullDiagOp(FlatS2EBFourier{T,P}(Ml,Ml)) * Squash
    
    # ϕ prior mask
    Mϕ = Squash
    
    ds = DataSet(f̃ + simulate(CN), CN̂, Cf, Cϕ, Md, Mf, Mϕ);
    target_lnP = (0Ð(f).+1)⋅(Md*(0Ð(f).+1)) / FFTgrid(T,P).Δℓ^2 / 2
    rundat = @dictpack Θpix nside T r μKarcminT d=>ds.d target_lnP cls f f̃ ϕ
    
    ## starting point
    fϕcur = f̃ϕcur = f̃ϕstart = Ł(FieldTuple(Squash*𝕎(Cf̃,CN̂)*ds.d,0ϕ))
    
    @show target_lnP
    
    if Nt1>0
        println(" --- t=1 steps ---")
        callback = tr -> outfile!=nothing && save(outfile,"rundat",rundat,"trace",tr)
        (f̃cur,ϕcur),tr1 = f̃ϕcur,tr1 = bcggd(1,f̃ϕstart,ds,L,LJ,Nsteps=Nt1,Ncg=Ncg1₀,β=2,callback=callback)
        fcur,ϕcur = fϕcur = FieldTuple(L(ϕcur)\f̃cur,ϕcur)
    else
        tr1 = []
    end
    
    println(" --- t=0 steps ---")
    callback = tr -> outfile!=nothing && @time save(outfile,"rundat",rundat,"trace",[tr1; tr])
    (fcur,ϕcur),tr2 = fϕcur,tr2 = bcggd(0,fϕcur,ds,L,LJ,Nsteps=Nt0,Ncg=Ncg0₀,β=2,callback=callback)
    f̃cur,ϕcur = f̃ϕcur = FieldTuple(L(ϕcur)*fcur,ϕcur)
    
    @show tr2[end][:lnP], target_lnP
    
    trace = [tr1; tr2]
    
    f̃cur, fcur, ϕcur, trace, rundat
    
end
