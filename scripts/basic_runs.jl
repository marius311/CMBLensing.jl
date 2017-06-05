using JLD
using CMBLensing
using CMBLensing: @dictpack, ode4, δlnΠᶠ_δfϕ
using CMBLensing.Minimize
using CMBLensing.Masking
using Optim
using Base.Iterators: repeated


function noisecls(μKarcminT,lmax=10000)
    cls = Dict{Symbol,Any}(:ℓ=>1:lmax)
    for x in [:tt,:ee,:bb]
        cls[x]=fill((x==:tt?1:2) * (μKarcminT*deg2rad(1/60))^2 * (4π),lmax)
    end
    cls[:te]=zeros(lmax)
    cls
end


function run1(;
    Θpix = 3,
    nside = 64,
    T = Float32,
    r = 0.05,
    Nt1 = 15,   # number of t=1 branch steps
    Nt0 = 15,   # number of t=0 branch steps
    Ncg1₀ = 5,  # initial Ncg for t=1 steps
    Ncg0₀ = 80, # initial Ncg for t=0 steps
    hessNϕ = false,
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
    Cn = FullDiagOp(FlatS2QUMap{T,P}(repeated(fill(μKarcminT^2 * Ωpix,(nside,nside)),2)...))
    Cn̂  = FullDiagOp(FlatS2EBFourier{T,P}(repeated(fill(μKarcminT^2 * Ωpix,(nside÷2+1,nside)),2)...))
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
    
    ds = DataSet(f̃ + simulate(Cn), Cn̂, Cf, Cϕ, Md, Mf, Mϕ)
    target_lnP = (0Ð(f).+1)⋅(Md*(0Ð(f).+1)) / FFTgrid(T,P).Δℓ^2 / 2
    rundat = @dictpack Θpix nside T r μKarcminT d=>ds.d target_lnP cls f f̃ ϕ
    
    
    # hessian
    if hessNϕ
        Nℓϕϕ = readdlm("../dat/noise_dd.dat")[:].*(2:3000.).^-2./100
        Nϕ = Cℓ_to_cov(T,P,S0,2:3000,Nℓϕϕ)
        approxℍ⁻¹ = FullDiagOp(FieldTuple(Squash*(@. (Md.a*Cn̂^-1 + Mf.a*Cf^-1)^-1).f, Mϕ*Nϕ.f))
    else
        approxℍ⁻¹ = nothing
    end
    
    ## starting point
    fϕcur = f̃ϕcur = f̃ϕstart = Ł(FieldTuple(Squash*𝕎(Cf̃,Cn̂)*ds.d,0ϕ))
    
    println("target_lnP = $(round(Int,target_lnP)) ± $(round(Int,sqrt(2*target_lnP)))")
    
    if Nt1>0
        println(" --- t=1 steps ---")
        callback = tr -> outfile!=nothing && save(outfile,"rundat",rundat,"trace",tr)
        (f̃cur,ϕcur),tr1 = f̃ϕcur,tr1 = bcggd(1,f̃ϕstart,ds,L,LJ,Nsteps=Nt1,Ncg=Ncg1₀,β=2,callback=callback,approxℍ⁻¹=approxℍ⁻¹)
        fcur,ϕcur = fϕcur = FieldTuple(L(ϕcur)\f̃cur,ϕcur)
    else
        tr1 = []
    end
    
    println(" --- t=0 steps ---")
    callback = tr -> outfile!=nothing && @time save(outfile,"rundat",rundat,"trace",[tr1; tr])
    (fcur,ϕcur),tr2 = fϕcur,tr2 = bcggd(0,fϕcur,ds,L,LJ,Nsteps=Nt0,Ncg=Ncg0₀,β=2,callback=callback,approxℍ⁻¹=approxℍ⁻¹)
    f̃cur,ϕcur = f̃ϕcur = FieldTuple(L(ϕcur)*fcur,ϕcur)
    
    @show tr2[end][:lnP], target_lnP
    
    trace = [tr1; tr2]
    
    f̃cur, fcur, ϕcur, trace, rundat
    
end



"""
Iterative conditional algorithm
"""
function run2(;
    Θpix = 3,
    nside = 64,
    T = Float32,
    r = 0.05,
    L = LenseFlow{ode4{7}},
    outfile = nothing,
    seed = nothing,
    mask = nothing,
    Cℓf = nothing,
    use = :TEB,
    ℓmax_data = 3000,
    μKarcminT = 1,
    ws = linspace(0,1,20).^3,
    Ncg = 100,
    Ncg0 = 5000)
    
    # Cℓs
    Cℓf==nothing && (Cℓf = class(lmax=8000,r=r))
    Cℓn = noisecls(μKarcminT)
    
    ## covariances
    P = Flat{Θpix,nside}
    Cϕ = Cℓ_to_cov(T,P,S0, Cℓf[:ℓ], Cℓf[:ϕϕ])
    if use==:TEB
        Cf =  Cℓ_to_cov(T,P,S0,S2,Cℓf[:ℓ], Cℓf[:tt],    Cℓf[:ee],    Cℓf[:bb],    Cℓf[:te])
        Cf̃  = Cℓ_to_cov(T,P,S0,S2,Cℓf[:ℓ], Cℓf[:ln_tt], Cℓf[:ln_ee], Cℓf[:ln_bb], Cℓf[:ln_te])
        Cn =  Cℓ_to_cov(T,P,S0,S2,Cℓn[:ℓ], Cℓn[:tt],    Cℓn[:ee],    Cℓn[:bb],    Cℓn[:te])
    elseif use==:EB
        Cf =  Cℓ_to_cov(T,P,S2,Cℓf[:ℓ], Cℓf[:ee],    Cℓf[:bb])
        Cf̃ =  Cℓ_to_cov(T,P,S2,Cℓf[:ℓ], Cℓf[:ln_ee], Cℓf[:ln_bb])
        Cn =  Cℓ_to_cov(T,P,S2,Cℓn[:ℓ], Cℓn[:ee],    Cℓn[:bb])
    elseif use==:T
        Cf =  Cℓ_to_cov(T,P,S0,Cℓf[:ℓ], Cℓf[:tt])
        Cf̃ =  Cℓ_to_cov(T,P,S0,Cℓf[:ℓ], Cℓf[:ln_tt])
        Cn =  Cℓ_to_cov(T,P,S0,Cℓn[:ℓ], Cℓn[:tt])
    else
        error("Unrecognized '$(use)'")
    end
    
    
    # data mask
    F,F̂,nF = Dict(:TEB=>(FlatIQUMap,FlatTEBFourier,3), :EB=>(FlatS2QUMap,FlatS2EBFourier,2), :T=>(FlatS0Map,FlatS0Fourier,1))[use]
    Mdf = FullDiagOp(F̂{T,P}(repeated(Cℓ_2D(P,1:ℓmax_data,ones(ℓmax_data)),nF)...))
    if mask!=nothing
        Mdr = FullDiagOp(F{T,P}(repeated(T.(sptlike_mask(nside,Θpix; (mask==true?():mask)...)),nF)...))
    else
        Mdr = 1
    end
    Md = Squash * Mdr * Mdf * Squash

    # field prior mask
    if iseven(nside)
        Ml = ones(Complex{T},nside÷2+1,nside)
        i = indexin([-FFTgrid(T,P).nyq],FFTgrid(T,P).k)[1]
        Ml[:,i] = Ml[i,:] = 0
        Mff = FullDiagOp(F̂{T,P}(repeated(Ml,nF)...))
    else
        Mff = 1
    end
    Mf = Squash * Mff * Squash
    
    # ϕ prior mask
    Mϕ = Squash
    
    ## simulate data
    seed!=nothing && srand(seed)
    f = simulate(Cf)
    ϕ = simulate(Cϕ)
    f̃ = L(ϕ)*f
    d = f̃ + simulate(Cn)

    target_lnP = mean(let n=simulate(Cn); -n⋅(Md'*(Cn\(Md*n)))/2 end for i=1:100)
    @show target_lnP
    rundat = @dictpack Θpix nside T r μKarcminT d target_lnP Cℓf Cℓn f f̃ ϕ

    trace = []

    ϕcur = 0ϕ
    fcur, f̃cur = nothing, nothing
    
    for w in ws
        
        Cfw = @. (1-w)*Cf̃ + w*Cf
        ds = DataSet(d, Cn, Cfw, Cϕ, Md, Mf, Mϕ)
        
        let L = (w==0?IdentityOp:L(ϕcur)),
            P = nan2zero.(sqrtm((nan2zero.(Mdf * Cn^-1) .+ nan2zero.(Mff * Cfw^-1)))^-1);
            A = L'*(Md'*(Cn^-1)*Md*L) + Mf'*Cfw^-1*Mf
            b = L'*(Md'*(Cn^-1)*Md*d)
            fcur,hist = pcg(P, A, b, fcur==nothing?0*b:(Squash*(P\fcur)), nsteps=(w==0?Ncg0:Ncg))
            f̃cur = L*fcur
        end

        ϕnew = Mϕ*Cϕ*(δlnΠᶠ_δfϕ(fcur,ϕcur,ds) * δfϕ_δf̃ϕ(L(ϕcur),fcur,f̃cur))[2]
        α = (res = optimize(α->(-lnP(1,f̃cur,(1-α)*ϕcur+α*ϕnew,ds,L)), T(0), T(1), abs_tol=1e-6)).minimizer
        ϕcur = (1-α)*ϕcur+α*ϕnew

        lnPw = -res.minimum
        lnP1 = lnP(1,f̃cur,(1-α)*ϕcur+α*ϕnew,DataSet(d, Cn, Cf, Cϕ, Md, Mf, Mϕ),L)
        push!(trace,@dictpack f̃cur fcur ϕcur ϕnew lnPw lnP1 α w hist)
        @printf("%.4f %.2f %.2f %.4f",w,lnPw,lnP1,α)
        
        outfile!=nothing && save(outfile,"rundat",rundat,"trace",trace)
            
    end
    
    f̃cur, fcur, ϕcur, trace, rundat
    
end
