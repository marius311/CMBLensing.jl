using JLD
using CMBLensing
using CMBLensing: @dictpack, jrk4, δlnΠᶠ_δfϕ
using Optim
using PyCall
using Base.Iterators: repeated

@pyimport scipy.interpolate as itp

function fit_wℓ(f_obs,f,f̃,Δℓ=400,s=1)
    (ℓ,Cℓb_obs),(ℓ,Cℓb),(ℓ,Cℓb̃) = (get_Cℓ(x,which=[:BB],Δℓ=Δℓ) for x in (f_obs,f,f̃))
    iℓ = @. !isnan(Cℓb_obs[:]) & (Cℓb_obs[:] > 0)
    ℓ = round.(Int,ℓ[iℓ])
    lnCℓb_obs, lnCℓb, lnCℓb̃ = @. (log(Cℓb_obs[:][iℓ]), log(Cℓb[:][iℓ]), log(Cℓb̃[:][iℓ]))
    z = @. (lnCℓb_obs-lnCℓb̃)/(lnCℓb-lnCℓb̃)
    iz = @. !isnan(z) & (z>0)
    itpz = itp.UnivariateSpline(ℓ[iz],log.(z[iz]),s=s)
    min.(1,exp.(itpz(1:10000)))
end


"""
Iterative conditional algorithm
"""
function run2(;
    Θpix = 3,
    nside = 64,
    T = Float32,
    r = 0.05,
    L = LenseFlow{jrk4{7}},
    outfile = nothing,
    seed = nothing,
    mask = nothing,
    Cℓf = nothing,
    use = :TEB,
    ℓmax_data = 3000,
    μKarcminT = 1,
    beamFWHM = 1.5,
    ℓknee = 100,
    ws = 1:20,
    Ncg = 100,
    Ncg0 = 5000,
    cgtol = 1e-4)
    
    # Cℓs
    Cℓf==nothing && (Cℓf = class(lmax=8000,r=r))
    Cℓf̃ = Dict(k=>Cℓf[Symbol("ln_$k")] for (k,v) in Cℓf if Symbol("ln_$k") in keys(Cℓf))
    Cℓn = noisecls(μKarcminT, beamFWHM=beamFWHM, ℓknee=ℓknee)
    
    ## covariances
    P = Flat{Θpix,nside}
    SS,ks = Dict(:TEB=>((S0,S2),(:tt,:ee,:bb,:te)), :EB=>((S2,),(:ee,:bb)), :T=>((S0,),(:tt,)))[use]
    Cϕ = Cℓ_to_cov(T,P,S0,    Cℓf[:ℓ], Cℓf[:ϕϕ])
    Cf = Cℓ_to_cov(T,P,SS..., Cℓf[:ℓ], (Cℓf[k] for k=ks)...)
    Cf̃ = Cℓ_to_cov(T,P,SS..., Cℓf[:ℓ], (Cℓf̃[k] for k=ks)...)
    Cn = Cℓ_to_cov(T,P,SS..., Cℓn[:ℓ], (Cℓn[k] for k=ks)...)
    
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
    local hist, fcur, f̃cur
    
    for (i,w) in enumerate(ws)
        
        if w==:auto
            wℓ = fit_wℓ(L(ϕcur)\f̃,f,f̃)[round.(Int,Cℓf[:ℓ])]
            w500 = wℓ[500]
            Cfw = Cℓ_to_cov(T,P,SS..., Cℓf[:ℓ], ((@. sign(Cℓf[k])*abs(Cℓf̃[k])^(1-wℓ)*abs(Cℓf[k])^wℓ) for k=ks)...);
        else
            w500 = w
            Cfw = Cf*w + Cf̃*(1-w)
        end
        
        ds = DataSet(d, Cn, Cfw, Cϕ, Md, Mf, Mϕ)
        
        let L = (w==0?IdentityOp:L(ϕcur)),
            P = nan2zero.(sqrtm((nan2zero.(Mdf * Cn^-1) .+ nan2zero.(Mff * Cfw^-1)))^-1)
            A = L'*(Md'*(Cn^-1)*Md*L) + Mf'*Cfw^-1*Mf
            b = L'*(Md'*(Cn^-1)*Md*d)
            fcur,hist = pcg(P, A, b, i==1?0*b:(Squash*(P\fcur)), nsteps=(w==0?Ncg0:Ncg), tol=cgtol)
            f̃cur = L*fcur
        end

        ϕnew = Mϕ*Cϕ*(δlnΠᶠ_δfϕ(fcur,ϕcur,ds) * δfϕ_δf̃ϕ(L(ϕcur),fcur,f̃cur))[2]
        α = (res = optimize(α->(-lnP(1,f̃cur,(1-α)*ϕcur+α*ϕnew,ds,L)), T(0), T(1), abs_tol=1e-6)).minimizer
        ϕcur = (1-α)*ϕcur+α*ϕnew

        lnPw = -res.minimum
        lnP1 = lnP(1,f̃cur,(1-α)*ϕcur+α*ϕnew,DataSet(d, Cn, Cf, Cϕ, Md, Mf, Mϕ),L)
        push!(trace,@dictpack f̃cur fcur ϕcur ϕnew lnPw lnP1 α w hist)
        @printf("%i %.4f %.2f %.2f %i %.4f\n",i,w500,lnPw,lnP1,length(hist),α)
        
        outfile!=nothing && save(outfile,"rundat",rundat,"trace",trace)
            
    end
    
    f̃cur, fcur, ϕcur, trace, rundat
    
end
