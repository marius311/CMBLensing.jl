module CINFFTDiagnostic
using CMBLensing, NFFT, Zygote
function lambert_probe()
    ny,nx,nside=8,8,4
    for T in (Float32,Float64)
        p=ProjLambert(;Ny=ny,Nx=nx,θpix=600,rotator=(0,0,0),T)
        hp=ProjHealpix(nside)
        pr=CMBLensing.Projector(p=>hp;method=:fft)
        θϕ=CMBLensing.pix2angRing.(Ref(CMBLensing.Resolution(nside)),1:12*nside^2)
        θ,ϕ=first.(θϕ),last.(θϕ)
        r=2 .* cos.(θ ./ 2)
        is=@. -r*cos(ϕ)/p.Δx+ny÷2+0.5
        js=@. -r*sin(ϕ)/p.Δx+nx÷2+0.5
        patch=pr.hpx_idxs_in_patch
        gn=reduce(hcat,[[(a-ny÷2-1)/ny,(b-nx÷2-1)/nx] for a=1:ny,b=1:nx])
        hn=reduce(hcat,[[(is[k]-ny÷2-1)/ny,(js[k]-nx÷2-1)/nx] for k in patch])
        gd,hd=NFFT.NDFTPlan(gn,(ny,nx)),NFFT.NDFTPlan(hn,(ny,nx))
        v=T.(sin.(collect(1.0:12*nside^2)./7))
        w=T.(cos.(collect(1.0:ny*nx)./5))
        exact=real.(gd*(hd'*complex.(Float64.(v[patch]))))./length(patch)
        exactgrad=real.(hd*(gd'*complex.(Float64.(w))))./length(patch)
        y=project(pr,CMBLensing.HealpixMap(v,hp)=>p).arr[:]
        g=Zygote.gradient(v->sum(w.*project(pr,CMBLensing.HealpixMap(v,hp)=>p).arr[:]),v)[1]
        println("Lambert T=",T," params=",pr.nfft_plan.params," first=",y[1]," exactfirst=",exact[1]," max_abs=",maximum(abs.(y-exact))," grad_max=",maximum(abs.(g[patch]-exactgrad)))
        for (label,kw) in [("default",(;)),("nonblocking",(;blocking=false)),("full",(;precompute=NFFT.FULL))]
            try
                gp=NFFT.plan_nfft(T.(gn),(ny,nx);kw...)
                hpplan=NFFT.plan_nfft(T.(hn),(ny,nx);kw...)
                yy=real.(gp*(hpplan'*complex.(v[patch])))./length(patch)
                gg=real.(hpplan*(gp'*complex.(w)))./length(patch)
                println("NFFT T=",T," variant=",label," params=",gp.params," first=",yy[1]," max_abs=",maximum(abs.(yy-exact))," grad_max=",maximum(abs.(gg-exactgrad)))
            catch e
                println("variant error ",label," ",sprint(showerror,e))
            end
            flush(stdout)
        end
    end
end
end
