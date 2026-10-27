using CMBLensing, FFTW, LinearAlgebra, Random, Zygote

function main()
    println("Julia=",VERSION," FFTW provider=",FFTW.fftw_provider," CMBLensing=",pathof(CMBLensing))
    println("RUN_ID=",get(ENV,"CMB_AUDIT_SOURCE","unspecified"))
    for T in (Float32,Float64), (Ny,Nx) in ((6,8),(7,9)), pol in (1,2), B in (Map,Fourier)
        rng=MersenneTwister(50123+Ny+Nx+pol)
        proj=ProjLambert(;Ny,Nx,θpix=60*180/π,T)
        scalar()=FlatMap(randn(rng,T,Ny,Nx),proj)
        polarized()=FlatQUMap(scalar().Ix,scalar().Ix,proj)
        ϕ,η=B(T(0.35)*scalar()),B(T(0.05)*scalar())
        f=B(pol==1 ? scalar() : polarized())
        g=B(pol==1 ? scalar() : polarized())
        δ=B(pol==1 ? scalar() : polarized())
        label="T=$T N=$(Ny)x$(Nx) pol=$pol B=$B"
        try
            old=copy.((ϕ.arr,η.arr,f.arr,g.arr,δ.arr))
            h=T==Float32 ? T(1e-3) : T(1e-5)
            obj=(p,x)->dot(g,BilinearLens(p)*x)
            val,back=Zygote.pullback(obj,ϕ,f)
            gradϕ,gradf=back(one(val))
            revϕ,revf=dot(gradϕ,η),dot(gradf,δ)
            mutation=maximum(maximum(abs.(a.-b)) for (a,b) in zip((ϕ.arr,η.arr,f.arr,g.arr,δ.arr),old))
            finiteϕ=(obj(ϕ+h*η,f)-obj(ϕ-h*η,f))/(2h)
            finitef=(obj(ϕ,f+h*δ)-obj(ϕ,f-h*δ))/(2h)
            finiteboth=(obj(ϕ+h*η,f+h*δ)-obj(ϕ-h*η,f-h*δ))/(2h)
            dx,dy=getindex.((∇*ϕ)./proj.Δx,:Ix)
            gap=minimum(minimum(abs.(d.-round.(d))) for d in (dx,dy))
            dη=getindex.((∇*η)./proj.Δx,:Ix)
            shift=h*maximum(maximum(abs.(d)) for d in dη)
            println("CASE $label revphi=$revϕ fdphi=$finiteϕ revf=$revf fdf=$finitef revboth=$(revϕ+revf) fdboth=$finiteboth gap=$gap step=$shift mutation=$mutation")
        catch err
            println("ERROR $label ",sprint(showerror,err,catch_backtrace()))
        end
        flush(stdout)
    end
end
main()
