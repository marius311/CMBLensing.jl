module Minimize

export bcggd, cg, pcg, gdsteps

using Base.Threads
using CMBLensing
using CMBLensing: ode4, @dictpack
using Optim
using Parameters

"""
Simple generic conjugate gradient implementation that works on Vectors, Fields, etc... 
"""
function cg(A, b, x=0*b; nsteps=length(b))
    r = b - A*x
    p = r
    res = dot(r,r)
    reshist = Vector{typeof(res)}()

    for i = 1:nsteps
        Ap = A*p
        α = res / dot(p,Ap)
        x = x + α * p
        r = r - α * Ap
        res′ = dot(r,r)
        p = r + (res′ / res) * p
        push!(reshist,res)
        res = res′
    end
    x, reshist
end

""" 
Preconditioned conjugate gradient. P should be symmetric and roughly √A⁻¹
"""
pcg(P,A,b,x=0*b; kwargs...) = ((x,hist)=cg(P*A*P,P*b,x; kwargs...); (P*x, hist))


"""
Do Ngd gradient descent steps, with Ncg conjugate gradient steps towards solving
H⁻¹g at each iteration.

Arguments:
* t : which parametrization to use (i.e. t=0 for unlensed or t=1 for lensed)
* fₜϕ_cur : fₜϕ to start at
* Ngd, Ncg : number of gradient descent and conjugate gradient steps

Returns:
* lnP(fₜϕ), fₜϕ, trace
"""
function gdsteps(t, fₜϕ_cur, ds, ::Type{L}, Ngd, Ncg) where {L<:LenseOp}
    fₜcur,ϕcur = fₜϕ_cur
    trace = []
    
    for i=1:Ngd
        let L=L(ϕcur)
            # get negative gradient
            g = -δlnP_δfϕₜ(t,fₜϕ_cur...,ds,L)

            # do steps towards CG solution of -H⁻¹*g
            @unpack CN,Cf,Cϕ,Md,Mf,Mϕ = ds
            approxℍ⁻¹ = FullDiagOp(FieldTuple(Squash*(@. (Md.a*CN^-1 + Mf.a*Cf^-1)^-1).f, 1e-5*Mϕ*Cϕ.f));
            if Ncg==0
                Hinvg,cghist = approxℍ⁻¹*g, nothing
            else
                Hinvg,cghist = pcg(sqrt.(approxℍ⁻¹), HlnP(t,fₜϕ_cur...,ds,L,LenseFlow{ode4{2}}(ϕcur)), g, nsteps=Ncg)
            end

            # line search
            T = eltype(Hinvg)
            res = optimize(α->(-lnP(t,(fₜϕ_cur+exp(α)*Hinvg)...,ds)), T(log(1e-5)), T(log(1e2)), abs_tol=1e-2)
            α = exp(res.minimizer)

            fₜcur,ϕcur = (fₜϕ_cur += α*Hinvg)
            
            push!(trace, @dictpack t fϕ=>fₜϕ_cur lnP=>res.minimum g Hinvg α Ncg Nls=>res.f_calls cghist)
        end
    end
    res.minimum, fₜϕ_cur, trace
end


"""
Branching ConjugateGradient-GradientDescent

Algortihm:
    In parallel do:
    a) 2 steps of gradient descent with Ncg steps of conjugate gradient towards H⁻¹g at each iteration
    b) 1 steps of gradient descent with β*Ncg steps of conjugate gradient towards H⁻¹g

    Take whichever is a lower lnP. If we take (b), then update Ncg = β*Ncg. 
    
    β is chosen such that (a) and (b) take roughly the same ammount of time to
    compute (usually β≈2 but could be higher if there's alot of overhead for a GD step)

Arguments:
* t : which parametrization to use (i.e. t=0 for unlensed or t=1 for lensed)
* fₜϕ_start : fₜϕ to start at
* Nsteps : how many iterations of the above algorithm to do
* Ncg : the starting value of Ncg
* β : scaling factor described above
"""
function bcggd(t, fₜϕ_start, ds, ::Type{L}; Nsteps=10, Ncg=10, β=2) where {L<:LenseOp}
    trace = []
    fₜϕ_cur = fₜϕ_start
    for i=1:Nsteps
        @threads for j=1:2
            j==1 && global t1 = @elapsed (global (lnP1, fₜϕ_cur1, tr1) = gdsteps(t,fₜϕ_cur,ds,L,2,Ncg))
            j==2 && global t2 = @elapsed (global (lnP2, fₜϕ_cur2, tr2) = gdsteps(t,fₜϕ_cur,ds,L,1,β*Ncg))
        end
        @show i, lnP1, lnP2, t1, t2
        if lnP2<lnP1
            push!(trace,tr2...)
            @show Ncg *= β
            fₜϕ_cur = fₜϕ_cur2
        else
            push!(trace,tr1...)
            fₜϕ_cur = fₜϕ_cur1
        end
    end
    fₜϕ_cur, trace
end


end
