export bcggd, cg, pcg, pcg2, gdsteps

"""
Simple generic conjugate gradient implementation that works on Vectors, Fields, etc... 
"""
function cg(A, b, x=0*b; nsteps=length(b), tol=sqrt(eps()), progress=false, callback=nothing)
    r = b - A*x
    p = r
    bestres = res = dot(r,r)
    bestx = x
    reshist = Vector{typeof(res)}()

    dt = (progress==false ? Inf : progress)
    @showprogress dt "CG: " for i = 1:nsteps
        Ap = A*p
        α = res / dot(p,Ap)
        x = x + α * p
        r = r - α * Ap
        res′ = dot(r,r)
        if res′<bestres
            bestres,bestx = res′,x
        end
        if callback!=nothing
            callback(i,x,res)
        end
        push!(reshist,res)
        if res′<tol; break; end
        p = r + (res′ / res) * p
        res = res′
    end
    bestx, reshist
end

""" 
Preconditioned conjugate gradient. P should be symmetric and roughly √A⁻¹
"""
pcg(P,A,b,x=P*b; callback=nothing, kwargs...) = begin
    x, hist = cg(P*A*P, P*b, x; callback=(callback==nothing ? nothing : (i,x,res)->callback(i,P*x,res)), kwargs...)
    P*x, hist
end


"""
    pcg2(M, A, b, x=M\\b; nsteps=length(b), tol=sqrt(eps()), progress=false, callback=nothing, hist=nothing, histmod=1)

Compute x = A\\b (where A is positive definite) by conjugate gradient. M is the
preconditioner and should approximate A, and M \\ x should be fast.

The solver will stop either after `nsteps` iterations or when `dot(r,r)<tol` (where r
is the residual A*x-b at that step), whichever occurs first.

Info from the iterations of the solver can be returned if `hist` is specified.
`hist` can be `:x`, `:res`, or a tuple `(:x,:res)`, specifying which of `x` (the
estimated solution) and/or `res` (the norm of the residual of this solution) to
include. `histmod` can be used to include every N-th iteration only. 
"""
function pcg2(M, A, b, x=M\b; nsteps=length(b), tol=sqrt(eps()), progress=false, callback=nothing, hist=nothing, histmod=1)
    r = b - A*x
    z = M \ r
    p = z
    bestres = res = dot(r,z)
    bestx = x
    _hist = []

    dt = (progress==false ? Inf : progress)
    @showprogress dt "CG: " for i = 1:nsteps
        Ap   = A*p
        α    = res / dot(p,Ap)
        x    = x + α * p
        r    = r - α * Ap
        z    = M \ r
        res′ = dot(r,z)
        p    = z + (res′ / res) * p
        res  = res′
        
        if res<bestres
            bestres,bestx = res,x
        end
        if callback!=nothing
            callback(i,x,res)
        end
        if hist!=nothing && (i%histmod)==0
            push!(_hist, getindex.(@(dictpack(x,res)),hist))
        end
        if res<tol
            break
        end
    end
    hist == nothing ? bestx : (bestx, vcat(_hist...))
end

"""
Do Ngd gradient descent steps, with Ncg conjugate gradient steps towards solving
H⁻¹g at each iteration.

Arguments:
* t : which parametrization to use (i.e. t=0 for unlensed or t=1 for lensed)
* fₜϕ_cur : fₜϕ to start at
* Ngd, Ncg : number of gradient descent and conjugate gradient steps
* L : Lensing operator to use for gradient descent
* LJ : Lensing operator to use for the Hessian calculation

Returns:
* lnP(fₜϕ), fₜϕ, trace
"""
function gdsteps(t, fₜϕ_cur, ds, Ngd, Ncg, ::Type{L}, ::Type{LJ}=L; approxℍ⁻¹=nothing) where {L<:LenseOp, LJ<:LenseOp}
    fₜcur,ϕcur = fₜϕ_cur
    trace = []
    if approxℍ⁻¹ == nothing
        @unpack CN,Cf,Cϕ,Md,Mf,Mϕ = ds
        approxℍ⁻¹ = FullDiagOp(FieldTuple(Squash*(@. (Md.a*CN^-1 + Mf.a*Cf^-1)^-1).f, 1e-5*Mϕ*Cϕ.f))
    end
    
    for i=1:Ngd
        # get negative gradient
        g = -δlnP_δfϕₜ(t,fₜϕ_cur...,ds,L)

        # do steps towards CG solution of -H⁻¹*g
        if Ncg==0
            Hinvg,cghist = approxℍ⁻¹*g, nothing
        else
            Hinvg,cghist = pcg(sqrt.(approxℍ⁻¹), HlnP(t,fₜϕ_cur...,ds,L,LJ), g, nsteps=Ncg)
        end

        # line search
        T = eltype(Hinvg)
        res = optimize(α->(-lnP(t,(fₜϕ_cur+exp(α)*Hinvg)...,ds,L)), T(log(1e-5)), T(log(1e2)), abs_tol=1e-2)
        α = exp(res.minimizer)

        fₜcur,ϕcur = (fₜϕ_cur += α*Hinvg)
        
        push!(trace, @dictpack t fϕ=>fₜϕ_cur lnP=>res.minimum g Hinvg α Ncg Nls=>res.f_calls cghist)
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
* L : Lensing operator to use for gradient descent
* LJ : Lensing operator to use for the Hessian calculation
"""
function bcggd(t, fₜϕ_start, ds, ::Type{L}, ::Type{LJ}=L; Nsteps=10, Ncg=10, β=2, callback=(tr->nothing), kwargs...) where {L<:LenseOp, LJ<:LenseOp}
    trace = []
    fₜϕ_cur = fₜϕ_start
    for i=1:Nsteps
        ttot = @elapsed @threads for j=1:2
            j==1 && global t1 = @elapsed (global (lnP1, fₜϕ_cur1, tr1) = gdsteps(t,fₜϕ_cur,ds,2,Ncg,L,LJ; kwargs...))
            j==2 && global t2 = @elapsed (global (lnP2, fₜϕ_cur2, tr2) = gdsteps(t,fₜϕ_cur,ds,1,β*Ncg,L,LJ; kwargs...))
        end
        @show i, lnP1, lnP2, t1, t2, ttot
        if lnP2<lnP1
            callback(push!(trace,tr2...))
            Ncg *= β
            println("Increasing Ncg to $Ncg")
            fₜϕ_cur = fₜϕ_cur2
        else
            callback(push!(trace,tr1...))
            fₜϕ_cur = fₜϕ_cur1
        end
    end
    fₜϕ_cur, trace
end
