abstract type ODESolver end
# only one single ODE solver implemented for now, a simple custom RK4:
abstract type RK4Solver{nsteps} <: ODESolver  end



"""
Solve for y(t₁) with 4th order Runge-Kutta assuming dy/dt = F(t,y) and y(t₀) = y₀

Arguments
* F! : a function F!(v,t,y) which sets v=F(t,y)
"""
function RK4Solver(F!::Function, y₀, t₀, t₁, nsteps)
    h = (t₁-t₀)/nsteps
    y = copy(y₀)
    k₁, k₂, k₃, k₄, y′ = @repeated(similar(y₀),5)
    for t in range(t₀,t₁,length=nsteps+1)[1:end-1]
        @! k₁ = F(t, y)
        @! k₂ = F(t + (h/2), (@. y′ = y + (h/2)*k₁))
        @! k₃ = F(t + (h/2), (@. y′ = y + (h/2)*k₂))
        @! k₄ = F(t +   (h), (@. y′ = y +   (h)*k₃))
        
        # due to https://github.com/JuliaLang/julia/issues/27988, if this were
        # written the natural way as:
        #    @. y .+= h*(k₁ + 2k₂ + 2k₃ + k₄)/6
        # it has god-awful performance for FieldTuples (although is fine for
        # FlatS0s). until a solution for that issue comes around, a workaround
        # is to write out the broadcasting kernel by hand:
        broadcast!((y,h,k₁,k₂,k₃,k₄)->(y+h*(k₁+2k₂+2k₃+k₄)/6), y, (y,h,k₁,k₂,k₃,k₄)...)
    end
    return y
end
odesolve(::Type{RK4Solver{N}},F!,y₀,t₀,t₁) where {N} = RK4Solver(F!,y₀,t₀,t₁,N)




@doc doc"""
    conjugate_gradient(M, A, b, x=M\b; nsteps=length(b), tol=sqrt(eps()), progress=false, callback=nothing, hist=nothing, histmod=1)

Compute `x=A\b` (where `A` is positive definite) by conjugate gradient. `M` is the
preconditioner and should be `M≈A`, and `M\x` should be fast.

The solver will stop either after `nsteps` iterations or when `dot(r,r)<tol`
(where `r=A*x-b` is the residual  at that step), whichever occurs first.

Info from the iterations of the solver can be returned if `hist` is specified.
`hist` can be one or a tuple of:

* `:i` — current iteration number
* `:x` — current solution
* `:r` — current residual `r=A*x-b`
* `:res` — the norm of `r`
* `:t` — the time elapsed (in seconds) since the start of the algorithm

`histmod` can be used to include every N-th iteration only in `hist`. 
"""
function conjugate_gradient(M, A, b, x=0*b; nsteps=length(b), tol=sqrt(eps()), progress=false, callback=nothing, hist=nothing, histmod=1)
    gethist() = hist == nothing ? nothing : NamedTuple{hist}(getindex.(Ref(@dictpack(i,x,p,r,res,t)),hist))
    t₀ = time()
    i = 1
    r = b - A*x
    z = M \ r
    p = z
    bestres = res = res₀ = dot(r,z)
    @assert !isnan(res)
    bestx = x
    t    = time() - t₀
    _hist = [gethist()]

    prog = Progress(100, (progress!=false ? progress : Inf), "Conjugate Gradient: ")
    for outer i = 2:nsteps
        Ap   = A * p
        α    = res / dot(p,Ap)
        x    = x + α * p
        r    = r - α * Ap
        z    = M \ r
        res′ = dot(r,z)
        p    = z + (res′ / res) * p
        res  = res′
        t    = time() - t₀
        
        if res<bestres
            bestres,bestx = res,x
        end
        if callback!=nothing
            callback(i,x,res)
        end
        if hist!=nothing && (i%histmod)==0
            push!(_hist, gethist())
        end
        if res<tol
            break
        end
        
        # update progress bar to whichever we've made the most progress on,
        # logarithmically reaching the toleranace limit or doing the maximum
        # number of steps
        progress_nsteps = round(Int,100*(i-1)/(nsteps-1))
        progress_tol = round(Int,100^((log10(res/res₀)) / log10(tol/res₀)))
        ProgressMeter.update!(prog, max(progress_nsteps,progress_tol))
    end
    ProgressMeter.finish!(prog)
    hist == nothing ? bestx : (bestx, _hist)
end
