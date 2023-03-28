abstract type ODESolver end

struct RK4Solver <: ODESolver
    nsteps :: Int
end

struct OutOfPlaceRK4Solver <: ODESolver
    nsteps :: Int
end

function (solver::RK4Solver)(F!::Function, y₀, (t₀, t₁)::Pair)
    h, h½, h⅙ = (t₁-t₀)/solver.nsteps ./ (1,2,6)
    y = copy(y₀)
    k₁, k₂, k₃, k₄, y′ = @repeated(similar(y₀),5)
    for t in range(t₀,t₁,length=solver.nsteps+1)[1:end-1]
        @! k₁ = F(t, y)
        @! k₂ = F(t + h½, (@. y′ = y + h½*k₁))
        @! k₃ = F(t + h½, (@. y′ = y + h½*k₂))
        @! k₄ = F(t + h,  (@. y′ = y + h*k₃))
        @. y += h*(k₁ + 2(k₂ + k₃) + k₄)/6
    end
    map(unsafe_free!, (k₁, k₂, k₃, k₄, y′))
    return y
end

function (solver::OutOfPlaceRK4Solver)(F::Function, y₀, (t₀, t₁)::Pair)
    h, h½, h⅙ = (t₁-t₀)/solver.nsteps ./ (1,2,6)
    y = copy(y₀)
    for i in 0:solver.nsteps-1
        t = i*h
        k₁ = F(t, y)
        k₂ = F(t + h½, y + h½*k₁)
        k₃ = F(t + h½, y + h½*k₂)
        k₄ = F(t + h,  y + h*k₃)
        y += @. h*(k₁ + 2(k₂ + k₃) + k₄)/6
    end
    return y
end


@doc doc"""
    conjugate_gradient(
        M, A, b, x=M\b; 
        nsteps       = length(b), 
        tol          = sqrt(eps()), 
        progress     = false, 
        callback     = nothing, 
        history_keys = nothing, 
        history_mod  = 1
    )

Compute `x=A\b` (where `A` is positive definite) by conjugate
gradient. `M` is the preconditioner and should be `M≈A`, and `M\x`
should be fast.

The solver will stop either after `nsteps` iterations or when
`dot(r,r)<tol` (where `r=A*x-b` is the residual  at that step),
whichever occurs first.

Info from the iterations of the solver can be returned if
`history_keys` is specified. `history_keys` can be one or a tuple of:

* `:i` — current iteration number
* `:x` — current solution
* `:r` — current residual `r=A*x-b`
* `:res` — the norm of `r`
* `:t` — the time elapsed (in seconds) since the start of the
  algorithm

`history_mod` can be used to include every N-th iteration only in
`history_keys`. 
"""
@⌛ function conjugate_gradient(
    M,
    A,
    b,
    x = zero(b);
    nsteps       = length(b),
    tol          = sqrt(eps()),
    progress     = false,
    callback     = nothing,
    history_keys = nothing,
    history_mod  = 1
)
    get_history() = isnothing(history_keys) ? nothing : select((;i,x,p,r,res,t),history_keys)
    t₀ = time()
    T = real(eltype(x)) # allow `dot` to return a higher precision but keep vector its original eltype
    i = 1
    r = b - A*x
    z = M \ r
    p = z
    bestres = res = res₀ = T(dot(r,z))
    @assert !isnan(res)
    bestx = x
    t    = time() - t₀
    history = [get_history()]

    prog = Progress(100, (progress!=false ? progress : Inf), "Conjugate Gradient: ")
    for outer i = 2:nsteps
        Ap   = @⌛ A * p
        α    = res / dot(p,Ap)
        x    = x + T(α) * p
        r    = r - T(α) * Ap
        z    = M \ r
        res′ = T(dot(r,z))
        p    = z + (res′ / res) * p
        res  = res′
        t    = time() - t₀
        
        if all(res<bestres)
            bestres,bestx = res,x
        end
        if !isnothing(callback)
            callback(i,x,res)
        end
        if (i%history_mod) == 0
            push!(history, get_history())
        end
        if all(res<tol)
            break
        end
        
        # update progress bar to whichever we've made the most progress on,
        # logarithmically reaching the toleranace limit or doing the maximum
        # number of steps
        if progress
            progress_nsteps = round(Int,100*(i-1)/(nsteps-1))
            progress_tol = res isa AbstractFloat ? round(Int,100^min(1, (log10(res/res₀)) / log10(tol/res₀))) : 0
            ProgressMeter.update!(prog, max(progress_nsteps,progress_tol))
        end
    end
    ProgressMeter.finish!(prog)
    (bestx, history)
end





@doc doc"""

    itp = LinearInterpolation(xdat::AbstractVector, ydat::AbstractVector; extrapolation_bc=NaN)
    itp(x) # interpolate at x
    
A simple 1D linear interpolation code which is fully Zygote differentiable in
either `xdat`, `ydat`, or the evaluation point `x`.
"""
function LinearInterpolation(xdat::AbstractVector, ydat::AbstractVector; extrapolation_bc=NaN)
    
    @assert issorted(xdat)
    @assert extrapolation_bc isa Number || extrapolation_bc == :line
    
    m = diff(ydat) ./ diff(collect(xdat))
    
    function (x::Number)
        
        if x<xdat[1] || x>xdat[end]
            if extrapolation_bc isa Number
                return extrapolation_bc
            elseif extrapolation_bc == :line
                if x<xdat[1]
                    i = 1 
                elseif x>xdat[end]
                    i = length(m)
                end
            end
        else
            # sets i such that x is between xdat[i] and xdat[i+1]
            i = max(1, searchsortedfirst(xdat,x) - 1)
        end
        
        # do interpolation
        ydat[i] + m[i]*(x-xdat[i])
        
    end
    
end


@doc doc"""

    gmres(A, b; maxiter, Pl=I)

Solve `A \ b` with `maxiter` iterations of the [generalized minimal
residual](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method)
algorithm. `Pl` is a left-preconditioner which should approximate `inv(A)`. 

Note: the implemenation is memory inefficient and uses O(n * maxiter) memory, where
`n,n=size(A)` (may not be a big deal for small `maxiter`), although is totally generic
and works with CPU or GPU and dense or sparse matrices, unlike IterativeSolver's
`gmres`.
"""
function gmres(A, b; Pl=I, maxiter)
    
    n = maxiter
    T = promote_op(matprod, eltype(A), eltype(b))
    
    # build Krylov matrix K = [(Pl*A)*b (Pl*A)²*b ...]
    K = similar(b, T, length(b), n+1)
    mul!(view(K, :, 1), Pl, b)
    for i=2:n+1
        mul!(view(K,:,i), A, view(K,:,i-1))
        if Pl !== I
            mul!(view(K,:,i), Pl, K[:,i]) # copy needed here
        end
    end
    
    # solve least-squares problem |Pl * A * K * α - Pl * b|²
    α = qr(view(K, :, 2:n+1)) \ view(K, :, 1)
    
    # return solution, K * α
    view(K, :, 1:n) * α
    
end

"""
    finite_second_derivative(x)

Second derivative of a vector `x` via finite differences, including at end points.
"""
function finite_second_derivative(x)
    map(eachindex(x)) do i
        if i==1
            x[3]-2x[2]+x[1]
        elseif i==length(x)
            x[end]-2x[end-1]+x[end-2]
        else
            x[i+1]-2x[i]+x[i-1]
        end
    end
end

"""
    longest_run_of_trues(x)

The slice corresponding to the longest run of `true`s in the vector `x`. 
"""
function longest_run_of_trues(x)
    nothing2length(y) = isnothing(y) ? length(x) : y
    next_true = nothing2length.(findnext.(Ref(.!x), eachindex(x)))
    (len,start) = findmax(next_true .- eachindex(x))
    start:start+len
end