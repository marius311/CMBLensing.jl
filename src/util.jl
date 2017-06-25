export @repeated, nan2zero

import Base: ==
==(x,y,z,ws...) = x==y && ==(y,z,ws...)

"""
Simply replaces every occurence of ∷ with <:Any
"""
macro ∷(ex)
    esc(postwalk(x->(x==:(∷) ? :(<:Any) : x),ex))
end


"""
@! x = f!(args...) is equivalent to f!(x,args...)
"""
macro !(ex)
    if @capture(ex, x_ = f_(args__; kwargs_...))
        esc(:($f($x,$(args...); $kwargs...)))
    elseif @capture(ex, x_ = f_(args__))
        esc(:($f($x,$(args...))))
    else
        error("Usage: @! x = f!(...)")
    end
end


"""
@symarg f(a::T1,b::T2) = body

is equivalent to

f(a::T1,b::T2) = body
f(b::T2,a::T1) = body

TODO: phase out use of this entirely, it tends to lead to ambiguities....
"""
macro symarg(ex)
    if @capture ex ((f_(a_::T1_,b_::T2_) = body_) | (function f_(a_::T1_,b_::T2_) body_ end))
        esc(:($f($a::$T1,$b::$T2)=$body; $f($b::$T2,$a::$T1)=$body))
    elseif @capture ex ((f_(::T1_,::T2_) = body_) | (function f_(::T1_,::T2_) body_ end))
        esc(:($f(::$T1,::$T2)=$body; $f(::$T2,::$T1)=$body))
    else
        error("@symarg couldn't understand function.")
    end
end

nan2zero{T}(x::T) = !isfinite(x)?zero(T):x
nan2zero(x::Diagonal{T}) where {T} = Diagonal{T}(nan2zero.(x.diag))

""" Return a tuple with the expression repeated n times """
macro repeated(ex,n)
    :(tuple($(repeated(esc(ex),n)...)))
end

""" 
Pack some variables in a dictionary 

```
> x = 3
> y = 4
> @dictpack x y z=>5
Dict(:x=>3,:y=>4,:z=>5)
```
"""
macro dictpack(exs...)
    kv(ex::Symbol) = :($(QuoteNode(ex))=>$(esc(ex)))
    kv(ex) = isexpr(ex,:call) && ex.args[1]==:(=>) ? :($(QuoteNode(ex.args[2]))=>$(esc(ex.args[3]))) : error()
    :(Dict($((kv(ex) for ex=exs)...)))
end


"""
Take an expression like

    for a in A, b in B
        ...
        @. r += term
    end

and rewrite it so each term is computed in parallel and the results are added
together in a threadsafe manner.
"""
macro threadsum(ex)
    
    if Threads.nthreads()==1; return esc(ex); end
    
    @capture(ex, for a_ in A_, b_ in B_
        begin temps__; @. r_ += inc_; end
    end) || error("usage: @threadsum for a in A, b in B; begin ...; @. r += ...; end")
    
    quote
        m = SpinLock()
        @threads for ab in [($(esc(a)),$(esc(b))) for $(esc(a)) in $(esc(A)) for $(esc(b)) in $(esc(B))]
            $(esc(a)),$(esc(b)) = ab
            $(esc.(temps)...)
            inc = @. $(esc(inc))
            lock(m)
            $(esc(r)) .+= inc
            unlock(m)
        end
    end
end


"""
Threaded `map`, like `pmap`, but using `@threads`. 

If Threads.nthreads()==1 then this macro-exapands to just use `map`, so there's
zero overhead and no impact to type-stability. The threaded case however is not
type-stable, although this is intentional b/c for some weird reason that actually
makes my use-case slower.
"""
macro tmap(f,args...)
    if Threads.nthreads()==1
        :(map($(esc(f)),$(esc.(args)...)))
    else
        quote
            cargs = collect(zip($(esc.(args)...)))
            n = length(cargs)
            # TODO: figure out why inferring the result actually makes things *slower* ?
            # T = Core.Inference.return_type(f, Tuple{typeof.(cargs[1])...})
            T = Any
            ans = Vector{T}(n)
            @threads for i=1:n
                ans[i] = $(esc(f))(cargs[i]...)
            end
            ans
        end
    end
end
