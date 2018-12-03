export @repeated, nan2zero

import Base: ==
==(x,y,z,ws...) = x==y && ==(y,z,ws...)


""" 
Return the type's fields as a tuple
"""
@generated fieldvalues(x) = :(tuple($((:(x.$f) for f=fieldnames(x))...)))


"""
Replaces every occurence of ∷ with `<:Any`
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
        if f == :*
            f = :mul!
        end
        esc(:($f($x,$(args...))))
    else
        error("Usage: @! x = f!(...)")
    end
end


"""
@commutative f(a::T1,b::T2) = body

is equivalent to

f(a::T1,b::T2) = body
f(b::T2,a::T1) = body

TODO: phase out use of this entirely, it tends to lead to ambiguities....
"""
macro commutative(ex)
    if @capture ex ((f_(a_::T1_,b_::T2_) = body_) | (function f_(a_::T1_,b_::T2_) body_ end))
        esc(:($f($a::$T1,$b::$T2)=$body; $f($b::$T2,$a::$T1)=$body))
    elseif @capture ex ((f_(::T1_,::T2_) = body_) | (function f_(::T1_,::T2_) body_ end))
        esc(:($f(::$T1,::$T2)=$body; $f(::$T2,::$T1)=$body))
    else
        error("@commutative couldn't understand function definition.")
    end
end

nan2zero(x::T) where {T} = !isfinite(x) ? zero(T) : x
nan2zero(x::Diagonal{T}) where {T} = Diagonal{T}(nan2zero.(x.diag))
nan2inf(x::T) where {T} = !isfinite(x) ? T(Inf) : x


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
Pack some variables in a named tuple 

```
> x = 3
> y = 4
> @dictpack x y z=>5
(x=3,y=4,z=5)
```
"""
macro ntpack(exs...)
    kv(ex::Symbol) = :($(esc(ex))=$(esc(ex)))
    kv(ex) = esc(ex)
    Expr(:tuple, (kv(ex) for ex=exs)...)
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


# these allow inv and sqrt of SMatrices of Diagonals to work correctly, which
# we use for the T-E block of the covariance. hopefully some of this can be cut
# down on in the futue with some PRs into StaticArrays.
import StaticArrays: arithmetic_closure
import Base: sqrt, inv, /
arithmetic_closure(::Type{Diagonal{T}}) where {T} = Diagonal{arithmetic_closure(T)}
inv(d::Diagonal) = Diagonal(1 ./ d.diag)
/(a::Number, b::Diagonal) = Diagonal(a ./ diag(b))


# some usefule tuple manipulation functions
using Base: tuple_type_cons, tuple_type_head, tuple_type_tail, first, tail
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)
# see https://discourse.julialang.org/t/any-way-to-make-this-one-liner-type-stable/10636/2
map_tupleargs(f,::Type{T}) where {T<:Tuple} = 
    (f(tuple_type_head(T)), map_tupleargs(f,tuple_type_tail(T))...)
map_tupleargs(f,::Type{T},::Type{S}) where {T<:Tuple,S<:Tuple} = 
    (f(tuple_type_head(T),tuple_type_head(S)), map_tupleargs(f,tuple_type_tail(T),tuple_type_tail(S))...)
map_tupleargs(f,::Type{T},s::Tuple) where {T<:Tuple} = 
    (f(tuple_type_head(T),first(s)), map_tupleargs(f,tuple_type_tail(T),tail(s))...)
map_tupleargs(f,::Type{<:Tuple{}}...) = ()
map_tupleargs(f,::Type{<:Tuple{}},::Tuple) = ()


# I really don't like that 0.7 got rid of the much more succinct `linspace`, so
# bring it back
linspace(start,stop,length::Integer) = range(start,stop=stop,length=length)


# returns the base parametric type with all type parameters stripped out
basetype(::Type{T}) where {T} = T.name.wrapper

# amazing Julia doesn't have this yet...
eachcol(A) = @views [A[:,i] for i=1:size(A,2)]
eachrow(A) = @views [A[i,:] for i=1:size(A,1)]
