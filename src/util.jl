
import Base: ==
==(x,y,z,ws...) = x==y && ==(y,z,ws...)


""" 
Return the type's fields as a tuple
"""
@generated fieldvalues(x) = Expr(:tuple, (:(x.$f) for f=fieldnames(x))...)
@generated fields(x) = Expr(:tuple, (:($f=x.$f) for f=fieldnames(x))...)
firstfield(x) = first(fieldvalues(x))



"""
@! x = f(args...) is equivalent to x = f!(x,args...)
"""
macro !(ex)
    if @capture(ex, x_ = f_(args__; kwargs_...))
        esc(:($(Symbol(string(f,"!")))($x,$(args...); $kwargs...)))
    elseif @capture(ex, x_ = f_(args__))
        if f == :*
            f = :mul
        elseif f==:\
            f = :ldiv
        end
        esc(:($x = $(Symbol(string(f,"!")))($x,$(args...))::typeof($x))) # ::typeof part helps inference sometimes
    else
        error("Usage: @! x = f(...)")
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
Pack some variables into a NamedTuple

```
> x = 3
> y = 4
> @namedtuple(x, y, z=5)
(x=3,y=4,z=5)
```
"""
macro namedtuple(exs...)
    kv(ex::Symbol) = :($(esc(ex))=$(esc(ex)))
    kv(ex) = isexpr(ex,:(=)) ? :($(esc(ex.args[1]))=$(esc(ex.args[2]))) : error()
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
import Base: sqrt, inv, /, permutedims
arithmetic_closure(::Type{Diagonal{T}}) where {T} = Diagonal{arithmetic_closure(T)}
/(a::Number, b::Diagonal) = Diagonal(a ./ diag(b))
permutedims(A::SMatrix{2,2}) = @SMatrix[A[1] A[3]; A[2] A[4]]


# some usefule tuple manipulation functions:

# see: https://discourse.julialang.org/t/efficient-tuple-concatenation/5398/10
# and https://github.com/JuliaLang/julia/issues/27988
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)

# see https://discourse.julialang.org/t/any-way-to-make-this-one-liner-type-stable/10636/2
using Base: tuple_type_cons, tuple_type_head, tuple_type_tail, first, tail
map_tupleargs(f,::Type{T}) where {T<:Tuple} = 
    (f(tuple_type_head(T)), map_tupleargs(f,tuple_type_tail(T))...)
map_tupleargs(f,::Type{T},::Type{S}) where {T<:Tuple,S<:Tuple} = 
    (f(tuple_type_head(T),tuple_type_head(S)), map_tupleargs(f,tuple_type_tail(T),tuple_type_tail(S))...)
map_tupleargs(f,::Type{T},s::Tuple) where {T<:Tuple} = 
    (f(tuple_type_head(T),first(s)), map_tupleargs(f,tuple_type_tail(T),tail(s))...)
map_tupleargs(f,::Type{<:Tuple{}}...) = ()
map_tupleargs(f,::Type{<:Tuple{}},::Tuple) = ()


# returns the base parametric type with all type parameters stripped out
basetype(::Type{T}) where {T} = T.name.wrapper




function ensuresame(args...)
    @assert all(args .== Ref(args[1]))
    args[1]
end


"""
Can be used to safely get method parameters which may not be defined due to e.g.:
https://discourse.julialang.org/t/dispatching-on-the-result-of-unwrap-unionall-seems-weird/25677
"""
macro safe_get(ex)
    head(x) = x isa Symbol ? x : head(x.args[1])
    x = head(ex)
    :($(Expr(:isdefined, esc(x))) ? $(esc(ex)) : $(QuoteNode(x)))
end


tuple_type_len(::Type{<:NTuple{N,Any}}) where {N} = N


ensure1d(x::Union{Tuple,AbstractArray}) = x
ensure1d(x) = (x,)


# see https://discourse.julialang.org/t/dispatching-on-the-result-of-unwrap-unionall-seems-weird/25677
# for why we need this
# to use, just decorate the custom show_datatype with it, and make sure the args
# are named `io` and `t`.
macro show_datatype(ex)
    def = splitdef(ex)
    def[:body] = quote
        isconcretetype(t) ? $(def[:body]) : invoke(Base.show_datatype, Tuple{IO,DataType}, io, t)
    end
    esc(combinedef(def))
end



"""
    # symmetric in any of its final arguments except for bar:
    @sym_memo foo(bar, @sym(args...)) = <body> 
    # symmetric in (i,j), but not baz
    @sym_memo foo(baz, @sym(i, j)) = <body> 
    
The `@sym_memo` macro should be applied to a definition of a function
which is symmetric in some of its arguments. The arguments in which its
symmetric are specified by being wrapping them in @sym, and they must come at
the very end. The resulting function will be memoized and permutations of the
arguments which are equal due to symmetry will only be computed once.
"""
macro sym_memo(funcdef)
    
    
    sfuncdef = splitdef(funcdef)
    
    asymargs = sfuncdef[:args][1:end-1]
    symargs = collect(@match sfuncdef[:args][end] begin
        Expr(:macrocall, [head, _, ex...]), if head==Symbol("@sym") end => ex
        _ => error("final argument(s) should be marked @sym")
    end)
    sfuncdef[:args] = [asymargs..., symargs...]
    
    sfuncdef[:body] = quote
        symargs = [$(symargs...)]
        sorted_symargs = sort(symargs)
        if symargs==sorted_symargs
            $((sfuncdef[:body]))
        else
            $(sfuncdef[:name])($(asymargs...), sorted_symargs...)
        end
    end
    
    esc(:(@memoize $(combinedef(sfuncdef))))
    
end


@doc doc"""
```
@subst sum(x*$(y+1) for x=1:2)
```
    
becomes

```
let tmp=(y+1)
    sum(x*tmp for x=1:2)
end
```

to aid in writing clear/succinct code that doesn't recompute things
unnecessarily.
"""
macro subst(ex)
    
    subs = []
    ex = postwalk(ex) do x
        if isexpr(x, Symbol(raw"$"))
            var = gensym()
            push!(subs, :($(esc(var))=$(esc(x.args[1]))))
            var
        else
            x
        end
    end
    
    quote
        let $(subs...)
            $(esc(ex))
        end
    end

end


"""
    @invokelatest expr...
    
Rewrites all non-broadcasted function calls anywhere within an expression to use
Base.invokelatest. This means functions can be called that have a newer world
age, at the price of making things non-inferrable.
"""
macro invokelatest(ex)
    function walk(x)
        if isdef(x)
            x.args[2:end] .= map(walk, x.args[2:end])
            x
        elseif @capture(x, f_(args__; kwargs__)) && !startswith(string(f),'.')
            :(Base.invokelatest($f, $(map(walk,args)...); $(map(walk,kwargs)...)))
        elseif @capture(x, f_(args__)) && !startswith(string(f),'.')
            :(Base.invokelatest($f, $(map(walk,args)...)))
        elseif isexpr(x)
            x.args .= map(walk, x.args)
            x
        else
            x
        end
    end
    esc(walk(ex))
end


"""
    @ondemand(Package.function)(args...; kwargs...)
    @ondemand(Package.Submodule.function)(args...; kwargs...)

Calls Package.function or Package.Submodule.function, but Package will be loaded
on-demand if it is not already loaded. The call is not inferrable.
"""
macro ondemand(ex)
    getmod(x) = @capture(x, Mod_.func_) ? getmod(Mod) : x
    Mod = getmod(ex)
    quote
        @eval import $Mod
        (args...; kwargs...) -> Base.invokelatest($(esc(ex)), args...; kwargs...)
    end
end
