

""" 
Return the type's fields as a tuple
"""
@generated fieldvalues(x) = Expr(:tuple, (:(x.$f) for f=fieldnames(x))...)
@generated fields(x) = Expr(:tuple, (:($f=x.$f) for f=fieldnames(x))...)
firstfield(x) = first(fieldvalues(x))


"""
Rewrites `@! x = f(args...)` to `x = f!(x,args...)`

Special cases for `*` and `\\` forward to `mul!` and `ldiv!`, respectively.
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


nan2zero(x::T) where {T} = !isfinite(x) ? zero(T) : x
nan2zero(x::Diagonal{T}) where {T} = Diagonal{T}(nan2zero.(x.diag))
nan2inf(x::T) where {T} = !isfinite(x) ? T(Inf) : x


""" Return a tuple with the expression repeated n times """
macro repeated(ex,n)
    :(tuple($(repeated(esc(ex),n)...)))
end

""" 
Pack some variables in a dictionary 

```julia
> x = 3
> y = 4
> @dict x y z=>5
Dict(:x=>3,:y=>4,:z=>5)
```
"""
macro dict(exs...)
    kv(ex::Symbol) = :($(QuoteNode(ex))=>$(esc(ex)))
    kv(ex) = isexpr(ex,:call) && ex.args[1]==:(=>) ? :($(QuoteNode(ex.args[2]))=>$(esc(ex.args[3]))) : error()
    :(Dict($((kv(ex) for ex=exs)...)))
end

""" 
Pack some variables into a NamedTuple. E.g.:

```julia
> x = 3
> y = 4
> @namedtuple(x, y, z=5)
(x=3,y=4,z=5)
```
"""
macro namedtuple(exs...)
    Base.depwarn("@namedtuple(x,y) is deprecated and will be removed soon, just use the built-in Julia (;x,y) now.", nothing)
    if length(exs)==1 && isexpr(exs[1],:tuple)
        exs = exs[1].args
    end
    kv(ex::Symbol) = :($(esc(ex))=$(esc(ex)))
    kv(ex) = isexpr(ex,:(=)) ? :($(esc(ex.args[1]))=$(esc(ex.args[2]))) : error()
    Expr(:tuple, (kv(ex) for ex=exs)...)
end



# these allow pinv and sqrt of SMatrices of Diagonals to work correctly, which
# we use for the T-E block of the covariance. hopefully some of this can be cut
# down on in the futue with some PRs into StaticArrays.
permutedims(A::SMatrix{2,2}) = @SMatrix[A[1] A[3]; A[2] A[4]]
function sqrt(A::SMatrix{2,2,<:Diagonal})
    a,b,c,d = A
    s = @. sqrt(a*d-b*c)
    t = pinv(@. sqrt(a+d+2s))
    @SMatrix[t*(a+s) t*b; t*c t*(d+s)]
end
function pinv(A::SMatrix{2,2,<:Diagonal})
    a,b,c,d = A
    idet = pinv(@. a*d-b*c)
    @SMatrix[d*idet -(b*idet); -(c*idet) a*idet]
end


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
@generated function basetype(t::UnionAll)
    unwrap_expr(s::UnionAll, t=:t) = unwrap_expr(s.body, :($t.body))
    unwrap_expr(::DataType, t) = t
    :($(unwrap_expr(t.parameters[1])).name.wrapper)
end


function ensuresame(args...)
    @assert all(args .== Ref(args[1]))
    args[1]
end


tuple_type_len(::Type{<:NTuple{N,Any}}) where {N} = N


ensure1d(x::Union{Tuple,AbstractArray}) = x
ensure1d(x) = (x,)


# see
# https://discourse.julialang.org/t/dispatching-on-the-result-of-unwrap-unionall-seems-weird/25677
# for why we need this to use, just decorate the custom show_datatype
# with it, and make sure the arg is named `t`
function typealias(t)
    if isconcretetype(t)
        ta = typealias_def(t)
        if !isnothing(ta)
            return ta
        end
    end
    sprint(io -> invoke(Base.show_datatype, Tuple{IO,DataType}, io, t))
end
typealias_def(t) = nothing



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
    @ondemand(Package.function)(args...; kwargs...)
    @ondemand(Package.Submodule.function)(args...; kwargs...)

Just like calling `Package.function` or `Package.Submodule.function`, but
`Package` will be loaded on-demand if it is not already loaded. The call is no
longer inferrable.
"""
macro ondemand(ex)
    get_root_package(x) = @capture(x, a_.b_) ? get_root_package(a) : x
    quote
        @eval import $(get_root_package(ex))
        (args...; kwargs...) -> Base.invokelatest($(esc(ex)), args...; kwargs...)
    end
end



get_kwarg_names(func::Function) = Vector{Symbol}(Base.kwarg_decl(first(methods(func))))

# maps a function recursively across all arguments of a Broadcasted expression,
# using the function `broadcasted` to reconstruct the `Broadcasted` object at
# each point.
map_bc_args(f, bc::Broadcasted) = broadcasted(bc.f, map(arg->map_bc_args(f, arg), bc.args)...)
map_bc_args(f, arg) = f(arg)


# adapting a closure adapts the captured variables
# this could probably be a PR into Adapt.jl
@generated function adapt_structure(to, f::F) where {F<:Function}
    if fieldcount(F) == 0
        :f
    else
        quote
            captured_vars = $(Expr(:tuple, (:(adapt(to, f.$x)) for x=fieldnames(F))...))
            $(Expr(:new, :($(F.name.wrapper){map(typeof,captured_vars)...}), (:(captured_vars[$i]) for i=1:fieldcount(F))...))
        end
    end
end

adapt_structure(to, d::Dict) = Dict(k => adapt(to, v) for (k,v) in d)

@doc doc"""

    cpu(xs)

Recursively move an object to CPU memory (i.e. the opposite of `cu`)
"""
cpu(xs) = adapt_structure(Array, xs)


function corrify(H)
    σ = sqrt.(abs.(diag(H)))
    for i=1:checksquare(H)
        H[i,:] ./= σ
        H[:,i] ./= σ
    end
    H
end



struct FailedPyimport
    err
end
getproperty(p::FailedPyimport, ::Symbol) = throw(getfield(p,:err))

@doc doc"""

    safe_pyimport(s)

Like `pyimport`, but if `s` fails to import, instead of an error right away, the
error will be thrown the first time the user tries to access the contents of the
module.
"""
function safe_pyimport(s)
    try
        @ondemand(PyCall.pyimport)(s)
    catch err
        FailedPyimport(err)
    end
end


@doc doc"""
    @ismain()
    
Return true if the current file is being run as a script.
"""
macro ismain()
    (__source__ != nothing) && (String(__source__.file) == abspath(PROGRAM_FILE))
end


@doc doc"""
    seed_for_storage!(storage[, seed])
    seed_for_storage!((storage1, storage2, ...)[, seed])
    
Set the global random seed for the RNG which controls `storage`-type. 
"""
seed_for_storage!(::Type{<:Array}, seed=nothing) = 
    Random.seed!((seed == nothing ? () : (seed,))...)
seed_for_storage!(storage::Any, seed=nothing) = 
    error("Don't know how to set seed for storage=$storage")
seed_for_storage!(storages::Tuple, seed=nothing) = 
    seed_for_storage!.(storages, seed)



@init @require MPIClusterManagers="e7922434-ae4b-11e9-05c5-9780451d2c66" begin

    using .MPIClusterManagers: MPI, start_main_loop, TCP_TRANSPORT_ALL, MPI_TRANSPORT_ALL

    """
    init_MPI_workers()

    Initialize MPI processes as Julia workers. Should be called from all MPI
    processes, and will only return on the master process. 

    `transport` should be `"MPI"` or `"TCP"`, which is by default read from the
    environment variable `JULIA_MPI_TRANSPORT`, and otherwise defaults to `"TCP"`.

    If CUDA is loaded and functional in the Main module, additionally calls
    [`assign_GPU_workers()`](@ref)
    """
    function init_MPI_workers(;
        stdout_to_master = false, 
        stderr_to_master = false,
        transport = get(ENV,"JULIA_MPI_TRANSPORT","TCP")
    )
        
        if !MPI.Initialized()
            MPI.Init()
        end
        size = MPI.Comm_size(MPI.COMM_WORLD)
        rank = MPI.Comm_rank(MPI.COMM_WORLD)

        if size>1
            # workers don't return from this call:
            start_main_loop(
                Dict("TCP"=>TCP_TRANSPORT_ALL,"MPI"=>MPI_TRANSPORT_ALL)[transport],
                stdout_to_master=stdout_to_master,
                stderr_to_master=stderr_to_master
            )
            
            if @isdefined(CUDA) && CUDA.functional()
                assign_GPU_workers()
            end
            @everywhere begin
                typ = (myid()==1) ? "(master)" : "(worker)"
                dev = (@isdefined(CUDA) && CUDA.functional()) ? device() : "CPU"
                @info "MPI process $(myid()) $typ is running on $(gethostname())::$dev"
            end
        end

    end

    
end


firsthalf(x) = x[1:end÷2]
lasthalf(x) = x[end÷2:end]


function sum_kbn(A::AbstractArray{T,N}; dims=:) where {T,N}
    if (dims == (:)) || (N == length(dims))
        KahanSummation.sum_kbn(adapt(Array,A))
    else
        dropdims(mapslices(sum_kbn, adapt(Array,A), dims=dims), dims=dims) :: Array{T,N-length(dims)}
    end
end
@adjoint sum_kbn(A) = sum_kbn(A), Δ -> (fill!(similar(A),Δ),)


# courtesy of Takafumi Arakaki
versionof(pkg::Module) = Pkg.dependencies()[Base.PkgId(pkg).uuid].version

# for mixed eltype, which Loess stupidly does not support
Loess.loess(x::AbstractVector, y::AbstractVector; kwargs...) = 
    loess(collect.(zip(promote.(x,y)...))...; kwargs...)


expnorm(x) = exp.(x .- maximum(x))


"""

    @⌛ code ...
    @⌛ function_definition() = .... 

Label a section of code to be timed. The first form uses the code
itselfs as a label, the second uses the function name, and its the
body of the function which is timed. 

To run the timer and print output, returning the result of the
calculation, use

    @show⌛ run_code()

Timing uses `TimerOutputs.get_defaulttimer()`. 
"""
macro ⌛(ex)
    source_str = last(splitpath(string(__source__.file)))*":"*string(__source__.line)
    if isdef(ex)
        sdef = splitdef(ex)
        sdef[:body] = quote
            CMBLensing.@timeit $("$(string(sdef[:name]))(…)  ($source_str)") $(sdef[:body])
        end
        esc(combinedef(sdef))
    else
        :(@timeit $("$(Base._truncate_at_width_or_chars(string(prewalk(rmlines,ex)),26))  ($source_str)") $(esc(ex)))
    end
end


"""
See [`@⌛`](@ref)
"""
macro show⌛(ex)
    quote
        reset_timer!(get_defaulttimer())
        result = $(esc(ex))
        show(get_defaulttimer())
        result
    end
end


drop_tail_singleton_dims(x::AbstractArray{T,N}) where {T,N} = size(x,N)==1 ? drop_tail_singleton_dims(dropdims(x,dims=N)) : x
drop_tail_singleton_dims(x::AbstractArray{T,0}) where {T} = x
