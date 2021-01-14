

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


nan2zero(x::T) where {T} = isfinite(x) ? x : zero(T)
@adjoint nan2zero(x::T) where {T} = nan2zero(x), Δ -> (isfinite(x) ? Δ : zero(T),)


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


# https://discourse.julialang.org/t/dispatching-on-the-result-of-unwrap-unionall-seems-weird/25677
# has some background related to this function. we can simplify this in 1.6
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


# adapting a closure adapts the captured variables. this is textbook
# type-piracy and prone to infinite loops if we have self-referential
# objects... but its pretty handy
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

    cpu(x)

Recursively move an object to CPU memory. See also [`gpu`](@ref).
"""
cpu(x) = adapt_structure(Array, x)

@doc doc"""

    gpu(x)

Recursively move an object to GPU memory. Note that, unlike `cu(x)`,
this does not change the `eltype` of any underlying arrays. See also
[`cpu`](@ref).
"""
function gpu end # defined in gpu.jl only when CUDA.jl is loaded



function corrify(H)
    σ = sqrt.(abs.(diag(H)))
    for i=1:checksquare(H)
        H[i,:] ./= σ
        H[:,i] ./= σ
    end
    H
end



struct LazyPyImport
    pkg
end
getproperty(p::LazyPyImport, s::Symbol) = getproperty(@ondemand(PyCall.pyimport)(pkg), s)

@doc doc"""

    lazy_pyimport(s)

Like `pyimport(s)`, but doesn't actually load anything (not even
PyCall) until a property of the returned module is accessed, allowing
this to go in `__init__` and still delay loading PyCall, as well as
preventing a Julia module load error if a Python module failed to load.
"""
lazy_pyimport(s) = LazyPyImport(s)


@doc doc"""
    @ismain()
    
Return true if the current file is being run as a script.
"""
macro ismain()
    (__source__ != nothing) && (String(__source__.file) == abspath(PROGRAM_FILE))
end


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

USE_SUM_KBN = true
use_sum_kbn!(flag) = USE_SUM_KBN=flag

# type-stable combination of summing and dropping dims, which uses
# either sum or sum_kbn (to reduce roundoff error), depending on
# CMBLensing.USE_SUM_KBN constant
function sum_dropdims(A::AbstractArray{T,N}; dims=:) where {T,N}
    if (dims == (:)) || (N == length(dims))
        if USE_SUM_KBN
            sum_kbn(cpu(A))
        else
            sum(A)
        end :: T
    else
        if USE_SUM_KBN
            dropdims(mapslices(sum_kbn, cpu(A), dims=dims), dims=dims) 
        else
            dropdims(mapslices(sum,         A,  dims=dims), dims=dims)
        end :: Array{T,N-length(dims)}
    end
end
@adjoint sum_dropdims(A) = sum_dropdims(A), Δ -> (fill!(similar(A),Δ),)


# courtesy of Takafumi Arakaki
versionof(pkg::Module) = Pkg.dependencies()[Base.PkgId(pkg).uuid].version

# for mixed eltype, which Loess stupidly does not support
Loess.loess(x::AbstractVector, y::AbstractVector; kwargs...) = 
    loess(collect.(zip(promote.(x,y)...))...; kwargs...)


expnorm(x) = exp.(x .- maximum(x))



# MacroTool's is broken https://github.com/FluxML/MacroTools.jl/issues/154
_isdef(ex) = @capture(ex, function f_(arg__) body_ end)

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
    if _isdef(ex)
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



# used in a couple of places to create a Base.promote_rule-like system
# where you can specify a set of rules for promotion via dispatch but
# don't need to write a method for both orders
select_known_rule(rule, x, y) = select_known_rule(rule, x, y, rule(x,y), rule(y,x))
select_known_rule(rule, x, y, R₁::Any,       R₂::Unknown) = R₁
select_known_rule(rule, x, y, R₁::Unknown,   R₂::Any)     = R₂
select_known_rule(rule, x, y, R₁::Any,       R₂::Any)     = (R₁ == R₂) ? R₁ : error("Conflicting rules.")
select_known_rule(rule, x, y, R₁::Unknown,   R₂::Unknown) = unknown_rule_error(rule, x, y)
