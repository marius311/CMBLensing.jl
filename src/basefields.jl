
#
# BaseFields are just wrappers around arrays tagged with additional
# information in the form of:
#  * the `B` parameter, for information which is needed at
#    compile-time (right now, just the basis)
#  * the `metadata` field, for information which is only needed at
#    run-time
#
struct BaseField{B, M, T, A<:AbstractArray{T}} <: Field{B, T}
    arr :: A
    metadata :: M
    (::Type{F})(arr::A, metadata::M) where {B,M,T,A<:AbstractArray{T},F<:BaseField{B}} = new{B,M,T,A}(arr, metadata)
end

## array interface
# even though the underlying data in BaseFields is multi-dimensional,
# they act as a 1D vector with all their entries splayed out in a
# column
size(f::BaseField) = (length(f.arr),)
lastindex(f::BaseField, i::Int) = lastindex(f.arr, i)
@propagate_inbounds getindex(f::BaseField, I...) = getindex(f.arr, I...)
@propagate_inbounds setindex!(f::BaseField, X, I...) = (setindex!(f.arr, X, I...); f)
similar(f::BaseField{B}, ::Type{T}) where {B,T} = BaseField{B}(similar(f.arr, T), f.metadata)


## broadcasting 

# the broadcast style of any expression which involves at least one
# BaseField will end up as ArrayStyle{BaseField}
BroadcastStyle(::Type{F}) where {F<:BaseField} = ArrayStyle{BaseField}()

function materialize(bc::Broadcasted{ArrayStyle{F}}) where {F<:BaseField}

    # recursively go through the broadcast expression and figure out
    # the final `B` and `metadata` of the result, using the
    # promote_b_metadata rules
    b, metadata = @⌛ promote_fields_b_metadata(bc)
    B = typeof(b)

    # "preprocess" all the arguments, which unwraps all of the
    # BaseFields in the expression to just the underlying arrays.
    # `preprocess` can dispatch on the now-known (B, metadata)
    bc′ = @⌛ preprocess((B, metadata), bc)
    
    # since the arguments are all now just arrays, convert to default
    # array broadcasting style. it works below to assume
    # dimensionality of 4, which is the maximum if you had (Nx, Ny,
    # Nspin, Nbatch)
    bc″ = @⌛ convert(Broadcasted{DefaultArrayStyle{4}}, bc′)

    # compute the broadcast, and wrap in the appropriate result type
    @⌛ BaseField{B}(materialize(bc″), metadata)

end

function materialize!(dst::BaseField{B}, bc::Broadcasted{ArrayStyle{F}}) where {B, F<:BaseField}
    
    # for inplace broadcasting, we don't need to compute the
    # (B,metadata) from the broadcasted object, we just take it from
    # the destination BaseField
    bc′ = preprocess((B, dst.metadata), bc)
    
    # same as before
    bc″ = convert(Broadcasted{DefaultArrayStyle{ndims(dst.arr)}}, bc′)
    materialize!(dst.arr, bc″)
    
    # return field itself
    dst

end

# the default preprocessing which just unwraps the underlying array.
# this doesn't use the known (B,metadata) in the first argument, but
# custom BaseFields are free to override this and dispatch on it if
# they need
preprocess(::Any, f::BaseField) = f.arr

# go through some arguments and promote down to a single (B,
# metadata), recursively going into Broadcasted objects. the thing we
# return is actually (B(), metadata), ie an *instance* of B, rather
# than the type, otherwise we would lose the information from the type
# system. more explicitly, if we returned (B, metadata), the return
# type would be inferred as eg Tuple{DataType, M}, whereas this way
# its Tuple{Map, M}, Tuple{Fourier, M}, etc...
promote_fields_b_metadata(x,               rest...) = _promote_b_metadata(get_b_metadata(x),                     promote_fields_b_metadata(rest...))
promote_fields_b_metadata(bc::Broadcasted, rest...) = _promote_b_metadata(promote_fields_b_metadata(bc.args...), promote_fields_b_metadata(rest...))
promote_fields_b_metadata(x)                        = get_b_metadata(x)
promote_fields_b_metadata()                         = nothing

# get the (b,metadata) out of the argument, if it has one
get_b_metadata(::Any)                     = nothing
get_b_metadata(x::BaseField{B}) where {B} = (B(), x.metadata)

# like promote_b_metadata, but checks both argument orders, to allow
# the user to only define one order
_promote_b_metadata(x, y) = _select_known_rule(promote_b_metadata(x, y), promote_b_metadata(y, x))
# if both argument orders defined, they should be the same (although
# this is not checked for performance reasons), so just pick either 
_select_known_rule(x::Any,      ::Any)     = x
# if only one is defined, pick that one
_select_known_rule(x::Any,      ::Unknown) = x
_select_known_rule( ::Unknown, x::Any)     = x
# otherwise the user would need to specify a rule
function _select_known_rule(::Unknown, ::Unknown)
    error("Unknown")
end

# default promotion rules for (b, metadata). if only one argument was
# a BaseField, then return the (b, metadata) for that one. if they're
# identical its fine to return either. otherwise, the user needs to
# specify a rule or its an error.
function promote_b_metadata((b,metadata)::Tuple, (b′,metadata′)::Tuple)
    if b===b′ && (metadata===metadata′ || metadata==metadata′)
        (b, metadata)
    else
        Unknown()
    end
end
promote_b_metadata((b,metadata)::Tuple, ::Nothing) = (b, metadata)
promote_b_metadata(::Any, ::Any) = Unknown()


## properties
getproperty(f::BaseField, s::Symbol) = getproperty(f,Val(s))
function getproperty(f::BaseField, ::Val{s}) where {s}
    if hasfield(typeof(getfield(f,:metadata)),s) && !hasfield(typeof(f),s)
        getfield(getfield(f,:metadata),s)
    else
        getfield(f,s)
    end
end
propertynames(f::BaseField) = (fieldnames(typeof(f))..., fieldnames(typeof(f.metadata))...)