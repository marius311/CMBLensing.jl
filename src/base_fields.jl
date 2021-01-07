
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
    function (::Type{F})(arr::A, metadata::M) where {B,M,T,A<:AbstractArray{T},F<:BaseField{B}}
        (metadata isa AbstractArray) && error("Array-type metadata is disallowed to avoid ambiguities.")
        new{B,M,T,A}(arr, metadata)
    end
end


## array interface
# even though the underlying data in BaseFields is multi-dimensional,
# they act as a 1D vector with all their entries splayed out in a
# column
size(f::BaseField) = (length(f.arr),)
lastindex(f::BaseField, i::Int) = lastindex(f.arr, i)
@propagate_inbounds getindex(f::BaseField, I::Union{Int,Colon,AbstractArray}...) = getindex(f.arr, I...)
@propagate_inbounds setindex!(f::BaseField, X, I::Union{Int,Colon,AbstractArray}...) = (setindex!(f.arr, X, I...); f)
similar(f::BaseField{B}, ::Type{T}) where {B,T} = BaseField{B}(similar(f.arr, T), f.metadata)
copyto!(dst::BaseField, src::BaseField) = (copyto!(dst.arr, src.arr); dst)


## promotion
function promote(f₁::BaseField, f₂::BaseField)
    b, metadata = promote_generic(get_b_metadata(f₁), get_b_metadata(f₂))
    B = typeof(b)
    B(convert_metadata(metadata,f₁)), B(convert_metadata(metadata,f₂))
end
convert_metadata(metadata, f::BaseField) = f.metadata === metadata ? f : error("need promotion rule")


## broadcasting 

# this makes it so the broadcast style of any expression which
# involves at least one BaseField will end up as BaseFieldStyle{S}.
# the S carries around what the BroadcastStyle would have been for the
# underlying arrays, and is promoted as per the usual rules by the
# result_style call below
struct BaseFieldStyle{S} <: AbstractArrayStyle{1} end
BroadcastStyle(::Type{F}) where {B,M,T,A,F<:BaseField{B,M,T,A}} = BaseFieldStyle{typeof(BroadcastStyle(A))}()
BroadcastStyle(::BaseFieldStyle{S₁}, ::BaseFieldStyle{S₂}) where {S₁,S₂} = BaseFieldStyle{typeof(result_style(S₁(), S₂()))}()
BroadcastStyle(S::BaseFieldStyle, ::DefaultArrayStyle{0}) = S

# this describes how to compute the broadcast across anything that ends
# up as BaseFieldStyle
function materialize(bc::Broadcasted{BaseFieldStyle{S}}) where {S}

    # recursively go through the broadcast expression and figure out
    # the final `B` and `metadata` of the result, using the
    # promote_bcast_rule rules
    b, metadata = promote_fields_bcast(bc)
    B = typeof(b)

    # "preprocess" all the arguments, which unwraps all of the
    # BaseFields in the expression to just the underlying arrays.
    # `preprocess` can dispatch on the now-known (B, metadata)
    bc′ = preprocess((B, metadata), bc)
    
    # convert the expression to style S, which is what it would have
    # been for the equivalent broadcast over the underlying arrays
    bc″ = convert(Broadcasted{S}, bc′)

    # run the normal array broadcast, and wrap in the appropriate
    # result type
    BaseField{B}(materialize(bc″), metadata)

end

function materialize!(dst::BaseField{B}, bc::Broadcasted{BaseFieldStyle{S}}) where {B,S}
    
    # for inplace broadcasting, we don't need to compute the
    # (B,metadata) from the broadcasted object, we just take it from
    # the destination BaseField
    bc′ = preprocess((B, dst.metadata), bc)
    
    # same as before
    bc″ = convert(Broadcasted{S}, bc′)
    materialize!(dst.arr, bc″)
    
    # return field itself
    dst

end

# the default preprocessing, which just unwraps the underlying array.
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
promote_fields_bcast(x,               rest...) = promote_bcast(get_b_metadata(x),                promote_fields_bcast(rest...))
promote_fields_bcast(bc::Broadcasted, rest...) = promote_bcast(promote_fields_bcast(bc.args...), promote_fields_bcast(rest...))
promote_fields_bcast(x)                        = get_b_metadata(x)
promote_fields_bcast()                         = nothing

# get the (b,metadata) out of the argument, if it has one
get_b_metadata(::Any)                     = nothing
get_b_metadata(x::BaseField{B}) where {B} = (B(), x.metadata)

# like promote_bcast_rule, but checks both argument orders, to allow
# the user to only define one order. if both argument orders defined,
# they should be the same (although this is not checked for
# performance reasons), so just pick either. if only one is defined,
# pick that one. otherwise the user would need to specify a rule
promote_bcast(x, y) = _select_known_rule(x, y, promote_bcast_rule(x, y), promote_bcast_rule(y, x))
_select_known_rule(x, y, R::Any,      ::Any)     = R
_select_known_rule(x, y, R::Any,      ::Unknown) = R
_select_known_rule(x, y,  ::Unknown, R::Any)     = R
_select_known_rule(x, y,  ::Unknown,  ::Unknown) = error("need rule")

# default rules for promoting the (b, metadata) from all the arguments
# in the broadcasted expression. only one order needs to be defined.
function promote_bcast_rule((b₁,metadata₁)::Tuple, (b₂,metadata₂)::Tuple)
    if (
        b₁ === b₂ && (
            metadata₁ === metadata₂ || # `===` is much faster so try to short-circuit to that before checking `==`
            metadata₁ == metadata₂
        )
    )
        (b₁, metadata₁)
    else
        # if the b and metadata aren't identical, the individual field
        # types need to specify how to combine them. see e.g.
        # flat_proj.jl
        Unknown()
    end
end
promote_bcast_rule((b,metadata)::Tuple, ::Nothing) = (b, metadata)
promote_bcast_rule(::Nothing,           ::Nothing) = nothing
promote_bcast_rule(::Any,               ::Any)     = Unknown()


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

## other CMBLensing-specific
global_rng_for(::Type{BaseField{B,M,T,A}}) where {B,M,T,A} = global_rng_for(A)
fieldinfo(f::BaseField) = f
get_storage(f::BaseField) = typeof(f.arr)
adapt_structure(to, f::BaseField{B}) where {B} = BaseField{B}(adapt(to, f.arr), adapt(to, f.metadata))