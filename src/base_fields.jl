
#
# BaseFields are just wrappers around arrays tagged with additional
# information in the form of:
#  * the `B` parameter, for information which is needed at
#    compile-time (right now, just the basis)
#  * the `metadata` field, for information which is only needed at
#    run-time
#

abstract type Proj end
broadcastable(proj::Proj) = Ref(proj)

struct BaseField{B, M<:Proj, T, A<:AbstractArray{T}} <: Field{B, T}
    arr :: A
    metadata :: M
    function (::Type{F})(arr::A, metadata::M) where {B,M<:Proj,T,A<:AbstractArray{T},F<:BaseField{B}}
        new{B,M,T,A}(arr, metadata) :: F
    end
end

typealias_def(::Type{F}) where {B,M,T,A,F<:BaseField{B,M,T,A}} = "BaseField{$(typealias(B)),$(typealias(A)),$(typealias(M))}"


## array interface
# even though the underlying data in BaseFields is multi-dimensional,
# they act as a 1D vector with all their entries splayed out in a
# column
size(f::BaseField) = (length(f.arr),)
lastindex(f::BaseField, i::Int) = lastindex(f.arr, i)
@propagate_inbounds getindex(f::BaseField, I::Union{Int,Colon,AbstractArray}...) = getindex(f.arr, I...)
@propagate_inbounds setindex!(f::BaseField, X, I::Union{Int,Colon,AbstractArray}...) = (setindex!(f.arr, X, I...); f)
similar(f::BaseField{B}, ::Type{T}) where {B,T} = BaseField{B}(similar(f.arr, T), f.metadata)
copy(f::BaseField{B}) where {B} = BaseField{B}(copy(f.arr), f.metadata)


## promotion
function promote(f₁::BaseField{B₁}, f₂::BaseField{B₂}) where {B₁,B₂}
    get_metadata_strict(f₁.metadata, f₂.metadata) # for now just check compatible
    B = typeof(promote_basis_generic(B₁(), B₂()))
    B(f₁), B(f₂)
end


## broadcasting 

# any broadcast expression that contains at least one BaseField will
# have a broadcast style of BaseFieldStyle{S,B}. the S is the
# broadcast style for the underlying arrays and B is the B parameter
# of the result BaseField. S and B are promoted recursively across all
# the arguments according to the set of rules below. 
struct BaseFieldStyle{S,B} <: AbstractArrayStyle{1} end
BroadcastStyle(::Type{F}) where {B,M,T,A,F<:BaseField{B,M,T,A}} = 
    BaseFieldStyle{typeof(BroadcastStyle(A)),B}()
BroadcastStyle(::BaseFieldStyle{S₁,B₁}, ::BaseFieldStyle{S₂,B₂}) where {S₁,B₁,S₂,B₂} = 
    BaseFieldStyle{typeof(result_style(S₁(), S₂())), typeof(promote_basis_strict(B₁(),B₂()))}()
BroadcastStyle(S::BaseFieldStyle, ::DefaultArrayStyle{0}) = S

# with the Broadcasted object created, we now compute the answer
function materialize(bc::Broadcasted{BaseFieldStyle{S,B}}) where {S,B}

    # first, recursively go through the broadcast arguments and figure
    # out the metadata of the result, using the
    # promote_metadata_strict rules
    metadata = get_metadata_strict(bc)

    # then "preprocess" all the arguments. this unwraps all of the
    # BaseFields in the expression into just the underlying arrays,
    # and turns things which were ImplicitFields into actual arrays
    # (which are computed by dispatching on the now-known S, B, and
    # metadata of the result)
    bc′ = preprocess((BaseFieldStyle{S,B}(), metadata), bc)
    
    # the arguments of bc′ are now all normal arrays, so convert it to
    # the broadcast style S that goes along with them
    bc″ = convert(Broadcasted{S}, bc′)

    # run the normal array broadcast, and wrap in the right
    # result type
    BaseField{B}(materialize(bc″), metadata)

end

function materialize!(dst::BaseField{B}, bc::Broadcasted{BaseFieldStyle{S,B′}}) where {B,B′,S}
    
    (B == B′) || error("Can't broadcast a $(typealias(B′)) into a $(typealias(B))")

    # for inplace broadcasting, we don't need to compute B or the
    # metadata from the broadcasted object, we just take it from the
    # destination BaseField. otherwise its the same as materialize above
    bc′ = preprocess((BaseFieldStyle{S,B}(), dst.metadata), bc)
    bc″ = convert(Broadcasted{S}, bc′)
    materialize!(dst.arr, bc″)
    dst

end

# the default preprocessing, which just unwraps the underlying array.
# this doesn't dispatch on the first argument, but custom BaseFields
# are free to override this and dispatch on it if they need
preprocess(::Any, f::BaseField) = f.arr

# we re-wrap each Broadcasted object as we go through preprocessing
# because some array types do special things here (e.g. CUDA wraps
# bc.f in a CUDA.cufunc)
preprocess(dest::Tuple{BaseFieldStyle{S,B},M}, bc::Broadcasted) where {S,B,M} = 
    broadcasted(S(), bc.f, preprocess_args(dest, bc.args)...)

# recursively go through a Broadcasted object's arguments and compute
# the final metadata according to the promote_metadata_strict rules.
# we use the select_known_rule machinery (see util.jl) to make it so
# promote_metadata_strict_rule only needs one argument order defined
# (similar to Base.promote_rule)
get_metadata_strict(x, rest...)      = promote_metadata_strict(get_metadata_strict(x), get_metadata_strict(rest...))
get_metadata_strict(bc::Broadcasted) = get_metadata_strict(bc.args...)
get_metadata_strict(f ::BaseField)   = f.metadata
get_metadata_strict(  ::Any)         = nothing
get_metadata_strict()                = nothing

promote_metadata_strict(x) = x
promote_metadata_strict(x, y) = select_known_rule(promote_metadata_strict_rule, x, y)
promote_metadata_strict_rule(metadata,   ::Nothing) = metadata
promote_metadata_strict_rule(::Nothing,  ::Nothing) = nothing
promote_metadata_strict_rule(::Any,      ::Any) = Unknown()


## properties
getproperty(f::BaseField, s::Symbol)           = getproperty(f,Val(s))
getproperty(f::BaseField,  ::Val{:arr})        = getfield(f,:arr)
getproperty(f::BaseField,  ::Val{:metadata})   = getfield(f,:metadata)
getproperty(f::BaseField,  ::Val{s}) where {s} = getfield(getfield(f,:metadata),s)
propertynames(f::BaseField) = (fieldnames(typeof(f))..., fieldnames(typeof(f.metadata))...)


## CMBLensing-specific stuff
global_rng_for(::Type{BaseField{B,M,T,A}}) where {B,M,T,A} = global_rng_for(A)
fieldinfo(f::BaseField) = f # for backwards compatibility
get_storage(f::BaseField) = typeof(f.arr)
adapt_structure(to, f::BaseField{B}) where {B} = BaseField{B}(adapt(to, f.arr), adapt(to, f.metadata))
hash(f::BaseField, h::UInt64) = foldr(hash, (typeof(f), cpu(f.arr), f.metadata), init=h)

# 
default_proj(::Type{F}) where {F<:BaseField{<:Any,<:Proj}} = Base.unwrap_unionall(F).parameters[2].ub
make_field_aliases("Base", Proj)

### basis-like definitions
LenseBasis(::Type{<:BaseS0})    = Map
LenseBasis(::Type{<:BaseS2})    = QUMap
LenseBasis(::Type{<:BaseS02})   = IQUMap
DerivBasis(::Type{<:BaseS0})    = Fourier
DerivBasis(::Type{<:BaseS2})    = QUFourier
DerivBasis(::Type{<:BaseS02})   = IQUFourier
HarmonicBasis(::Type{<:BaseS0}) = Fourier
HarmonicBasis(::Type{<:BaseQU}) = QUFourier
HarmonicBasis(::Type{<:BaseEB}) = EBFourier


# useful for enumerating some cases below and in plotting
_sub_components = [
    (:BaseMap,        (:Ix=>:, :I=>:)),
    (:BaseFourier,    (:Il=>:, :I=>:)),
    (:BaseQUMap,      (:Qx=>1, :Ux=>2, :Q =>1, :U=>2, :P=>:)),
    (:BaseQUFourier,  (:Ql=>1, :Ul=>2, :Q =>1, :U=>2, :P=>:)),
    (:BaseEBMap,      (:Ex=>1, :Bx=>2, :E =>1, :B=>2, :P=>:)),
    (:BaseEBFourier,  (:El=>1, :Bl=>2, :E =>1, :B=>2, :P=>:)),
    (:BaseIQUMap,     (:Ix=>1, :Qx=>2, :Ux=>3, :I=>1, :Q=>2, :U=>3, :P=>2:3, :IP=>:)),
    (:BaseIQUFourier, (:Il=>1, :Ql=>2, :Ul=>3, :I=>1, :Q=>2, :U=>3, :P=>2:3, :IP=>:)),
    (:BaseIEBMap,     (:Ix=>1, :Ex=>2, :Bx=>3, :I=>1, :E=>2, :B=>3, :P=>2:3, :IP=>:)),
    (:BaseIEBFourier, (:Il=>1, :El=>2, :Bl=>3, :I=>1, :E=>2, :B=>3, :P=>2:3, :IP=>:)),
]

# sub-components like f.Ix, f.Q, f.P, etc...
for (F, props) in _sub_components
    for (k, I) in props
        if String(k)[end] in "xl"
            if I == (:)
                @eval getproperty(f::$F, ::Val{$(QuoteNode(k))}) = getfield(f,:arr)
            else
                @eval getproperty(f::$F, ::Val{$(QuoteNode(k))}) = view(getfield(f,:arr), pol_slice(f,$I)...)
            end
        else
            if I == (:)
                @eval getproperty(f::$F, ::Val{$(QuoteNode(k))}) = f
            else
                B = (k == :P) ? Basis2Prod{basis(@eval($F)).parameters[end-1:end]...} : basis(@eval($F)).parameters[end]
                @eval getproperty(f::$F, ::Val{$(QuoteNode(k))}) = BaseField{$B}(_reshape_batch(view(getfield(f,:arr), pol_slice(f,$I)...)), f.metadata)
            end
        end
    end
end
