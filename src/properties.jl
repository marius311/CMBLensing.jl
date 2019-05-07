# note, this file must be included last due to the use of @generated functions,
# but hopefully eventually get rid of them / simplify the whole thing


## properties

# this allows you write eg f.Tl even when f::FlatS0Map, and it automatically
# first converts f to FlatS0Fourier *then* takes the Tl field (this is
# type-stable and has no overhead otherwise)

# under the hood, the code guesses what the given field could be converted to by
# finding other Field types that differ only in basis, scans through the
# fieldnames of those types, then converts to the appropriate one


# gets other concrete fields types which share the same spin and pixelzation,
# differing only in basis, meaning you should be able to convert F to these
# types
concrete_subtypes(::Type{F}) where {F} = isabstracttype(F) ? vcat(map(concrete_subtypes,subtypes(F))...) : F
convertable_fields(::Type{F}) where {B,S,P,F<:Field{B,S,P}} = filter(x->(!(x<:AdjField) || !isstructtype(x)),concrete_subtypes(Field{B′,S,P} where B′))

# use convertable_fields to get possible properties
propertynames(::Type{F}) where {F<:Field} = unique(tuplejoin(fieldnames(F), map(fieldnames,convertable_fields(F))...))
propertynames(::F) where {F<:Field} = propertynames(F)

# implement getproperty using possible conversions
getproperty(f::Field, s::Symbol) = getproperty(f,Val(s))
@generated function getproperty(f::F,::Val{s}) where {F<:Field, s}
    l = filter(F′->(s in fieldnames(F′)), convertable_fields(F))
    if s in fieldnames(F)
        :(getfield(f,s))
    elseif (length(l)==1)
        :(getfield($(l[1])(f),s))
    elseif (length(l)==0)
        error("type $F has no property $s")
    else
        error("Ambiguous property. Multiple types that $F could be converted to have a field $s: $l")
    end
end
function getindex(f::Field, s::Symbol)
    Base.depwarn("Syntax: f[:$s] is deprecated. Use f.$s or getproperty(f,:$s) instead.", "getindex")
    getproperty(f,Val(s))
end
