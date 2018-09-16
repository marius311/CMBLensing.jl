export FieldTuple

abstract type BasisTuple{T} <: Basis end
abstract type SpinTuple{T} <: Spin end
abstract type PixTuple{T} <: Pix end

struct FieldTuple{FS<:Tuple,B<:BasisTuple,S<:SpinTuple,P<:PixTuple} <: Field{B,S,P}
    fs::FS
end
FieldTuple{FS,B,S,P}(fs::Field...) where {FS<:Tuple,B<:BasisTuple,S<:SpinTuple,P<:PixTuple} = FieldTuple{FS,B,S,P}(fs)
FieldTuple{FS,B,S,P}(ft::FieldTuple) where {FS<:Tuple,B<:BasisTuple,S<:SpinTuple,P<:PixTuple} = convert(FieldTuple{FS,B,S,P}, ft)
function FieldTuple(fs::Field...)
    B = BasisTuple{Tuple{map(basis,fs)...}}
    S = SpinTuple{Tuple{map(spin,fs)...}}
    P = PixTuple{Tuple{map(pix,fs)...}}
    FieldTuple{typeof(fs),B,S,P}(fs)
end

shortname(::Type{<:FieldTuple{FS}}) where {FS} = "FieldTuple{$(join(map(shortname,FS.parameters),","))}"

# broadcasting
broadcast_data(::Type{FT}, ft::FT) where {FS,FT<:FieldTuple{FS}} = ft.fs
broadcast_data(::Type{FT}, f::Union{Field,LinOp}) where {FS,FT<:FieldTuple{FS}} = ntuple(_->f,nfields(FS))
broadcast_data(::Type{FT}, L::FullDiagOp{FT}) where {FS,FT<:FieldTuple{FS}} = L.f.fs
@commutative promote_containertype{FT<:FieldTuple,F<:Field}(::Type{FT}, ::Type{F}) = FT
promote_containertype(::Type{FT}, ::Type{FT}) where {FT<:FieldTuple} = FT # needed for ambiguity
BroadcastStyle(::Style{F0}, ::Style{FT}) where {F0<:Field{Map,S0},FT<:FieldTuple} = Style{FT}()

# promotion / conversion
promote_rule(::Type{<:FieldTuple{FS1}},::Type{<:FieldTuple{FS2}}) where {FS1,FS2} = 
    FieldTuple{Tuple{map_tupleargs(promote_type,FS1,FS2)...}}
convert(::Type{<:FieldTuple{FS}}, ft::FieldTuple) where {FS} = 
    FieldTuple(map_tupleargs((F,f)->F(f),FS,ft.fs)...)

# basis conversion
(::Type{BasisTuple{BS}})(ft::FieldTuple) where {BS} = FieldTuple(map_tupleargs((B,f)->B(f), BS, ft.fs)...)
(::Type{B})(ft::FieldTuple) where {B<:Basis}     = FieldTuple(map(B,ft.fs)...)
(::Type{B})(ft::FieldTuple) where {B<:Basislike} = FieldTuple(map(B,ft.fs)...) # needed for ambiguity

# basic functionality
white_noise(::Type{FT}) where {FS,FT<:FieldTuple{FS}} = FT(map_tupleargs(white_noise, FS))
zero(::Type{FT}) where {FS,FT<:FieldTuple{FS}} = FT(map_tupleargs(zero, FS))
dot(a::FieldTuple, b::FieldTuple) = sum(map(dot, a.fs, b.fs))
Ac_mul_B(a::FieldTuple, b::FieldTuple) = sum(map(Ac_mul_B, a.fs, b.fs))
eltype(ft::FieldTuple) = promote_type(map(eltype,ft.fs)...)
length(ft::FieldTuple) = sum(map(length, ft.fs))
ud_grade(ft::FieldTuple, args...; kwargs...) = FieldTuple((ud_grade(f,args...;kwargs...) for f in ft)...)

# iterating
start(ft::FieldTuple) = start(ft.fs)
next(ft::FieldTuple, state) = next(ft.fs, state)
done(ft::FieldTuple, state) = done(ft.fs, state)

# indexing
getindex(ft::FieldTuple, i::Union{Int,UnitRange}) = getindex(ft.fs, i)

# automatically forwards properties to tuple fields
getproperty(ft::FT, ::Val{:fs}) where {s,FS,FT<:FieldTuple{FS}} = getfield(ft,:fs)
@generated function getproperty(ft::FT, ::Val{s}) where {s,FS,FT<:FieldTuple{FS}}
    l = filter(((i,F),)->(s in propertynames(F)), collect(enumerate(FS.parameters)))
    if (length(l)==1)
        :(getproperty(ft.fs[$(l[1][1])], $(QuoteNode(s))))
    elseif (length(l)==0)
        error("type $(shortname(FT)) has no property $s")
    else
        error("Ambiguous property. Multiple types in the FieldTuple have a property $s")
    end
end

# the generic * method only works if a & b are in the same basis, so we need this here
@commutative *(a::Field, b::FieldTuple) = a.*b
*(a::FieldTuple, b::FieldTuple) = a.*b 
