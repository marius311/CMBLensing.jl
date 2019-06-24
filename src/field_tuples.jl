

abstract type BasisTuple{T} <: Basis end

## FieldTuple type 
# a thin wrapper around a NamedTuple which additionally forwards all
# broadcasts one level deeper
struct FieldTuple{B<:Basis,FS<:NamedTuple,T} <: Field{B,Spin,Pix,T}
    fs::FS
end
FieldTuple(;kwargs...) = FieldTuple((;kwargs...))
FieldTuple(fs::NamedTuple) = FieldTuple{BasisTuple{Tuple{map(basis,values(fs))...}}}(fs)
FieldTuple{B}(fs::FS) where {B, FS<:NamedTuple} = FieldTuple{B,FS,ensuresame(map(eltype,values(fs))...)}(fs)
(::Type{FT})(;kwargs...) where {B,FT<:FieldTuple{B}} = FieldTuple{B}((;kwargs...))::FT
(::Type{FT})(ft::FieldTuple) where {B,FT<:FieldTuple{B}} = FieldTuple{B}(ft.fs)::FT



## printing
getindex(f::FieldTuple,::Colon) = vcat(values(f.fs)...)[:]
show(io::IO, ::Type{<:FieldTuple{B,<:NamedTuple{Names},T}}) where {B,Names,T} =
    print(io, "FieldTuple{$(Names), $(B.name.name), $T}")
# todo: let Atom display individual components in drop-down


## array interface
size(f::FieldTuple) = (sum(map(length, f.fs)),)
copyto!(dest::FT, src::FT) where {FT<:FieldTuple} = (map(copyto!,dest.fs,src.fs); dest)
similar(f::FT) where {FT<:FieldTuple} = FT(map(similar,f.fs))
similar(::Type{FT},::Type{T}) where {T,B,Names,FS,FT<:FieldTuple{B,<:NamedTuple{Names,FS}}} = 
    FieldTuple{B}(NamedTuple{Names}(map_tupleargs(F->similar(F,T), FS)))

## broadcasting
@propagate_inbounds @inline getindex(f::FieldTuple, i) = getindex(f.fs, i)
broadcastable(f::FieldTuple) = f
BroadcastStyle(::Type{FT}) where {FT<:FieldTuple} = ArrayStyle{FT}()
BroadcastStyle(::ArrayStyle{FT}, ::DefaultArrayStyle{0}) where {FT<:FieldTuple} = ArrayStyle{FT}()
BroadcastStyle(::ArrayStyle{FT}, ::DefaultArrayStyle{1}) where {FT<:FieldTuple} = ArrayStyle{FT}()
BroadcastStyle(::ArrayStyle{FT}, ::Style{Tuple}) where {FT<:FieldTuple} = ArrayStyle{FT}()
instantiate(bc::Broadcasted{<:ArrayStyle{<:FieldTuple}}) = bc
fieldtuple_data(f::FieldTuple) = values(f.fs)
fieldtuple_data(f::Field) = (f,)
fieldtuple_data(x) = x
similar(bc::Broadcasted{ArrayStyle{FT}}, ::Type{T}) where {T, FT<:FieldTuple} = similar(FT,T)
function copyto!(dest::FT, bc::Broadcasted{Nothing}) where {Names, FT<:FieldTuple{<:Any,<:NamedTuple{Names}}}
    bc′ = flatten(bc)
    bc″ = Broadcasted{Style{Tuple}}((dest,args...)->broadcast!(bc′.f,dest,args...), (fieldtuple_data(dest), map(fieldtuple_data,bc′.args)...))
    copy(bc″)
    dest
end


### conversion
# no conversion needed
(::Type{B})(f::F)  where {B<:Basis,F<:FieldTuple{B}} = f
# FieldTuple is in BasisTuple
(::Type{B′})(f::F) where {B′<:BasisTuple,B<:BasisTuple,F<:FieldTuple{B}} = error("not implemented yet")
(::Type{B′})(f::F) where {B′<:Basis,     B<:BasisTuple,F<:FieldTuple{B}} = FieldTuple(map(B′,f.fs))
(::Type{B′})(f::F) where {B′<:Basislike, B<:BasisTuple,F<:FieldTuple{B}} = FieldTuple(map(B′,f.fs))
# FieldTuple is in a concrete basis
(::Type{B′})(f::F) where {B′<:Basis,     B<:Basis,     F<:FieldTuple{B}} = FieldTuple(map(B′,f.fs))
(::Type{B′})(f::F) where {B′<:Basislike, B<:Basis,     F<:FieldTuple{B}} = B′(F)(f)




### properties
getproperty(f::FieldTuple, s::Symbol) = getproperty(f::FieldTuple, Val(s))
getproperty(f::FieldTuple, ::Val{:fs}) = getfield(f,:fs)
getproperty(f::FieldTuple, ::Val{s}) where {s} = getproperty(getfield(f,:fs),s)


# generic AbstractVector inv/pinv/dot don't work with FieldTuples because those
# implementations depends on getindex which we don't implement for FieldTuples
for func in [:inv, :pinv]
    @eval $(func)(D::Diagonal{<:Any,FT}) where {FT<:FieldTuple} = 
        Diagonal(FT(map(firstfield, map($(func), map(Diagonal,D.diag.fs)))))
end
dot(a::FieldTuple, b::FieldTuple) = sum(map(dot, a.fs, b.fs))
