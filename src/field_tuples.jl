

abstract type BasisTuple{T} <: Basis end
# abstract type SpinTuple{T} <: Spin end
# abstract type PixTuple{T} <: Pix end


## FieldTuple type 
# a thin wrapper around a NamedTuple which additionally forwards all
# broadcasts one level deeper
struct FieldTuple{B,FS<:NamedTuple} <: Field{B,Spin,Pix,Real}
    fs::FS
end
FieldTuple(;kwargs...) = FieldTuple((;kwargs...))
FieldTuple(fs::NamedTuple) = FieldTuple{BasisTuple{Tuple{map(basis,values(fs))...}}}(fs)
FieldTuple{B}(fs::FS) where {B, FS<:NamedTuple} = FieldTuple{B,FS}(fs)
(::Type{FT})(;kwargs...) where {B,FT<:FieldTuple{B}} = FieldTuple{B}((;kwargs...))::FT
(::Type{FT})(ft::FieldTuple) where {B,FT<:FieldTuple{B}} = FieldTuple{B}(ft.fs)::FT


## printing
showarg(io::IO, ft::FT, toplevel) where {FT<:FieldTuple} = showarg(io,FT)
showarg(io::IO, ::Type{<:FieldTuple{B,<:NamedTuple{Names}}}) where {B,Names} =
    print(io, "$(Names) FieldTuple{$B}")
function show(io::IO, ::MIME"text/plain", ft::FieldTuple)
    print(io, "$(sum(map(length,values(ft.fs))))-element ")
    showarg(io, ft, false)
    println(io, ":")
    for (k,f) in pairs(ft.fs)
        print(io, " ", k, " = ")
        summary(IOContext(io, :limit=>true), f)
        show_vector(io, f)
        println(io)
    end
end


## array interface
size(f::FieldTuple) = (sum(map(length, f.fs)),)
similar(f::FT) where {FT<:FieldTuple}= FT(map(similar,f.fs))
copyto!(dest::FT, src::FT) where {FT<:FieldTuple} = (map(copyto!,dest.fs,src.fs); dest)


## broadcasting
@propagate_inbounds @inline getindex(f::FieldTuple, i) = getindex(f.fs, i)
broadcastable(f::FieldTuple) = f
BroadcastStyle(::Type{FT}) where {FT<:FieldTuple} = Style{FT}()
BroadcastStyle(::Style{FT}, ::DefaultArrayStyle{0}) where {FT<:FieldTuple} = Style{FT}()
BroadcastStyle(::Style{FT}, ::DefaultArrayStyle{1}) where {FT<:FieldTuple} = Style{FT}()
BroadcastStyle(::Style{FT}, ::Style{Tuple}) where {FT<:FieldTuple} = Style{FT}()
instantiate(bc::Broadcasted{<:Style{<:FieldTuple}}) = bc
tuple_data(f::FieldTuple) = values(f.fs)
tuple_data(f::Field) = (f,)
tuple_data(x) = x
function copy(bc::Broadcasted{Style{FT}}) where {Names, FT<:FieldTuple{<:Any,<:NamedTuple{Names}}}
    bc′ = flatten(bc)
    bc″ = Broadcasted{Style{Tuple}}((args...)->broadcast(bc′.f,args...), map(tuple_data,bc′.args))
    FT(NamedTuple{Names}(copy(bc″)))
end
function copyto!(dest::FT, bc::Broadcasted{Style{FT}}) where {Names, FT<:FieldTuple{<:Any,<:NamedTuple{Names}}}
    bc′ = flatten(bc)
    bc″ = Broadcasted{Style{Tuple}}((dest,args...)->broadcast!(bc′.f,dest,args...), (tuple_data(dest), map(tuple_data,bc′.args)...))
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
