

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
function showarg(io::IO, ft::FieldTuple{B,<:NamedTuple{Names}}, toplevel) where {B,Names}
    print(io, "$(Names) FieldTuple{$B}")
end
function show(io::IO, ::MIME"text/plain", ft::FieldTuple)
    print(io, "$(sum(map(length,values(ft.fs))))-element ")
    showarg(io, ft, false)
    println(io, ":")
    for (k,f) in pairs(ft.fs)
        print(io, k, " = ")
        summary(IOContext(io, :limit=>true), f)
        show_vector(io, f)
        println(io)
    end
end


## array interface
size(f::FieldTuple) = (sum(map(length, f.fs)),)
# @propagate_inbounds @inline getindex(f::FlatS0, I...) = getindex(broadcast_data(f), I...)
# @propagate_inbounds @inline setindex!(f::FlatS0, X, I...) = setindex!(broadcast_data(f), X, I...)
similar(f::FT) where {FT<:FieldTuple}= FT(map(similar,f.fs))



## broadcasting
@propagate_inbounds @inline getindex(f::FieldTuple, i) = getindex(f.fs, i)
broadcastable(f::FieldTuple) = f
BroadcastStyle(::Type{FT}) where {FT<:FieldTuple} = Style{FT}()
BroadcastStyle(::Style{FT}, ::DefaultArrayStyle{0}) where {FT<:FieldTuple} = Style{FT}()
BroadcastStyle(::Style{FT}, ::Style{Tuple}) where {FT<:FieldTuple} = Style{FT}()
instantiate(bc::Broadcasted{<:Style{<:FieldTuple}}) = bc
tuple_data(f::FieldTuple) = values(f.fs)
tuple_data(f::Field) = (f,)
tuple_data(x) = x
function copy(bc::Broadcasted{Style{FT}}) where {Names, FT<:FieldTuple{<:Any,<:NamedTuple{Names}}}
    bc′ = flatten(bc)
    bc″ = Broadcasted{Style{Tuple}}((args...)->broadcast(bc′.f,args...), map(tuple_data,bc′.args))
    FT(NamedTuple{Names}(copy(preprocess(bc))))
end
function copyto!(dest::FT, bc::Broadcasted{Style{FT}}) where {Names, FT<:FieldTuple{<:Any,<:NamedTuple{Names}}}
    bc′ = flatten(bc)
    bc″ = Broadcasted{Style{Tuple}}((dest,args...)->broadcast!(bc′.f,dest,args...), (tuple_data(dest), map(tuple_data,bc′.args)...))
    copy(bc″)
    dest
end


### conversion
# (::Type{B})(::Type{<:FieldTuple{FS}}) where {FS,B<:Basislike} = BasisTuple{Tuple{map_tupleargs(F->B(F),FS)...}}
# (::Type{BasisTuple{BS}})(ft::FieldTuple) where {BS} = FieldTuple(map_tupleargs((B,f)->B(f), BS, ft.fs)...)
(::Type{B})(ft::FieldTuple) where {B<:Basis}     = FieldTuple(map(B,ft.fs))
# (::Type{B})(ft::FieldTuple) where {B<:Basislike} = FieldTuple(map(B,ft.fs)...) # needed for ambiguity
# (::Type{B})(ft′::FieldTuple, ft::FieldTuple) where {B<:Basis}     = (map(B, ft′.fs, ft.fs); ft′)
# (::Type{B})(ft′::FieldTuple, ft::FieldTuple) where {B<:Basislike} = (map(B, ft′.fs, ft.fs); ft′) # needed for ambiguity
# Basis(ft::FieldTuple) where {B<:Basis} = ft # needed for ambiguity


### properties
getproperty(f::FieldTuple, s::Symbol) = getproperty(f::FieldTuple, Val(s))
getproperty(f::FieldTuple, ::Val{:fs}) = getfield(f,:fs)
getproperty(f::FieldTuple, ::Val{s}) where {s} = getproperty(getfield(f,:fs),s)
