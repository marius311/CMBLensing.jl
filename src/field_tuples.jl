

abstract type BasisTuple{T} <: Basis end
# abstract type SpinTuple{T} <: Spin end
# abstract type PixTuple{T} <: Pix end


## FieldTuple type 
# a thin wrapper around a NamedTuple which additionally forwards all
# broadcasts one level deeper
struct FieldTuple{FS<:NamedTuple,B} <: Field{B,Spin,Pix,Real}
    fs::FS
end
FieldTuple(fs::FS) where {FS} = FieldTuple{FS,BasisTuple{Tuple{map(basis,fs)...}}}(fs)



## printing
function showarg(io::IO, ft::FieldTuple{<:NamedTuple{Names}}, toplevel) where {Names}
    print(io, "$(Names) FieldTuple")
end
function show(io::IO, ::MIME"text/plain", ft::FieldTuple{<:NamedTuple})
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
function copy(bc::Broadcasted{Style{FT}}) where {Names, FT<:FieldTuple{<:NamedTuple{Names}}}
    bc′ = flatten(bc)
    bc″ = Broadcasted{Style{Tuple}}((args...)->broadcast(bc′.f,args...), map(tuple_data,bc′.args))
    FT(NamedTuple{Names}(copy(preprocess(bc))))
end
function copyto!(dest::FT, bc::Broadcasted{Style{FT}}) where {Names, FT<:FieldTuple{<:NamedTuple{Names}}}
    bc′ = flatten(bc)
    bc″ = Broadcasted{Style{Tuple}}((dest,args...)->broadcast!(bc′.f,dest,args...), (tuple_data(dest), map(tuple_data,bc′.args)...))
    copy(bc″)
    dest
end
