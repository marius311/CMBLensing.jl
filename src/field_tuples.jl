


## FieldTuple type 
# a thin wrapper around a NamedTuple which additionally forwards all
# broadcasts one level deeper
struct FieldTuple{FS<:NamedTuple} <: Field{Basis,Spin,Pix,Real}
    fs::FS
end

## printing
summary(io::IO, ft::FieldTuple{<:NamedTuple{Names}}) where {Names}  = 
    print(io, "$(sum(map(length,values(ft.fs))))-element $(Names) FieldTuple")
    
function show(io::IO, ::MIME"text/plain", ft::FieldTuple{<:NamedTuple})
    summary(io, ft)
    println(io, ":")
    for (k,f) in pairs(ft.fs)
        print(io, k, " = ")
        summary(IOContext(io, :limit=>true), f)
        show_vector(io, f)
        println(io)
    end
end

## broadcasting 
@propagate_inbounds @inline getindex(f::FieldTuple, i) = getindex(f.fs, i)
broadcastable(f::FieldTuple) = f
BroadcastStyle(::Type{<:FieldTuple}) = Style{FieldTuple}()
BroadcastStyle(::Style{FieldTuple}, ::DefaultArrayStyle{0}) = Style{FieldTuple}()
BroadcastStyle(::Style{FieldTuple}, ::Style{Tuple}) = Style{FieldTuple}()
instantiate(bc::Broadcasted{<:Style{FieldTuple}}) = bc
tuple_data(f::FieldTuple) = values(f.fs)
tuple_data(f::Field) = (f,)
tuple_data(x) = x
function copy(bc::Broadcasted{Style{FieldTuple}})
    bc′ = flatten(bc)
    bc″ = Broadcasted{Style{Tuple}}((args...)->broadcast(bc′.f,args...), map(tuple_data,bc′.args))
    FieldTuple(NamedTuple{keys(bc′)}(copy(bc″)))
end
keys(f::FieldTuple) = keys(f.fs)
keys(bc::Broadcasted{Style{FieldTuple}}) = _keys(nothing, bc.args...)
_keys(cur, x::FieldTuple, rest...) = _keys(_samekeys(cur,keys(x)), rest...)
_keys(cur, x, rest...) = _keys(cur, rest...)
_keys(cur) = cur
_samekeys(a::Nothing, b) = b
_samekeys(a::T, b::T) where {T} = a==b ? a : error("Can't broadcast over FieldTuples with different keys: $a and $b")
