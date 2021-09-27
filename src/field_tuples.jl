
### FieldTuple types 

# FieldTuple is a thin wrapper around a Tuple or NamedTuple holding some Fields
# and behaving like a Field itself
struct FieldTuple{FS<:Union{Tuple,NamedTuple},B,T} <: Field{B,T}
    fs :: FS
    function FieldTuple(fs::FS) where {FS<:Union{Tuple,NamedTuple}}
        B = BasisProd{Tuple{map(basis,values(fs))...}}
        T = promote_type(map(eltype,values(fs))...)
        new{FS,B,T}(fs)
    end
end
# FieldTuple(args...) or FieldTuple(;kwargs...) calls the inner constructor
# which takes a single Tuple/NamedTuple:
(::Type{FT})(;kwargs...) where {FT<:FieldTuple} = FT((;kwargs...))
(::Type{FT})(f1::Field,f2::Field,fs::Field...) where {FT<:FieldTuple} = FT((f1,f2,fs...))

### printing
getindex(f::FieldTuple,::Colon) = vcat(getindex.(values(f.fs),:)...)[:]
getindex(D::DiagOp{<:FieldTuple}, i::Int, j::Int) = (i==j) ? D.diag[:][i] : diagzero(D, i, j)
typealias_def(::Type{<:FieldTuple{NamedTuple{Names,FS},T}}) where {Names,FS<:Tuple,T} =
    "Field-($(join(map(string,Names),",")))-$FS"
typealias_def(::Type{<:FieldTuple{FS,T}}) where {FS<:Tuple,T} =
    "Field-$(tuple_type_len(FS))-$FS"

### array interface
size(f::FieldTuple) = (mapreduce(length, +, f.fs, init=0),)
copy(f::FieldTuple) = FieldTuple(map(copy,f.fs))
iterate(ft::FieldTuple, args...) = iterate(ft.fs, args...)
getindex(f::FieldTuple, i::Union{Int,UnitRange}) = getindex(f.fs, i)
fill!(ft::FieldTuple, x) = (map(f->fill!(f,x), ft.fs); ft)
get_storage(f::FieldTuple) = only(unique(map(get_storage, f.fs)))
adapt_structure(to, f::FieldTuple) = FieldTuple(map(f->adapt(to,f),f.fs))
similar(ft::FieldTuple) = FieldTuple(map(similar,ft.fs))
similar(ft::FieldTuple, ::Type{T}) where {T<:Number} = FieldTuple(map(f->similar(f,T),ft.fs))
similar(ft::FieldTuple, Nbatch::Int) = FieldTuple(map(f->similar(f,Nbatch),ft.fs))
sum(f::FieldTuple; dims=:) = dims == (:) ? sum(sum, f.fs) : error("sum(::FieldTuple, dims=$dims not supported")

### broadcasting
# see base_fields.jl for more explanation of all these pieces, its the
# exact same principle 
struct FieldTupleStyle{S,Names} <: AbstractArrayStyle{1} end
function BroadcastStyle(::Type{<:FieldTuple{TS}}) where {TS<:Tuple}
    FieldTupleStyle{Tuple{map_tupleargs(typeof∘BroadcastStyle,TS)...}, Nothing}()
end
function BroadcastStyle(::Type{<:FieldTuple{NamedTuple{Names,TS}}}) where {Names,TS<:Tuple}
    FieldTupleStyle{Tuple{map_tupleargs(typeof∘BroadcastStyle,TS)...}, Names}()
end
function BroadcastStyle(::FieldTupleStyle{S₁,Names}, ::FieldTupleStyle{S₂,Names}) where {S₁,S₂,Names}
    FieldTupleStyle{Tuple{map_tupleargs((s₁,s₂)->typeof(result_style(s₁(),s₂())), S₁, S₂)...}, Names}()
end
BroadcastStyle(S::FieldTupleStyle, ::DefaultArrayStyle{0}) = S


@generated function materialize(bc::Broadcasted{FieldTupleStyle{S,Names}}) where {S,Names}
    wrapper = Names == Nothing ? :tuple : :(NamedTuple{$Names})
    exprs = map_tupleargs(S, tuple(1:tuple_type_len(S)...)) do Sᵢ, i
        :(materialize(convert(Broadcasted{$Sᵢ}, preprocess(($(S.parameters[i])(),FieldTupleComponent{$i}()), bc))))
    end
    :(FieldTuple($wrapper($(exprs...))))
end
@generated function materialize!(dst::FieldTuple, bc::Broadcasted{FieldTupleStyle{S,Names}}) where {S,Names}
    exprs = map_tupleargs(S, tuple(1:tuple_type_len(S)...)) do Sᵢ, i
        :(materialize!(dst.fs[$i], convert(Broadcasted{$Sᵢ}, preprocess(($(S.parameters[i])(),FieldTupleComponent{$i}()), bc))))
    end
    :(begin $(exprs...) end; dst)
end

struct FieldTupleComponent{i} end

preprocess(::Tuple{<:Any,FieldTupleComponent{i}}, ft::FieldTuple) where {i} = ft.fs[i]

preprocess(dest::Tuple{FieldTupleStyle{S},<:Any}, bc::Broadcasted) where {S} = 
    broadcasted(S(), bc.f, preprocess_args(dest, bc.args)...)


### promotion
function promote(ft1::FieldTuple, ft2::FieldTuple)
    fts = map(promote, ft1.fs, ft2.fs)
    FieldTuple(map(first,fts)), FieldTuple(map(last,fts))
end

### conversion
Basis(ft::FieldTuple) = ft
(::Type{B})(ft::FieldTuple) where {B<:Basis}     = FieldTuple(map(B, ft.fs))
(::Type{B})(ft::FieldTuple) where {B<:Basislike} = FieldTuple(map(B, ft.fs))
(::Type{B})(ft::FieldTuple{<:Tuple}) where {Bs,B<:BasisProd{Bs}} = 
    FieldTuple(map_tupleargs((B,f)->B(f), Bs, ft.fs))
(::Type{B})(ft::FieldTuple{<:NamedTuple{Names}}) where {Names,Bs,B<:BasisProd{Bs}} = 
    FieldTuple(NamedTuple{Names}(map_tupleargs((B,f)->B(f), Bs, values(ft.fs))))



### properties
getproperty(f::FieldTuple, s::Symbol) = getproperty(f, Val(s))
getproperty(f::FieldTuple, ::Val{:fs}) = getfield(f, :fs)
getproperty(f::FieldTuple, ::Val{s}) where {s} = getproperty(getfield(f,:fs), s)
propertynames(f::FieldTuple) = (:fs, propertynames(f.fs)...)

### simulation
white_noise(ξ::FieldTuple, rng::AbstractRNG) = FieldTuple(map(f -> white_noise(f, rng), ξ.fs))

### Diagonal-ops
# need a method specific for FieldTuple since we don't carry around
# the basis in a way that works with the default implementation
(*)(D::DiagOp{<:FieldTuple}, f::FieldTuple) = FieldTuple(map((d,f)->Diagonal(d)*f, D.diag.fs, f.fs))
(\)(D::DiagOp{<:FieldTuple}, f::FieldTuple) = FieldTuple(map((d,f)->Diagonal(d)\f, D.diag.fs, f.fs))


# promote before recursing for these 
dot(a::FieldTuple, b::FieldTuple) = reduce(+, map(dot, getfield.(promote(a,b),:fs)...), init=0)
hash(ft::FieldTuple, h::UInt64) = foldr(hash, (typeof(ft), ft.fs), init=h)

# logdet & trace
@auto_adjoint logdet(L::Diagonal{<:Union{Real,Complex}, <:FieldTuple}) = reduce(+, map(logdet∘Diagonal, L.diag.fs), init=0)
tr(L::Diagonal{<:Union{Real,Complex}, <:FieldTuple}) = reduce(+, map(tr∘Diagonal, L.diag.fs), init=0)

# misc
batch_length(ft::FieldTuple) = only(unique(map(batch_length, ft.fs)))
batch_index(ft::FieldTuple, I) = FieldTuple(map(f -> batch_index(f, I), ft.fs))
function global_rng_for(::Type{<:FieldTuple{<:Union{FS,NamedTuple{Names,FS}}}}) where {Names,FS<:Tuple} 
    only(unique(map_tupleargs(global_rng_for, FS)))
end
getindex(ft::FieldTuple, k::Symbol) = ft.fs[k]