
### FieldTuple types 

# FieldTuple is a thin wrapper around a Tuple or NamedTuple holding some Fields
# and behaving like a Field itself
struct FieldTuple{FS<:Union{Tuple,NamedTuple},T} <: Field{Basis,T}
    fs::FS
    function FieldTuple(fs::FS) where {FS<:Union{Tuple,NamedTuple}}
        T = promote_type(map(eltype,values(fs))...)
        new{FS,T}(fs)
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
size(f::FieldTuple) = (sum(map(length, f.fs)),)
copyto!(dest::FieldTuple, src::FieldTuple) = (map(copyto!,dest.fs,src.fs); dest)
iterate(ft::FieldTuple, args...) = iterate(ft.fs, args...)
getindex(f::FieldTuple, i::Union{Int,UnitRange}) = getindex(f.fs, i)
fill!(ft::FieldTuple, x) = (map(f->fill!(f,x), ft.fs); ft)
adapt_structure(to, f::FieldTuple) = FieldTuple(map(f->adapt(to,f),f.fs))
similar(ft::FieldTuple) = FieldTuple(map(similar,ft.fs))
similar(ft::FieldTuple, ::Type{T}) where {T<:Number} = FieldTuple(map(f->similar(f,T),ft.fs))
function sum(f::FieldTuple; dims=:)
    if dims == (:)
        sum(sum,f.fs)
    elseif all(dims .> 1)
        f
    else
        error("Invalid dims in sum(::FieldTuple, dims=$(dims)).")
    end
end

### broadcasting
struct FieldTupleStyle{S} <: AbstractArrayStyle{1} end
BroadcastStyle(::Type{<:FieldTuple{FS}}) where {FS<:Tuple} = 
    FieldTupleStyle{Tuple{map_tupleargs(typeof∘BroadcastStyle,FS)...}}()
BroadcastStyle(::FieldTupleStyle{S₁}, ::FieldTupleStyle{S₂}) where {S₁,S₂} = 
    FieldTupleStyle{Tuple{map_tupleargs((s₁,s₂)->typeof(result_style(s₁(),s₂())), S₁, S₂)...}}()
BroadcastStyle(S::FieldTupleStyle, ::DefaultArrayStyle{0}) = S

struct FieldTupleComponent{i} end
preprocess(::FieldTupleComponent{i}, ft::FieldTuple) where {i} = ft.fs[i]
@generated function materialize(bc::Broadcasted{FieldTupleStyle{S}}) where {S}
    exprs = map_tupleargs(S, tuple(1:tuple_type_len(S)...)) do Sᵢ, i
        :(materialize(convert(Broadcasted{$Sᵢ}, preprocess(FieldTupleComponent{$i}(), bc))))
    end
    :(FieldTuple($(exprs...)))
end
@generated function materialize!(dst::FieldTuple, bc::Broadcasted{FieldTupleStyle{S}}) where {S}
    exprs = map_tupleargs(S, tuple(1:tuple_type_len(S)...)) do Sᵢ, i
        :(materialize!(dst.fs[$i], convert(Broadcasted{$Sᵢ}, preprocess(FieldTupleComponent{$i}(), bc))))
    end
    :(begin $(exprs...) end; dst)
end


### promotion
function promote(ft1::FieldTuple, ft2::FieldTuple)
    fts = map(promote,ft1.fs,ft2.fs)
    FieldTuple(map(first,fts)), FieldTuple(map(last,fts))
end

### conversion
(::Type{B})(ft::FieldTuple) where {B<:Basis}     = FieldTuple(map(B, ft.fs))
(::Type{B})(ft::FieldTuple) where {B<:Basislike} = FieldTuple(map(B, ft.fs))



# ### properties
# getproperty(f::FieldTuple, s::Symbol) = getproperty(f, Val(s))
# getproperty(f::FieldTuple, ::Val{:fs}) = getfield(f,:fs)
# getproperty(f::FieldTuple, ::Val{s}) where {s} = getproperty(getfield(f,:fs),s)
# propertynames(f::FieldTuple) = (:fs, propertynames(f.fs)...)

# ### simulation
# white_noise(rng::AbstractRNG, ::Type{<:FieldTuple{B,FS}}) where {B,FS<:Tuple} = 
#     FieldTuple(map(x->white_noise(rng,x), tuple(FS.parameters...)))
# white_noise(rng::AbstractRNG, ::Type{<:FieldTuple{B,NamedTuple{Names,FS}}}) where {B,Names,FS<:Tuple} = 
#     FieldTuple(NamedTuple{Names}(map(x->white_noise(rng,x), tuple(FS.parameters...))))

# # generic AbstractVector inv/pinv don't work with FieldTuples because those
# # implementations depends on get/setindex which we don't implement for FieldTuples
# for func in [:inv, :pinv]
#     @eval $(func)(D::DiagOp{FT}) where {FT<:FieldTuple} = 
#         Diagonal(FT(map(firstfield, map($(func), map(Diagonal,D.diag.fs)))))
# end

# # promote before recursing for these 
# ≈(a::FieldTuple, b::FieldTuple) = all(map(≈, getfield.(promote(a,b),:fs)...))
# dot(a::FieldTuple, b::FieldTuple) = sum(map(dot, getfield.(promote(a,b),:fs)...))
# hash(ft::FieldTuple, h::UInt) = hash(ft.fs, h)

# function ud_grade(f::FieldTuple, args...; kwargs...)
#     FieldTuple(map(f->ud_grade(f,args...; kwargs...), f.fs))
# end

# # logdet & trace
# logdet(L::Diagonal{<:Union{Real,Complex}, <:FieldTuple}) = sum(map(logdet∘Diagonal,L.diag.fs))
# tr(L::Diagonal{<:Union{Real,Complex}, <:FieldTuple}) = sum(map(tr∘Diagonal,L.diag.fs))

# # misc
# fieldinfo(ft::FieldTuple) = fieldinfo(only(unique(map(typeof, ft.fs)))) # todo: make even more generic
# batch_length(ft::FieldTuple) = only(unique(map(batch_length, ft.fs)))
# global_rng_for(::Type{<:FieldTuple{B,FS}}) where {B,FS} = global_rng_for(eltype(FS))

# ### adjoint tuples

# # represents a field which is adjoint over just the "tuple" indices. multiplying
# # such a field by a non-adjointed one should be the inner product over just the
# # tuple indices, and hence return a tuple-less, i.e a spin-0, field. 
# # note: these are really only lightly used in one place in LenseFlow, so they
# # have almost no real functionality, the code here is in fact all there is. 
# struct TupleAdjoint{T<:Field}
#     f :: T
# end
# tuple_adjoint(f::Field) = TupleAdjoint(f)

# *(a::TupleAdjoint{F}, b::F) where {F<:Field{<:Any,S0}} = a.f .* b
# *(a::TupleAdjoint{FT}, b::FT) where {FT<:FieldTuple} = sum(map((a,b)->tuple_adjoint(a)*b, a.f.fs, b.fs))

# mul!(dst::Field{<:Any,S0}, a::TupleAdjoint{FT}, b::FT) where {FT<:Field{<:Any,S0}} = dst .= a.f .* b
# mul!(dst::Field{<:Any,S0}, a::TupleAdjoint{FT}, b::FT) where {FT<:FieldTuple{<:Any,<:NamedTuple{<:Any,NTuple{2}}}} = 
#     (@. dst = a.f[1]*b[1] + a.f[2]*b[2])
# # todo: make this generic case efficient:    
# mul!(dst::Field{<:Any,S0}, a::TupleAdjoint{FT}, b::FT) where {FT<:FieldTuple} = 
#     dst .= sum(map((a,b)->mul!(copy(dst),tuple_adjoint(a),b), a.f.fs, b.fs))
