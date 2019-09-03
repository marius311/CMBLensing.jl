

abstract type BasisTuple{T} <: Basis end
promote_type(::Type{BasisTuple{BT1}}, ::Type{BasisTuple{BT2}}) where {BT1,BT2} = BasisTuple{Tuple{map_tupleargs(promote_type,BT1,BT2)...}}


### FieldTuple type 
# a thin wrapper around a Tuple or NamedTuple which additionally forwards all
# broadcasts one level deeper
struct FieldTuple{B<:Basis,FS<:Union{Tuple,NamedTuple},T} <: Field{B,Spin,Pix,T}
    fs::FS
end
FieldTuple{B}(fs::FS) where {B, FS} = FieldTuple{B,FS,promote_type(map(eltype,values(fs))...)}(fs)
# constructors for FieldTuples with names
FieldTuple(;kwargs...) = FieldTuple((;kwargs...))
FieldTuple(fs::NamedTuple) = FieldTuple{BasisTuple{Tuple{map(basis,values(fs))...}}}(fs)
(::Type{<:FT})(f1,f2,fs...) where {Names,FT<:FieldTuple{<:Any,<:NamedTuple{Names}}} = FieldTuple(NamedTuple{Names}((f1,f2,fs...)))
(::Type{FT})(;kwargs...) where {B,FT<:FieldTuple{B}} = FieldTuple{B}((;kwargs...))::FT
(::Type{FT})(ft::FieldTuple) where {B,FT<:FieldTuple{B}} = FieldTuple{B}(ft.fs)::FT
# constructors for FieldTuples without names
FieldTuple(f1,f2,fs...) = FieldTuple((f1,f2,fs...))
FieldTuple(fs::Tuple) = FieldTuple{BasisTuple{Tuple{map(basis,values(fs))...}},typeof(fs),promote_type(map(eltype,values(fs))...)}(fs)


### printing
getindex(f::FieldTuple,::Colon) = vcat(getindex.(values(f.fs),:)...)[:]
getindex(D::DiagOp{<:FieldTuple}, i::Int, j::Int) = (i==j) ? D.diag[:][i] : diagzero(D, i, j)
@show_datatype show_datatype(io::IO, t::Type{FT}) where {B,Names,T,FS,FT<:FieldTuple{B,NamedTuple{Names,FS},T}} =
    print(io, "Field$(tuple_type_len(FS))Tuple{$(Names), $(B), $(T)}")
@show_datatype show_datatype(io::IO, t::Type{FT}) where {B,T,FS<:Tuple,FT<:FieldTuple{B,FS,T}} =
    print(io, "Field$(tuple_type_len(FS))Tuple{$(B), $(T)}")

### array interface
size(f::FieldTuple) = (sum(map(length, f.fs)),)
copyto!(dest::FT, src::FT) where {FT<:FieldTuple} = (map(copyto!,dest.fs,src.fs); dest)
similar(f::FT) where {FT<:FieldTuple} = FT(map(similar,f.fs))
similar(f::FT, ::Type{T}) where {T, B, FT<:FieldTuple{B}} = FieldTuple{B}(map(f->similar(f,T),f.fs))
iterate(ft::FieldTuple, args...) = iterate(ft.fs, args...)
getindex(f::FieldTuple, i::Union{Int,UnitRange}) = getindex(f.fs, i)
fill!(ft::FieldTuple, x) = (map(f->fill!(f,x), ft.fs); ft)


### broadcasting
struct FieldTupleStyle{B,Names,FS<:Tuple} <: AbstractArrayStyle{1} end
(::Type{FTS})(::Val{1}) where {FTS<:FieldTupleStyle} = FTS()
BroadcastStyle(::Type{FT}) where {B,FS<:Tuple,FT<:FieldTuple{B,FS}} = FieldTupleStyle{B,Nothing,Tuple{map_tupleargs(typeof∘BroadcastStyle,FS)...}}()
BroadcastStyle(::Type{FT}) where {B,Names,FS,NT<:NamedTuple{Names,FS},FT<:FieldTuple{B,NT}} = FieldTupleStyle{B,Names,Tuple{map_tupleargs(typeof∘BroadcastStyle,FS)...}}()
similar(::Broadcasted{FTS}, ::Type{T}) where {T, FTS<:FieldTupleStyle} = similar(FTS,T)
similar(::Type{FieldTupleStyle{B,Nothing,FS}}, ::Type{T}) where {B,FS,T} = FieldTuple{B}(map_tupleargs(F->similar(F,T), FS))
similar(::Type{FieldTupleStyle{B,Names,FS}}, ::Type{T}) where {B,Names,FS,T} = FieldTuple{B}(NamedTuple{Names}(map_tupleargs(F->similar(F,T), FS)))
instantiate(bc::Broadcasted{<:FieldTupleStyle}) = bc
fieldtuple_data(f::FieldTuple) = values(f.fs)
fieldtuple_data(f::Field) = (f,)
fieldtuple_data(x) = x
function copyto!(dest::FieldTuple, bc::Broadcasted{Nothing})
    bc′ = flatten(bc)
    bc″ = Broadcasted{Style{Tuple}}((dest,args...)->broadcast!(bc′.f,dest,args...), (fieldtuple_data(dest), map(fieldtuple_data,bc′.args)...))
    copy(bc″)
    dest
end

### promotion
function promote(ft1::FieldTuple, ft2::FieldTuple)
    fts = map(promote,ft1.fs,ft2.fs)
    FieldTuple(map(first,fts)), FieldTuple(map(last,fts))
end

### conversion
# The basis we're converting to is always B′. The FieldTuple's basis is B (if
# its different). Each of these might be a concrete basis or a BasisTuple, and
# the FieldTuple might be named or not. And we have Basis(f) which should be a
# no-op. This giant matrix of possibilities presents some ambiguity problems,
# hence why the rules below are so lengthy. Perhaps there's a more succinct
# way to do it, but for now this works.
# 
# 
# cases where no conversion is needed
Basis(::FieldTuple{<:Basis}) = f
Basis(::FieldTuple{<:BasisTuple}) = f
(::Type{B′})(f::F)  where {B′<:Basis,      F<:FieldTuple{B′}} = f
(::Type{B′})(f::F)  where {B′<:BasisTuple, F<:FieldTuple{B′,<:Tuple}} = f
(::Type{B′})(f::F)  where {B′<:BasisTuple, Names,F<:FieldTuple{B′,<:NamedTuple{Names}}} = f
# cases where FieldTuple is in BasisTuple
(::Type{B′})(f::F) where {B′<:BasisTuple,B<:BasisTuple,F<:FieldTuple{B,<:Tuple}} = 
    FieldTuple(map((B,f)->B(f), tuple(B′.parameters[1].parameters...), f.fs))
(::Type{B′})(f::F) where {B′<:BasisTuple,B<:BasisTuple,Names,F<:FieldTuple{B,<:NamedTuple{Names}}} = 
    FieldTuple(NamedTuple{Names}(map((B,f)->B(f), tuple(B′.parameters[1].parameters...), values(f.fs))))
(::Type{B′})(f::F) where {B′<:Basis,     B<:BasisTuple,F<:FieldTuple{B}} = FieldTuple(map(B′,f.fs))
(::Type{B′})(f::F) where {B′<:Basislike, B<:BasisTuple,F<:FieldTuple{B}} = FieldTuple(map(B′,f.fs))
# cases FieldTuple is in a concrete basis
(::Type{B′})(f::F) where {B′<:Basis,     B<:Basis,     F<:FieldTuple{B}} = FieldTuple(map(B′,f.fs))
(::Type{B′})(f::F) where {B′<:Basislike, B<:Basis,     F<:FieldTuple{B}} = B′(F)(f)




### properties
getproperty(f::FieldTuple, s::Symbol) = getproperty(f, Val(s))
getproperty(f::FieldTuple, ::Val{:fs}) = getfield(f,:fs)
getproperty(f::FieldTuple, ::Val{s}) where {s} = getproperty(getfield(f,:fs),s)


### simulation
white_noise(::Type{<:FieldTuple{B,FS}}) where {B,FS<:Tuple} = 
    FieldTuple(map(white_noise, tuple(FS.parameters...)))
white_noise(::Type{<:FieldTuple{B,NamedTuple{Names,FS}}}) where {B,Names,FS<:Tuple} = 
    FieldTuple(NamedTuple{Names}(map(white_noise, tuple(FS.parameters...))))
    

# generic AbstractVector inv/pinv don't work with FieldTuples because those
# implementations depends on get/setindex which we don't implement for FieldTuples
for func in [:inv, :pinv]
    @eval $(func)(D::DiagOp{FT}) where {FT<:FieldTuple} = 
        Diagonal(FT(map(firstfield, map($(func), map(Diagonal,D.diag.fs)))))
end

# promote before recursing for these 
≈(a::FieldTuple, b::FieldTuple) = all(map(≈, getfield.(promote(a,b),:fs)...))
dot(a::FieldTuple, b::FieldTuple) = sum(map(dot, getfield.(promote(a,b),:fs)...))
hash(ft::FieldTuple, h::UInt) = hash(ft.fs, h)

### adjoint tuples

# represents a field which is adjoint over just the "tuple" indices. multiplying
# such a field by a non-adjointed one should be the inner product over just the
# tuple indices, and hence return a tuple-less, i.e a spin-0, field. 
# note: these are really only lightly used in one place in LenseFlow, so they
# have almost no real functionality, the code here is in fact all there is. 
struct TupleAdjoint{T<:Field}
    f :: T
end
tuple_adjoint(f::Field) = TupleAdjoint(f)

mul!(dst::Field{<:Any,S0}, a::TupleAdjoint{FT}, b::FT) where {FT<:Field{<:Any,S0}} = dst .= a.f .* b
mul!(dst::Field{<:Any,S0}, a::TupleAdjoint{FT}, b::FT) where {FT<:FieldTuple{<:Any,<:NamedTuple{<:Any,NTuple{2}}}} = 
    (@. dst = a.f[1]*b[1] + a.f[2]*b[2])
# todo: make this generic case efficient:    
mul!(dst::Field{<:Any,S0}, a::TupleAdjoint{FT}, b::FT) where {FT<:FieldTuple} = 
    dst .= sum(map((a,b)->mul!(copy(dst),tuple_adjoint(a),b), a.f.fs, b.fs))
