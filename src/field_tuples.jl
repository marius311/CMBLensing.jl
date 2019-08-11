

abstract type BasisTuple{T} <: Basis end

## FieldTuple type 
# a thin wrapper around a Tuple or NamedTuple which additionally forwards all
# broadcasts one level deeper
struct FieldTuple{B<:Basis,FS<:Union{Tuple,NamedTuple},T} <: Field{B,Spin,Pix,T}
    fs::FS
end
# constructors for FieldTuples with names
FieldTuple(;kwargs...) = FieldTuple((;kwargs...))
FieldTuple(fs::NamedTuple) = FieldTuple{BasisTuple{Tuple{map(basis,values(fs))...}}}(fs)
FieldTuple{B}(fs::FS) where {B, FS<:NamedTuple} = FieldTuple{B,FS,promote_type(map(eltype,values(fs))...)}(fs)
(::Type{<:FT})(f1,f2,fs...) where {Names,FT<:FieldTuple{<:Any,<:NamedTuple{Names}}} = FieldTuple(NamedTuple{Names}((f1,f2,fs...)))
(::Type{FT})(;kwargs...) where {B,FT<:FieldTuple{B}} = FieldTuple{B}((;kwargs...))::FT
(::Type{FT})(ft::FieldTuple) where {B,FT<:FieldTuple{B}} = FieldTuple{B}(ft.fs)::FT
# constructors for FieldTuples without names
FieldTuple(f1,f2,fs...) = FieldTuple((f1,f2,fs...))
FieldTuple(fs::Tuple) = FieldTuple{BasisTuple{Tuple{map(basis,values(fs))...}},typeof(fs),promote_type(map(eltype,values(fs))...)}(fs)


## printing
getindex(f::FieldTuple,::Colon) = vcat(getindex.(values(f.fs),:)...)[:]
getindex(D::DiagOp{<:FieldTuple}, i::Int, j::Int) = (i==j) ? D.diag[:][i] : diagzero(D, i, j)
@show_datatype show_datatype(io::IO, t::Type{FT}) where {B,Names,T,FS,FT<:FieldTuple{B,NamedTuple{Names,FS},T}} =
    print(io, "Field$(tuple_type_len(FS))Tuple{$(Names), $(B), $(T)}")
@show_datatype show_datatype(io::IO, t::Type{FT}) where {B,T,FS<:Tuple,FT<:FieldTuple{B,FS,T}} =
    print(io, "Field$(tuple_type_len(FS))Tuple{$(B), $(T)}")

## array interface
size(f::FieldTuple) = (sum(map(length, f.fs)),)
copyto!(dest::FT, src::FT) where {FT<:FieldTuple} = (map(copyto!,dest.fs,src.fs); dest)
similar(f::FT) where {FT<:FieldTuple} = FT(map(similar,f.fs))
similar(f::FT, ::Type{T}) where {T, FT<:FieldTuple} = similar(FT,T)
similar(::Type{FT},::Type{T}) where {T,B,Names,FS,FT<:FieldTuple{B,<:NamedTuple{Names,FS}}} = 
    FieldTuple{B}(NamedTuple{Names}(map_tupleargs(F->similar(F,T), FS)))
similar(::Type{FT},::Type{T}) where {T,B,FS<:Tuple,FT<:FieldTuple{B,FS}} = 
    FieldTuple(map_tupleargs(F->similar(F,T), FS))
iterate(ft::FieldTuple, args...) = iterate(ft.fs, args...)
getindex(f::FieldTuple, i::Union{Int,UnitRange}) = getindex(f.fs, i)
fill!(ft::FieldTuple, x) = (map(f->fill!(f,x), ft.fs); ft)


## broadcasting
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
# no conversion needed
(::Type{B})(f::F)  where {B<:Basis,     F<:FieldTuple{B}} = f
# FieldTuple is in BasisTuple
(::Type{B′})(f::F) where {BS′,B′<:BasisTuple{BS′},B<:BasisTuple,F<:FieldTuple{B,<:Tuple}} = 
    FieldTuple(map((B,f)->B(f), tuple(BS′.parameters...), f.fs))
(::Type{B′})(f::F) where {BS′,B′<:BasisTuple{BS′},B<:BasisTuple,Names,F<:FieldTuple{B,<:NamedTuple{Names}}} = 
    FieldTuple(NamedTuple{Names}(map((B,f)->B(f), tuple(BS′.parameters...), values(f.fs))))
(::Type{B′})(f::F) where {B′<:Basis,     B<:BasisTuple,F<:FieldTuple{B}} = FieldTuple(map(B′,f.fs))
(::Type{B′})(f::F) where {B′<:Basislike, B<:BasisTuple,F<:FieldTuple{B}} = FieldTuple(map(B′,f.fs))
# FieldTuple is in a concrete basis
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

≈(a::FieldTuple, b::FieldTuple) = all(map(≈, a.fs, b.fs))
dot(a::FieldTuple, b::FieldTuple) = sum(map(dot, a.fs, b.fs))


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

mul!(dst::Field{<:Any,S0}, a::TupleAdjoint{FT}, b::FT) where {FT<:FieldTuple} = 
    dst .= sum(map(*, a.f.fs, b.fs))
mul!(dst::Field{<:Any,S0}, a::TupleAdjoint{FT}, b::FT) where {B<:Basis,FT<:FieldTuple{B, <:NamedTuple{<:Any,NTuple{2}}}} = 
    (@. dst = a.f[1]*b[1] + a.f[2]*b[2])
mul!(dst::Field{<:Any,S0}, a::TupleAdjoint{FT}, b::FT) where {FT<:Field{<:Any,S0}} = dst .= a.f .* b
