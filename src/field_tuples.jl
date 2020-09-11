
### FieldTuple types 

# FieldTuple is a thin wrapper around a Tuple or NamedTuple holding some Fields
# and behaving like a Field itself
struct FieldTuple{B<:Basis,FS<:Union{Tuple,NamedTuple},T} <: Field{B,Spin,Pix,T}
    fs::FS
    # the constructor for FieldTuples is a bit complex because there's alot of
    # different ways we want to allow constructing them. see the
    # FieldTuples/Constructors section of runtests.jl for a list. the use of
    # @generated is an easier way to deal with all the different cases and avoid
    # ambiguities vs. trying to do this via dispatch.
    @generated function (::Type{FT})(fs::Union{Tuple,NamedTuple}) where {FT<:FieldTuple}
        T =  (eltype(FT)==Any) ? :(promote_type(map(eltype,values(fs))...))     : eltype(FT)
        B = (basis(FT)==Basis) ? :(BasisTuple{Tuple{map(basis,values(fs))...}}) : basis(FT)
        getnames(::Type{FT}) where {Names, FT<:FieldTuple{<:Any, <:NamedTuple{Names}}} = Names
        getnames(::Any)  = nothing
        Names = getnames(FT)
        fs′ = (Names==nothing) ? :fs : :(NamedTuple{$Names}(fs))
        quote
            fs′ = $fs′
            $(Expr(:new, :(FieldTuple{$B,typeof(fs′),$T}), fs′))
        end
    end
end
# FieldTuple(args...) or FieldTuple(;kwargs...) calls the inner constructor
# which takes a single Tuple/NamedTuple:
(::Type{FT})(;kwargs...) where {FT<:FieldTuple} = FT((;kwargs...))
(::Type{FT})(f1::Field,f2::Field,fs::Field...) where {FT<:FieldTuple} = FT((f1,f2,fs...))
# converting a FieldTuple to a different kind of FieldTuple just changes the
# basis then asserts we end up with the right type:
(::Type{FT})(ft::FieldTuple) where {B<:Basis,FT<:FieldTuple{B}} = FieldTuple{B}(ft.fs) :: FT


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
iterate(ft::FieldTuple, args...) = iterate(ft.fs, args...)
getindex(f::FieldTuple, i::Union{Int,UnitRange}) = getindex(f.fs, i)
fill!(ft::FieldTuple, x) = (map(f->fill!(f,x), ft.fs); ft)
adapt_structure(to, f::FieldTuple{B}) where {B} = FieldTuple{B}(map(f->adapt(to,f),f.fs))
similar(ft::FT) where {FT<:FieldTuple} = FT(map(f->similar(f),ft.fs))
function similar(ft::FT, ::Type{T}, dims::Dims) where {T<:Number, B, FT<:FieldTuple{B}}
    @assert size(ft)==dims "Tried to make a field similar to $FT but dims should have been $(size(ft)), not $dims."
    FieldTuple{B}(map(f->similar(f,T),ft.fs))
end
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
struct FieldTupleStyle{B,Names,FS} <: AbstractArrayStyle{1} end
(::Type{FTS})(::Val{1}) where {FTS<:FieldTupleStyle} = FTS()
BroadcastStyle(::Type{FT}) where {B,FS<:Tuple,FT<:FieldTuple{B,FS}} = FieldTupleStyle{B,Nothing,Tuple{map_tupleargs(typeof∘BroadcastStyle,FS)...}}()
BroadcastStyle(::Type{FT}) where {B,Names,FS,NT<:NamedTuple{Names,FS},FT<:FieldTuple{B,NT}} = FieldTupleStyle{B,Names,Tuple{map_tupleargs(typeof∘BroadcastStyle,FS)...}}()
BroadcastStyle(::FieldTupleStyle{B,Names,FS1}, ::FieldTupleStyle{B,Names,FS2}) where {B,Names,FS1,FS2} = begin
    FS = Tuple{map_tupleargs((S1,S2)->typeof(result_style(S1(),S2())), FS1, FS2)...}
    FieldTupleStyle{B,Names,FS}()
end
BroadcastStyle(S1::FieldTupleStyle{B1}, S2::FieldTupleStyle{B2}) where {B1,B2} =
    invalid_broadcast_error(B1,S1,B2,S2)
similar(::Broadcasted{FTS}, ::Type{T}) where {T, FTS<:FieldTupleStyle} = similar(FTS,T)
similar(::Type{FieldTupleStyle{B,Nothing,FS}}, ::Type{T}) where {B,FS,T} = FieldTuple{B}(map_tupleargs(F->similar(F,T), FS))
similar(::Type{FieldTupleStyle{B,Names,FS}}, ::Type{T}) where {B,Names,FS,T} = FieldTuple{B}(NamedTuple{Names}(map_tupleargs(F->similar(F,T), FS)))
instantiate(bc::Broadcasted{<:FieldTupleStyle}) = bc
fieldtuple_data(f::FieldTuple, i) = f.fs[i]
fieldtuple_data(f::Field, i) = f
fieldtuple_data(x, i) = x
function copyto!(dest::FieldTuple, bc::Broadcasted{Nothing})
    for (i,d) in enumerate(dest.fs)
        copyto!(d, map_bc_args(arg->fieldtuple_data(arg,i), bc))
    end
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
Basis(f::FieldTuple{<:Basis}) = f
Basis(f::FieldTuple{<:BasisTuple}) = f
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

# in-place conversions
(::Type{B′})(f′::FieldTuple, f::FieldTuple) where {B′<:BasisTuple} = 
    (map((B,f′,f)->B(f′,f), tuple(B′.parameters[1].parameters...), f′.fs, f.fs); f′)




### properties
getproperty(f::FieldTuple, s::Symbol) = getproperty(f, Val(s))
getproperty(f::FieldTuple, ::Val{:fs}) = getfield(f,:fs)
getproperty(f::FieldTuple, ::Val{s}) where {s} = getproperty(getfield(f,:fs),s)
propertynames(f::FieldTuple) = (:fs, propertynames(f.fs)...)

### simulation
white_noise(rng::AbstractRNG, ::Type{<:FieldTuple{B,FS}}) where {B,FS<:Tuple} = 
    FieldTuple(map(x->white_noise(rng,x), tuple(FS.parameters...)))
white_noise(rng::AbstractRNG, ::Type{<:FieldTuple{B,NamedTuple{Names,FS}}}) where {B,Names,FS<:Tuple} = 
    FieldTuple(NamedTuple{Names}(map(x->white_noise(rng,x), tuple(FS.parameters...))))

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

function ud_grade(f::FieldTuple, args...; kwargs...)
    FieldTuple(map(f->ud_grade(f,args...; kwargs...), f.fs))
end


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

*(a::TupleAdjoint{F}, b::F) where {F<:Field{<:Any,S0}} = a.f .* b
*(a::TupleAdjoint{FT}, b::FT) where {FT<:FieldTuple} = sum(map((a,b)->tuple_adjoint(a)*b, a.f.fs, b.fs))

mul!(dst::Field{<:Any,S0}, a::TupleAdjoint{FT}, b::FT) where {FT<:Field{<:Any,S0}} = dst .= a.f .* b
mul!(dst::Field{<:Any,S0}, a::TupleAdjoint{FT}, b::FT) where {FT<:FieldTuple{<:Any,<:NamedTuple{<:Any,NTuple{2}}}} = 
    (@. dst = a.f[1]*b[1] + a.f[2]*b[2])
# todo: make this generic case efficient:    
mul!(dst::Field{<:Any,S0}, a::TupleAdjoint{FT}, b::FT) where {FT<:FieldTuple} = 
    dst .= sum(map((a,b)->mul!(copy(dst),tuple_adjoint(a),b), a.f.fs, b.fs))
