# the following code defines a two-tuple of Fields, where basically everything
# is just forwarded down to the individual fields
#
# todo: define FieldNTuple by just looping over this with an @eval and $N
# inserted in-place of 2

export FieldTuple

abstract type Basis2Tuple{B1<:Basis,B2<:Basis} <: Basis end

#
# I really wish I could define this just as 
# Field2Tuple{F1<:Field{∷,∷,B1},F2<:Field{∷,∷,B2}} <: Field{Pix,Spin,Basis2Tuple{B1,B2}}
# but this doesn't exist in Julia (yet?), so instead I use this "hack"
# see also: https://discourse.julialang.org/t/could-julia-have-implicit-type-parameters/2914/5
#
@∷ struct Field2Tuple{F1<:Field,F2<:Field,B1,B2} <: Field{Pix,Spin,Basis2Tuple{B1,B2}}
    f1 :: F1
    f2 :: F2
    Field2Tuple(f1::F1,f2::F2) where {B1,B2,F1<:Field{∷,∷,B1},F2<:Field{∷,∷,B2}} = new{F1,F2,B1,B2}(f1,f2)
    Field2Tuple{F1,F2,B1,B2}(f1::F1,f2::F2) where {B1,B2,F1<:Field{∷,∷,B1},F2<:Field{∷,∷,B2}} = new{F1,F2,B1,B2}(f1,f2)
end

const F2T = Field2Tuple
FieldTuple(a,b) = F2T(a,b)
shortname(::Type{<:F2T{F1,F2}}) where {F1,F2} = "Field2Tuple{$(shortname(F1)),$(shortname(F2))}"

# Field2Tuple's data
broadcast_length(::Type{<:F2T{F1,F2}}) where {F1,F2} = broadcast_length(F1) + broadcast_length(F2)
broadcast_data(::Type{F2T{F1,F2}}, f::F2T) where {F1,F2}  = (broadcast_data(F1,f.f1)...,broadcast_data(F2,f.f2)...)
# How to broadcast other things as a Field2Tuple
broadcast_data(::Type{F2T{F1,F2}}, f::Union{Field,LinOp}) where {F1,F2} = (broadcast_data(F1,f)...,broadcast_data(F2,f)...)
# needed for ambiguity (todo: get rid of needing this...)
broadcast_data(::Type{F2T{F1,F2}}, op::FullDiagOp{<:F2T{F1,F2}}) where {F1,F2} = broadcast_data(F2T{F1,F2},op.f)

# the final data type when broadcasting things with Field2Tuple
containertype(::F2T{F1,F2}) where {F1,F2} = F2T{F1,F2}
promote_containertype(::Type{F2T{F1a,F1b}}, ::Type{F2T{F2a,F2b}}) where {F1a,F2a,F1b,F2b} = F2T{promote_containertype(F1a,F2a),promote_containertype(F1b,F2b)}
@commutative promote_containertype{F<:Field,F1,F2}(::Type{F},::Type{F2T{F1,F2}}) = F2T{promote_containertype(F,F1),promote_containertype(F,F2)}
@commutative *(a::Field,b::F2T) = a.*b
*(a::F2T,b::F2T) = a.*b 


# Reconstruct Field2Tuple from broadcasted data
function (::Type{F2T{F1,F2}})(args...) where {F1,F2}
    n = length.(fieldnames.((F1,F2)))
    F2T(F1(args[1:n[1]]...),F2(args[n[1]+1:end]...))
end

# promotion / conversion
promote_rule(::Type{<:F2T{F1a,F2a}},::Type{<:F2T{F1b,F2b}}) where {F1a,F2a,F1b,F2b} = F2T{promote_type(F1a,F1b),promote_type(F2a,F2b)}
(::Type{<:F2T{F1,F2}})(f::F2T) where {F1,F2} = F2T(F1(f.f1),F2(f.f2))
convert(::Type{<:F2T{F1,F2}},f::F2T) where {F1,F2} = F2T(F1(f.f1),F2(f.f2))

# Basis conversions
(::Type{Basis2Tuple{B1,B2}})(f::Field2Tuple) where {B1,B2} = F2T(B1(f.f1),B2(f.f2))
(::Type{B})(f::Field2Tuple) where {B<:Basis} = F2T(B(f.f1),B(f.f2))
(::Type{B})(f::Field2Tuple) where {B<:Basislike} = F2T(B(f.f1),B(f.f2)) #needed for ambiguity

# vector conversion
getindex(f::F2T,::Colon) = vcat(f.f1[:],f.f2[:])
function fromvec(::Type{F}, vec::AbstractVector) where {F1,F2,F<:F2T{F1,F2}}
    n = length.((F1,F2))
    F(fromvec(F1,(@view vec[1:n[1]])),fromvec(F2,(@view vec[n[1]+1:end])))
end
length(::Type{<:F2T{F1,F2}}) where {F1,F2} = length(F1)+length(F2)

# dot product
dot(a::F2T{F1,F2},b::F2T{F1,F2}) where {F1,F2} = dot(a.f1,b.f1)+dot(a.f1,b.f1)

# transpose multiplication (avoid storing temporary array)
Ac_mul_B(a::F2T,b::F2T) = (x=a.f1'*b.f1; x+=a.f2'*b.f2)

# for simulating
white_noise(::Type{<:F2T{F1,F2}}) where {F1,F2} = F2T(white_noise(F1),white_noise(F2))

zero(::Type{<:F2T{F1,F2}}) where {F1,F2} = F2T(zero(F1),zero(F2))
