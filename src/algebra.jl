
# addition/subtraction works between any fields and scalar, promotion is done
# automatically if fields are in different bases
for op in (:+,:-), (T1,T2) in ((:Field,:Scalar),(:Scalar,:Field),(:Field,:Field))
    @eval ($op)(a::$T1, b::$T2) = broadcast($op,($T1==$T2 ? promote : tuple)(a,b)...)
end

# multiplication/division is not strictly be defined for abstract vectors, but
# make it work anyway if the two fields are exactly the same type, in which case
# its clear we wanted broadcasted multiplication/division. 
for f in (:*, :/)
    @eval function ($f)(A::F, B::F) where {F<:Field}
        broadcast($f, A, B)
    end
end


# dot(a::Field,b::Field) = dot(promote(a,b)...)


### basis conversion

# B(f) where B is a basis converts f to that basis. This is the fallback if the
# field is already in the right basis.
(::Type{B})(f::Field{B}) where {B} = f

# The abstract `Basis` type means "any basis", hence this conversion rule:
Basis(f::Field) = f

# B(f′, f) converts f to basis B and stores the result inplace in f′. If f is
# already in basis B, we just return f (but note, we never actually set f′ in
# this case, which is more efficient, but necessitates some care when using this
# construct)
(::Type{B})(f′::Field{B}, f::Field{B}) where {B} = f



# F(f) where F is some Field type defaults to just using the basis conversion
# and asserting that we end up with the right type, F
# convert(::Type{F}, f::Field{B1}) where {B1,B2,F<:Field{B2}} = B2(f)::F

# this used to be the default in 0.6, bring it back because we use F(f) alot to
# mean convert f to a type of F
# (::Type{F})(f::Field) where {F<:Field} = convert(F,f)



# # a generic lazy adjoint
# # note: the adjoint of a LinOp{B,S,P} is not necessarily ::LinOp{B,S,P}
# # (consider e.g. the pixelization operator which produces a different P)
# struct AdjOp{L<:LinOp} <: LinOp{Basis,Spin,Pix} 
#     op::L
# end
# adjoint(L::LinOp) = AdjOp(L)
# adjoint(L::AdjOp) = L.op
# inv(L::AdjOp) = AdjOp(inv(L))
# ud_grade(lz::AdjOp, args...; kwargs...) = AdjOp(ud_grade(lz.a,args...; kwargs...))
# 
# 
# ### linear algebra of Vectors and Matrices of Fields
# 
# # alot of work needed here to make various StaticArray stuff work / infer
# # correctly... maybe at some point evaluate if its really worth it?
# 
# 
# 
# 
# ud_grade(s::Scalar, args...; kwargs...) = s
