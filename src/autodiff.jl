
# this does basis promotion, unlike Zygote's default for AbstractArrays
Zygote.accum(a::Field, b::Field) = a+b
# this may create a LazyBinaryOp, unlike Zygote's
Zygote.accum(a::FieldOp, b::FieldOp) = a+b


# constant functions, as far as AD is concerned
@nograd ProjLambert
@nograd fieldinfo
@nograd hasfield


const SpatialBasis{B,I,P} = Union{B, Basis2Prod{P,B}, Basis3Prod{I,P,B}}

# adjoint constructors
@adjoint function (::Type{F})(arr::A, metadata::M) where {B<:SpatialBasis{Map},M<:FieldMetadata,T,A<:AbstractArray{T},F<:BaseField{B}}
    F(arr, metadata), Δ -> (Δ.arr, nothing)
end
@adjoint function (::Type{F})(arr::A, metadata::M) where {B<:SpatialBasis{Fourier},M<:FieldMetadata,T,A<:AbstractArray{T},F<:BaseField{B}}
    F(arr, metadata), Δ -> (Δ.arr .* adapt(Δ.storage, rfft_degeneracy_fac(metadata.Ny) ./ Zfac(B(), metadata)), nothing)
end

@adjoint (::Type{FT})(fs) where {FT<:FieldTuple} = FT(fs), Δ -> (Δ.fs,)



## Fields

# A Diagonal(::Field) is a really subtle object in terms of AD bc of
# all the implicit FFT operations (ie auto basis-conversions) that it
# inserts. this leads to needing a factor of Npix(=Ny*Nx) for
# Diagonal(::Fourier) in a few cases below to make the logic all work.
# this is handled by this Zfac function.
Zfac(::SpatialBasis{Map},     proj::FlatProj) = 1
Zfac(::SpatialBasis{Fourier}, proj::FlatProj) = proj.Ny * proj.Nx
Zfac(L::DiagOp{<:Field{B}}) where {B} = Zfac(B(), L.diag.metadata)

# Zygote has lots of rules for AbstractVectors / AbstractMatrices that
# don't quite work right due to the auto-basis conversions done for
# Fields, or which work right but trigger scalar indexing (thus don't
# work on GPU). this leads us to need a few more custom rules below
# than might be ideal, although its not too bad. see also: 
# https://discourse.julialang.org/t/how-to-deal-with-zygote-sometimes-pirating-its-own-adjoints-with-worse-ones

# ℝᴺ -> ℝ¹ 
@adjoint sum(f::Field{B}) where {B} = sum(f), Δ -> (Δ*one(f),)
@adjoint norm(f::Field) = Zygote.pullback(f->sqrt(dot(f,f)), f)
@adjoint dot(f::Field{B1}, g::Field{B2}) where {B1,B2} = dot(f,g), Δ -> (Δ*B1(g), Δ*B2(f))
@adjoint (*)(f::Adjoint{<:Any,<:Field}, g::Field) = Zygote.pullback((f,g)->dot(f',g),f,g)
# ℝᴺˣᴺ -> ℝ¹ 
@adjoint logdet(L::ParamDependentOp, θ) = Zygote._pullback(θ->logdet(L(;θ...)), θ)
@adjoint logdet(L::DiagOp) = logdet(L), Δ -> (Δ * Zfac(L) * pinv(L)',)

# basis conversion
@adjoint (::Type{B})(f::Field{B′}) where {B<:Basis, B′} = B(f), Δ -> (B′(Δ),)

# algebra
@adjoint (+)(f::Field{B1}, g::Field{B2}) where {B1,B2} = f+g, Δ -> (B1(Δ), B2(Δ))

@adjoint (*)(a::Real, L::DiagOp) = a*L, Δ -> (tr(L'*Δ)/Zfac(L), a*Δ)
@adjoint (*)(L::DiagOp, a::Real) = a*L, Δ -> (a*Δ, tr(L'*Δ)/Zfac(L))

# operators
@adjoint function (*)(D::DiagOp{<:Field{B}}, v::Field{B′}) where {B,B′}
    D*v, Δ -> (Diagonal(B(Δ) .* conj.(B(v))), B′(D'*Δ))
end
@adjoint function (\)(D::DiagOp{<:Field{B}}, v::Field{B′}) where {B,B′}
    z = D \ v
    function back(Δ)
        v̄ = D' \ Δ
        -Diagonal(B(v̄) .* conj.(B(z))), B′(v̄)
    end
    z, back
end
# this makes it so we only have to define a L*f and L\f adjoint (like
# above) for any new operators, and we get f'*L and f'/L for free
# without necessarily needing to fully implement the AbstractMatrix
# interface for L
@adjoint *(f::Adjoint{<:Any,<:Field}, L::Union{DiagOp,ImplicitOp}) = Zygote.pullback((f,L)->(L'*f')', f, L)
@adjoint /(f::Adjoint{<:Any,<:Field}, L::Union{DiagOp,ImplicitOp}) = Zygote.pullback((f,L)->(L'\f')', f, L)
# special case for some ops which are constant by definition
@adjoint *(L::Union{FuncOp,DiagOp{<:∇diag}}, f::Field{B}) where {B} = L*f, Δ->(nothing, B(L'*Δ))


# don't use the AbstractMatrix * AbstractVector rule here, just go
# through the actual definition of these things
@adjoint *(L::LazyBinaryOp,     f::Field) = Zygote.pullback((L,f)->L.X*(L.Y*f), L, f)
@adjoint *(L::ParamDependentOp, f::Field) = Zygote.pullback((L,f)->L()*f,       L, f)


# these make things like gradient(f->f.arr[1], f) return a Field rather than a NamedTuple
@adjoint Zygote.literal_getproperty(f::BaseField{B}, ::Val{:arr}) where {B} = getfield(f,:arr), Δ -> (BaseField{B}(Δ, f.metadata),)
@adjoint Zygote.literal_getproperty(f::FieldTuple,   ::Val{:fs}) = getfield(f,:fs), Δ -> (FieldTuple(map((f,f̄) -> isnothing(f̄) ? zero(f) : f̄, getfield(f,:fs), Δ)),)


## FieldVectors

# following two definitions are almost definitely not totally right w.r.t
# putting stuff in the correct basis, although they're working for everything
# I've needed thus far

@adjoint function *(x::FieldOrOpRowVector, y::FieldVector)
    z = x * y
    # when x is a vector of Fields
    back(Δ) = (Δ * y', x' * Δ)
    # when x is a vector of Diagonals. in this case, Δ * basis(Δ)(y)' create an
    # OuterProdOp in the same basis as the Diagonals in x
    back(Δ::Field{B}) where {B} = (Δ * basis(Δ)(y)'), (x' * Δ)
    z, back
end

@adjoint function *(A::FieldOrOpMatrix, x::FieldOrOpVector)
    z = A * x
    back(Δ::FieldVector) = (B=basis(eltype(z)); (B(Δ) * B(x)', A' * Δ))
    back(Δ::FieldOrOpVector) = (Δ * x', A' * Δ) # not sure if anything special needed here?
    back(Δ) = (Δ * x', A' * Δ)
    z, back
end

# don't know why Zygote's default adjoint for this breaks in various ways but this is simple enough
@adjoint +(I::UniformScaling, L::Union{FieldOp, FieldOrOpMatrix}) = I+L, Δ->(nothing, Δ)

# Zygote/lib/array.jl:311 would suggest this should be:
#    M⁻¹, Δ->(-M⁻¹' * Δ * M⁻¹' + (- M * M⁻¹ * Δ' * M⁻¹ * M⁻¹' + Δ' * M⁻¹ * M⁻¹') + (M⁻¹' * M⁻¹ * Δ' - M⁻¹' * M⁻¹ * Δ' * M⁻¹ * M),)
# I haven't derived their version, but numerically the one gives the right answer where as their doesn't...
@adjoint function pinv(L::Union{FieldOp, FieldOrOpMatrix})
    L⁻¹ = pinv(L)
    L⁻¹, Δ->(-L⁻¹' * Δ * L⁻¹',)
end

@adjoint sqrt(L::DiagOp) = (z=sqrt(L);), Δ -> ((pinv(z)/2)'*Δ,)


# some stuff which arguably belongs in Zygote or ChainRules
# see also: https://github.com/FluxML/Zygote.jl/issues/316

@adjoint broadcasted(::typeof(\), x ::Numeric, y::Numeric) =
    broadcast(\, x, y), Δ -> (nothing, unbroadcast(x, @. -Δ*y/x^2), unbroadcast(y, @. Δ/x))

@adjoint (::Type{SA})(tup) where {SA<:SArray} = SA(tup), Δ->(tuple(Δ...),)

# workaround for https://github.com/FluxML/Zygote.jl/issues/686
@static if versionof(Zygote) > v"0.4.15"
    Zygote._zero(xs::StaticArray, T) = SizedArray{Tuple{size(xs)...},Union{T,Nothing}}(map(_->nothing, xs))
end

# workaround for Zygote not working through cat when dims is a Val
# adapted from solution by Seth Axen 
@adjoint function cat(Xs::AbstractArray...; dims)
    cat(Xs...; dims = dims), Δ -> begin
        start = ntuple(_ -> 0, ndims(Δ))
        catdims = Base.dims2cat(dims)
        dXs = map(Xs) do x
            move = ntuple(d -> (d<=length(catdims) && catdims[d]) ? size(x,d) : 0, ndims(Δ))
            x_in_Δ = ntuple(d -> (d<=length(catdims) && catdims[d]) ? (start[d]+1:start[d]+move[d]) : Colon(), ndims(Δ))
            start = start .+ move
            dx = reshape(Δ[x_in_Δ...], size(x))
        end
    end
end


# finite difference Hessian using Zygote gradients
function hessian(f, xs::Vector; ε=1f-3)
    hcat(finite_difference(xs->vcat(gradient(f,xs)[1]...),xs,ε=ε)...)
end

function finite_difference(f, xs::Vector; ε=1f-3, progress=false)
    @showprogress (progress ? 1 : Inf) map(1:length(xs)) do i
        xs₊ = copy(xs); xs₊[i] += ε
        xs₋ = copy(xs); xs₋[i] -= ε
        (f(xs₊) .- f(xs₋)) ./ (2ε)
    end
end
