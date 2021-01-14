    

const PolBasis{I,P,B} = Union{B, Basis2Prod{P,B}, Basis3Prod{I,P,B}}

Base.iterate(::Val{x}) where x = Base.iterate(x)

# adjoint constructors
@nograd ProjLambert

@adjoint function (::Type{F})(arr::A, metadata::M) where {B<:PolBasis{<:Any,<:Any,Map},M<:FieldMetadata,T,A<:AbstractArray{T},F<:BaseField{B}}
    F(arr, metadata), Δ->(Δ.arr, nothing)
end

@adjoint function (::Type{F})(arr::A, metadata::M) where {B<:PolBasis{<:Any,<:Any,Fourier},M<:FieldMetadata,T,A<:AbstractArray{T},F<:BaseField{B}}
    function back(Δ)
        @unpack Ny, Nx, storage = Δ
        fac = adapt(storage, rfft_degeneracy_fac(Ny) ./ (Ny*Nx))
        (Δ.arr .* fac, nothing)
    end
    F(arr, metadata), back
end
    
# @adjoint (::Type{FT})(fs::Tuple) where {FT<:FieldTuple} = FT(fs), Δ -> (values(Δ.fs),)


# lazy outer products of Fields, which comes up alot in automatic differentiation
*(x::Field, y::Adjoint{<:Any, <:Field}) = OuterProdOp(x, y.parent)

# this does basis promotion, unlike Zygote's default for AbstractArrays
Zygote.accum(a::Field, b::Field) = a+b
# this may create a LazyBinaryOp, unlike Zygote's
Zygote.accum(a::FieldOp, b::FieldOp) = a+b

## Fields

# we have to define several custom adjoints which account for the automatic
# basis conversion which CMBLensing does. part of the reason we have to do this
# explicilty is because of:
# https://discourse.julialang.org/t/how-to-deal-with-zygote-sometimes-pirating-its-own-adjoints-with-worse-ones


# ℝᴺ -> ℝ¹ 
@adjoint sum(f::Field{B}) where {B} = sum(f), Δ -> (Δ*one(f),)
@adjoint norm(f::Field) = Zygote.pullback(f->sqrt(dot(f,f)), f)
@adjoint dot(f::Field{B1}, g::Field{B2}) where {B1,B2} = dot(f,g), Δ -> (Δ*B1(g), Δ*B2(f))
@adjoint *(f::Adjoint{<:Any,<:Field}, g::Field) = Zygote.pullback((f,g)->dot(f',g),f,g)
# ℝᴺˣᴺ -> ℝ¹ 
@adjoint logdet(L::ParamDependentOp, θ) = Zygote._pullback(θ->logdet(L(;θ...)), θ) # dont need to take out offset here like in 
@adjoint logdet(L::DiagOp{<:Field{B}}) where {B<:PolBasis{<:Any,<:Any,Map}} = logdet(L), Δ -> (Δ * pinv(L)',)
@adjoint logdet(L::DiagOp{<:Field{B}}) where {B<:PolBasis{<:Any,<:Any,Fourier}} = begin
    @unpack Ny, Nx = L.diag
    logdet(L), Δ -> (Δ * pinv(L)',)
end

@adjoint Diagonal(f::FlatField{B}) where {B<:PolBasis{<:Any,<:Any,Fourier}} = begin
    @unpack Ny, Nx = f
    function back(Δ)
        @show typeof(Δ)
        (diag(Δ) * (Nx*Ny),)
    end
    function back(Δ::OuterProdOp)
        @show typeof(Δ)
        (diag(Δ),)
    end
    Diagonal(f), back
end



# basis conversion
@adjoint (::Type{B})(f::Field{B′}) where {B<:Basis, B′} = B(f), Δ -> (B′(Δ),)

# algebra
@adjoint +(f::Field{B1}, g::Field{B2}) where {B1,B2} = f+g, Δ -> (B1(Δ), B2(Δ))
@adjoint *(a::Real, f::Field{B}) where {B} = a*f, Δ -> (f'*Δ, B(Δ*a))
@adjoint *(a::Real, L::DiagOp) = a*L, Δ -> (tr(L'*Δ), a*Δ) # need to use trace here since it unfolds the diagonal

# operators
@adjoint *(D::DiagOp{<:Field{B}}, v::Field{B′}) where {B,B′} = D*v, Δ->(B(Δ)*B(v)', B′(D'*Δ))
@adjoint \(D::DiagOp{<:Field{B}}, v::Field{B′}) where {B,B′} = begin
    z = D \ v
    function back(Δ)
        v̄ = D' \ Δ
        -B(v̄)*B(z)', B′(v̄)
    end
    z, back
end
@adjoint *(∇::DiagOp{<:∇diag}, f::Field{B}) where {B} = ∇*f, Δ->(nothing, B(∇'*Δ))
# this makes it so we only have to define adjoints for L*f, and the f'*L adjoint just uses that
@adjoint *(f::Adjoint{<:Any,<:Field}, D::Diagonal) = Zygote.pullback((f,D)->(D'*f')', f, D)

# properties
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
if versionof(Zygote) > v"0.4.15"
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

# functions with no gradient which Zygote would otherwise fail on

@nograd fieldinfo


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
