
# accum is basically supposed to do addition, but Zygotes default for
# Arrays does a broadcast which doesnt do a potentially needed basis
# conversion.
function Zygote.accum(x::Field, y::Field)
    x === nothing ? y : 
    y === nothing ? x :
    x + y
end
Zygote.accum(x::Field, y::Field, zs::Field...) = Zygote.accum(Zygote.accum(x, y), zs...)
# TODO: we might need to a vararg version of this too:
Zygote.accum(a::FieldOp, b::FieldOp) = a + b


# constant functions, as far as AD is concerned
@nograd ProjLambert
@nograd fieldinfo
@nograd hasfield
@nograd basetype
@nograd get_storage


# AD for Fourier Fields can be really subtle because such objects are
# still supposed to represent a pixel-space field, despite that
# they're stored as the half-plane real FFT coefficients. this leads
# to needing factors of Npix(=Ny*Nx) and rfft_degeneracy_fac (which
# gives weight of 2 to coefficient which would be there twice in the
# full-plane FFT) in a few cases below to make the logic all work. the
# Npix factor is handled by the Zfac function.
Zfac(::SpatialBasis{Map},     proj::FlatProj) = 1
Zfac(::SpatialBasis{Fourier}, proj::FlatProj) = proj.Ny * proj.Nx
Zfac(L::DiagOp{<:Field{B}}) where {B} = Zfac(B(), L.diag.metadata)


## constructors and getproperty
# these are needed to allow you to "bail out" to the underlyig arrays
# via f.arr and reconstruct them with FlatField(f.arr, f.metadata). 
@adjoint function (::Type{F})(arr::A, metadata::M) where {B<:SpatialBasis{Map},M<:Proj,T,A<:AbstractArray{T},F<:BaseField{B}}
    F(arr, metadata), Δ -> (Δ.arr, nothing)
end
@adjoint function (::Type{F})(arr::A, metadata::M) where {B<:SpatialBasis{Fourier},M<:Proj,T,A<:AbstractArray{T},F<:BaseField{B}}
    F(arr, metadata), Δ -> (Δ.arr .* adapt(Δ.storage, T.(rfft_degeneracy_fac(metadata.Ny) ./ Zfac(B(), metadata))), nothing)
end
# the factors here need to cancel the ones in the corresponding constructors above
@adjoint function Zygote.literal_getproperty(f::BaseField{B}, ::Val{:arr}) where {B<:SpatialBasis{Map}}
    getfield(f,:arr), Δ -> (BaseField{B}(Δ, f.metadata),)
end
@adjoint function Zygote.literal_getproperty(f::BaseField{B,M,T}, ::Val{:arr}) where {B<:SpatialBasis{Fourier},M,T}
    getfield(f,:arr), Δ -> (BaseField{B}(Δ ./ adapt(typeof(Δ), T.(rfft_degeneracy_fac(f.Ny) ./ Zfac(B(), f.metadata))), f.metadata),)
end
# preserve field type for sub-component property getters
function _getproperty_subcomponent_pullback(f, k)
    g = zero(f)
    function getproperty_pullback(Δ)
        getproperty(g, k) .= Δ
        (g, nothing)
    end
    getproperty(f, k), getproperty_pullback
end
@adjoint function Zygote.literal_getproperty(f::BaseField{B}, k::Union{typeof.(Val.((:I,:Q,:U,:E,:B)))...}) where {B₀, B<:SpatialBasis{B₀}}
    _getproperty_subcomponent_pullback(f, k)
end
@adjoint function Zygote.literal_getproperty(f::BaseS02{Basis3Prod{𝐈,B₂,B₀}}, k::Val{:P}) where {B₂,B₀}
    _getproperty_subcomponent_pullback(f, k)
end
# if accumulting from one branch that was just a f.metadata
Zygote.accum(f::BaseField, nt::NamedTuple{(:arr,:metadata)}) = (@assert(isnothing(nt.arr)); f)

# FieldTuple
@adjoint (::Type{FT})(fs) where {FT<:FieldTuple} = FT(fs), Δ -> (Δ.fs,)
@adjoint Zygote.literal_getproperty(f::FieldTuple, ::Val{:fs}) = getfield(f,:fs), Δ -> (FieldTuple(map((f,f̄) -> isnothing(f̄) ? zero(f) : f̄, getfield(f,:fs), Δ)),)

# BatchedReals
@adjoint Zygote.literal_getproperty(br::BatchedReal, ::Val{:vals}) = getfield(br,:vals), Δ -> (batch(real.(Δ)),)



## Field algebra

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
@adjoint logdet(L::ParamDependentOp, θ) = Zygote.pullback((L,θ)->logdet(L(θ)), L, θ)
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
# todo: need to fix this to allow gradient w.r.t. entries of a BlockDiagIEB
@adjoint *(L::BlockDiagIEB, f::Field{B}) where {B} = L*f, Δ->(nothing, B(L'*Δ))
@adjoint \(L::BlockDiagIEB, f::Field{B}) where {B} = L\f, Δ->(nothing, B(L'\Δ))


## FieldVectors

# following two definitions are almost definitely not totally right w.r.t
# putting stuff in the correct basis, although they're working for everything
# I've needed thus far

@adjoint function *(x::FieldOrOpRowVector, y::FieldVector)
    z = x * y
    # when x is a vector of Fields
    back(Δ::Real) = ((Δ * y)', x' * Δ)
    # when x is a vector of Diagonals. in this case, Δ * basis(Δ)(y)'
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
    L⁻¹, Δ->(@thunk(-L⁻¹' * (Δ * L⁻¹')),)
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
# see https://github.com/FluxML/Zygote.jl/pull/881
@adjoint function cat(Xs::AbstractArray...; dims)
    cat(Xs...; dims = dims), Δ -> begin
        start = ntuple(_ -> 0, ndims(Δ))
        catdims = Base.dims2cat(dims)
        dXs = map(Xs) do x
            move = ntuple(d -> (d<=length(catdims) && catdims[d]) ? size(x,d) : 0, ndims(Δ))
            x_in_Δ = ntuple(d -> (d<=length(catdims) && catdims[d]) ? (start[d]+1:start[d]+move[d]) : Colon(), ndims(Δ))
            start = start .+ move
            dx = Δ[x_in_Δ...]
        end
    end
end

# todo: basetype needed because it seems sometimes the array comes
# back complex when it was real on the forward pass. seems likely that
# thats a consequences of some incorrect/missing ProjectTo's
# somewhere, find them...
@adjoint adapt(to, x::A) where {A<:AbstractArray} = adapt(to, x), Δ -> (nothing, adapt(basetype(A), Δ))

# finite difference Hessian using Zygote gradients
# todo: delete, just use FiniteDifferences
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


# new ChainRules ProjectTo interface. with these here, I think theres
# a good chance many of the above rules can simply be deleted, but
# haven't gone through yet to figure out which ones yet
ProjectTo(::F) where {F<:Field} = ProjectTo{F}()
ProjectTo(::L) where {L<:FieldOp} = ProjectTo{L}()
(project::ProjectTo{F})(dx::Field) where {B, F<:Field{B}} = B(dx)
(project::ProjectTo{L})(dx::FieldOp) where {L<:FieldOp} = dx

Zygote.wrap_chainrules_output(dxs::LazyBinaryOp) = dxs

# needed to allow AD through field broadcasts
Zygote.unbroadcast(x::BaseField{B}, x̄::BaseField) where {B} = 
    BaseField{B}(Zygote.unbroadcast(x.arr, x̄.arr), x.metadata)
