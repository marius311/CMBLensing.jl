# accum is basically supposed to do addition, but Zygotes default for
# Arrays does a broadcast which doesnt do a potentially needed basis
# conversion.
function _plus_accum(x, y)
    x === nothing ? y : 
    y === nothing ? x :
    x + y
end
_plus_accum(x, y, zs...) = Zygote.accum(Zygote.accum(x, y), zs...)
Zygote.accum(x::Field, y::Field) = _plus_accum(x, y)
Zygote.accum(x::Field, y::Field, zs::Field...) = _plus_accum(x, y, zs...)
Zygote.accum(x::FieldOp, y::FieldOp) = _plus_accum(x, y)
Zygote.accum(x::FieldOp, y::FieldOp, zs::FieldOp...) = _plus_accum(x, y, zs...)


# constant functions, as far as AD is concerned
ChainRulesCore.@non_differentiable ProjLambert(::Any...)
ChainRulesCore.@non_differentiable fieldinfo(::Any...)
ChainRulesCore.@non_differentiable hasfield(::Any...)
ChainRulesCore.@non_differentiable basetype(::Any...)
ChainRulesCore.@non_differentiable get_storage(::Any...)


# AD for Fourier Fields can be really subtle because such objects are
# still supposed to represent a pixel-space field, despite that
# they're stored as the half-plane real FFT coefficients. this leads
# to needing factors of Npix(=Ny*Nx) and λ_rfft (which
# gives weight of 2 to coefficient which would be there twice in the
# full-plane FFT) in a few cases below to make the logic all work. the
# Npix factor is handled by the Zfac function.
Zfac(::SpatialBasis{Map},       proj::CartesianProj) = 1
Zfac(::SpatialBasis{Fourier},   proj::CartesianProj) = proj.Ny * proj.Nx
Zfac(::SpatialBasis{AzFourier}, proj::CartesianProj) = 1 # no factor needed here bc of √nφ in AzFourier(::Map)
Zfac(L::DiagOp{<:Field{B}}) where {B} = Zfac(B(), L.diag.metadata)


## constructors and getproperty
# these are needed to allow you to "bail out" to the underlyig arrays
# via f.arr and reconstruct them with FlatField(f.arr, f.metadata). 
@adjoint function (::Type{F})(arr::A, metadata::M) where {B<:SpatialBasis{Map},M<:Proj,T,A<:AbstractArray{T},F<:BaseField{B}}
    F(arr, metadata), Δ -> (Δ.arr, nothing)
end
@adjoint function (::Type{F})(arr::A, metadata::M) where {B<:SpatialBasis{Fourier},M<:Proj,T,A<:AbstractArray{T},F<:BaseField{B}}
    F(arr, metadata), Δ -> (Δ.arr .* Δ.λ_rfft ./ Zfac(B(), metadata), nothing)
end
@adjoint function (::Type{F})(arr::A, metadata::M) where {B<:SpatialBasis{AzFourier},M<:Proj,T,A<:AbstractArray{T},F<:BaseField{B}}
    F(arr, metadata), Δ -> (Δ.arr .* Δ.λ_rfft ./ Zfac(B(), metadata), nothing)
end

# EquiRect's scalar azimuthal transform is an orthonormal rFFT along
# dimension 2. Unlike Lambert projections, its metadata does not store rFFT
# degeneracies. Construct those weights in the storage backend of the field.
function _equirect_az_rfft_weights(f::BaseField{AzFourier,<:ProjEquiRect})
    weights = reshape(real(eltype(f)).(rfft_degeneracy_fac(f.Nx)), 1, :)
    adapt(f.metadata.storage, weights)
end
@adjoint function (::Type{F})(arr::A, metadata::M) where {M<:ProjEquiRect,T,A<:AbstractArray{T},F<:BaseField{AzFourier}}
    field = F(arr, metadata)
    function equirect_az_constructor_pullback(Δ)
        arr_gradient = similar(arr)
        arr_gradient .= Δ.arr .* _equirect_az_rfft_weights(field)
        (arr_gradient, nothing)
    end
    field, equirect_az_constructor_pullback
end
# the factors here need to cancel the ones in the corresponding constructors above
@adjoint function Zygote.literal_getproperty(f::BaseField{B}, ::Val{:arr}) where {B<:SpatialBasis{Map}}
    getfield(f,:arr), Δ -> (BaseField{B}(Δ, f.metadata),)
end
@adjoint function Zygote.literal_getproperty(f::BaseField{B,M,T}, ::Val{:arr}) where {B<:SpatialBasis{Fourier},M,T}
    getfield(f,:arr), Δ -> (BaseField{B}(Δ ./ f.λ_rfft .* Zfac(B(), f.metadata), f.metadata),)
end
@adjoint function Zygote.literal_getproperty(f::BaseField{B,M,T}, ::Val{:arr}) where {B<:SpatialBasis{AzFourier},M,T}
    getfield(f,:arr), Δ -> (BaseField{B}(Δ ./ f.λ_rfft .* Zfac(B(), f.metadata), f.metadata),)
end
@adjoint function Zygote.literal_getproperty(f::BaseField{AzFourier,<:ProjEquiRect}, ::Val{:arr})
    function equirect_az_array_pullback(Δ)
        arr_gradient = similar(f.arr)
        arr_gradient .= Δ ./ _equirect_az_rfft_weights(f)
        (BaseField{AzFourier}(arr_gradient, f.metadata),)
    end
    getfield(f,:arr), equirect_az_array_pullback
end
# needed to preserve field type for sub-component property getters
@adjoint function Zygote.getproperty(f::BaseField, k::Union{typeof.(Val.((:I,:Q,:U,:E,:B,:P,:IP)))...})
    function field_getproperty_pullback(Δ)
        g = (similar(f, promote_type(eltype(f), eltype(Δ))) .= 0)
        getproperty(g, k) .= Δ
        (g, nothing)
    end
    getproperty(f, k), field_getproperty_pullback
end
# if accumulting from one branch that was just a f.metadata
Zygote.accum(f::BaseField, nt::NamedTuple{(:arr,:metadata)}) = (@assert(isnothing(nt.arr)); f)

# FieldTuple
@adjoint (::Type{FT})(fs) where {FT<:FieldTuple} = FT(fs), Δ -> (Δ.fs,)
@adjoint function Zygote.literal_getproperty(f::FieldTuple, ::Val{:fs})
    getfield(f,:fs), Δ -> (FieldTuple(map((f,f̄) -> isnothing(f̄) ? zero(f) : f̄, getfield(f,:fs), Δ)),)
end
@adjoint function Zygote.getproperty(f::FieldTuple, ::Val{k}) where {k}
    function fieldtuple_getproperty_pullback(Δ)
        g = (similar(f, promote_type(eltype(f), eltype(Δ))) .= 0)
        getproperty(g, k) .= Δ
        (g, nothing)
    end
    getproperty(f,Val(k)), fieldtuple_getproperty_pullback
end

# BatchedReals
@adjoint Zygote.literal_getproperty(br::BatchedReal, ::Val{:vals}) = getfield(br,:vals), Δ -> (batch(real.(Δ)),)



## Field algebra

# Zygote has lots of rules for AbstractVectors / AbstractMatrices that
# don't quite work right due to the auto-basis conversions done for
# Fields, or which work right but trigger scalar indexing (thus don't
# work on GPU). this leads us to need a few more custom rules below
# than might be ideal, although its not too bad. see also: 
# https://discourse.julialang.org/t/how-to-deal-with-zygote-sometimes-pirating-its-own-adjoints-with-worse-ones

# have to be careful for reductions to ℝ¹ since if
# set_sum_accuracy_mode(Float64) the return value might be higher
# percision than input fields. (also it might be a Dual for
# higher-order diff)

# ℝᴺ -> ℝ¹ 
@adjoint sum(f::Field{B,T}) where {B,T} = sum(f), Δ -> (real(T)(Δ) * one(f),)
# `sum` sums the stored half-plane coefficients. Their field inner product
# has an rFFT degeneracy weight and an FFT normalization, so the cotangent
# is not a Fourier field filled with ones. Keep complex seeds for objectives
# such as `imag(sum(f))`.
@adjoint function sum(f::BaseField{B,M,T}) where {B<:SpatialBasis{Fourier},M<:ProjLambert,T}
    function sum_fourier_pullback(Δ)
        seed = T <: Real ? T(real(Δ)) : T(Δ)
        arr = similar(f.arr)
        arr .= seed .* Zfac(B(), f.metadata) ./ f.λ_rfft
        (BaseField{B}(arr, f.metadata),)
    end
    sum(f), sum_fourier_pullback
end
# Each component's sum has its own storage and inner-product convention.
# In particular, a Fourier component cannot use a field filled with ones.
@adjoint function sum(ft::FieldTuple)
    function sum_fieldtuple_pullback(Δ)
        (Δ === nothing || Δ isa ChainRulesCore.AbstractZero) && return (nothing,)
        gradients = map(getfield(ft, :fs)) do component
            value, back = Zygote.pullback(sum, component)
            component_gradient, = back(ProjectTo(value)(Δ))
            component_gradient === nothing || component_gradient isa ChainRulesCore.AbstractZero ?
                zero(component) : component_gradient
        end
        (FieldTuple(gradients),)
    end
    sum(ft), sum_fieldtuple_pullback
end
@adjoint norm(f::Field) = Zygote.pullback(f->sqrt(dot(f,f)), f)
@adjoint dot(f::Field{B1,T1}, g::Field{B2,T2}) where {B1,B2,T1,T2} = dot(f,g), Δ -> (real(T1)(Δ)*B1(g), real(T2)(Δ)*B2(f))
@adjoint (*)(f::Adjoint{<:Any,<:Field}, g::Field) = Zygote.pullback((f,g)->dot(f',g),f,g)
# ℝᴺˣᴺ -> ℝ¹ 
@adjoint logdet(L::ParamDependentOp, θ) = Zygote.pullback((L,θ)->logdet(L(θ)), L, θ)
@adjoint logdet(L::DiagOp{F,T}) where {F<:Field, T} = logdet(L), Δ -> (real(T)(Δ) * Zfac(L) * pinv(L)',)

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

## FieldVectors

# following two definitions are almost definitely not totally right w.r.t
# putting stuff in the correct basis, although they're working for everything
# I've needed thus far

@adjoint adjoint(x::FieldOrOpArray) = x', Δ -> (Δ',)

@adjoint function *(x::FieldOrOpRowVector, y::FieldVector)
    z = x * y
    # when x is a vector of Fields
    back(Δ::Real) = ((Δ * y)', x' * Δ)
    # when x is a vector of Diagonals.
    back(Δ::Field{B}) where {B} = (Δ * B(y)'), (x' * Δ)
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
    field_pinv_pullback(Δ) = (@thunk(-L⁻¹' * (Δ * L⁻¹')),)
    L⁻¹, field_pinv_pullback
end

@adjoint function sqrt(L::DiagOp)
    z = sqrt(L)
    field_sqrt_pullback(Δ) = (Diagonal((pinv(z)/2)'*Δ),)
    z, field_sqrt_pullback
end


# some stuff which arguably belongs in Zygote or ChainRules
# see also: https://github.com/FluxML/Zygote.jl/issues/316

@adjoint broadcasted(::typeof(\), x ::Numeric, y::Numeric) =
    broadcast(\, x, y), Δ -> (nothing, unbroadcast(x, @. -Δ*y/x^2), unbroadcast(y, @. Δ/x))

@adjoint (::Type{SA})(tup) where {SA<:SArray} = SA(tup), Δ->(tuple(Δ...),)


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

# Find storage through the standard array wrappers that can nest around it.
# Other wrappers keep the previous adaptation behavior.
_field_adapt_storage_root(x) = nothing
_field_adapt_storage_root(x::DenseArray) = x
_field_adapt_storage_root(x::Union{SubArray,Base.ReshapedArray,Base.PermutedDimsArray}) =
    _field_adapt_storage_root(parent(x))

# A field-backed diagonal operator is itself an AbstractArray, but its
# wrapper type is not a storage backend. Move its cotangent back to the
# backend of the field's underlying array.
@adjoint function adapt(to, x::DiagOp{<:BaseField})
    adapted = adapt(to, x)
    function adapt_field_diagonal_pullback(Δ)
        if Δ === nothing || Δ isa ChainRulesCore.AbstractZero
            return (nothing, Δ)
        end
        arr = diag(x).arr
        root = _field_adapt_storage_root(arr)
        backend = basetype(typeof(root === nothing ? parent(arr) : root))
        (nothing, adapt(backend, Δ))
    end
    adapted, adapt_field_diagonal_pullback
end

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

# ProjectTo and _eltype_projectto support for ChainRules compatibility
# This is needed for StaticArrays and other packages that rely on ProjectTo
ChainRulesCore._eltype_projectto(::Type{F}) where {F<:BaseField} = ProjectTo{F}()
ChainRulesCore._eltype_projectto(::Type{SA}) where {SA<:SArray} = ProjectTo{SA}()

# Allow multiplication of SVector{2} (like from gradients) with Fields
*(v::SVector{2, <:Real}, f::Field) = @SVector[v[1]*f, v[2]*f]

# Reduce broadcast dimensions without narrowing precision. Contributions must
# be added before projection, so cancellation is not lost to early rounding.
function _svector_unbroadcast(x::AbstractArray, δ::AbstractArray)
    if length(x) == length(δ)
        reshape(δ, size(x))
    else
        dims = ntuple(d -> size(x, d) == 1 ? d : ndims(δ) + 1, ndims(δ))
        reshape(sum(δ; dims=dims), size(x))
    end
end

# A plain scalar is shared by all batches; a BatchedReal has one value per
# batch (or a single broadcast value).
_svector_reduce_coefficient(v::Real, δ::Real) = δ
_svector_reduce_coefficient(v::Real, δ::BatchedReal) = sum(δ.vals)
_svector_reduce_coefficient(v::BatchedReal, δ::BatchedReal) = batch(_svector_unbroadcast(v.vals, δ.vals))
_svector_reduce_coefficient(v::BatchedReal, δ::Real) = batch(_svector_unbroadcast(v.vals, [δ]))
_svector_project_coefficient(v::Real, δ::Real) = ProjectTo(v)(δ)
_svector_project_coefficient(v::BatchedReal, δ::BatchedReal) = batch(ProjectTo(v.vals)(δ.vals))

# EquiRect's dot reduces all array dimensions; preserve the batch dimension
# here so scalar and BatchedReal coefficients can be projected independently.
_svector_field_dot(f, δ) = dot(f, δ)
function _svector_field_dot(f::EquiRectField, δ::EquiRectField)
    z = conj.(Ł(f)) .* Ł(δ)
    batch(sum_dropdims(real.(z.arr), dims=nonbatch_dims(z)))
end
_svector_coefficient_field_tangent(v::Real, f, δ) = _svector_reduce_coefficient(v, _svector_field_dot(f, δ))
function _svector_coefficient_field_tangent(v::Real, f::FieldTuple, δ::FieldTuple)
    f′, δ′ = promote(f, δ)
    # Components may have different batch sizes; reduce each to the
    # coefficient's shape before adding their contributions.
    reduce(+, map((x, y) -> _svector_coefficient_field_tangent(v, x, y),
                  getfield(f′, :fs), getfield(δ′, :fs)); init=0)
end

# Broadcasted scalar multiplication can expand an unbatched field. Reduce
# each branch to the input shape and basis, then project precision after adding.
_svector_reduce_field(f::Field{B}, δ::Field) where {B} = B(δ)
function _svector_reduce_field(f::BaseField{B}, δ::BaseField) where {B}
    δ′ = B(δ)
    BaseField{B}(_svector_unbroadcast(f.arr, δ′.arr), δ′.metadata)
end
_svector_reduce_field(f::FieldTuple, δ::FieldTuple) = FieldTuple(map(_svector_reduce_field, getfield(f, :fs), getfield(δ, :fs)))
_svector_reduce_field(f::Real, δ::Real) = _svector_reduce_coefficient(f, δ)
_svector_reduce_field(f::AbstractArray, δ::AbstractArray) = _svector_unbroadcast(f, δ)

_svector_project_field(f::Field{B}, δ::Field) where {B} = B(δ)
_svector_project_field(f::BaseField{B}, δ::BaseField) where {B} = Zygote.unbroadcast(f, B(δ))
_svector_project_field(f::FieldTuple, δ::FieldTuple) = FieldTuple(map(_svector_project_field, getfield(f, :fs), getfield(δ, :fs)))
_svector_project_field(f::Real, δ::Real) = _svector_project_coefficient(f, δ)
_svector_project_field(f::AbstractArray, δ::AbstractArray) = ProjectTo(f)(δ)

# Field tangents use the field inner product, even for Fourier storage.
@adjoint function *(v::SVector{2, <:Real}, f::Field)
    result = v * f
    function svector_field_pullback(Δ)
        is_zero_tangent(x) = x === nothing || x isa ChainRulesCore.AbstractZero
        is_zero_tangent(Δ) && return (nothing, nothing)
        δ₁, δ₂ = Δ
        v̄₁ = is_zero_tangent(δ₁) ? nothing : _svector_coefficient_field_tangent(v[1], f, δ₁)
        v̄₂ = is_zero_tangent(δ₂) ? nothing : _svector_coefficient_field_tangent(v[2], f, δ₂)
        isnothing(v̄₁) && isnothing(v̄₂) && return (nothing, nothing)
        v̄₁ = isnothing(v̄₁) ? zero(v[1]) : _svector_project_coefficient(v[1], v̄₁)
        v̄₂ = isnothing(v̄₂) ? zero(v[2]) : _svector_project_coefficient(v[2], v̄₂)
        v_grad = SVector{2,typejoin(typeof(v̄₁), typeof(v̄₂))}((v̄₁, v̄₂))
        # Reduce each branch before adding: their output batch sizes may differ.
        f̄₁ = is_zero_tangent(δ₁) ? nothing : _svector_reduce_field(f, v[1] * δ₁)
        f̄₂ = is_zero_tangent(δ₂) ? nothing : _svector_reduce_field(f, v[2] * δ₂)
        return (v_grad, _svector_project_field(f, _plus_accum(f̄₁, f̄₂)))
    end
    return result, svector_field_pullback
end

# needed to allow AD through field broadcasts
Zygote.unbroadcast(x::BaseField{B}, x̄::BaseField) where {B} = 
    BaseField{B}(Zygote.unbroadcast(x.arr, x̄.arr), x.metadata)



## ForwardDiff rules 
# mainly FFTs which don't work, some work upstream here
# https://github.com/JuliaDiff/ForwardDiff.jl/pull/541 but its not
# incorporated merged

# FFT is a linear operator so FFT of a vector of duals is just an FFT
# of the values and partials separately

function apply_plan(op, plan, arr::AbstractArray{Dual{T,V,N}}) where {T,V,N}
    value_arr = op(plan, value.(arr))
    partials_arrs = ntuple(i -> op(plan, partials.(arr, i)), Val(N))
    return (value_arr, partials_arrs)
end

function apply_plan(op, plan, arr::AbstractArray{<:Complex{Dual{T,V,N}}}) where {T,V,N}
    value_arr = op(plan, complex.(value.(real.(arr)), value.(imag.(arr))))
    partials_arrs = ntuple(i -> op(plan, complex.(partials.(real.(arr), i), partials.(imag.(arr), i))), Val(N))
    return (value_arr, partials_arrs)
end

function arr_of_duals(::Type{T}, value_arr::AbstractArray{<:Real}, partials_arrs) where {T}
    return broadcast(value_arr, partials_arrs...) do value, partials...
        Dual{T}(real(value), Partials(map(real, partials)))
    end
end

function arr_of_duals(::Type{T}, value_arr::AbstractArray{<:Complex}, partials_arrs) where {T}
    return broadcast(value_arr, partials_arrs...) do value, partials...
        complex(
            Dual{T}(real(value), Partials(map(real, partials))),
            Dual{T}(imag(value), Partials(map(imag, partials)))
        )
    end
end

for P in [AbstractFFTs.Plan, AbstractFFTs.ScaledPlan]
    for op in [:(Base.:*), :(Base.:\)]
        @eval function ($op)(plan::$P, arr::AbstractArray{<:Union{Dual{T},Complex{<:Dual{T}}}}) where {T}
            arr_of_duals(T, apply_plan($op, plan, arr)...)
        end
    end
end

LinearAlgebra.mul!(dst::AbstractArray{<:Complex{<:Dual}}, plan::AbstractFFTs.Plan, src::AbstractArray{<:Dual}) = (dst .= plan * src)
LinearAlgebra.mul!(dst::AbstractArray{<:Dual}, plan::AbstractFFTs.ScaledPlan, src::AbstractArray{<:Complex{<:Dual}}) = (dst .= plan * src)


# to allow creating a plan within ForwardDiff'ed code. even for
# arrays of Duals, the plan is still a plan for Float32/64 since we
# apply it to values/partials separtarely above
AbstractFFTs.complexfloat(arr::AbstractArray{<:Dual}) = complex.(arr)
AbstractFFTs.realfloat(arr::AbstractArray{<:Dual}) = arr
AbstractFFTs.plan_fft(arr::AbstractArray{<:Complex{<:Dual}}, region) = plan_fft(complex.(value.(real.(arr)), value.(imag.(arr))), region)
AbstractFFTs.plan_rfft(arr::AbstractArray{<:Dual}, region; kws...) = plan_rfft(value.(arr), region; kws...)

# super edge-case ambiguity in Zygote when first arg is a LazyBinaryOp...
Zygote.z2d(dx::AbstractArray{Union{}}, ::AbstractArray) = dx

# to allow stuff like Float32(::Dual) to work
# might be coming upstream https://github.com/JuliaDiff/ForwardDiff.jl/pull/538
(::Type{S})(x::Dual{T,V,N}) where {T,V,N,S<:Union{Float32,Float64}} = Dual{T,S,N}(x)
