
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
@nograd ProjLambert
@nograd fieldinfo
@nograd hasfield
@nograd basetype
@nograd get_storage


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
    F(arr, metadata), Δ -> (Δ.arr .* Δ.λ_rfft, nothing)
end
@adjoint function (::Type{F})(arr::A, metadata::M) where {B<:SpatialBasis{AzFourier},M<:Proj,T,A<:AbstractArray{T},F<:BaseField{B}}
    F(arr, metadata), Δ -> (Δ.arr .* Δ.λ_rfft, nothing)
end
# the factors here need to cancel the ones in the corresponding constructors above
@adjoint function Zygote.literal_getproperty(f::BaseField{B}, ::Val{:arr}) where {B<:SpatialBasis{Map}}
    getfield(f,:arr), Δ -> (BaseField{B}(Δ, f.metadata),)
end
@adjoint function Zygote.literal_getproperty(f::BaseField{B,M,T}, ::Val{:arr}) where {B<:SpatialBasis{Fourier},M,T}
    getfield(f,:arr), Δ -> (BaseField{B}(Δ ./ f.λ_rfft, f.metadata),)
end
@adjoint function Zygote.literal_getproperty(f::BaseField{B,M,T}, ::Val{:arr}) where {B<:SpatialBasis{AzFourier},M,T}
    getfield(f,:arr), Δ -> (BaseField{B}(Δ ./ f.λ_rfft, f.metadata),)
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



## ForwardDiff rules 
# mainly FFTs which don't work, some work upstream here
# https://github.com/JuliaDiff/ForwardDiff.jl/pull/541 but its not
# incorporated merged

using ForwardDiff: Dual, Partials, value, partials


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
