
export BilinearLens

@doc doc"""

    BilinearLens(ϕ)

`BilinearLens` is a lensing operator that computes lensing with
bilinear interpolation. The action of the operator, as well as its
adjoint, inverse, inverse-adjoint, and gradient w.r.t. `ϕ` can all be
computed. The log-determinant of the operation is non-zero and can't
be computed. 

Internally, `BilinearLens` forms a sparse matrix with the
interpolation weights, which can be applied and adjoint-ed extremely
fast (e.g. at least an order of magnitude faster than
[`LenseFlow`](@ref)). Inverse and inverse-adjoint lensing is somewhat
slower since it requires an iterative solve, here performed with the
[preconditioned generalized minimal
residual](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method)
algorithm. 

The potential gradient differentiates the interpolation weights. Bilinear
interpolation is not generally differentiable at pixel boundaries, including
zero deflection. At a boundary the pullback uses the derivative of the cell
on the positive-index side in each coordinate.
When an unbatched potential or input field is shared across a batch, its
pullback sums the contributions from the batch. Evaluate batches separately
when individual sensitivities are needed.

"""
mutable struct BilinearLens{T,Φ<:Field{<:Any,T},S} <: ImplicitOp{T}
    ϕ :: Φ
    sparse_repr :: S
    anti_lensing_sparse_repr :: Union{S, Nothing}
end


function BilinearLens(ϕ::FlatField{B1,M1,CT,AA}) where {B1,M1,CT,AA<:AbstractArray}
    
    # If the potential's primal value is zero, use the identity operator.
    if _primal_iszero(norm(ϕ))
        return BilinearLens(ϕ,I,I)
    end
    
    @unpack Nbatch,Nx,Ny,Δx = ϕ
    T = real(ϕ.T)
    Nbatch > 1 && error("BilinearLens with batched ϕ not implemented yet.")
    
    # the (i,j)-th pixel is deflected to (ĩs[i],j̃s[j])
    j̃s,ĩs = getindex.((∇*ϕ)./Δx, :Ix)
    ĩs .=  ĩs  .+ (1:Ny)
    j̃s .= (j̃s' .+ (1:Nx))'
    
    # sub2ind converts a 2D index to 1D index, including wrapping at edges
    indexwrap(i,N) = mod(i - 1, N) + 1
    sub2ind(i,j) = Base._sub2ind((Ny,Nx),indexwrap(i,Ny),indexwrap(j,Nx))

    # compute the 4 non-zero entries in L[I,:] (ie the Ith row of the sparse
    # lensing representation, L) and add these to the sparse constructor
    # matrices, M, and V, accordingly. this function is split off so it can be
    # called directly or used as a CUDA kernel
    function compute_row!(I, ĩ, j̃, M, V)

        # (i,j) indices of the 4 nearest neighbors
        left,right = floor(Int,ĩ) .+ (0, 1)
        top,bottom = floor(Int,j̃) .+ (0, 1)
        
        # 1-D indices of the 4 nearest neighbors
        M[4I-3:4I] .= @SVector[sub2ind(left,top), sub2ind(right,top), sub2ind(left,bottom), sub2ind(right,bottom)]
        
        # weights of these neighbors in the bilinear interpolation
        Δx⁻, Δx⁺ = ((left,right) .- ĩ)
        Δy⁻, Δy⁺ = ((top,bottom) .- j̃)
        A = @SMatrix[
            1 Δx⁻ Δy⁻ Δx⁻*Δy⁻;
            1 Δx⁺ Δy⁻ Δx⁺*Δy⁻;
            1 Δx⁻ Δy⁺ Δx⁻*Δy⁺;
            1 Δx⁺ Δy⁺ Δx⁺*Δy⁺
        ]
        V[4I-3:4I] .= inv(A)[1,:]
    end
    
    # a surprisingly large fraction of the computation for large Nside, so memoize it:
    @memoize getK(Nx,Ny) = Int32.((4:4*Nx*Ny+3) .÷ 4)

    K = Vector{Int32}(getK(Nx,Ny))
    M = similar(K)
    V = similar(K,T)
    for I in 1:length(ĩs)
        compute_row!(I, ĩs[I], j̃s[I], M, V)
    end
    spr = sparse(K,M,V,Nx*Ny,Nx*Ny)
    return BilinearLens(ϕ, spr, nothing)
end


# lazily computing the sparse representation for anti-lensing

function get_anti_lensing_sparse_repr!(Lϕ::BilinearLens)
    if Lϕ.anti_lensing_sparse_repr == nothing
        Lϕ.anti_lensing_sparse_repr = BilinearLens(-Lϕ.ϕ).sparse_repr
    end
    Lϕ.anti_lensing_sparse_repr
end


getϕ(Lϕ::BilinearLens) = Lϕ.ϕ
(Lϕ::BilinearLens)(ϕ::FlatField) = BilinearLens(ϕ)
hash(L::BilinearLens, h::UInt64) = foldr(hash, (typeof(L), getϕ(L)), init=h)


# applying various forms of the operator

function *(Lϕ::BilinearLens, f::FlatField)
    Lϕ.sparse_repr===I && return f
    Łf = Ł(f)
    f̃ = similar(Łf)
    for batch in 1:size(f.arr,4), pol in 1:size(f.arr,3)
        mul!(@views(f̃.arr[:,:,pol,batch][:]), Lϕ.sparse_repr, @views(Łf.arr[:,:,pol,batch][:]))
    end
    f̃
end

function *(Lϕ::Adjoint{<:Any,<:BilinearLens}, f::FlatField)
    parent(Lϕ).sparse_repr===I && return f
    Łf = Ł(f)
    f̃ = similar(Łf)
    for batch in 1:size(f.arr,4), pol in 1:size(f.arr,3)
        mul!(@views(f̃.arr[:,:,pol,batch][:]), parent(Lϕ).sparse_repr', @views(Łf.arr[:,:,pol,batch][:]))
    end
    f̃
end

function \(Lϕ::BilinearLens, f̃::FlatField)
    Lϕ.sparse_repr===I && return f̃
    Łf̃ = Ł(f̃)
    f = similar(Łf̃)
    for batch in 1:size(f.arr,4), pol in 1:size(f.arr,3)
        @views(f.arr[:,:,pol,batch][:]) .= gmres(
            Lϕ.sparse_repr, @views(Łf̃.arr[:,:,pol,batch][:]),
            Pl = get_anti_lensing_sparse_repr!(Lϕ), maxiter = 5
        )
    end
    f
end

function \(Lϕ::Adjoint{<:Any,<:BilinearLens}, f̃::FlatField)
    parent(Lϕ).sparse_repr===I && return f̃
    Łf̃ = Ł(f̃)
    f = similar(Łf̃)
    for batch in 1:size(f.arr,4), pol in 1:size(f.arr,3)
        @views(f.arr[:,:,pol,batch][:]) .= gmres(
            parent(Lϕ).sparse_repr', @views(Łf̃.arr[:,:,pol,batch][:]),
            Pl = get_anti_lensing_sparse_repr!(parent(Lϕ))', maxiter = 5
        )
    end
    f
end


for op in (:*, :\)
    @eval function ($op)(Lϕ::Union{BilinearLens, Adjoint{<:Any,<:BilinearLens}}, f::FieldTuple)
        FieldTuple(map(f->($op)(Lϕ,f), f.fs))
    end
end


# gradients

@adjoint BilinearLens(ϕ) = BilinearLens(ϕ), Δ -> (Δ,)

# Derivative of the same periodic, bilinear map used to construct sparse_repr.
# A linear index includes spin and batch axes; the displacement field itself
# is unbatched.
@inline function _bilinear_lens_component(img, dx, dy, k, Δx, ::Val{D}) where D
    Ny, Nx, Npol = size(img,1), size(img,2), size(img,3)
    i = mod(k-1, Ny) + 1
    j = mod(div(k-1, Ny), Nx) + 1
    pol = mod(div(k-1, Ny*Nx), Npol) + 1
    batch = div(k-1, Ny*Nx*Npol) + 1

    y = i + dy[i,j]
    x = j + dx[i,j]
    iy, ix = floor(Int,y), floor(Int,x)
    u, v = y-iy, x-ix
    i0, i1 = mod(iy-1,Ny)+1, mod(iy,Ny)+1
    j0, j1 = mod(ix-1,Nx)+1, mod(ix,Nx)+1
    f00, f10 = img[i0,j0,pol,batch], img[i1,j0,pol,batch]
    f01, f11 = img[i0,j1,pol,batch], img[i1,j1,pol,batch]
    D == 1 ? ((1-u)*(f01-f00) + u*(f11-f10))/Δx :
             ((1-v)*(f10-f00) + v*(f11-f01))/Δx
end

function _bilinear_lens_gradients(Lϕ::BilinearLens, f::Field)
    Łf = Ł(f)
    @unpack Δx = Lϕ.ϕ
    dx, dy = getindex.((∇*Lϕ.ϕ)./Δx, :Ix)
    gx, gy = similar(Łf), similar(Łf)
    indices = reshape(Base.OneTo(length(Łf.arr)), size(Łf.arr))
    broadcast!(_bilinear_lens_component, gx.arr, Ref(Łf.arr), Ref(dx), Ref(dy), indices, Ref(Δx), Ref(Val(1)))
    broadcast!(_bilinear_lens_component, gy.arr, Ref(Łf.arr), Ref(dx), Ref(dy), indices, Ref(Δx), Ref(Val(2)))
    @SVector[gx, gy]
end

# Reduce each component to the potential's shape before adding it. In
# particular, the same potential is shared across all input field batches.
function _bilinear_spin_dot(a::FlatField, b::FlatField)
    # Broadcasting permits a shared input to receive a batched cotangent,
    # or inputs and cotangents to have different storage precision.
    product = a .* b
    r = sum(product.arr; dims=3)
    LambertMap(dropdims(r; dims=(ndims(r)==3 ? 3 : ())), product.metadata)
end

# The potential is unbatched. Sum the map-space contributions before taking
# their divergence: differentiation is linear, and this avoids requiring a
# batched FFT for a quantity that will be reduced immediately afterwards.
function _bilinear_sum_batch(f::LambertMap)
    ndims(f.arr) == 4 || return f
    LambertMap(dropdims(sum(f.arr; dims=4); dims=(3,4)), f.metadata)
end

# MKL's inverse real FFT reads only the positive-x half of its self-conjugate
# y-frequency rows. Its adjoint under the field's weighted Fourier inner
# product must therefore double those entries and zero their negative-x
# partners. This also handles arbitrary complex Fourier input directions.
function _bilinear_mkl_irfft_adjoint(f::FlatField)
    f̃ = Fourier(f)
    # Treat spin and any other trailing axes independently.
    a = reshape(f̃.arr, size(f̃.arr,1), size(f̃.arr,2), :)
    Nx = f.Nx
    npairs = (Nx-1) ÷ 2
    rows = iseven(f.Ny) ? (1, size(a,1)) : (1,)
    for i in rows
        @views a[i,2:(npairs+1),:] .*= 2
        @views a[i,(Nx-npairs+1):Nx,:] .= 0
        @views a[i,1,:] .= real.(a[i,1,:])
        iseven(Nx) && (@views a[i,Nx÷2+1,:] .= real.(a[i,Nx÷2+1,:]))
    end
    f̃
end

function _bilinear_lens_pullback(Lϕ::BilinearLens, f::FlatField, Δ::Field)
    gradients = _bilinear_lens_gradients(Lϕ, f)
    ŁΔ = Ł(Δ)
    contributions = map(g -> _bilinear_sum_batch(_bilinear_spin_dot(ŁΔ, g)), gradients)
    ϕ̄ = if Lϕ.ϕ isa FlatFourier && _fft_arr_type(ŁΔ.arr) <: Array && fftw_provider() == "mkl"
        ∇' * map(_bilinear_mkl_irfft_adjoint, contributions)
    else
        ∇' * contributions
    end
    f̄ = Lϕ' * Δ
    if Lϕ.sparse_repr !== I && basis(f) <: SpatialBasis{Fourier} && _fft_arr_type(f̄.arr) <: Array && fftw_provider() == "mkl"
        # An unbatched input may receive a batched cotangent. Reduce it while
        # still in map space so that MKL need not plan a batched real FFT.
        if batch_length(f) == 1 && batch_length(f̄) > 1
            fmap = Ł(f)
            f̄ = BaseField{basis(f̄)}(_svector_unbroadcast(fmap.arr, f̄.arr), f̄.metadata)
        end
        f̄ = _bilinear_mkl_irfft_adjoint(f̄)
    end
    _svector_reduce_field(Lϕ.ϕ, ϕ̄), _svector_project_field(f, f̄)
end
function _bilinear_lens_pullback(Lϕ::BilinearLens, f::FieldTuple, Δ::FieldTuple)
    parts = map((x, y) -> _bilinear_lens_pullback(Lϕ, x, y), f.fs, Δ.fs)
    reduce(+, map(first, values(parts))), FieldTuple(map(last, parts))
end

@adjoint function *(Lϕ::BilinearLens, f::Field)
    f̃ = Lϕ * f
    function back(Δ)
        ϕ̄, f̄ = _bilinear_lens_pullback(Lϕ, f, Δ)
        _svector_project_field(Lϕ.ϕ, ϕ̄), f̄
    end
    f̃, back
end


# gpu

adapt_structure(storage, Lϕ::BilinearLens) = BilinearLens(adapt(storage, fieldvalues(Lϕ))...)
