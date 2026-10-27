module CMBLensingCUDAEquiRectExt

using ..CMBLensing, ..CUDA, ..KernelAbstractions
using CMBLensing: AzBasis, Basis, BlockDiagEquiRect, EquiRectField, ProjEquiRect, promote_metadata_strict
using ChainRulesCore: @thunk, NoTangent
import ChainRulesCore: rrule
using LinearAlgebra: Adjoint
import LinearAlgebra: dot
using Tullio

# Older Tullio versions generate CUDAKernels actors, and older CUDA versions
# lack CUDABackend. Keep their existing methods untouched.
@static if CMBLensing.versionof(Tullio) >= v"0.3.7" && isdefined(CUDA, :CUDABackend) && isdefined(CUDA, :AnyCuArray)

# These more-specific methods mirror the array contractions in
# proj_equirect.jl. Expanding @tullio in this extension supplies the CUDA and
# KernelAbstractions bindings needed to generate its CUDA actor without
# replacing the CPU methods during package precompilation.
function Base.:*(M::BlockDiagEquiRect{B,T,P,A}, f::F) where {B<:AzBasis,T<:Real,P<:ProjEquiRect{T},A<:CUDA.AnyCuArray,F<:EquiRectField{B}}
    promote_metadata_strict(M.proj, f.proj)
    CMBLensing.BaseField{B}(@tullio(Bf[p,iₘ] := M.blocks[p,q,iₘ] * f.arr[q,iₘ]), f.proj)
end

function Base.:*(M::Adjoint{S,<:BlockDiagEquiRect{B,T,P,A}}, f::F) where {S,B<:AzBasis,T<:Real,P<:ProjEquiRect{T},A<:CUDA.AnyCuArray,F<:EquiRectField{B}}
    promote_metadata_strict(M.parent.proj, f.proj)
    CMBLensing.BaseField{B}(@tullio(Bf[p,iₘ] := conj(M.parent.blocks[q,p,iₘ]) * f.arr[q,iₘ]), f.proj)
end

function rrule(::typeof(*), M::BlockDiagEquiRect{B,T,P,A}, f::EquiRectField{B′}) where {B<:Basis,T<:Real,P<:ProjEquiRect{T},A<:CUDA.AnyCuArray,B′<:Basis}
    function times_pullback(Δ)
        BΔ, Bf = B(Δ), B(f)
        NoTangent(), @thunk(BlockDiagEquiRect{B}(@tullio(M̄[p,q,iₘ] := Bf.arr[p,iₘ] * conj(BΔ.arr[q,iₘ])), M.proj)'), B′(M' * BΔ)
    end
    M * f, times_pullback
end

function Base.:*(M₁::BlockDiagEquiRect{B,T,P,A}, M₂::BlockDiagEquiRect{B}) where {B<:AzBasis,T<:Real,P<:ProjEquiRect{T},A<:CUDA.AnyCuArray}
    promote_metadata_strict(M₁.proj, M₂.proj)
    BlockDiagEquiRect{B}(@tullio(M₃[p,q,iₘ] := M₁.blocks[p,j,iₘ] * M₂.blocks[j,q,iₘ]), M₁.proj)
end

function Base.:*(M₁::Adjoint{S,<:BlockDiagEquiRect{B,T,P,A}}, M₂::BlockDiagEquiRect{B}) where {S,B<:AzBasis,T<:Real,P<:ProjEquiRect{T},A<:CUDA.AnyCuArray}
    promote_metadata_strict(M₁.parent.proj, M₂.proj)
    BlockDiagEquiRect{B}(@tullio(M₃[p,q,iₘ] := conj(M₁.parent.blocks[j,p,iₘ]) * M₂.blocks[j,q,iₘ]), M₁.parent.proj)
end

function Base.:*(M₁::BlockDiagEquiRect{B,T,P,A}, M₂::Adjoint{S,<:BlockDiagEquiRect{B}}) where {S,B<:AzBasis,T<:Real,P<:ProjEquiRect{T},A<:CUDA.AnyCuArray}
    promote_metadata_strict(M₁.proj, M₂.parent.proj)
    BlockDiagEquiRect{B}(@tullio(M₃[p,q,iₘ] := M₁.blocks[p,j,iₘ] * conj(M₂.parent.blocks[q,j,iₘ])), M₁.proj)
end

function dot(M₁::Adjoint{S,<:BlockDiagEquiRect{B,T,P,A}}, M₂::BlockDiagEquiRect{B}) where {S,B<:AzBasis,T<:Real,P<:ProjEquiRect{T},A<:CUDA.AnyCuArray}
    # Reduce the blocks in parallel, then use the GPU array's scalar reduction
    # rather than scalar-indexing a zero-dimensional CuArray.
    sum(@tullio(a[iₘ] := M₁.parent.blocks[q,p,iₘ] * M₂.blocks[p,q,iₘ]))
end

end

end # module
