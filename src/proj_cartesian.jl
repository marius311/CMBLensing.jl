

# Cartesian projection (concrete examples include Lambert or
# EquiRect), defined by the fact that the pixels are stored in a
# matrix. The projection type must have a constructor accepting at
# least (Ny,Nx,T,storage) keyword arguments. 
abstract type CartesianProj <: Proj end
make_field_aliases("Cartesian", CartesianProj)


### constructors

_reshape_batch(arr::AbstractArray{T,3}) where {T} = reshape(arr, size(arr,1), size(arr,2), 1, size(arr,3))
_reshape_batch(arr) = arr

## constructing from arrays
# spin-0
function (::Type{F})(Ix::A; Proj=default_proj(F), kwargs...) where {F<:BaseField{Map,<:CartesianProj}, T, A<:AbstractArray{T}}
    BaseField{Map}(_reshape_batch(Ix), Proj(;Ny=size(Ix,1), Nx=size(Ix,2), T, storage=basetype(A), kwargs...))
end
function (::Type{F})(Il::A; Ny, Proj=default_proj(F), kwargs...) where {F<:BaseField{Fourier,<:CartesianProj}, T, A<:AbstractArray{T}}
    BaseField{Fourier}(_reshape_batch(Il), Proj(;Ny, Nx=size(Il,2), T, storage=basetype(A), kwargs...))
end
# spin-2
function (::Type{F})(X::A, Y::A, metadata::CartesianProj) where {B<:Basis2Prod{<:Union{𝐐𝐔,𝐄𝐁}}, F<:BaseField{B,<:CartesianProj}, T, A<:AbstractArray{T}}
    BaseField{B}(cat(_reshape_batch(X), _reshape_batch(Y), dims=Val(3)), metadata)
end
function (::Type{F})(X::A, Y::A; Proj=default_proj(F), kwargs...) where {B<:Basis2Prod{<:Union{𝐐𝐔,𝐄𝐁}}, F<:BaseField{B,<:CartesianProj}, T, A<:AbstractArray{T}}
    BaseField{B,<:CartesianProj}(X, Y, Proj(;Ny=size(X,1), Nx=size(X,2), T, storage=basetype(A), kwargs...))
end
# spin-(0,2)
function (::Type{F})(X::A, Y::A, Z::A, metadata::CartesianProj) where {B<:Basis3Prod{𝐈,<:Union{𝐐𝐔,𝐄𝐁}}, F<:BaseField{B,<:CartesianProj}, T, A<:AbstractArray{T}}
    BaseField{B}(cat(_reshape_batch(X), _reshape_batch(Y), _reshape_batch(Z), dims=Val(3)), metadata)
end
function (::Type{F})(X::A, Y::A, Z::A; Proj=default_proj(F), kwargs...) where {B<:Basis3Prod{𝐈,<:Union{𝐐𝐔,𝐄𝐁}}, F<:BaseField{B,<:CartesianProj}, T, A<:AbstractArray{T}}
    BaseField{B,<:CartesianProj}(X, Y, Z, Proj(;Ny=size(X,1), Nx=size(X,2), T, storage=basetype(A), kwargs...))
end

## constructing from other fields
function (::Type{F})(X::BaseField{B₀,P}, Y::BaseField{B₀,P}) where {B₀<:Union{Map,Fourier}, B<:Basis2Prod{<:Union{𝐐𝐔,𝐄𝐁},B₀}, P<:CartesianProj, F<:BaseField{B,<:CartesianProj}}
    BaseField{B,P}(cat(X.arr, Y.arr, dims=Val(3)), get_metadata_strict(X, Y)) :: F
end
function (::Type{F})(X::BaseField{B₀,P}, Y::BaseField{Basis2Prod{Pol,B₀},P}) where {B₀<:Union{Map,Fourier}, Pol<:Union{𝐐𝐔,𝐄𝐁}, B<:Basis3Prod{𝐈,Pol,B₀}, P<:CartesianProj, F<:BaseField{B,<:CartesianProj}}
    BaseField{B,P}(cat(X.arr, Y.arr, dims=Val(3)), get_metadata_strict(X, Y)) :: F
end
function (::Type{F})(X::BaseField{B₀,P}, Y::BaseField{B₀,P}, Z::BaseField{B₀,P}) where {B₀<:Union{Map,Fourier}, B<:Basis3Prod{𝐈,<:Union{𝐐𝐔,𝐄𝐁},B₀}, P<:CartesianProj, F<:BaseField{B,<:CartesianProj}}
    BaseField{B,P}(cat(X.arr, Y.arr, Z.arr, dims=Val(3)), get_metadata_strict(X, Y, Z)) :: F
end


### array interface
# most is inherited from BaseField. the main thing we have specify
# here has to do with which dimension is the "batch" dimension
# (dimension 4), since that is not assumed in BaseField
similar(f::CartesianField{B}, Nbatch::Int) where {B} = CartesianField{B}(similar(f.arr, size(f.arr,1), size(f.arr,2), size(f.arr,3), Nbatch), f.metadata)
nonbatch_dims(f::CartesianField) = ntuple(identity,min(3,ndims(f.arr)))
require_unbatched(f::CartesianField) = (f.Nbatch==1) || error("This function not implemented for batched fields.")
pol_slice(f::CartesianField, i) = (:, :, i, ..)


### properties
# generic
getproperty(f::CartesianField, ::Val{:Nbatch}) = size(getfield(f,:arr), 4)
getproperty(f::CartesianField, ::Val{:Npol})   = size(getfield(f,:arr), 3)
getproperty(f::CartesianField, ::Val{:T})      = eltype(f)
getproperty(f::CartesianField, ::Val{:proj})   = getfield(f, :metadata)
