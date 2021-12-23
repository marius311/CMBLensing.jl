

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

# constructing from other fields
function (::Type{F})(X::BaseField{B₀,P}, Y::BaseField{B₀,P}) where {B₀<:Union{Map,Fourier}, B<:Basis2Prod{<:Union{𝐐𝐔,𝐄𝐁},B₀}, P<:CartesianProj, F<:BaseField{B,<:CartesianProj}}
    BaseField{B,P}(cat(X.arr, Y.arr, dims=Val(3)), get_metadata_strict(X, Y)) :: F
end
function (::Type{F})(X::BaseField{B₀,P}, Y::BaseField{Basis2Prod{Pol,B₀},P}) where {B₀<:Union{Map,Fourier}, Pol<:Union{𝐐𝐔,𝐄𝐁}, B<:Basis3Prod{𝐈,Pol,B₀}, P<:CartesianProj, F<:BaseField{B,<:CartesianProj}}
    BaseField{B,P}(cat(X.arr, Y.arr, dims=Val(3)), get_metadata_strict(X, Y)) :: F
end
function (::Type{F})(X::BaseField{B₀,P}, Y::BaseField{B₀,P}, Z::BaseField{B₀,P}) where {B₀<:Union{Map,Fourier}, B<:Basis3Prod{𝐈,<:Union{𝐐𝐔,𝐄𝐁},B₀}, P<:CartesianProj, F<:BaseField{B,<:CartesianProj}}
    BaseField{B,P}(cat(X.arr, Y.arr, Z.arr, dims=Val(3)), get_metadata_strict(X, Y, Z)) :: F
end

# consistency checks called from BaseField constructor
function check_field_consistency(::SpatialBasis{Map}, arr, proj::P) where {P<:CartesianProj}
    size(arr)[1:2] == (proj.Ny, proj.Nx) || error("$(basetype(P)) metadata (Ny=$(proj.Ny), Nx=$(proj.Nx)) is inconsistent with pixel array size $(size(arr)).")
end
function check_field_consistency(::SpatialBasis{Fourier}, arr, proj::P) where {P<:CartesianProj}
    size(arr)[1:2] == (proj.Ny÷2+1, proj.Nx) || error("$(basetype(P)) metadata (Ny=$(proj.Ny), Nx=$(proj.Nx)) is inconsistent with half-plane Fourier array size $(size(arr)).")
end



### array interface
# most is inherited from BaseField. the main thing we have specify
# here has to do with which dimension is the "batch" dimension
# (dimension 4), since that is not assumed in BaseField
similar(f::CartesianField{B}, Nbatch::Int) where {B} = 
    CartesianField{B}(similar(f.arr, size(f.arr,1), size(f.arr,2), size(f.arr,3), Nbatch), f.metadata)
nonbatch_dims(f::CartesianField) = ntuple(identity, min(3, ndims(f.arr)))
require_unbatched(f::CartesianField) = (f.Nbatch==1) || error("This function not implemented for batched fields.")
pol_slice(f::CartesianField, i) = (:, :, i, ..)


### properties
# metadata
getproperty(f::CartesianField, ::Val{:Nbatch}) = size(getfield(f,:arr), 4)
getproperty(f::CartesianField, ::Val{:Npol})   = size(getfield(f,:arr), 3)
getproperty(f::CartesianField, ::Val{:T})      = eltype(f)
getproperty(f::CartesianField, ::Val{:proj})   = getfield(f, :metadata)
# sub-components
getproperty(f::CartesianField{B}, k::Union{typeof.(Val.((:Ix,:Qx,:Ux,:Ex,:Bx,:Il,:Ql,:Ul,:El,:Bl)))...}) where {B} = 
    view(getfield(f,:arr), pol_slice(f, pol_index(B(), k))...)
getproperty(f::CartesianField{B}, k::Union{typeof.(Val.((:I,:Q,:U,:E,:B)))...}) where {B₀, B<:SpatialBasis{B₀}} =
    BaseField{B₀}(_reshape_batch(view(getfield(f,:arr), pol_slice(f, pol_index(B(), k))...)), getfield(f,:metadata))
getproperty(f::CartesianS02{Basis3Prod{𝐈,B₂,B₀}}, ::Val{:P}) where {B₂,B₀} = 
    BaseField{Basis2Prod{B₂,B₀}}(view(getfield(f,:arr), pol_slice(f, 2:3)...), getfield(f,:metadata))
getproperty(f::CartesianS2, ::Val{:P}) = f


### indices
function getindex(f::CartesianS0, k::Symbol; full_plane=false)
    maybe_unfold = full_plane ? x->unfold(x,fieldinfo(f).Ny) : identity
    @match k begin
        :I  => f
        :Ix => Map(f).Ix
        :Il => maybe_unfold(Fourier(f).Il)
        _   => throw(ArgumentError("Invalid CartesianS0 index: $k"))
    end
end
function getindex(f::CartesianS2{Basis2Prod{B₁,B₂}}, k::Symbol; full_plane=false) where {B₁,B₂}
    maybe_unfold = (full_plane && k in [:El,:Bl,:Ql,:Ul]) ? x->unfold(x,fieldinfo(f).Ny) : identity
    B = @match k begin
        (:P)         => identity
        (:E  || :B)  => Basis2Prod{𝐄𝐁,B₂}
        (:Q  || :U)  => Basis2Prod{𝐐𝐔,B₂}
        (:Ex || :Bx) => Basis2Prod{𝐄𝐁,Map}
        (:El || :Bl) => Basis2Prod{𝐄𝐁,Fourier}
        (:Qx || :Ux) => Basis2Prod{𝐐𝐔,Map}
        (:Ql || :Ul) => Basis2Prod{𝐐𝐔,Fourier}
        _ => throw(ArgumentError("Invalid CartesianS2 index: $k"))
    end
    maybe_unfold(getproperty(B(f),k))
end
function getindex(f::CartesianS02{Basis3Prod{B₁,B₂,B₃}}, k::Symbol; full_plane=false) where {B₁,B₂,B₃}
    maybe_unfold = (full_plane && k in [:Il,:El,:Bl,:Ql,:Ul]) ? x->unfold(x,fieldinfo(f).Ny) : identity
    B = @match k begin
        (:I  || :P)  => identity
        (:E  || :B)  => Basis3Prod{𝐈,𝐄𝐁,B₃}
        (:Q  || :U)  => Basis3Prod{𝐈,𝐐𝐔,B₃}
        (:Ix)        => Basis3Prod{𝐈,B₂,Map}
        (:Il)        => Basis3Prod{𝐈,B₂,Fourier}
        (:Ex || :Bx) => Basis3Prod{𝐈,𝐄𝐁,Map}
        (:El || :Bl) => Basis3Prod{𝐈,𝐄𝐁,Fourier}
        (:Qx || :Ux) => Basis3Prod{𝐈,𝐐𝐔,Map}
        (:Ql || :Ul) => Basis3Prod{𝐈,𝐐𝐔,Fourier}
        _ => throw(ArgumentError("Invalid CartesianS02 index: $k"))
    end
    maybe_unfold(getproperty(B(f),k))
end
