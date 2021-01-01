
## FlatS2 types
const FlatQUMap{M<:FlatProj, T, A<:AbstractArray{T}} = BaseField{QUMap, M, T, A}
const FlatQUFourier{M<:FlatProj, T, A<:AbstractArray{T}} = BaseField{QUFourier, M, T, A}
const FlatEBMap{M<:FlatProj, T, A<:AbstractArray{T}} = BaseField{EBMap, M, T, A}
const FlatEBFourier{M<:FlatProj, T, A<:AbstractArray{T}} = BaseField{EBFourier, M, T, A}
## FlatS2 unions
const FlatS2{M,T,A}=Union{FlatEBMap{M,T,A},FlatEBFourier{M,T,A},FlatQUMap{M,T,A},FlatQUFourier{M,T,A}}
const FlatQU{M,T,A}=Union{FlatQUMap{M,T,A},FlatQUFourier{M,T,A}}
const FlatEB{M,T,A}=Union{FlatEBMap{M,T,A},FlatEBFourier{M,T,A}}
const FlatS2Map{M,T,A}=Union{FlatQUMap{M,T,A},FlatEBMap{M,T,A}}
const FlatS2Fourier{M,T,A}=Union{FlatQUFourier{M,T,A},FlatEBFourier{M,T,A}}

## convenience constructors
function FlatQUMap(Qx::A, Ux::A; θpix=θpix₀) where {T, A<:AbstractArray{T}}
    FlatQUMap(
        drop_tail_singleton_dims(cat(
            reshape(Qx, size(Qx,1), size(Qx,2), 1, size(Qx,3)),
            reshape(Ux, size(Ux,1), size(Ux,2), 1, size(Ux,3)),
            dims = 3
        )),
        ProjLambert(T, basetype(A), θpix, size(Qx,1), size(Qx,2))
    )
end

# FlatQUFourier(Ql::A, Ul::A; θpix=1) where {T, A<:AbstractArray{T}} = 
#     FlatQUFourier(Ql[:,:,:,:], ProjLambert(real(T), basetype(A), θpix, size(Ix)[1:2]...))

## properties
getproperty(f::FlatQUMap, ::Val{:Qx}) = @view(getfield(f,:arr)[:,:,1,..])
getproperty(f::FlatQUMap, ::Val{:Ux}) = @view(getfield(f,:arr)[:,:,2,..])

## basis conversion
# out-of-place
QUFourier(f::FlatQUMap) = FlatQUFourier(m_rfft(f.arr, (1,2)), f.metadata)
EBFourier(f::FlatQUMap) = FlatEBFourier(m_rfft(f.arr, (1,2)), f.metadata)
QUMap(f::FlatQUFourier) = FlatQUMap(m_irfft(f.arr, f.Ny, (1,2)), f.metadata)
EBMap(f::FlatQUFourier) = FlatEBMap(m_irfft(f.arr, f.Ny, (1,2)), f.metadata)
# in-plane
QUMap(f′::FlatQUMap, f::FlatQUFourier) = (m_irfft!(f′.arr, f.arr, (1,2)); f′)
QUFourier(f′::FlatQUFourier, f::FlatQUMap) = (m_rfft!(f′.arr, f.arr, (1,2)); f′)


# for (F,F0,(X,Y),T) in [
#         (:FlatQUMap,     :FlatMap,     (:Qx,:Ux), :T),
#         (:FlatQUFourier, :FlatFourier, (:Ql,:Ul), :(Complex{T})),
#         (:FlatEBMap,     :FlatMap,     (:Ex,:Bx), :T),
#         (:FlatEBFourier, :FlatFourier, (:El,:Bl), :(Complex{T}))
#     ]
#     doc = """
#         # main constructor:
#         $F(
#             $X::AbstractArray, $Y::AbstractArray; $(F0==:FlatFourier ? "\n        Nside, # required, size of the map in pixels" : "")
#             θpix,  # optional, resolution in arcmin (default: 1)
#             ∂mode, # optional, fourier∂ or map∂ (default: fourier∂)
#         )

#         # more low-level:
#         $F{P}($X::AbstractArray, $Y::AbstractArray) # specify pixelization P explicilty
#         $F{P,T}($X::AbstractArray, $Y::AbstractArray) # additionally, convert elements to type $T
#         $F{P,T,M<:AbstractArray{$T}}($X::M, $Y::M) # specify everything explicilty
        
#     Construct a `$F` object. The top form of the constructor is most convenient
#     for interactive work, while the others may be more useful for low-level code.
#     """
#     @eval begin
#         @doc $doc $F
#         $F($X::AbstractRank2or3Array, $Y::AbstractRank2or3Array; kwargs...) =
#             $F(($F0($X; kwargs...), $F0($Y; kwargs...)))
#     end
# end





# ### properties
# function propertynames(f::FlatS2)
#     (:fs, propertynames(f.fs)..., 
#      (Symbol(string(k,(f isa FlatMap ? "x" : "l"))) for (k,f) in pairs(f.fs))...)
# end
# getproperty(f::FlatQUMap,     ::Val{:Qx}) = getfield(f,:fs).Q.Ix
# getproperty(f::FlatQUMap,     ::Val{:Ux}) = getfield(f,:fs).U.Ix
# getproperty(f::FlatQUFourier, ::Val{:Ql}) = getfield(f,:fs).Q.Il
# getproperty(f::FlatQUFourier, ::Val{:Ul}) = getfield(f,:fs).U.Il
# getproperty(f::FlatEBMap,     ::Val{:Ex}) = getfield(f,:fs).E.Ix
# getproperty(f::FlatEBMap,     ::Val{:Bx}) = getfield(f,:fs).B.Ix
# getproperty(f::FlatEBFourier, ::Val{:El}) = getfield(f,:fs).E.Il
# getproperty(f::FlatEBFourier, ::Val{:Bl}) = getfield(f,:fs).B.Il
# getproperty(f::FlatS2,        ::Val{:P})  = f
# function getindex(f::FlatS2, k::Symbol; full_plane=false)
#     maybe_unfold = (full_plane && k in [:El,:Bl,:Ql,:Ul]) ? x->unfold(x,fieldinfo(f).Ny) : identity
#     B = @match k begin
#         (:P)         => identity
#         (:E  || :B)  => @match f begin
#             _ :: FlatQUMap     => EBMap
#             _ :: FlatQUFourier => EBFourier
#             _                  => identity
#         end
#         (:Q  || :U)  => @match f begin
#             _ :: FlatEBMap     => QUMap
#             _ :: FlatEBFourier => QUFourier
#             _                  => identity
#         end
#         (:Ex || :Bx) => EBMap
#         (:El || :Bl) => EBFourier
#         (:Qx || :Ux) => QUMap
#         (:Ql || :Ul) => QUFourier
#         _ => throw(ArgumentError("Invalid FlatS2 index: $k"))
#     end
#     maybe_unfold(getproperty(B(f),k))
# end

# function getindex(D::DiagOp{<:FlatEBFourier}, k::Symbol)
#     @unpack El, Bl = diag(D)
#     @unpack sin2ϕ, cos2ϕ, P = fieldinfo(diag(D))
#     f = @match k begin
#         (:QQ)        => FlatFourier{P}(@. Bl*sin2ϕ^2 + El*cos2ϕ^2)
#         (:QU || :UQ) => FlatFourier{P}(@. (El - Bl) * sin2ϕ * cos2ϕ)
#         (:UU)        => FlatFourier{P}(@. Bl*cos2ϕ^2 + El*sin2ϕ^2)
#         _            => getproperty(D.diag, k)
#     end
#     Diagonal(f)
# end

# ### basis conversion

# QUFourier(f::FlatQUMap) = FlatQUFourier(map(Fourier,f.fs))
# QUFourier(f::FlatEBMap) = f |> EBFourier |> QUFourier
# QUFourier(f::FlatEBFourier{P,T}) where {P,T} = begin
#     @unpack sin2ϕ, cos2ϕ = fieldinfo(f)
#     Ql = @. - f.El * cos2ϕ + f.Bl * sin2ϕ
#     Ul = @. - f.El * sin2ϕ - f.Bl * cos2ϕ
#     FlatQUFourier(Q=FlatFourier{P}(Ql), U=FlatFourier{P}(Ul))
# end

# QUMap(f::FlatQUFourier)  = FlatQUMap(map(Map,f.fs))
# QUMap(f::FlatEBMap)      = f |> EBFourier |> QUFourier |> QUMap
# QUMap(f::FlatEBFourier)  = f |> QUFourier |> QUMap

# EBFourier(f::FlatEBMap) = FlatEBFourier(map(Fourier,f.fs))
# EBFourier(f::FlatQUMap) = f |> QUFourier |> EBFourier
# EBFourier(f::FlatQUFourier{P,T}) where {P,T} = begin
#     @unpack sin2ϕ, cos2ϕ = fieldinfo(f)
#     El = @. - f.Ql * cos2ϕ - f.Ul * sin2ϕ
#     Bl = @.   f.Ql * sin2ϕ - f.Ul * cos2ϕ
#     FlatEBFourier(E=FlatFourier{P}(El), B=FlatFourier{P}(Bl))
# end

# EBMap(f::FlatEBFourier) = FlatEBMap(map(Map,f.fs))
# EBMap(f::FlatQUMap)     = f |> QUFourier |> EBFourier |> EBMap
# EBMap(f::FlatQUFourier) = f |> EBFourier |> EBMap

# QUFourier(f′::FlatQUFourier, f::FlatQUMap) = (map(Fourier,f′.fs,f.fs); f′)
# QUMap(f′::FlatQUMap, f::FlatQUFourier) = (map(Map,f′.fs,f.fs); f′)

# Map(f::FlatQUFourier) = QUMap(f)
# Map(f::FlatEBFourier) = EBMap(f)
# Map(f::FlatS2Map) = f
# Fourier(f::FlatQUMap) = QUFourier(f)
# Fourier(f::FlatEBMap) = EBFourier(f)
# Fourier(f::FlatS2Fourier) = f


# ### simulation and power spectra
# function white_noise(rng::AbstractRNG, ::Type{F2}) where {F2<:FlatS2}
#     F = (((::Type{<:FieldTuple{B,NamedTuple{Names,NTuple{2,F}}}}) where {B,Names,F}) -> F)(F2)
#     FlatEBMap(E=white_noise(rng,F), B=white_noise(rng,F))
# end
# function Cℓ_to_Cov(::Type{P}, ::Type{T}, ::Type{S2}, CℓEE::InterpolatedCℓs, CℓBB::InterpolatedCℓs; kwargs...) where {P,T,M}
#     Diagonal(FlatEBFourier(E=diag(Cℓ_to_Cov(P,T,S0,CℓEE;kwargs...)), B=diag(Cℓ_to_Cov(P,T,S0,CℓBB;kwargs...))))
# end
# function Cℓ_to_Cov(::Type{P}, ::Type{T}, ::Type{S2}, (CℓEE, ℓedges, θname)::Tuple, CℓBB::InterpolatedCℓs; units=fieldinfo(P).Ωpix) where {P,T}
#     C₀ = Cℓ_to_Cov(P, T, S2, CℓEE, CℓBB, units=units)
#     Cbins = Diagonal.(FlatEBFourier.(MidPasses(ℓedges) .* [diag(C₀).E], [zero(diag(C₀).B)]))
#     BinRescaledOp(C₀,Cbins,θname)
# end



# function get_Cℓ(f1::FlatS2, f2::FlatS2=f1; which=(:EE,:BB), kwargs...)
#     Cℓ = (;[Symbol(x1*x2) => get_Cℓ(getindex(f1,Symbol(x1)),getindex(f2,Symbol(x2)); kwargs...) for (x1,x2) in split.(string.(ensure1d(which)),"")]...)
#     which isa Symbol ? Cℓ[1] : Cℓ
# end


# function ud_grade(f::FlatS2{P}, args...; kwargs...) where {P} 
#     f′ = FieldTuple(map(f->ud_grade(f, args...; kwargs...), f.fs))
#     B′ = (f′[1] isa FlatMap) ? (f isa FlatQU ? QUMap : EBMap) : (f isa FlatQU ? QUFourier : EBFourier)
#     FieldTuple{B′}(f′)
# end
