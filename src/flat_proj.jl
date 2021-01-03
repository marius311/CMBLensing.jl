
abstract type FlatProj end

# default angular resolution used by a number of convenience constructors
θpix₀ = 1


@kwdef struct ProjLambert{T, V<:AbstractVector{T}, M<:AbstractMatrix{T}} <: FlatProj
    θpix
    storage
    Δx       :: T
    Ny       :: Int
    Nx       :: Int
    Ωpix     :: T
    nyq      :: T
    Δℓx      :: T
    Δℓy      :: T
    ℓy       :: V
    ℓx       :: V
    ℓmag     :: M
    sin2ϕ    :: M
    cos2ϕ    :: M
    units     = 1
    θϕ_center = nothing
end

@memoize function ProjLambert(T, storage, θpix, Ny, Nx)

    Δx           = T(deg2rad(θpix/60))
    Δℓx          = T(2π/(Nx*Δx))
    Δℓy          = T(2π/(Ny*Δx))
    nyq          = T(2π/(2Δx))
    Ωpix         = T(Δx^2)
    ℓy           = adapt(storage, (ifftshift(-Ny÷2:(Ny-1)÷2) .* Δℓy)[1:Ny÷2+1])
    ℓx           = adapt(storage, (ifftshift(-Nx÷2:(Nx-1)÷2) .* Δℓx))
    ℓmag         = @. sqrt(ℓx'^2 + ℓy^2)
    ϕ            = @. angle(ℓx' + im*ℓy)
    sin2ϕ, cos2ϕ = @. sin(2ϕ), cos(2ϕ)
    if iseven(Ny)
        sin2ϕ[end, end:-1:(Nx÷2+2)] .= sin2ϕ[end, 2:Nx÷2]
    end
    
    ProjLambert(;θpix, Ny, Nx, storage, Δx, Δℓx, Δℓy, nyq, Ωpix, ℓx, ℓy, ℓmag, sin2ϕ, cos2ϕ)

end

typealias(::Type{<:ProjLambert{T}}) where {T} = "ProjLambert{$T}"



### promotion

# used in broadcasting to decide the result of broadcasting across two
# fields with a given `metadata` and basis, `b` (where b is an
# instance of type-parameter B), and 
function promote_bcast_b_metadata(
    (b,metadata₁) :: Tuple{B,<:ProjLambert{T₁}}, 
    (_,metadata₂) :: Tuple{B,<:ProjLambert{T₂}}
) where {B,T₁,T₂}

    # even though if metadata₁ === metadata₂ we could technically
    # return either, it helps inference if we always return the
    # technically-"wider" one. this line is optimized away at compile
    # time anyway so doesn't slow us down if metadata₁ === metadata₂
    # is indeed true
    wider_metadata = promote_type(T₁,T₂) == T₁ ? metadata₁ : metadata₂

    if (
        metadata₁ === metadata₂ || (
            metadata₁.θpix    == metadata₂.θpix &&
            metadata₁.Ny      == metadata₂.Ny   &&
            metadata₁.Nx      == metadata₂.Nx      
        )
    )
        (b, wider_metadata)
    else
        error("""Can't broadcast two fields with the following differing metadata:
        1: $(select(fields(metadata₁),(:θpix,:Ny,:Nx)))
        2: $(select(fields(metadata₂),(:θpix,:Ny,:Nx)))
        """)
    end

end


# used in non-broadcasted algebra to decide the result of performing
# some operation across two fields with a given `metadata`. this is
# free to do more generic promotion than promote_bcast_b_metadata. the
# result should be a common metadata which we convert both fields to
# then do a succesful broadcast
function promote_metadata(metadata₁::ProjLambert, metadata₂::ProjLambert)
    if (
        metadata₁ === metadata₂ || (
            metadata₁.θpix         == metadata₂.θpix    &&
            metadata₁.Ny           == metadata₂.Ny      &&
            metadata₁.Nx           == metadata₂.Nx      &&
            metadata₁.storage      == metadata₂.storage
        )
    )
        (metadata₁, metadata₂)
    else
        error("""Can't promote two fields with the following differing metadata:
        1: $(select(fields(metadata₁),(:θpix,:Ny,:Nx,:storage)))
        2: $(select(fields(metadata₂),(:θpix,:Ny,:Nx,:storage)))
        """)
    end
    # in the future, could add rules here to allow e.g. automatically
    # upgrading one field if they have different resolutions, etc...
end







# function Cℓ_to_2D(::Type{P}, ::Type{T}, Cℓ) where {T,N,P<:Flat{N}}
#     Complex{T}.(nan2zero.(Cℓ.(fieldinfo(P,T).kmag[1:fieldinfo(P,T).Ny÷2+1,:])))
# end


# @doc doc"""
#     pixwin(θpix, ℓ)

# Returns the pixel window function for square flat-sky pixels of width `θpix` (in
# arcmin) evaluated at some `ℓ`s. This is the scaling of k-modes, the scaling of
# the power spectrum will be pixwin^2. 
# """
# pixwin(θpix, ℓ) = @. sinc(ℓ*deg2rad(θpix/60)/2π)
