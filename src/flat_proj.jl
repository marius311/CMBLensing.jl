
abstract type FlatProj end

# default angular resolution used by a number of convenience constructors
θpix₀ = 1


struct ProjLambert{T, V<:AbstractVector{T}, M<:AbstractMatrix{T}} <: FlatProj
    θpix
    storage
    Δx       :: T
    Ny       :: Int
    Nx       :: Int
    Ωpix     :: T
    nyquist  :: T
    Δℓx      :: T
    Δℓy      :: T
    ℓy       :: V
    ℓx       :: V
    ℓmag     :: M
    sin2ϕ    :: M
    cos2ϕ    :: M
    units
    θϕ_center
end

@memoize function ProjLambert(;Ny, Nx, θpix=θpix₀, T=Float32, storage=Array)

    Δx           = T(deg2rad(θpix/60))
    Δℓx          = T(2π/(Nx*Δx))
    Δℓy          = T(2π/(Ny*Δx))
    nyquist      = T(2π/(2Δx))
    Ωpix         = T(Δx^2)
    ℓy           = adapt(storage, (ifftshift(-Ny÷2:(Ny-1)÷2) .* Δℓy)[1:Ny÷2+1])
    ℓx           = adapt(storage, (ifftshift(-Nx÷2:(Nx-1)÷2) .* Δℓx))
    ℓmag         = @. sqrt(ℓx'^2 + ℓy^2)
    ϕ            = @. angle(ℓx' + im*ℓy)
    sin2ϕ, cos2ϕ = @. sin(2ϕ), cos(2ϕ)
    if iseven(Ny)
        sin2ϕ[end, end:-1:(Nx÷2+2)] .= sin2ϕ[end, 2:Nx÷2]
    end

    units     = 1
    θϕ_center = nothing
    
    ProjLambert(θpix,storage,Δx,Ny,Nx,Ωpix,nyquist,Δℓx,Δℓy,ℓy,ℓx,ℓmag,sin2ϕ,cos2ϕ,units,θϕ_center)
    
end

typealias_def(::Type{<:ProjLambert{T}}) where {T} = "ProjLambert{$T}"



### promotion

# used in broadcasting to decide the result of broadcasting across two
# fields with a given `metadata` and basis, `b` (where b is an
# instance of the type-parameter B) 
function promote_bcast(
    (b₁,metadata₁) :: Tuple{B₁,<:ProjLambert{T₁}}, 
    (b₂,metadata₂) :: Tuple{B₂,<:ProjLambert{T₂}}
) where {B₁,B₂,T₁,T₂}

    # even though if metadata₁ === metadata₂ we could technically
    # return either, it helps inference if we always return the
    # technically-"wider" one. this line is optimized away at compile
    # time anyway so doesn't slow us down if metadata₁ === metadata₂
    # is indeed true
    wider_metadata = promote_type(T₁,T₂) == T₁ ? metadata₁ : metadata₂
    b = promote_bcast(b₁,b₂)

    if (
        b isa Basis && (
            metadata₁ === metadata₂ || (
                metadata₁.θpix == metadata₂.θpix &&
                metadata₁.Ny   == metadata₂.Ny   &&
                metadata₁.Nx   == metadata₂.Nx      
            )
        )
    )
        (b, wider_metadata)
    else
        error("""Can't broadcast two fields with the following differing metadata:
        1: $((;B₁, select(fields(metadata₁),(:θpix,:Ny,:Nx))...))
        2: $((;B₂, select(fields(metadata₂),(:θpix,:Ny,:Nx))...))
        """)
    end

end


# used in non-broadcasted algebra to decide the result of performing
# some operation across two fields with a given `metadata`. this is
# free to do more generic promotion than promote_bcast_rule. the
# result should be a common metadata which we can convert both fields
# to then do a succesful broadcast
function promote_generic(
    (b₁,metadata₁) :: Tuple{B₁,<:ProjLambert{T₁}}, 
    (b₂,metadata₂) :: Tuple{B₂,<:ProjLambert{T₂}}
) where {B₁,B₂,T₁,T₂}

    b = promote_generic(b₁, b₂)

    if (
        metadata₁ === metadata₂ || (
            metadata₁.θpix    == metadata₂.θpix    &&
            metadata₁.Ny      == metadata₂.Ny      &&
            metadata₁.Nx      == metadata₂.Nx      &&
            metadata₁.storage == metadata₂.storage
        )
    )
        (b, metadata₁)
    else
        error("""Can't promote two fields with the following differing metadata:
        1: $(select(fields(metadata₁),(:θpix,:Ny,:Nx,:storage)))
        2: $(select(fields(metadata₂),(:θpix,:Ny,:Nx,:storage)))
        """)
    end
    # in the future, could add rules here to allow e.g. automatically
    # upgrading one field if they have different resolutions, etc...
end



### preprocessing

function preprocess((_,proj)::Tuple{<:Any,<:ProjLambert}, ∇d::∇diag)
    # turn both vectors into 2-D matrix so this function is
    # type-stable (note: reshape does not actually make a copy here,
    # so this doesn't impact performance)
    if ∇d.coord == 1
        broadcasted(*, ∇d.prefactor * im, reshape(proj.ℓx, 1, :))
    elseif ∇d.coord == 2
        broadcasted(*, ∇d.prefactor * im, reshape(proj.ℓy, :, 1))
    else
        error()
    end
end

function preprocess((_,proj)::Tuple{<:Any,<:ProjLambert}, ::∇²diag)
    broadcasted(+, broadcasted(^, proj.ℓx', 2), broadcasted(^, proj.ℓy, 2))
end

function preprocess((_,proj)::Tuple{<:Any,<:ProjLambert}, bp::BandPass)
    Cℓ_to_2D(bp.Wℓ, proj)
end

function Cℓ_to_2D(Cℓ, proj::ProjLambert{T}) where {T}
    Complex{T}.(nan2zero.(Cℓ.(proj.ℓmag)))
end


### adapting

# dont adapt the fields in proj, instead re-call into the memoized
# ProjLambert so we always get back the singleton ProjLambert object
# for the given set of parameters (helps reduce memory usage and
# speed-up subsequent broadcasts which would otherwise not hit the
# "===" branch of the "promote_*" methods)
function adapt_structure(storage, proj::ProjLambert{T}) where {T}
    @unpack Ny, Nx, θpix = proj
    T′ = eltype(storage)
    ProjLambert(;Ny, Nx, θpix, T=(T′==Any ? T : real(T′)), storage)
end



# @doc doc"""
#     pixwin(θpix, ℓ)

# Returns the pixel window function for square flat-sky pixels of width `θpix` (in
# arcmin) evaluated at some `ℓ`s. This is the scaling of k-modes, the scaling of
# the power spectrum will be pixwin^2. 
# """
# pixwin(θpix, ℓ) = @. sinc(ℓ*deg2rad(θpix/60)/2π)
