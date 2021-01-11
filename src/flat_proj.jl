
abstract type FlatProj end

# default angular resolution used by a number of convenience constructors
θpix₀ = 1

# holds the core things which must be identical between fields for
# broadcasting between them to be possible
struct ProjLambertCore{storage} <: FlatProj
    θpix
    Ny       :: Int
    Nx       :: Int
end

# holds extended information, which might differ between fields that
# you can still broadcast across (eg T can be different)
struct ProjLambert{storage, T, V<:AbstractVector{T}, M<:AbstractMatrix{T}} <: FlatProj
    θpix
    Ny       :: Int
    Nx       :: Int
    Δx       :: T
    Ωpix     :: T
    nyquist  :: T
    Δℓx      :: T
    Δℓy      :: T
    ℓy       :: V
    ℓx       :: V
    ℓmag     :: M
    sin2ϕ    :: M
    cos2ϕ    :: M
end

function ProjLambert(; T=nothing, storage, θpix=θpix₀, Ny, Nx)
    proj = ProjLambertCore{storage}(θpix, Ny, Nx)
    isnothing(T) ? proj : extended_metadata(proj, T)
end

function core_metadata(proj::ProjLambert{storage}) where {storage}
    ProjLambertCore{storage}(proj.θpix, proj.Ny, proj.Nx)
end

@memoize function extended_metadata(proj::ProjLambertCore{storage}, ::Type{T}) where {T,storage}

    @unpack θpix, Ny, Nx = proj

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

    ProjLambert{storage,T,storage{T,1},storage{T,2}}(
        θpix,Ny,Nx,Δx,Ωpix,nyquist,Δℓx,Δℓy,ℓy,ℓx,ℓmag,sin2ϕ,cos2ϕ
    )
    
end

typealias_def(::Type{<:ProjLambert{storage}}) where {storage} = "ProjLambert{$storage}"



### promotion

# used in non-broadcasted algebra to decide the result of performing
# some operation across two fields with a given `metadata`. this is
# free to do more generic promotion than promote_bcast_rule. the
# result should be a common metadata which we can convert both fields
# to then do a succesful broadcast
function promote_basis_generic(
    (b₁,metadata₁) :: Tuple{B₁,<:ProjLambert{T₁}}, 
    (b₂,metadata₂) :: Tuple{B₂,<:ProjLambert{T₂}}
) where {B₁,B₂,T₁,T₂}

    b = promote_basis_generic(b₁, b₂)

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

function preprocess((_,proj)::Tuple{<:Any,<:ProjLambert{storage}}, br::BatchedReal) where {storage}
    adapt(storage, reshape(br.vals, 1, 1, 1, :))
end

function preprocess((_,proj)::Tuple{BaseFieldStyle{S,B},<:ProjLambert}, ∇d::∇diag) where {S,B}

    (B <: Union{Fourier,QUFourier,IQUFourier}) ||
        error("Can't broadcast ∇ as a $(typealias(B)), its not diagonal in this basis.")
    
    # turn both vectors into 2-D matrix so this function is
    # type-stable (note: reshape does not actually make a copy here,
    # so this doesn't impact performance)
    if ∇d.coord == 1
        broadcasted(*, ∇d.prefactor * im, reshape(proj.ℓx, 1, :))
    else
        broadcasted(*, ∇d.prefactor * im, reshape(proj.ℓy, :, 1))
    end

end

function preprocess((_,proj)::Tuple{BaseFieldStyle{S,B},<:ProjLambert}, ::∇²diag) where {S,B}

    (B <: Union{Fourier,QUFourier,IQUFourier}) ||
        error("Can't broadcast ∇² as a $(typealias(B)), its not diagonal in this basis.")

    # need complex here to avoid problem with ^ below being Base.pow instead of CUDA.pow
    # todo: find better solution
    broadcasted(complex, broadcasted(+, broadcasted(^, proj.ℓx', 2), broadcasted(^, proj.ℓy, 2)))
end

function preprocess((_,proj)::Tuple{BaseFieldStyle{S,B},<:ProjLambert}, bp::BandPass) where {S,B}

    (B <: Union{Fourier,<:Basis2Prod{<:Any,Fourier},<:Basis3Prod{<:Any,<:Any,Fourier}}) ||
        error("Can't broadcast a BandPass as a $(typealias(B)), its not diagonal in this basis.")

    Cℓ_to_2D(bp.Wℓ, proj)
end

function Cℓ_to_2D(Cℓ, proj::ProjLambert{storage,T}) where {storage,T}
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



@doc doc"""
    pixwin(θpix, ℓ)

Returns the pixel window function for square flat-sky pixels of width `θpix` (in
arcmin) evaluated at some `ℓ`s. This is the scaling of k-modes, the scaling of
the power spectrum will be pixwin^2. 
"""
pixwin(θpix, ℓ) = @. sinc(ℓ*deg2rad(θpix/60)/2π)

