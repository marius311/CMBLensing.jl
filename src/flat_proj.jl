
abstract type FlatProj end

struct ProjLambert{T, V<:AbstractVector{T}, M<:AbstractMatrix{T}} <: FlatProj
    # these must be the same to broadcast together
    Ny        :: Int
    Nx        :: Int
    θpix      :: Float64
    center    :: Tuple{Float64,Float64}
    # these can be different and still broadcast (including different types)
    storage
    Δx        :: T
    Ωpix      :: T
    nyquist   :: T
    Δℓx       :: T
    Δℓy       :: T
    ℓy        :: V
    ℓx        :: V
    ℓmag      :: M
    sin2ϕ     :: M
    cos2ϕ     :: M
end

real_type(T) = promote_type(real(T), Float32)
@init @require Unitful="1986cc42-f94f-5a68-af5c-568840ba703d" real_type(::Type{<:Unitful.Quantity{T}}) where {T} = real_type(T)

ProjLambert(;Ny, Nx, θpix=1, center=(0,0), T=Float32, storage=Array) = 
    ProjLambert(Ny, Nx, Float64(θpix), Float64.(center), real_type(T), storage)

@memoize function ProjLambert(Ny, Nx, θpix, center, T, storage)

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

    ProjLambert(Ny,Nx,θpix,center,storage,Δx,Ωpix,nyquist,Δℓx,Δℓy,ℓy,ℓx,ℓmag,sin2ϕ,cos2ϕ)
    
end

typealias_def(::Type{<:ProjLambert{T}}) where {T} = "ProjLambert{$T}"



### promotion

# used in broadcasting to decide the resulting metadata when
# broadcasting over two fields
function promote_metadata_strict(metadata₁::ProjLambert{T₁}, metadata₂::ProjLambert{T₂} ) where {T₁,T₂}

    if (
        metadata₁.θpix === metadata₂.θpix &&
        metadata₁.Ny   === metadata₂.Ny   &&
        metadata₁.Nx   === metadata₂.Nx      
    )
        
        # always returning the "wider" metadata even if T₁==T₂ helps
        # inference and is optimized away anyway
        promote_type(T₁,T₂) == T₁ ? metadata₁ : metadata₂
        
    else
        error("""Can't broadcast two fields with the following differing metadata:
        1: $(select(fields(metadata₁),(:θpix,:Ny,:Nx)))
        2: $(select(fields(metadata₂),(:θpix,:Ny,:Nx)))
        """)
    end

end


# used in non-broadcasted algebra to decide the resulting metadata
# when performing some operation across two fields. this is free to do
# more generic promotion than promote_metadata_strict (although this
# is currently not used, but in the future could include promoting
# resolution, etc...). the result should be a common metadata which we
# can convert both fields to then do a succesful broadcast
promote_metadata_generic(metadata₁::ProjLambert, metadata₂::ProjLambert) = 
    promote_metadata_strict(metadata₁, metadata₂)


### preprocessing
# defines how ImplicitFields and BatchedReals behave when broadcasted
# with ProjLambert fields 

function preprocess((_,proj)::Tuple{<:Any,<:ProjLambert{T,V}}, br::BatchedReal) where {T,V}
    adapt(V, reshape(br.vals, 1, 1, 1, :))
end

function preprocess((_,proj)::Tuple{BaseFieldStyle{S,B},<:ProjLambert}, ∇d::∇diag) where {S,B}

    (B <: Union{Fourier,QUFourier,IQUFourier}) ||
        error("Can't broadcast ∇² as a $(typealias(B)), its not diagonal in this basis.")

    # turn both into 2D matrices so this function is type-stable
    # (reshape doesnt actually make a copy here, so this doesn't
    # impact performance)
    if ∇d.coord == 1
        broadcasted(*, ∇d.prefactor * im, reshape(proj.ℓx, 1, :))
    else
        broadcasted(*, ∇d.prefactor * im, reshape(proj.ℓy, :, 1))
    end
end

function preprocess((_,proj)::Tuple{BaseFieldStyle{S,B},<:ProjLambert}, ::∇²diag) where {S,B}
    
    (B <: Union{Fourier,<:Basis2Prod{<:Any,Fourier},<:Basis3Prod{<:Any,<:Any,Fourier}}) ||
        error("Can't broadcast a BandPass as a $(typealias(B)), its not diagonal in this basis.")

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



@doc doc"""
    pixwin(θpix, ℓ)

Returns the pixel window function for square flat-sky pixels of width `θpix` (in
arcmin) evaluated at some `ℓ`s. This is the scaling of k-modes, the scaling of
the power spectrum will be pixwin^2. 
"""
pixwin(θpix, ℓ) = @. sinc(ℓ*deg2rad(θpix/60)/2π)



# ### serialization
# makes it so the arrays in ProjLambert objects aren't actually
# serialized to disk by JLD2, instead just (Ny, Nx, θpix, center, T)
# are stored, and the arrays are reconstructed when loading a field

function JLD2.writeas(::Type{P}) where {P<:ProjLambert}
    Tuple{Val{ProjLambert},Int,Int,Float64,Tuple{Float64,Float64},DataType}
end
function JLD2.wconvert(
         :: Type{Tuple{Val{ProjLambert},Int,Int,Float64,Tuple{Float64,Float64},DataType}}, 
    proj :: ProjLambert{T}
) where {T}
    @unpack Ny, Nx, θpix, center = proj
    (Val(ProjLambert), Ny, Nx, θpix, center, T)
end
function JLD2.rconvert(
         :: Type{<:ProjLambert}, 
    proj :: Tuple{Val{ProjLambert},Int,Int,Float64,Tuple{Float64,Float64},DataType}
)
    (_, Ny, Nx, θpix, center, T) = proj
    ProjLambert(; Ny, Nx, θpix, center, T)
end