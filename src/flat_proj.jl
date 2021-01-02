
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
