
# Some initial support for Healpix fields. 
# The main functionality of broadcasting, indexing, and projection for
# a few field types is implemented, but not much beyond that. 

@init global hp = lazy_pyimport("healpy")

struct ProjHealpix <: Proj
    Nside :: Int
end
make_field_aliases("Healpix", ProjHealpix)
typealias_def(::Type{<:ProjHealpix}) = "ProjHealpix"


## constructing from arrays
# spin-0
function HealpixMap(I::A) where {T, A<:AbstractArray{T}}
    HealpixMap(I, ProjHealpix(hp.npix2nside(length(I))))
end
# spin-2
function HealpixField{B}(X::A, Y::A) where {T, A<:AbstractArray{T}, B<:Basis2Prod{<:Union{𝐐𝐔,𝐄𝐁},Map}}
    HealpixField{B}(cat(X, Y, dims=Val(2)), ProjHealpix(hp.npix2nside(length(X))))
end
# spin-(0,2)
function HealpixField{B}(I::A, X::A, Y::A) where {T, A<:AbstractArray{T}, B<:Basis3Prod{𝐈,<:Union{𝐐𝐔,𝐄𝐁},Map}}
    HealpixField{B}(cat(I, X, Y, dims=Val(2)), ProjHealpix(hp.npix2nside(length(I))))
end

### pretty printing
typealias_def(::Type{F}) where {B,M<:ProjHealpix,T,A,F<:HealpixField{B,M,T,A}} = "Healpix$(typealias(B)){$(typealias(A))}"
function Base.summary(io::IO, f::HealpixField)
    @unpack Nside = f
    print(io, "$(length(f))-element Nside=$Nside ")
    Base.showarg(io, f, true)
end

getproperty(f::HealpixField, ::Val{:proj}) = getfield(f,:metadata)
pol_slice(::HealpixField, i) = (:, i)


### broadcasting

function promote_metadata_strict(metadata₁::ProjHealpix, metadata₂::ProjHealpix)
    if metadata₁ === metadata₂
        metadata₁
    else
        error("Can't broadcast Healpix maps with two different Nsides ($metadata₁.Nside, $metadata₂.Nside).")
    end
end

dot(a::HealpixField, b::HealpixField) = sum((Ł(a) .* Ł(b)).arr)


### Projection

# adding more flat projections in the future should be as easy as
# defining pairs of functions like the following two for other cases


## ProjEquiRect
function θϕ_to_ij(proj::ProjEquiRect, θ, ϕ)
    @unpack Ny, Nx, θspan, φspan = proj
    i =        (θ - θspan[1])             / abs(-(θspan...)) * Ny
    j = rem2pi((ϕ - φspan[1]), RoundDown) / abs(-(φspan...)) * Nx
    (i, j)
end

function ij_to_θϕ(proj::ProjEquiRect, i, j)
    @unpack Ny, Nx, θspan, φspan = proj
    θ = abs(-(θspan...)) / Ny * i + θspan[1]
    ϕ = abs(-(φspan...)) / Nx * j + φspan[1]
    (θ, ϕ)
end

# rotation angle of the coordinate basis between the sphere and
# projection, at (θ,ϕ)
function get_ψpol(proj::ProjEquiRect, θ, ϕ)
    0
end


## ProjLambert
# 
# Notes:
# * choices of negative signs are such that the default projection is
#   centered on the center of a healpy.mollview plot and oriented the
#   same way
# * we have to define these at the "broadcasted" level so we can do
#   just a single call to Rotator(), otherwise it'd be really slow

function broadcasted(::typeof(ij_to_θϕ), proj::ProjLambert, is, js)
    @unpack Δx, Ny, Nx, rotator = proj
    R = hp.Rotator(rotator)
    θϕs = broadcast(is, js) do i, j
        x = Δx * (j - Nx÷2 - 0.5)
        y = Δx * (i - Ny÷2 - 0.5)
        r = sqrt(x^2 + y^2)
        θ = 2*acos(r/2)
        ϕ = atan(-x, -y)
        θ, ϕ
    end
    θϕs = reshape(R.get_inverse()(first.(θϕs)[:], last.(θϕs)[:]), 2, size(θϕs)...)
    tuple.(θϕs[1,:,:], θϕs[2,:,:])
end

function broadcasted(::typeof(θϕ_to_ij), proj::ProjLambert, θs, ϕs)
    @unpack Δx, Ny, Nx, rotator = proj
    R = hp.Rotator(rotator)
    (θs, ϕs) = eachrow(R(θs, ϕs))
    broadcast(θs, ϕs) do θ, ϕ
        r = 2cos(θ/2)
        x = -r*sin(ϕ)
        y = -r*cos(ϕ)
        i = y / Δx + Ny÷2 + 0.5
        j = x / Δx + Nx÷2 + 0.5
        (i, j)
    end
end

function broadcasted(::typeof(get_ψpol), proj::ProjLambert, θs, ϕs)
    @unpack rotator, T = proj
    R = hp.Rotator((0,-90,0)) * hp.Rotator(rotator)
    @assert size(θs) == size(ϕs)
    T.(reshape(R.angle_ref(materialize(θs)[:], materialize(ϕs)[:]), size(θs)...))
end


# stores some precomputed quantities for doing projections
struct Projector{method}
    proj_in
    proj_out
    θs
    ϕs
    is
    js
    ψpol
    hpx_idxs_in_patch
    nfft_plan
    nfft_plan_grid
end


## Healpix => Cartesian


# some NFFT stuff needed for method=:fft projections
cu_nfft_loaded = false
@init begin
    @require NFFT="efe261a4-0d2b-5849-be55-fc731d526b0d" begin
        using .NFFT: plan_nfft, AbstractNFFTPlan
        Zygote.@adjoint function *(plan::Union{Adjoint{<:Any,<:AbstractNFFTPlan}, AbstractNFFTPlan}, x::AbstractArray{T}) where {T}
            function mul_nfft_plan_pullback(Δ)
                (nothing, T.(adjoint(plan) * complex(Δ)))
            end
            plan * x, mul_nfft_plan_pullback
        end
    end        
    @require CuNFFT="a9291f20-7f4c-4d50-b30d-4e07b13252e1" global cu_nfft_loaded = true
end


"""
    project(healpix_map::HealpixField => cart_proj::CartesianProj)

Project `healpix_map` to a cartisian projection specified by
`cart_proj`. E.g.:

    healpix_map = HealpixMap(rand(12*2048^2))
    flat_proj = ProjLambert(Ny=128, Nx=128, θpix=3, T=Float32)
    f = project(healpix_map => flat_proj; rotator=pyimport("healpy").Rotator((0,28,23)))

The use of `=>` is to help remember in which order the arguments are
specified. If `healpix_map` is a `HealpixQU`, Q/U polarization angles
are rotated to be aligned with the local coordinates (sometimes called
"polarization flattening").
"""
function project((hpx_map, cart_proj)::Pair{<:HealpixField,<:CartesianProj}; method::Symbol=:bilinear)
    project(Projector(hpx_map.proj => cart_proj; method), hpx_map => cart_proj)
end

function project(projector::Projector, (hpx_map, cart_proj)::Pair{<:HealpixMap,<:CartesianProj})
    @unpack (Ny, Nx, T) = cart_proj
    @unpack (θs, ϕs) = projector
    BaseMap(T.(reshape(hp.get_interp_val(collect(hpx_map), θs, ϕs), Ny, Nx)), cart_proj)
end

function project(projector::Projector, (hpx_map, cart_proj)::Pair{<:HealpixQUMap,<:CartesianProj})
    @unpack (T) = cart_proj
    @unpack (ψpol) = projector
    Q = project(projector, hpx_map.Q => cart_proj)
    U = project(projector, hpx_map.U => cart_proj)
    QU_pol_flattened = @. (Q.arr + im * U.arr) * exp(im * 2 * T(ψpol))
    FlatQUMap(real.(QU_pol_flattened), imag.(QU_pol_flattened), cart_proj)
end

function project(projector::Projector, (hpx_map, cart_proj)::Pair{<:HealpixIQUMap,<:CartesianProj})
    I = project(projector, hpx_map.I => cart_proj)
    P = project(projector, hpx_map.P => cart_proj)
    BaseField{IQUMap}(cat(I.Ix, P.Qx, P.Ux, dims=3), cart_proj)
end

function Projector((hpx_proj,cart_proj)::Pair{<:ProjHealpix,<:CartesianProj}; method::Symbol=:bilinear)
    @unpack (Ny, Nx) = cart_proj
    @unpack (Nside) = hpx_proj
    θϕs = ij_to_θϕ.(cart_proj, 1:Ny, (1:Nx)')
    θs, ϕs = first.(θϕs), last.(θϕs)
    ψpol = get_ψpol.(cart_proj, first.(θϕs), last.(θϕs))
    Projector{method}(hpx_proj, cart_proj, θs[:], ϕs[:], nothing, nothing, ψpol, nothing, nothing, nothing)
end



## Cartesian => Healpix

"""
    project(flat_map::FlatField => healpix_proj::ProjHealpix; [rotator])

Reproject a `flat_map` back onto the sphere in a Healpix projection
specified by `healpix_proj`. E.g.

    flat_map = FlatMap(rand(128,128))
    f = project(flat_map => ProjHealpix(2048); rotator=pyimport("healpy").Rotator((0,28,23)))

The use of `=>` is to help remember in which order the arguments are
specified. Optional keyword argument `rotator` is a `healpy.Rotator`
object specifying a rotation which rotates the north pole to the
center of the desired field. 
"""
function project((cart_map, hpx_proj)::Pair{<:CartesianField, <:ProjHealpix}; method::Symbol=:bilinear)
    project(Projector(cart_map.proj => hpx_proj; method), cart_map => hpx_proj)
end

function Projector((cart_proj,hpx_proj)::Pair{<:CartesianProj,<:ProjHealpix}; method::Symbol=:bilinear)
    @unpack (Nside) = hpx_proj
    @unpack (Ny, Nx, T, storage) = cart_proj
    (θs, ϕs) = hp.pix2ang(Nside, 0:(12*Nside^2-1))
    ijs = θϕ_to_ij.(cart_proj, θs, ϕs)
    is, js = first.(ijs), last.(ijs)
    ψpol = adapt(storage, get_ψpol.(cart_proj, θs, ϕs))
    hpx_idxs_in_patch = adapt(storage, [k for (k,(i,j)) in enumerate(zip(is, js)) if 1<=i<=Ny && 1<=j<=Nx])
    if method == :fft
        @isdefined(plan_nfft) || error("Load the `NFFT` package to make `method=:fft` available.")
        (storage isa Type && storage <: Array) || cu_nfft_loaded || error("Load the `CuNFFT` package to make `method=:fft` available on GPU.")
        # ij indices mapped to [-0.5,0.5] and in the format NFFT wants
        # them for 1) a cartesian grid and 2) where the healpix
        # pixel centers fall in this grid
        nfft_ijs_grid  = adapt(storage, reduce(hcat, [[T((i-Ny÷2-1)/Ny), T((j-Nx÷2-1)/Nx)] for i=1:Ny, j=1:Nx]))
        nfft_ijs       = adapt(storage, reduce(hcat, [[T((i-Ny÷2-1)/Ny), T((j-Nx÷2-1)/Nx)] for (i,j) in zip(is, js) if 1 <= i <= Ny && 1 <= j <= Nx]))
        # two plans needed for FFT resampling
        arr_type = typeof(nfft_ijs)
        nfft_plan_grid = plan_nfft(arr_type, nfft_ijs_grid, (Ny, Nx))
        nfft_plan      = plan_nfft(arr_type, nfft_ijs, (Ny, Nx))
    else
        nfft_plan = nfft_plan_grid = nothing
    end
    Projector{method}(cart_proj, hpx_proj, nothing, nothing, is, js, ψpol, hpx_idxs_in_patch, nfft_plan, nfft_plan_grid)
end

function project(projector::Projector{:bilinear}, (cart_field, hpx_proj)::Pair{<:CartesianS0, <:ProjHealpix})
    @unpack (is, js) = projector
    HealpixMap(broadcast(@ondemand(Images.bilinear_interpolation), Ref(cpu(Map(cart_field).Ix)), is, js), hpx_proj)
end

function project(projector::Projector{:fft}, (cart_field, hpx_proj)::Pair{<:CartesianS0, <:ProjHealpix})
    @assert projector.proj_in == cart_field.proj && projector.proj_out == hpx_proj
    @unpack (Ny, Nx, T) = cart_field
    @unpack (Nside) = hpx_proj
    @unpack (nfft_plan, nfft_plan_grid, hpx_idxs_in_patch) = projector
    splayed_pixels = Map(cart_field).arr[:]
    hpx_map = Zygote.Buffer(splayed_pixels, 12*Nside^2) # need Buffer for AD bc we mutate this array below
    Zygote.@ignore fill!(hpx_map.data, 0)
    hpx_patch = real.(nfft_plan * (adjoint(nfft_plan_grid) * complex(splayed_pixels))) ./ (Ny * Nx)
    hpx_map[hpx_idxs_in_patch] = hpx_patch
    HealpixMap(copy(hpx_map), hpx_proj)
end

function project(projector::Projector, (cart_field, hpx_proj)::Pair{<:CartesianS2, <:ProjHealpix})
    @unpack (T) = cart_field
    @unpack (ψpol) = projector
    Q = project(projector, Ł(cart_field).Q => hpx_proj).arr
    U = project(projector, Ł(cart_field).U => hpx_proj).arr
    Q_flat = @. Q * cos(2ψpol) + U * sin(2ψpol)
    U_flat = @. U * cos(2ψpol) - Q * sin(2ψpol)
    HealpixQUMap([Q_flat; U_flat], hpx_proj)
end

function project(projector::Projector, (cart_field, hpx_proj)::Pair{<:CartesianS02, <:ProjHealpix})
    I = project(projector, cart_field.I => hpx_proj)
    P = project(projector, cart_field.P => hpx_proj)
    HealpixIQUMap([I.arr; P.arr], hpx_proj)
end
