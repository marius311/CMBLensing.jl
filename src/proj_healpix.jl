
# Some initial support for Healpix fields. 
# The main functionality of broadcasting, indexing, and projection for
# a few field types is implemented, but not much beyond that. 

struct ProjHealpix <: Proj
    Nside :: Int
end
make_field_aliases("Healpix", ProjHealpix)
typealias_def(::Type{<:ProjHealpix}) = "ProjHealpix"


## constructing from arrays
# spin-0
function HealpixMap(I::AbstractArray{T}) where {T}
    HealpixMap(I, ProjHealpix(npix2nside(length(I))))
end
# spin-2
function HealpixField{B}(X::AbstractArray{T}, Y::AbstractArray{T}) where {T, B<:Basis2Prod{<:Union{𝐐𝐔,𝐄𝐁},Map}}
    HealpixField{B}(cat(X, Y, dims=Val(2)), ProjHealpix(npix2nside(length(X))))
end
# spin-(0,2)
function HealpixField{B}(I::AbstractArray{T}, X::AbstractArray{T}, Y::AbstractArray{T}) where {T, B<:Basis3Prod{𝐈,<:Union{𝐐𝐔,𝐄𝐁},Map}}
    HealpixField{B}(cat(I, X, Y, dims=Val(2)), ProjHealpix(npix2nside(length(I))))
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

function ij_to_θϕ(proj::ProjLambert, i, j)
    @unpack Δx, Ny, Nx, rotator, T = proj
    x = Δx * (j - Nx÷2 - T(0.5))
    y = Δx * (i - Ny÷2 - T(0.5))
    r = sqrt(x^2 + y^2)
    θ = 2*acos(r/2)
    ϕ = atan(-x, -y)
    # note transform to CoordinateTransformations' (θ,ϕ) convention
    z = SphericalFromCartesian()(RotZYX(T.(deg2rad.(rotator))...) \ CartesianFromSpherical()(Spherical(1, ϕ, T(π/2)-θ)))
    T(π/2)-z.ϕ, z.θ
end

function θϕ_to_ij(proj::ProjLambert, θ, ϕ)
    @unpack Δx, Ny, Nx, rotator, T = proj
    # note transform to CoordinateTransformations' (θ,ϕ) convention
    z = SphericalFromCartesian()(RotZYX(T.(deg2rad.(rotator))...) * CartesianFromSpherical()(Spherical(1, ϕ, T(π/2)-θ)))
    θ, ϕ = T(π/2)-z.ϕ, z.θ
    r = 2cos(θ/2)
    x = -r*sin(ϕ)
    y = -r*cos(ϕ)
    i = y / Δx + Ny÷2 + T(0.5)
    j = x / Δx + Nx÷2 + T(0.5)
    (i, j)
end

function get_ψpol(proj::ProjLambert, θ, ϕ)
    J = ForwardDiff.jacobian(@SVector[θ, ϕ]) do (θ, ϕ)
        SVector{2}(θϕ_to_ij(proj, θ, ϕ))
    end
    (atan(J[1,1], J[2,1]) + atan(-J[2,2],J[1,2]) - π)/2
end


# stores some precomputed quantities for doing projections
struct Projector{method}
    cart_proj
    hpx_proj
    θs                # θ,ϕ of cartesian pixel centers and ψpol at those positions
    ϕs                # ⋅
    ψpol_θϕs          # ⋅
    is                # i,j (fractional) indices of healpix pixel centers and ψpol at those positions
    js                # ⋅
    ψpol_ijs          # ⋅
    hpx_idxs_in_patch # healpix pixel indices inside the patch we're projecting
    nfft_plan
    nfft_plan_grid
end


## Healpix => Cartesian


# some NFFT stuff needed for method=:fft projections
cu_nfft_loaded = false
@init @require NFFT="efe261a4-0d2b-5849-be55-fc731d526b0d" begin
    using .NFFT: plan_nfft, AbstractNFFTPlan
    Zygote.@adjoint function *(plan::Union{Adjoint{<:Any,<:AbstractNFFTPlan}, AbstractNFFTPlan}, x::AbstractArray{T}) where {T}
        function mul_nfft_plan_pullback(Δ)
            (nothing, adjoint(plan) * complex(Δ))
        end
        plan * x, mul_nfft_plan_pullback
    end
    for P in [:(AbstractNFFTPlan{S}), :(Adjoint{Complex{S},<:AbstractNFFTPlan{S}})]
        for op in [:(Base.:*), :(Base.:\)]
            for D in [1, 2] # need explicit dimension to resolve method ambiguity
                @eval function ($op)(plan::$P, arr::AbstractArray{<:Complex{<:Dual{T}}, $D}) where {T, S}
                    arr_of_duals(T, apply_plan($op, plan, arr)...)
                end
            end
        end
    end
end
@init @require CuNFFT="a9291f20-7f4c-4d50-b30d-4e07b13252e1" global cu_nfft_loaded = true


@doc doc"""

    project(healpix_field::HealpixField => cart_proj::CartesianProj; [method = :bilinear])
    project(cart_field::FlatField => healpix_proj::ProjHealpix; [method=:bilinear])

Project a `healpix_field` to a cartesian projection specified by
`cart_proj`, or project a `cart_field` back up to sphere on the
Healpix pixelization specified by `healpix_proj`. E.g. 

```julia
# sphere to cartesian
healpix_field = HealpixMap(rand(12*2048^2))
cart_proj = ProjLambert(Ny=128, Nx=128, θpix=3, T=Float32, rotator=(0,30,0))
f = project(healpix_field => cart_proj)

# and back to sphere
project(f => ProjHealpix(512))
```

The `(Ny, Nx, θpix, rotator)` parameters of `cart_proj` control the
size and location of the projected region.

The use of `=>` is to help remember in which order the arguments are
specified. 

For either projection direction, if the field is a QU or IQU field,
polarization angles are rotated to be aligned with the local
coordinates (sometimes called "polarization flattening").

The projection interpolates the original map at the positions of the
centers of the projected map pixels. `method` controls how this
interpolation is done, and can be one of:

* `:bilinear` — Bilinear interpolation (default)
* `:fft` — FFT-based interpolation, which uses a non-uniform FFT to
  evaluate the discrete Fourier series of the field at arbitrary new
  positions. This is currently implemented only for cartesian to
  Healpix projection. To make this mode available, you must load the
  `NFFT` package first. For GPU fields, you must also load `CuNFFT`.
  Projection with `method=:fft` is both GPU compatible and
  automatically differentiable.

A pre-computation step can be cached by first doing, 

```julia
projector = CMBLensing.Projector(healpix_map.proj => cart_proj, method=:fft)
f = project(projector, healpix_map => cart_proj) 
```

which makes subsequent `project` calls significantly faster. Note the
`method` argument is specified in the precomputation step.

"""
function project((hpx_map, cart_proj)::Pair{<:HealpixField,<:CartesianProj}; method::Symbol=:bilinear)
    project(Projector(hpx_map.proj => cart_proj; method), hpx_map => cart_proj)
end

function project(projector::Projector{:bilinear}, (hpx_map, cart_proj)::Pair{<:HealpixMap,<:CartesianProj})
    @assert projector.hpx_proj == hpx_map.proj && projector.cart_proj == cart_proj
    @unpack (Ny, Nx, T) = cart_proj
    @unpack (θs, ϕs) = projector
    np = pyimport("numpy")
    BaseMap(T.(reshape(PyArray(pyimport("healpy").get_interp_val(np.array(collect(hpx_map)), np.array(θs), np.array(ϕs))), Ny, Nx)), cart_proj)
end

function project(projector::Projector{:fft}, (hpx_map, cart_proj)::Pair{<:HealpixMap,<:CartesianProj})
    @assert projector.hpx_proj == hpx_map.proj && projector.cart_proj == cart_proj
    @unpack (Ny, Nx, T) = cart_proj
    @unpack (Nside) = hpx_map
    @unpack (nfft_plan, nfft_plan_grid, hpx_idxs_in_patch) = projector
    splayed_pixels = real.(nfft_plan_grid * (adjoint(nfft_plan) * complex(ensure_dense(hpx_map[hpx_idxs_in_patch])))) ./ (length(hpx_idxs_in_patch))
    FlatMap(reshape(splayed_pixels, Ny, Nx), cart_proj)
end

function project(projector::Projector, (hpx_map, cart_proj)::Pair{<:HealpixQUMap,<:CartesianProj})
    @unpack (T) = cart_proj
    ψpol = projector.ψpol_θϕs
    Q = project(projector, Ł(hpx_map).Q => cart_proj).arr
    U = project(projector, Ł(hpx_map).U => cart_proj).arr
    Q_flat = @. Q * cos(2ψpol) - U * sin(2ψpol)
    U_flat = @. U * cos(2ψpol) + Q * sin(2ψpol)
    FlatQUMap(cat(Q_flat, U_flat, dims=3), cart_proj)
end

function project(projector::Projector, (hpx_map, cart_proj)::Pair{<:HealpixIQUMap,<:CartesianProj})
    I = project(projector, hpx_map.I => cart_proj)
    P = project(projector, hpx_map.P => cart_proj)
    BaseField{IQUMap}(cat(I.Ix, P.Qx, P.Ux, dims=3), cart_proj)
end

function Projector((hpx_proj,cart_proj)::Pair{<:ProjHealpix,<:CartesianProj}; method::Symbol=:bilinear)
    @unpack (Ny, Nx, T, storage) = cart_proj
    @unpack (Nside) = hpx_proj

    # θ,ϕ of cartesian pixel centers and ψpol at those positions
    θϕs = ij_to_θϕ.(cart_proj, 1:Ny, (1:Nx)')
    θs, ϕs = first.(θϕs), last.(θϕs)
    ψpol_θϕs = adapt(storage, get_ψpol.(cart_proj, first.(θϕs), last.(θϕs)))
    
    # i,j (fractional) indices of healpix pixel centers and ψpol at those positions
    θϕs′ = pix2angRing.(Ref(Resolution(Nside)), 1:(12*Nside^2))
    (θs′, ϕs′) = first.(θϕs′), last.(θϕs′)
    ijs = θϕ_to_ij.(cart_proj, θs′, ϕs′)
    is, js = first.(ijs), last.(ijs)
    ψpol_ijs = adapt(storage, get_ψpol.(cart_proj, θs′, ϕs′))

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

    Projector{method}(
        cart_proj, hpx_proj,
        θs[:], ϕs[:], ψpol_θϕs, 
        is, js, ψpol_ijs, 
        hpx_idxs_in_patch, nfft_plan, nfft_plan_grid
    )
end



## Cartesian => Healpix

function project((cart_map, hpx_proj)::Pair{<:CartesianField, <:ProjHealpix}; method::Symbol=:bilinear)
    project(Projector(cart_map.proj => hpx_proj; method), cart_map => hpx_proj)
end

function Projector((cart_proj,hpx_proj)::Pair{<:CartesianProj,<:ProjHealpix}; method::Symbol=:bilinear)
    Projector(hpx_proj => cart_proj; method) # precomputed quantities same inependent of order
end

function project(projector::Projector{:bilinear}, (cart_field, hpx_proj)::Pair{<:CartesianS0, <:ProjHealpix})
    @assert projector.cart_proj == cart_field.proj && projector.hpx_proj == hpx_proj
    @unpack (is, js) = projector
    HealpixMap(broadcast(@ondemand(Images.bilinear_interpolation), Ref(cpu(Map(cart_field).Ix)), is, js), hpx_proj)
end

function project(projector::Projector{:fft}, (cart_field, hpx_proj)::Pair{<:CartesianS0, <:ProjHealpix})
    @assert projector.cart_proj == cart_field.proj && projector.hpx_proj == hpx_proj
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
    ψpol = projector.ψpol_ijs
    Q = project(projector, Ł(cart_field).Q => hpx_proj).arr
    U = project(projector, Ł(cart_field).U => hpx_proj).arr
    Q_flat = @. Q * cos(2ψpol) + U * sin(2ψpol)
    U_flat = @. U * cos(2ψpol) - Q * sin(2ψpol)
    HealpixQUMap(cat(Q_flat, U_flat, dims=2), hpx_proj)
end

function project(projector::Projector, (cart_field, hpx_proj)::Pair{<:CartesianS02, <:ProjHealpix})
    I = project(projector, cart_field.I => hpx_proj)
    P = project(projector, cart_field.P => hpx_proj)
    HealpixIQUMap(cat(I.arr, P.arr, dims=2), hpx_proj)
end
