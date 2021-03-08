
# Some initial support for Healpix fields. 
# The main functionality of broadcasting, indexing, and projection for
# a few field types is implemented, but not much beyond that. 


@init global hp = lazy_pyimport("healpy")

struct ProjHealpix <: FieldMetadata
    Nside :: Int
end
typealias_def(::Type{<:ProjHealpix}) = "ProjHealpix"


# spin-0
const HealpixMap{        M<:ProjHealpix, T, A<:AbstractArray{T} } = BaseField{Map,        M, T, A}
const HealpixFourier{    M<:ProjHealpix, T, A<:AbstractArray{T} } = BaseField{Fourier,    M, T, A}
# spin-2
const HealpixQUMap{      M<:ProjHealpix, T, A<:AbstractArray{T} } = BaseField{QUMap,      M, T, A}
const HealpixQUFourier{  M<:ProjHealpix, T, A<:AbstractArray{T} } = BaseField{QUFourier,  M, T, A}
const HealpixEBMap{      M<:ProjHealpix, T, A<:AbstractArray{T} } = BaseField{EBMap,      M, T, A}
const HealpixEBFourier{  M<:ProjHealpix, T, A<:AbstractArray{T} } = BaseField{EBFourier,  M, T, A}
# spin-(0,2)
const HealpixIQUMap{     M<:ProjHealpix, T, A<:AbstractArray{T} } = BaseField{IQUMap,     M, T, A}
const HealpixIQUFourier{ M<:ProjHealpix, T, A<:AbstractArray{T} } = BaseField{IQUFourier, M, T, A}
const HealpixIEBMap{     M<:ProjHealpix, T, A<:AbstractArray{T} } = BaseField{IEBMap,     M, T, A}
const HealpixIEBFourier{ M<:ProjHealpix, T, A<:AbstractArray{T} } = BaseField{IEBFourier, M, T, A}

const HealpixField{B, M<:ProjHealpix, T, A<:AbstractArray{T}} = BaseField{B, M, T, A}

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
typealias_def(::Type{F}) where {B,M<:ProjHealpix,T,A,F<:HealpixField{B,M,T,A}} = "Healpix$(typealias(B)){$(typealias(A)),$(typealias(M))}"
function Base.summary(io::IO, f::HealpixField)
    @unpack Nside = f
    print(io, "$(length(f))-element Nside=$Nside ")
    Base.showarg(io, f, true)
end

# useful for enumerating some cases below
_healpix_sub_components = [
    (:HealpixMap,        ("Ix"=>:, "I"=>:)),
    (:HealpixFourier,    ("Il"=>:, "I"=>:)),
    (:HealpixQUMap,      ("Qx"=>1, "Ux"=>2, "Q" =>1, "U"=>2, "P"=>:)),
    (:HealpixQUFourier,  ("Ql"=>1, "Ul"=>2, "Q" =>1, "U"=>2, "P"=>:)),
    (:HealpixEBMap,      ("Ex"=>1, "Bx"=>2, "E" =>1, "B"=>2, "P"=>:)),
    (:HealpixEBFourier,  ("El"=>1, "Bl"=>2, "E" =>1, "B"=>2, "P"=>:)),
    (:HealpixIQUMap,     ("Ix"=>1, "Qx"=>2, "Ux"=>3, "I"=>1, "Q"=>2, "U"=>3, "P"=>2:3, "IP"=>:)),
    (:HealpixIQUFourier, ("Il"=>1, "Ql"=>2, "Ul"=>3, "I"=>1, "Q"=>2, "U"=>3, "P"=>2:3, "IP"=>:)),
    (:HealpixIEBMap,     ("Ix"=>1, "Ex"=>2, "Bx"=>3, "I"=>1, "E"=>2, "B"=>3, "P"=>2:3, "IP"=>:)),
    (:HealpixIEBFourier, ("Il"=>1, "El"=>2, "Bl"=>3, "I"=>1, "E"=>2, "B"=>3, "P"=>2:3, "IP"=>:)),
]
# sub-components
for (F, props) in _healpix_sub_components
    for (k,I) in props
        body = if k[end] in "xl"
            I==(:) ? :(getfield(f,:arr)) : :(view(getfield(f,:arr), :, $I))
        else
            I==(:) ? :f : :($(HealpixField{k=="P" ? Basis2Prod{basis(@eval($F)).parameters[end-1:end]...} : basis(@eval($F)).parameters[end]})(view(getfield(f,:arr), :, $I), f.metadata))
        end
        @eval getproperty(f::$F, ::Val{$(QuoteNode(Symbol(k)))}) = $body
    end
end
getproperty(f::HealpixField, ::Val{:proj}) = getfield(f,:metadata)



### broadcasting

function promote_metadata_strict(metadata₁::ProjHealpix, metadata₂::ProjHealpix)
    if metadata₁ === metadata₂
        metadata₁
    else
        error("Can't broadcast Healpix maps with two different Nsides ($metadata₁.Nside, $metadata₂.Nside).")
    end
end


### Projection

## Coordinate transformations

# choices of negative signs are such that the default projection is
# centered on the center of a healpy.mollview plot and oriented the
# same way.

function xy_to_θϕ(::ProjLambert, x, y)
    r = sqrt(x^2+y^2)
    θ = 2*acos(r/2)
    ϕ = atan(-x,-y)
    θ, ϕ
end

function θϕ_to_xy(::ProjLambert, θ, ϕ)
    r = 2cos(θ/2)
    x = -r*sin(ϕ)
    y = -r*cos(ϕ)
    x, y
end

# adding more flat projections in the future should be as easy as
# defining pairs of functions like the above two for other cases


function xy_to_θϕ(flat_proj::FlatProj; rotator)
    @unpack Δx, Ny, Nx = flat_proj
    xs = @. Δx * ((-Nx÷2:Nx÷2-1) + 0.5)
    ys = @. Δx * ((-Ny÷2:Ny÷2-1) + 0.5)
    θϕs = xy_to_θϕ.(Ref(flat_proj), xs, ys')
    (eachrow(rotator.get_inverse()(first.(θϕs)[:], last.(θϕs)[:]))...,)
end

function θϕ_to_xy((hpx_proj, flat_proj)::Pair{<:ProjHealpix, <:FlatProj}; rotator)
    @unpack Δx, Ny, Nx = flat_proj
    @unpack Nside = hpx_proj
    (θs, ϕs) = hp.pix2ang(Nside, 0:(12*Nside^2-1))
    (θs′, ϕs′) = eachrow(rotator(θs, ϕs))
    xys = θϕ_to_xy.(Ref(flat_proj), θs′, ϕs′)
    xs = @. first(xys) / Δx + Nx÷2 + 0.5
    ys = @.  last(xys) / Δx + Ny÷2 + 0.5
    (xs, ys)
end


## Healpix => Flat

"""
    project(healpix_map::HealpixField => flat_proj::FlatProj; [rotator])

Project `healpix_map` to a flat projection specified by `flat_proj`.
E.g.:

    healpix_map = HealpixMap(rand(12*2048^2))
    flat_proj = ProjLambert(Ny=128, Nx=128, θpix=3, T=Float32)
    f = project(healpix_map => flat_proj; rotator=pyimport("healpy").Rotator((0,28,23)))

The use of `=>` is to help remember in which order the arguments are
specified. Optional keyword argument `rotator` is a `healpy.Rotator`
object specifying a rotation which rotates the north pole to the
center of the desired field. If `healpix_map` is a `HealpixQU`, Q/U
polarization angles are rotated to be aligned with the local
coordinates (sometimes called "polarization flattening").
"""
function project((hpx_map, flat_proj)::Pair{<:HealpixMap,<:ProjLambert{T}}; rotator=hp.Rotator((0,90,0))) where {T}
    _project(xy_to_θϕ(flat_proj; rotator), hpx_map => flat_proj)
end

function project((hpx_map, flat_proj)::Pair{<:HealpixQUMap,<:ProjLambert{T}}; rotator=hp.Rotator((0,90,0))) where {T}
    @unpack Ny, Nx = flat_proj
    (θs, ϕs) = xy_to_θϕ(flat_proj; rotator)
    Q = _project((θs, ϕs), hpx_map.Q => flat_proj)
    U = _project((θs, ϕs), hpx_map.U => flat_proj)
    ψpol = permutedims(reshape((hp.Rotator((0,-90,0)) * rotator).angle_ref(θs, ϕs), Nx, Ny))
    QU_flat = @. (Q.arr + im * U.arr) * exp(im * 2 * ψpol)
    FlatQUMap(real.(QU_flat), imag.(QU_flat), flat_proj)
end

function _project((θs,ϕs)::Tuple, (hpx_map, flat_proj)::Pair{<:HealpixMap,<:ProjLambert{T}}) where {T}
    @unpack Ny, Nx = flat_proj
    FlatMap(T.(permutedims(reshape(hp.get_interp_val(collect(hpx_map), θs, ϕs), Nx, Ny))), flat_proj)
end


## Flat => Healpix

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
function project((flat_map, hpx_proj)::Pair{<:FlatS0, <:ProjHealpix}; rotator=hp.Rotator((0,90,0)))
    (xs, ys) = θϕ_to_xy(hpx_proj => flat_map.proj; rotator)
    HealpixMap(@ondemand(Images.bilinear_interpolation).(Ref(cpu(flat_map[:Ix])), ys, xs), hpx_proj)
end

