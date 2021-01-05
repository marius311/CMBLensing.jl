
struct ProjHealpix
    Nside :: Int
end


const HealpixMap{M<:ProjHealpix, T, A<:AbstractArray{T}} = BaseField{Map, M, T, A}

HealpixMap(Ix::Vector) = HealpixMap(Ix, ProjHealpix(hp.npix2nside(length(Ix))))

f = HealpixMap(rand(12*128^2))


@. 2f + f

pretty_type_name(::Type{<:HealpixMap}) = "HealpixMap"

typeof(f)


@init global hp = lazy_pyimport("healpy")


function θϕ_to_xy((θ,ϕ))
    r = 2cos(θ/2)
    x =  r*cos(ϕ)
    y = -r*sin(ϕ)
    x, y
end

function xy_to_θϕ((x,y))
    r = sqrt(x^2+y^2)
    θ = 2*acos(r/2)
    ϕ = -atan(y,x)
    θ, ϕ
end

function healpix_to_flat(healpix_map::Vector{T}, ::Type{P}; rot=(0,0,0)) where {Nside, θpix, T, P<:Flat{Nside,θpix}}
    
    Nside_sphere = hp.npix2nside(length(healpix_map))
    @unpack Δx = fieldinfo(P)

    # compute projection coordinate mapping
    xs = ys = Δx*((-Nside÷2:Nside÷2-1) .+ 0.5)
    xys = tuple.(xs,ys')[:]
    θϕs = xy_to_θϕ.(xys)
    (θs, ϕs) = first.(θϕs), last.(θϕs)
    
    # rotate the pole to the equator to match Healpy's azeqview convention, in
    # addition to applying the user rotation
    R = hp.Rotator((0,90,0), eulertype="ZYX") * hp.Rotator(rot, eulertype="ZYX")
    (θs′, ϕs′) = eachrow(R.get_inverse()(θs, ϕs))
    
    # interpolate map
    FlatMap{P,T}(reshape(hp.get_interp_val(healpix_map, θs′, ϕs′), Nside, Nside))
    
end

function flat_to_healpix(f::FlatMap, Nside_sphere; rot=(0,0,0))
    
    @unpack Δx, Nside = fieldinfo(f)

    (θs, ϕs) = hp.pix2ang(Nside_sphere, 0:(12*Nside_sphere^2-1))
    
    # rotate the pole to the equator to match Healpy's azeqview convention, in
    # addition to applying the user rotation
    R = hp.Rotator((0,90,0), eulertype="ZYX") * hp.Rotator(rot, eulertype="ZYX")
    (θs′, ϕs′) = eachrow(R(θs, ϕs))

    # compute projection coordinate mapping
    xys = θϕ_to_xy.(tuple.(θs′, ϕs′))
    xs = first.(xys) ./ Δx .+ Nside÷2 .+ 0.5
    ys = last.(xys) ./ Δx .+ Nside÷2 .+ 0.5

    # interpolate map
    @ondemand(Images.bilinear_interpolation).(Ref(f[:Ix]), xs, ys)
    
end
