
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

function healpix_to_flat(healpix_map::Vector{T}, proj::ProjLambert{T}; rot=(0,0,0)) where {T}
    
    Nside_sphere = hp.npix2nside(length(healpix_map))
    @unpack Δx = proj

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
    FlatMap(reshape(hp.get_interp_val(healpix_map, θs′, ϕs′), Nside, Nside), proj)
    
end

function healpix_pixel_centers_to_flat(f::FlatField, Nside_sphere; rots=[(0,90,0)], healpix_pixels=0:(12*Nside_sphere^2-1))

    @unpack Δx, Ny, Nx = f

    (θs, ϕs) = hp.pix2ang(Nside_sphere, healpix_pixels)
    
    # rotate the pole to the equator to match Healpy's azeqview convention, in
    # addition to applying the user rotation
    R = prod([hp.Rotator(rot, eulertype="ZYX") for rot in rots])
    (θs′, ϕs′) = eachrow(R(θs, ϕs))

    # compute projection coordinate mapping
    # (using Ny for xs and vice-versa intentional)
    xys = @. θϕ_to_xy(tuple(θs′, ϕs′))
    xs = @. first(xys) / Δx + Ny÷2 + 0.5
    ys = @.  last(xys) / Δx + Nx÷2 + 0.5

    (xs, ys)

end


function flat_to_healpix(f::FlatS0, Nside_sphere; kwargs...)
    # get pixel centers of Healpix pixels in 2D map
    xs,ys = healpix_pixel_centers_to_flat(f, Nside_sphere; kwargs...)
    # interpolate map
    @ondemand(Images.bilinear_interpolation).(Ref(f[:Ix]), xs, ys)
end

