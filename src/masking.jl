
function make_mask(
    rng::AbstractRNG, Nside, θpix; 
    edge_padding_deg = 2,
    edge_rounding_deg = 1,
    apodization_deg = 1,
    ptsrc_radius_arcmin = 7, # roughly similar to Story et al. 2015
    num_ptsrcs = round(Int, prod(Nside .* (1,1)) * (θpix/60)^2 * 120/100)  # SPT-like density of sources
)

    deg2npix(x) = round(Int, x/θpix*60)
    arcmin2npix(x) = round(Int, x/θpix)
    
    ptsrc = num_ptsrcs==0 ? 1 : .!bleed(sim_ptsrcs(rng, Nside, num_ptsrcs), arcmin2npix(ptsrc_radius_arcmin))
    boundary = boundarymask(Nside, deg2npix(edge_padding_deg))
    mask_array = if apodization_deg in (false, 0)
        boundary .& ptsrc
    else
        apod_ptsrc = num_ptsrcs==0 ? 1 : cos_apod(ptsrc, arcmin2npix(ptsrc_radius_arcmin));
        cos_apod(boundary, deg2npix(apodization_deg), deg2npix(edge_rounding_deg)) .* apod_ptsrc
    end
    
    FlatMap(Float32.(mask_array), θpix=θpix)
end

make_mask(args...; kwargs...) = make_mask(Random.default_rng(), args...; kwargs...)


# all padding/smoothing/etc... quantities below are in units of numbers of pixels

function boundarymask(Nside, pad)
    m = fill(true, Nside .* (1,1,))
    m[1:pad,:]          .= false
    m[:,1:pad]          .= false
    m[end-pad+1:end,:]  .= false
    m[:,end-pad+1:end]  .= false
    m
end

function bleed(img, w)
    @dynamic import ImageMorphology
    Ny,Nx = size(img)
    nearest = getfield.(ImageMorphology.feature_transform(img), :I)
    [norm(nearest[i,j] .- [i,j]) < w for i=1:Ny,j=1:Nx]
end

function cos_apod(img, w, smooth_distance=false)
    @dynamic import ImageMorphology, ImageFiltering
    Ny,Nx = size(img)
    nearest = getfield.(ImageMorphology.feature_transform(.!img),:I)
    distance = [norm(nearest[i,j] .- [i,j]) for i=1:Ny,j=1:Nx]
    if smooth_distance!=false
        distance = ImageFiltering.imfilter(distance, ImageFiltering.Kernel.gaussian(smooth_distance))
    end
    @. (1-cos(min(distance,w)/w*pi))/2
end

function round_edges(img, w)
    @dynamic import ImageFiltering
    .!(ImageFiltering.imfilter(img, ImageFiltering.Kernel.gaussian(w)) .< 0.5)
end

function sim_ptsrcs(rng,Nside,nsources)
    Ny, Nx = Nside .* (1,1)
    m = fill(false,Ny,Nx);
    for i=1:nsources
        m[rand(rng,1:Ny),rand(rng,1:Nx)] = true
    end
    m
end
