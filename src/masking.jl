module Masking

export sptlike_mask

using Images
using ImageFiltering

"""
An SPT-like mask, based mostly on,

For the boundary:

"[First we] calculate the distance from the
nearest masked pixel, and this distance map is smoothed
using a Gaussian beam of FWHM = 15'. This smoothing
is applied to soften the corners of the mask. The distance
map is then used to apodize the binary mask with a
Gaussian beam of FWHM = 30'" - Omori et al. 2017

and for point sources: 

"The SPTpol deep field contains a number of bright
point sources, which we mask. We identify all sources
detected at > 5σ at 150 GHz from Mocanu et al. (2013),
which corresponds to a flux-cut of approximately 6 mJy;
all pixels within 5' of each source are masked in both
the TOD filtering and the final maps. We extend this
mask to 10' for all very bright sources detected at > 75σ.
We also use a 10' radius to mask all galaxy clusters de-
tected in this field by Vanderlinde et al. (2010) using
the Sunyaev-Zel’dovich effect. This masking removes ap-
proximately 120 sources, cutting 5 deg² of the field. We
additionally multiply the maps by a sky mask that down-
weights the noisy edges of the maps." Story et al. 2015


Note: some of the above numbers are very slightly tweaked.

"""
function sptlike_mask(nside, Θpix; apod=false, nsources=round(Int,((nside*Θpix)/60)^2 * 120/100), paddeg=3)
    ptsrc = .!bleed(sim_ptsrcs(nside,nsources),7/Θpix)
    boundary = boundarymask(nside, Θpix, paddeg)
    if apod==false
        boundary .& ptsrc
    else
        apod==true && (apod=90)
        cos_apod(boundary,apod/Θpix,15/Θpix) .* cos_apod(ptsrc,7/Θpix);
    end
end



function boundarymask(nside,Θpix,paddeg=3)
    pad = round(Int,paddeg*60/Θpix)
    m = fill(true,nside,nside)
    m[1:pad,:] = m[end-pad:end,:] = m[:,1:pad] = m[:,end-pad:end] = false
    m
end

function bleed(img,w)
    nside,nside = size(img)
    nearest = getfield.(feature_transform(img),:I)
    [norm(nearest[i,j] .- [i,j]) < w for i=1:nside,j=1:nside]
end

function cos_apod(img,w,smooth_distance=false)
    nside,nside = size(img)
    nearest = getfield.(feature_transform(.!img),:I)
    distance = [norm(nearest[i,j] .- [i,j]) for i=1:nside,j=1:nside]
    if smooth_distance!=false
        distance = imfilter(distance, Kernel.gaussian(smooth_distance))
    end
    @. (1-cos(min(distance,w)/w*pi))/2
end

function sim_ptsrcs(nside,nsources)
    m = fill(false,nside,nside);
    for i=1:nsources
        m[rand(1:nside),rand(1:nside)] = true
    end
    m
end

end
