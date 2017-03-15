
try
    global hp
    @pyimport healpy as hp
catch err
    warn("Couldn't load Healpy, Healpix functionality will not be available.")
    println(err)
end

# Healpix pixelization with particular `Nside` value
abstract type Healpix{Nside} <: Pix end

struct HealpixS0Map{Nside} <: Field{Healpix{Nside},S0,Map}
    Tx::Vector{Float64}
end
HealpixS0Map(Tx::Vector{Float64}) = HealpixS0Map{hp.npix2nside(length(Tx))}(Tx)

struct HealpixS0Fourier{Nside} <: Field{Healpix{Nside},S0,Fourier}
    alm::Vector{Complex{Float64}}
end

Fourier{Nside}(f::HealpixS0Map{Nside}) = HealpixS0Fourier{Nside}(hp.map2alm(f.Tx))
Map{Nside}(f::HealpixS0Fourier{Nside}) = HealpixS0Map{Nside}(hp.alm2map(f.alm))
