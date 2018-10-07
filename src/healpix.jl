
struct HpxPix{Nside} <: Pix end

struct MaskedHpxS0Map{Nside, T} <: Field{Map, S0, HpxPix{Nside}}
    Tx::Vector{T}
    # mapping between ring indices and a unit range encompassing the non-masked
    # pixels on that ring
    unmasked_rings::OrderedDict{Int,UnitRange{Int}} 
end

function MaskedHpxS0Map(m::Vector{T}) where {T}
    Nside = hp.npix2nside(length(m))
    ring_ranges = ringinfo(Nside).ring_ranges
    unmasked_rings = OrderedDict{Int,UnitRange{Int}}()
    for (ring_index, ring_range) in enumerate(ring_ranges)
        unmasked = @. !(isnan(m[ring_range]) | (m[ring_range] ≈ hp.UNSEEN))
        if any(unmasked)
            r = ring_range[unmasked]
            unmasked_rings[ring_index] = minimum(r):maximum(r)
        end
        m[ring_range[.!unmasked]]=NaN
    end
    MaskedHpxS0Map{Nside,T}(m,unmasked_rings)
end

ringinfo(Nside) = ringinfo(Val(Nside))
@generated function ringinfo(::Val{Nside}) where {Nside}
    r = hp.ringinfo(Nside, collect(1:4Nside-1))
    ring_lengths = r[2]
    ring_starts = [1; cumsum(ring_lengths)[1:end-1] .+ 1]
    ring_ranges = [range(a,length=b) for (a,b) in tuple.(ring_starts, ring_lengths)]
    (ring_lengths=ring_lengths, ring_starts=ring_starts, ring_ranges=ring_ranges, cosθ=r[3], sinθ=r[4])
end


function similar(f::MaskedHpxS0Map{Nside,T}) where {Nside,T}
    MaskedHpxS0Map{Nside,T}(Vector{T}(undef,12Nside^2),f.unmasked_rings)
end
    
    
## derivatives
north_neighbor(Nside, i) = i-4Nside
south_neighbor(Nside, i) = i+4Nside
function *(::∇i{0,covariant}, f::MaskedHpxS0Map{Nside,T}) where {Nside,T,covariant}
    f′ = similar(f)
    Δθ = π/4Nside
    fac = covariant ? ringinfo(Nside).sinθ : 1 ./ ringinfo(Nside).sinθ
    for (ring_index,ring_range) in f.unmasked_rings
        @inbounds @simd for i in ring_range
            f′.Tx[i] = fac[ring_index] * (f.Tx[north_neighbor(Nside,i)] - f.Tx[south_neighbor(Nside,i)])/2Δθ
        end
    end
    f′
end
function *(::∇i{1}, f::MaskedHpxS0Map{Nside,T}) where {Nside,T}
    f′ = similar(f)
    Δϕ = 2π/4Nside
    for ring_range in values(f.unmasked_rings)
        @inbounds @simd for i in ring_range
            f′.Tx[i] = (f.Tx[i-1] - f.Tx[i+1])/2Δϕ
        end
    end
    f′
end

## plotting related
spin180(f::MaskedHpxS0Map) = spin180!(similar(f),f)
function spin180!(f′::MaskedHpxS0Map{Nside}, f::MaskedHpxS0Map{Nside}) where {Nside}
    @unpack ring_starts, ring_ranges = ringinfo(Nside)
    for (ring_index, (ring_start, ring_range)) in enumerate(zip(ring_starts, ring_ranges))
        if (ring_index) in keys(f.unmasked_rings)
            f′.Tx[ring_start .+ mod.((0:4Nside-1) .+ 2Nside, 4Nside)] = f.Tx[ring_start:(ring_start+4Nside-1)]
        else
            f′.Tx[ring_range] = NaN
        end
    end
    f′
end

plot(f::MaskedHpxS0Map, args...; kwargs...) = hp.mollview(spin180(f).Tx, args...; kwargs...)

## conversion to flat sky maps
function azeqproj(f::MaskedHpxS0Map{<:Any,T}, θpix, Nside) where {T}
    wasinteractive = pylab.isinteractive()
    pylab.ioff()
    Tx = hp.azeqview(spin180(f).Tx, reso=θpix, xsize=Nside, return_projected_map=true)
    close()
    wasinteractive && pylab.ion()
    FlatS0Map{T,Flat{θpix,Nside,fourier∂}}(Tx)
end


## this will eventually go elsewhere

function load_s4_map(filename, Nside=2048, ::Type{T}=Float64) where {T}
    Tx = convert(Vector{T}, hp.read_map(filename))
    Tx = hp.ud_grade(Tx, Nside)
    Tx = hp.rotator[:Rotator](rot=(0,-44.9))[:rotate_map](Tx)
    Tx = hp.rotator[:Rotator](rot=(180,0))[:rotate_map](Tx)
    MaskedHpxS0Map(Tx)
end
