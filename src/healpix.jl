
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
            @assert length(ring_range)==4Nside "MaskedHpxS0Map does not support non-masked pixels in the polar cap regions"
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
north_neighbor(Nside, i) = i-8Nside
south_neighbor(Nside, i) = i+8Nside
function *(::∇i{0,covariant}, f::MaskedHpxS0Map{Nside,T}) where {Nside,T,covariant}
    f′ = similar(f)
    Δθ = π/4Nside
    fac = covariant ? ringinfo(Nside).sinθ : 1 ./ ringinfo(Nside).sinθ
    for (ring_index,ring_range) in f.unmasked_rings
        @inbounds @simd for i in ring_range
            if north_neighbor(Nside,i) in get(f.unmasked_rings,ring_index-2,()) && south_neighbor(Nside,i) in get(f.unmasked_rings,ring_index+2,())
                f′.Tx[i] = fac[ring_index] * (f.Tx[north_neighbor(Nside,i)] - f.Tx[south_neighbor(Nside,i)])/2Δθ
            else
                f′.Tx[i] = 0
            end
        end
    end
    f′
end
function *(::AdjOp{∇i{0,covariant}}, f::MaskedHpxS0Map{Nside,T}) where {Nside,T,covariant}
    f′ = similar(f)
    Δθ = π/4Nside
    fac = covariant ? ringinfo(Nside).sinθ : 1 ./ ringinfo(Nside).sinθ
    for (ring_index,ring_range) in f.unmasked_rings
        @inbounds @simd for i in ring_range
            if north_neighbor(Nside,i) in get(f.unmasked_rings,ring_index-2,()) && south_neighbor(Nside,i) in get(f.unmasked_rings,ring_index+2,())
                f′.Tx[i] = (f.Tx[north_neighbor(Nside,i)]*fac[ring_index-1] - f.Tx[south_neighbor(Nside,i)]*fac[ring_index+1])/2Δθ
            else
                f′.Tx[i] = 0
            end
        end
    end
    f′
end
function *(::∇i{1}, f::MaskedHpxS0Map{Nside,T}) where {Nside,T}
    f′ = similar(f)
    Δϕ = 2π/4Nside
    for ring_range in values(f.unmasked_rings)
        @inbounds @simd for i in (ring_range.start+1):(ring_range.stop-1)
            f′.Tx[i] = (f.Tx[i-1] - f.Tx[i+1])/2Δϕ
        end
        f′.Tx[ring_range.start] = f′.Tx[ring_range.stop] = 0
    end
    f′
end
*(L::AdjOp{<:∇i{1}}, f::MaskedHpxS0Map) = -L'*f



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

function plot(f::MaskedHpxS0Map, args...; plot_type=:mollzoom, cmap="RdBu_r", kwargs...)
    cmap = get_cmap(cmap)
    cmap[:set_bad]("lightgray")
    cmap[:set_under]("w")
    getproperty(hp,plot_type)(spin180(f).Tx, args...; cmap=cmap, kwargs...)
end

## conversion to flat sky maps
function azeqproj(f::MaskedHpxS0Map{<:Any,T}, θpix, Nside) where {T}
    wasinteractive = pylab.isinteractive()
    pylab.ioff()
    Tx = hp.azeqview(spin180(f).Tx, reso=θpix, xsize=Nside, return_projected_map=true)
    close()
    wasinteractive && pylab.ion()
    FlatS0Map{T,Flat{θpix,Nside,fourier∂}}(Tx)
end

## broadcasting
broadcast_data(::Type{F}, f::F) where {F<:MaskedHpxS0Map} = (f.Tx,)
metadata(::Type{F}, f::F) where {F<:MaskedHpxS0Map} = (f.unmasked_rings,)
metadata_reduce((m1,)::Tuple, (m2,)::Tuple) = (@assert(m1==m2); (m1,))


LenseBasis(::Type{<:MaskedHpxS0Map}) = Map


function remask!(f::MaskedHpxS0Map)
    mask = fill(true, length(f.Tx))
    for ring_range in values(f.unmasked_rings)
        mask[ring_range] = false
    end
    f.Tx[mask.==true] = NaN
    f
end


## this will eventually go elsewhere

function load_s4_map(filename, Nside=2048, ::Type{T}=Float64) where {T}
    Tx = convert(Vector{T}, hp.read_map(filename))
    Tx = hp.ud_grade(Tx, Nside)
    Tx = hp.rotator[:Rotator](rot=(0,-44.9))[:rotate_map](Tx)
    Tx = hp.rotator[:Rotator](rot=(180,0))[:rotate_map](Tx)
    MaskedHpxS0Map(Tx)
end
