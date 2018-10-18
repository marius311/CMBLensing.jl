
struct HpxPix{Nside} <: Pix end


using StaticArrays: SArray

struct NeighborCache{Nside, T, QRs}
    neighbor_indices::Matrix{Int}
    qrs::QRs
    
    function NeighborCache{Nside,T}(imax) where {Nside,T}
        neighbor_indices = [(0:(imax-1))'; hp.get_all_neighbours(Nside,collect(0:(imax-1)))::Matrix{Int}] .+ 1
        (θs, ϕs) = convert.(Vector{T}, hp.pix2ang(Nside,collect(0:maximum(neighbor_indices))))::Tuple{Vector{T},Vector{T}}
        qrs = Tuple{SArray{Tuple{9,6},T,2,54},UpperTriangular{T,SArray{Tuple{6,6},T,2,36}}}[]
        @showprogress 1 for (i,(ni,θ,ϕ)) in enumerate(zip(eachcol(neighbor_indices), θs, ϕs))
            (Δθ, Δϕ) = (θs[ni] .- θ, ϕs[ni] .- ϕ)
            P = @. [ones(T,9) Δθ Δϕ Δθ^2 Δϕ^2 Δθ*Δϕ]
            Q,R = qr(SMatrix{9,6}(P))
            push!(qrs,(Q,UpperTriangular(R)))
        end
        new{Nside,T,typeof(qrs)}(neighbor_indices, qrs)
    end
end


struct MaskedHpxS0Map{Nside, T, NC} <: Field{Map, S0, HpxPix{Nside}}
    Tx::Vector{T}
    neighbor_cache::NC
end

function MaskedHpxS0Map(m::Vector{T}) where {T}
    Nside = hp.npix2nside(length(m))
    imax = maximum(findall(!isnan,m))
    nc = NeighborCache{Nside,T}(imax)
    MaskedHpxS0Map{Nside,T,typeof(nc)}(m[1:maximum(nc.neighbor_indices)], nc)
end

function MaskedHpxS0Map(m::Vector, nc::NC) where {Nside,T,NC<:NeighborCache{Nside,T}}
    @assert Nside == hp.npix2nside(length(m))
    imax = maximum(findall(!isnan,m))
    @assert nc.neighbor_indices[1,end] == imax
    MaskedHpxS0Map{Nside,T,NC}(convert(Vector{T},m[1:maximum(nc.neighbor_indices)]), nc)
end
    
    
ringinfo(Nside) = ringinfo(Val(Nside))
@generated function ringinfo(::Val{Nside}) where {Nside}
    r = hp.ringinfo(Nside, collect(1:4Nside-1))
    ring_lengths = r[2]
    ring_starts = [1; cumsum(ring_lengths)[1:end-1] .+ 1]
    ring_ranges = [range(a,length=b) for (a,b) in tuple.(ring_starts, ring_lengths)]
    (ring_lengths=ring_lengths, ring_starts=ring_starts, ring_ranges=ring_ranges, cosθ=r[3], sinθ=r[4], θ=acos.(r[3]))
end

function similar(f::MaskedHpxS0Map{Nside,T,NC}) where {Nside,T,NC}
    MaskedHpxS0Map{Nside,T,NC}(similar(f.Tx), f.neighbor_cache)
end

## derivatives
function _apply!(∇f::FieldVector, ::∇Op, f::F) where {Nside,T,F<:MaskedHpxS0Map{Nside,T}}
    for i in f.neighbor_cache.neighbor_indices[1,:]
        ni = @view f.neighbor_cache.neighbor_indices[:,i]
        Q,R = f.neighbor_cache.qrs[i]
        ∂θ, ∂ϕ = (R \ (Q' * f.Tx[ni]))[2:3]
        ∇f[1].Tx[i], ∇f[2].Tx[i] = ∂θ, ∂ϕ
    end
end


## plotting related
spin180(f::MaskedHpxS0Map) = spin180!(similar(f),f)
function spin180!(f′::MaskedHpxS0Map{Nside}, f::MaskedHpxS0Map{Nside}) where {Nside}
    @unpack ring_starts, ring_ranges = ringinfo(Nside)
    for (ring_index, (ring_start, ring_range)) in enumerate(zip(ring_starts, ring_ranges))
        if (ring_index) in keys(f.unmasked_rings)
            f′.Tx[ring_start .+ mod.((0:4Nside-1) .+ 2Nside, 4Nside)] = f.Tx[ring_start:(ring_start+4Nside-1)]
        end
    end
    f′
end

function plot(f::MaskedHpxS0Map, args...; plot_type=:mollzoom, cmap="RdBu_r", kwargs...)
    cmap = get_cmap(cmap)
    cmap[:set_bad]("lightgray")
    cmap[:set_under]("w")
    Tx = Vector(spin180(f).Tx)
    Tx[Tx.==0] .= NaN
    getproperty(hp,plot_type)(Tx, args...; cmap=cmap, kwargs...)
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

function sqrt_gⁱⁱ(f::MaskedHpxS0Map{Nside}) where {Nside}
    gθθ = similar(f)
    gϕϕ = similar(f)
    sinθ = ringinfo(Nside).sinθ
    for (ring_index,ring_range) in f.unmasked_rings
        gθθ.Tx[ring_range] = 1/sinθ[ring_index]
        gϕϕ.Tx[ring_range] = 1
    end
    @SMatrix[gϕϕ 0f; 0f gθθ]
end

function remask!(f::MaskedHpxS0Map)
    mask = fill(true, length(f.Tx))
    for ring_range in values(f.unmasked_rings)
        mask[ring_range] .= false
    end
    f.Tx[mask.==true] .= NaN
    f
end

function full(f::MaskedHpxS0Map{Nside,T}) where {Nside,T}
    fTx = fill(NaN,12*Nside^2)
    for (Tx,ring_range) in zip(f.Tx,values(f.unmasked_rings))
        fTx[ring_range] .= Tx
    end
    fTx
end


## this will eventually go elsewhere

function load_s4_map(filename, Nside=2048, ::Type{T}=Float64) where {T}
    Tx = convert(Vector{T}, hp.read_map(filename))
    Tx = hp.ud_grade(Tx, Nside)
    Tx = hp.Rotator((180,45,0),eulertype="ZYX")[:rotate_map](Tx)
    MaskedHpxS0Map(Tx)
end



function Base.Vector(x::AbstractSparseVector{Tv}, undef_val) where Tv
    n = length(x)
    n == 0 && return Vector{Tv}()
    nzind = nonzeroinds(x)
    nzval = nonzeros(x)
    r = fill(Tv(undef_val), n)
    for k in 1:nnz(x)
        i = nzind[k]
        v = nzval[k]
        r[i] = v
    end
    return r
end
