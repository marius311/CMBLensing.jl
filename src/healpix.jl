struct HealpyNotImported end
getproperty(::HealpyNotImported, ::Symbol) = error("Healpy needs to be installed to use this function.")
@init try 
    @eval @pyimport healpy as hp
catch
    @eval hp = HealpyNotImported()
end


# use generated function to memoize ring info
ringinfo(Nside) = ringinfo(Val(Nside))
@generated function ringinfo(::Val{Nside}) where {Nside}
    r = hp.ringinfo(Nside, collect(1:4Nside-1))
    ring_lengths = r[2]
    ring_starts = [1; cumsum(ring_lengths)[1:end-1] .+ 1]
    ring_ranges = [range(a,length=b) for (a,b) in tuple.(ring_starts, ring_lengths)]
    (ring_lengths=ring_lengths, ring_starts=ring_starts, ring_ranges=ring_ranges, cosθ=r[3], sinθ=r[4], θ=acos.(r[3]))
end

# GradientCache holds precomputed quantities that let us do fast spherical gradients 
struct GradientCache{Nside, T, Nobs, Ntot, NB, W}
    neighbors        :: NB
    W_covariant      :: W
    W_contravariant  :: W
    Wᵀ_covariant     :: W
    Wᵀ_contravariant :: W
    
    function GradientCache{Nside,T}(Nobs, order=Val(1)) where {Nside,T}
        N_coeffs    = (order == Val(1)) ? 3 : 6
        N_neighbors = (order == Val(1)) ? 4 : 8
        
        # the commented out line uses the pixel itself, but that gives us a non-zero trace so we don't use it
        # neighbors_mat = [(0:(Nobs-1))'; hp.get_all_neighbours(Nside,collect(0:(Nobs-1)))[(order == Val(1) ? (1:2:end) : 1:end),:]::Matrix{Int}] .+ 1
        neighbors_mat = hp.get_all_neighbours(Nside,collect(0:(Nobs-1)))[(order == Val(1) ? (2:2:end) : 1:end),:]::Matrix{Int} .+ 1
        Ntot = maximum(neighbors_mat)
        neighbors = SVector{N_neighbors}.(eachcol(Int32.(neighbors_mat)))

        (θs, ϕs) = convert.(Vector{T}, hp.pix2ang(Nside,collect(0:Ntot))::Tuple{Vector{Float64},Vector{Float64}})
        
        # derivative
        W_covariant, W_contravariant = [], []
        @showprogress 1 "∇  precomputation: " for (i,(ni,θ,ϕ)) in enumerate(zip(neighbors, θs, ϕs))
            Δθ = @.     θs[ni]-θ
            Δϕ = @. rem(ϕs[ni]-ϕ+T(π), T(2π), RoundDown) - T(π)
            if order == Val(1)
                P = @. [Δθ Δϕ ones(T,N_neighbors)]
            else
                P = @. [Δθ Δϕ ones(T,N_neighbors) Δθ^2 Δϕ^2 Δθ*Δϕ]
            end
            Q,R = qr(P)
            W = inv(R)[1:2,:]*Q'
            push!(W_covariant,     SMatrix{2,N_neighbors}(W))
            push!(W_contravariant, SMatrix{2,N_neighbors}(W ./ [1, sin(θ)^2]))
        end
        _W_covariant     = collect(typeof(W_covariant[1]),    W_covariant)
        _W_contravariant = collect(typeof(W_contravariant[1]),W_contravariant)
        
        # transpose derivative
        Wᵀ_covariant     = fill(NaN, length(neighbors), 2, N_neighbors)
        Wᵀ_contravariant = fill(NaN, length(neighbors), 2, N_neighbors)
        @showprogress 1 "∇' precomputation: " for (i,Ni) in collect(enumerate(neighbors))
            for (j,Nij) in enumerate(Ni)
                if Nij<=length(neighbors)
                    j′ = first(indexin(i,neighbors[Nij]))
                    Wᵀ_covariant[i,:,j]     = _W_covariant[Nij][:,j′]
                    Wᵀ_contravariant[i,:,j] = _W_contravariant[Nij][:,j′]
                end
            end
        end
        _Wᵀ_covariant     = SMatrix{2,N_neighbors,T}.(@views [Wᵀ_covariant[i,:,:]     for i=1:length(neighbors)])
        _Wᵀ_contravariant = SMatrix{2,N_neighbors,T}.(@views [Wᵀ_contravariant[i,:,:] for i=1:length(neighbors)])
        
        new{Nside,T,Nobs,Ntot,typeof(neighbors),typeof(_W_covariant)}(neighbors, _W_covariant, _W_contravariant, _Wᵀ_covariant, _Wᵀ_contravariant)
    end
    
end

# now define the Healpix Fields (called HealpixCap)

struct HpxPix{Nside} <: Pix end
abstract type HealpixCap{B,S,P<:HpxPix} <: Field{B,S,P} end


## Spin-0
struct HealpixS0Cap{Nside, T, Nobs, GC<:GradientCache{Nside, T, Nobs}} <: HealpixCap{Map, S0, HpxPix{Nside}}
    Ix::Vector{T}
    gradient_cache::GC
    function HealpixS0Cap(Ix::Vector, gc::GradientCache{Nside,T,Nobs,Ntot}) where {T,Nside,Nobs,Ntot}
        Ix = length(Ix)==Ntot ? Ix : Ix[1:Ntot]
        HealpixS0Cap{Nside,T,Nobs,typeof(gc)}(Ix, gc)
    end
    HealpixS0Cap{Nside,T,Nobs,GC}(Ix, gc) where {Nside,T,Nobs,GC} = new{Nside,T,Nobs,GC}(Ix,gc)
end
function HealpixS0Cap(Ix::Vector{T}; gradient_order=1) where {T}
    Nside = hp.npix2nside(length(Ix))
    Nobs = maximum(findall(!isnan,Ix))
    HealpixS0Cap(Ix, GradientCache{Nside,T}(Nobs,Val(gradient_order)))
end


## Spin-2
struct HealpixS2Cap{Nside, T, Nobs, GC<:GradientCache{Nside, T, Nobs}} <: HealpixCap{QUMap, S2, HpxPix{Nside}}
    Qx::Vector{T}
    Ux::Vector{T}
    gradient_cache::GC
end
function HealpixS2Cap(Qx::Vector{T}, Ux::Vector{T}) where {T}
    @assert length(Qx)==length(Ux)
    Nside = hp.npix2nside(length(Qx))
    Nobs = maximum(findall(!isnan,Qx))
    HealpixS2Cap(Qx, Ux, GradientCache{Nside,T}(Nobs))
end
function HealpixS2Cap(Qx::Vector, Ux::Vector, gc::GradientCache{Nside,T,Nobs,Ntot}) where {T,Nside,Nobs,Ntot}
    Qx = length(Qx)==Ntot ? Qx : Qx[1:Ntot]
    Ux = length(Ux)==Ntot ? Ux : Ux[1:Ntot]
    HealpixS2Cap{Nside,T,Nobs,typeof(gc)}(Qx, Ux, gc)
end



similar(f::F) where {F<:HealpixS0Cap} = F(similar(f.Ix), f.gradient_cache)
similar(f::F) where {F<:HealpixS2Cap} = F(similar(f.Qx), similar(f.Ux), f.gradient_cache)
copy(f::F)    where {F<:HealpixS0Cap} = F(copy(f.Ix),    f.gradient_cache)
copy(f::F)    where {F<:HealpixS2Cap} = F(copy(f.Ix),    copy(f.Ux),    f.gradient_cache)

## derivatives
function get_W(∇Op::Union{∇Op{covariant},Adjoint{∇i,∇Op{covariant}}}, gc) where {covariant}
    if ∇Op isa Adjoint
        W = covariant ? gc.Wᵀ_covariant : gc.Wᵀ_contravariant
    else
        W = covariant ? gc.W_covariant : gc.W_contravariant
    end
end

# function mul!(∇f::FieldVector{F}, ∇Op::Union{∇Op,Adjoint{∇i,∇Op}}, f::F) where {F<:HealpixS0Cap}
function mul!(∇f::FieldVector{F}, ∇Op::∇Op, f::F) where {F<:HealpixS0Cap}
    gc = f.gradient_cache
    W = get_W(∇Op, gc)
    @inbounds for i in eachindex(gc.neighbors)
        Ix = @view f.Ix[gc.neighbors[i]]
        #todo: replace this nan2zero with just not calculating the last ring
        ∇f[1].Ix[i], ∇f[2].Ix[i] = nan2zero.(W[i] * Ix) 
    end
    imax = gc.neighbors[end][1] + 2
    ∇f[1].Ix[imax:end] .= ∇f[2].Ix[imax:end] .= NaN
    ∇f
end
function mul!(∇f::FieldVector, ∇Op::Union{∇Op,Adjoint{∇i,∇Op}}, f::HealpixS2Cap)
    gc = f.gradient_cache
    W = get_W(∇op, gc)
    @inbounds for i in eachindex(gc.neighbors)
        Qx = @view f.Qx[gc.neighbors[i]]
        Ux = @view f.Ux[gc.neighbors[i]]
        ∇f[1].Qx[i], ∇f[2].Qx[i] = W[i] * Qx
        ∇f[1].Ux[i], ∇f[2].Ux[i] = W[i] * Ux
    end
    imax = gc.neighbors[end][1] + 1
    ∇f[1].Qx[imax:end] .= ∇f[2].Qx[imax:end] .= ∇f[1].Ux[imax:end] .= ∇f[2].Ux[imax:end] .= NaN
    ∇f
end
*(∇Op::Union{∇Op,Adjoint{∇i,<:∇Op}}, f::HealpixS0Cap) where {B} =  mul!(allocate_result(∇Op,f),∇Op,f)
DerivBasis(::Type{<:HealpixS0Cap}) = Map


*(∇Op::Adjoint{∇i,<:∇Op}, v::FieldVector{<:HealpixS0Cap}) = mul!(similar(v[1]), ∇Op, v)
function mul!(f′::F, ∇Op::Adjoint{∇i,<:∇Op}, v::FieldVector{F}, memf′::F=v[1]) where {F<:HealpixCap}
    gc = f′.gradient_cache
    W = get_W(∇Op, gc)
    @inbounds for i in eachindex(gc.neighbors)
        f′.Ix[i] = -nan2zero(  (W[i] * @view v[1].Ix[gc.neighbors[i]])[1] 
                             + (W[i] * @view v[2].Ix[gc.neighbors[i]])[2])
    end
    imax = gc.neighbors[end][1] + 2
    f′.Ix[imax:end] 
    f′
end


dot(a::HealpixS0Cap, b::HealpixS0Cap) = dot(nan2zero.(a.Ix),nan2zero.(b.Ix))


function plot(f::HealpixS0Cap, args...; cmap="RdBu_r", vlim=nothing, plot_type=(PyPlot.isinteractive() ? :mollzoom : :mollview), kwargs...)
    kwargs = Dict(kwargs...)
    cmap = get_cmap(cmap)
    if vlim!=nothing
        kwargs[:min], kwargs[:max] = -vlim, vlim
    end
    getproperty(hp,plot_type)(full(f), args...; cmap=cmap, @show(kwargs)...)
end

function plot(f::HealpixS2Cap, args...; kwargs...)
    plot(HealpixS0Cap(f.Qx, f.gradient_cache), args...; kwargs...)
    plot(HealpixS0Cap(f.Ux, f.gradient_cache), args...; kwargs...)
end
    
    
    
## conversion to flat sky maps
function azeqproj(f::HealpixS0Cap{<:Any,T}, θpix, Nside) where {T}
    wasinteractive = PyPlot.matplotlib.pylab.isinteractive()
    try
        PyPlot.matplotlib.pylab.ioff()
        Ix = hp.azeqview(full(f), rot=(0,90), reso=θpix, xsize=Nside, return_projected_map=true)
        close()
        FlatS0Map{T,Flat{θpix,Nside,fourier∂}}(Ix)
    finally
        wasinteractive && PyPlot.matplotlib.pylab.ion()
    end
end

## broadcasting
broadcast_data(::Type{F}, f::F) where {F<:HealpixS0Cap} = (f.Ix,)
broadcast_data(::Type{F}, f::F) where {F<:HealpixS2Cap} = (f.Qx, f.Ux)
metadata(::Type{F}, f::F) where {F<:HealpixCap} = (f.gradient_cache,)
metadata_reduce((m1,)::Tuple{GC}, (m2,)::Tuple{GC}) where {GC<:GradientCache} = (m1,)
metadata_reduce((m1,)::Tuple{GradientCache}, (m2,)::Tuple{GradientCache}) = error()

LenseBasis(::Type{<:HealpixS0Cap}) = Map
LenseBasis(::Type{<:HealpixS2Cap}) = QUMap

adjoint(f::HealpixS0Cap) = f
length(::Type{<:HealpixS0Cap{Nside,T,Nobs}}) where {Nside,T,Nobs} = Nobs

function full(f::HealpixS0Cap{Nside,T}) where {Nside,T}
    Ix = fill(T(NaN),12*Nside^2)
    Ix[1:length(f.Ix)] .= f.Ix
    Ix
end


## Harmonic Operators
 
struct IsotropicHarmonicCov{Nside, T, Nobs, Ntot, GC<:GradientCache{Nside, T, Nobs, Ntot}} <: LinOp{Map, S0, HpxPix}
    Cℓ :: Vector{T}
    gc :: GC
end

*(L::BandPassOp, f::HealpixS0Cap{Nside, T}) where {Nside, T} = 
    IsotropicHarmonicCov(T.(L.Wℓ), f.gradient_cache) * f

function mul!(f′::F, L::IsotropicHarmonicCov{Nside, T, Nobs}, f::F) where {Nside, T, Nobs, F<:HealpixS0Cap{Nside, T, Nobs}}
    zbounds = [cos(hp.pix2ang(Nside,Nobs)[1]), 1]
    ℓmax = length(L.Cℓ)-1
    alms = reshape(map2alm(f.Ix, Nside=Nside, ℓmax=ℓmax, zbounds=zbounds), ℓmax+1, ℓmax+1)
    alms .*= L.Cℓ
    alm2map!(f′.Ix, alms, Nside=Nside, zbounds=zbounds)
    f′.Ix[Nobs+1:end] .= 0
    f′
end

function ldiv!(f′::F, L::IsotropicHarmonicCov{Nside, T, Nobs}, f::F) where {Nside, T, Nobs, F<:HealpixS0Cap{Nside, T, Nobs}}
    zbounds = [cos(hp.pix2ang(Nside,Nobs)[1]), 1]
    ℓmax = length(L.Cℓ)-1
    alms = reshape(map2alm(f.Ix, Nside=Nside, ℓmax=ℓmax, zbounds=zbounds), ℓmax+1, ℓmax+1)
    @. alms = nan2zero(L.Cℓ \ alms)
    alm2map!(f′.Ix, alms, Nside=Nside, zbounds=zbounds)
    f′.Ix[Nobs+1:end] .= 0
    f′
end

# todo: implement proper broadcasting
+(Σa::I, Σb::I) where {I<:IsotropicHarmonicCov} = IsotropicHarmonicCov(Σa.Cℓ+Σb.Cℓ, Σa.gc)
*(Σa::I, Σb::I) where {I<:IsotropicHarmonicCov} = IsotropicHarmonicCov(Σa.Cℓ.*Σb.Cℓ, Σa.gc)
*(Σ::IsotropicHarmonicCov, α::Real) = IsotropicHarmonicCov(α*Σ.Cℓ, Σ.gc)
*(α::Real, Σ::IsotropicHarmonicCov) = IsotropicHarmonicCov(α*Σ.Cℓ, Σ.gc)
inv(Σ::IsotropicHarmonicCov) = IsotropicHarmonicCov(nan2zero.(inv.(Σ.Cℓ)), Σ.gc)
sqrt(Σ::IsotropicHarmonicCov) = IsotropicHarmonicCov(sqrt.(Σ.Cℓ), Σ.gc)
simulate(Σ::IsotropicHarmonicCov{Nside,T,Nobs,Ntot}) where {Nside,T,Nobs,Ntot} = 
    sqrt(Σ) * HealpixS0Cap(randn(T,Ntot)/hp.nside2resol(Nside), Σ.gc)


## this will eventually go elsewhere

function load_s4_map(filename; Nside=2048, T=Float32, which::_PolType = PolType.T)
    m = hp.read_map(filename, verbose=false, field=Dict(PolType.T=>0, PolType.QU=>[1,2], PolType.TQU=>[0,1,2])[which])
    m = hp.ud_grade(m, Nside)
    m = hcat(hp.Rotator((0,-135,0),eulertype="ZYX")[:rotate_map_pixel](m)...)
    m = T.(m)
    m[@. abs(m)>1e20] .= NaN
    if which == PolType.T
        HealpixS0Cap(m[1,:])
    elseif which == PolType.QU
        HealpixS2Cap(m[:,1], m[:,2])
    end
end


##

# Healpy doesn't yet have the zbounds option, so call into the Healpix Fortran
# libraries directly. 

# needing to manually dlopen this is probably a bug in Healpix or gfortan, not
# sure which...
using Libdl
Libdl.dlopen("libgomp",Libdl.RTLD_GLOBAL)

@generated function map2alm(mp::Vector{T}; Nside = hp.npix2nside(length(mp)), ℓmax=nothing, mmax=ℓmax, zbounds=[-1,1]) where {T<:Union{Float32,Float64}}
    fn_name = "__alm_tools_MOD_map2alm_sc_$((T==Float32) ? "s" : "d")"
    quote
        if ℓmax==nothing; ℓmax=3Nside-1; end
        alm = Array{Complex{T}}(undef, (1,ℓmax+1,mmax+1))
        ccall(  
           ($fn_name, "libhealpix"), Nothing,
           (Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{T}, Ref{Complex{T}}, Ref{Float64}, Ref{Nothing}),
           Nside, ℓmax, mmax, mp, alm, Float64.(zbounds), C_NULL
        )
        alm
    end
end

@generated function alm2map!(mp::Vector{T}, alm::Matrix{Complex{T}}; Nside, ℓmax=(size(alm,1)-1), mmax=(size(alm,2)-1), zbounds=[-1,1]) where {T<:Union{Float32,Float64}}
    fn_name = "__alm_tools_MOD_alm2map_sc_wrapper_$((T==Float32) ? "s" : "d")"
    quote
        ccall(
           ($fn_name, "libhealpix"), Nothing,
           (Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Complex{T}}, Ref{T}, Ref{Float64}, Ref{Nothing}),
           Nside, ℓmax, mmax, alm, mp, Float64.(zbounds), C_NULL
        )
        mp
    end
end

function alm2cl(alm::Matrix{Complex{T}}; ℓmax=(size(alm,1)-1)) where {T}
    InterpolatedCℓs(0:ℓmax, [(abs2(alm[ℓ+1,1]) + 2sum(abs2.(alm[ℓ+1, 2:ℓ+1])))/(2ℓ+1) for ℓ=0:ℓmax])
end

get_Cℓ(mp::HealpixS0Cap{Nside}; ℓmax=2Nside) where {Nside} = 
    alm2cl(map2alm(nan2zero.(full(mp)); ℓmax=ℓmax)[1,:,:])
