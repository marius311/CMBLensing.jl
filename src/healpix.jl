struct HealpyNotImported end
getproperty(::HealpyNotImported, ::Symbol) = error("Healpy needs to be installed to use this function.")
@init try 
    global hp = pyimport(:healpy)
catch
    global hp = HealpyNotImported()
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
struct GradientCache{Nside, T, Nobs, NB, W}
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
        # technically we dont need the +1 below, but this makes it so the entire
        # very last ring is stored (for uniformity's sake)
        Ntot = maximum(neighbors_mat)+1 
        neighbors = SVector{N_neighbors}.(eachcol(Int32.(neighbors_mat)))

        (θs, ϕs) = convert.(Vector{T}, hp.pix2ang(Nside,collect(0:Ntot))::Tuple{Vector{Float64},Vector{Float64}})
        
        # derivative
        W_covariant, W_contravariant = [], []
        @showprogress 1 "∇  precomputation: " for (i,(ni,θ,ϕ)) in enumerate(zip(neighbors, θs, ϕs))
            if all(ni .<= Nobs)
                Δθ = @.     θs[ni]-θ
                Δϕ = @. rem(ϕs[ni]-ϕ+T(π), T(2π), RoundDown) - T(π)
                if order == Val(1)
                    P = @. [Δθ Δϕ ones(T,N_neighbors)]
                else
                    P = @. [Δθ Δϕ ones(T,N_neighbors) Δθ^2 Δϕ^2 Δθ*Δϕ]
                end
                Q,R = qr(P)
                W = inv(R)[1:2,:]*Q'
            else
                W = zeros(T, 2, N_neighbors)
                neighbors[i] = map(j->(j>Nobs ? i : j), neighbors[i])
            end
            push!(W_covariant,     SMatrix{2,N_neighbors}(W))
            push!(W_contravariant, SMatrix{2,N_neighbors}(W ./ [1, sin(θ)^2]))
        end
        _W_covariant     = collect(typeof(W_covariant[1]),    W_covariant)
        _W_contravariant = collect(typeof(W_contravariant[1]),W_contravariant)
        
        # transpose derivative
        Wᵀ_covariant     = fill(T(0), length(neighbors), 2, N_neighbors)
        Wᵀ_contravariant = fill(T(0), length(neighbors), 2, N_neighbors)
        @showprogress 1 "∇' precomputation: " for (i,Ni) in collect(enumerate(neighbors))
            for (j,Nij) in enumerate(Ni)
                if Nij<=length(neighbors)
                    j′ = first(indexin(i,neighbors[Nij]))
                    if j′ != nothing
                        Wᵀ_covariant[i,:,j]     = _W_covariant[Nij][:,j′]
                        Wᵀ_contravariant[i,:,j] = _W_contravariant[Nij][:,j′]
                    end
                end
            end
        end
        _Wᵀ_covariant     = SMatrix{2,N_neighbors,T}.(@views [Wᵀ_covariant[i,:,:]     for i=1:length(neighbors)])
        _Wᵀ_contravariant = SMatrix{2,N_neighbors,T}.(@views [Wᵀ_contravariant[i,:,:] for i=1:length(neighbors)])
        
        new{Nside,T,Nobs,typeof(neighbors),typeof(_W_covariant)}(neighbors, _W_covariant, _W_contravariant, _Wᵀ_covariant, _Wᵀ_contravariant)
    end
    
end
function GradientCache(Nside, T; θmax, order=1)
    i = hp.ang2pix(Nside, deg2rad(θmax), 0)
    Nobs = ringinfo(Nside).ring_starts[findfirst(j->j>i, ringinfo(Nside).ring_starts)] - 1
    GradientCache{Nside,T}(Nobs, Val(order))
end


# now define the Healpix Fields (called HealpixCap)

struct HpxPix{Nside} <: Pix end
abstract type HealpixCap{Nside,T,Nobs,B,S,P<:HpxPix} <: Field{B,S,P} end


## Spin-0
struct HealpixS0Cap{Nside, T, Nobs, GC<:Union{Nothing,GradientCache{Nside,T,Nobs}}} <: HealpixCap{Nside, T, Nobs, Map, S0, HpxPix{Nside}}
    Ix::Vector{T}
    gradient_cache::GC
end
struct HealpixS2Cap{Nside, T, Nobs, GC<:Union{Nothing,GradientCache{Nside,T,Nobs}}} <: HealpixCap{Nside, T, Nobs, QUMap, S2, HpxPix{Nside}}
    QUx::Matrix{T}
    gradient_cache::GC
end
function (::Type{H})(x::VecOrMat{T}; gradient_order=nothing, Nside=hp.npix2nside(size(x,1))) where {T, H<:HealpixCap}
    Nobs = maximum(findall(!isnan,x[:,1]))
    gc = gradient_order==nothing ? nothing : GradientCache{Nside,T}(Nobs,Val(gradient_order))
    H{Nside,T,Nobs,typeof(gc)}(x, gc)
end


getproperty(f::HealpixS2Cap, ::Val{:Qx}) = f.QUx[:,1]
getproperty(f::HealpixS2Cap, ::Val{:Ux}) = f.QUx[:,2]

# convertable_fields is screwed up for HealpixCap so say by hand here there's
# none (todo: fix this)
convertable_fields(::Type{F}) where {B,S,P<:HpxPix,F<:Field{B,S,P}} = []

similar(f::F) where {F<:HealpixCap} = F(similar(first(fieldvalues(f))), f.gradient_cache)
copy(f::F)    where {F<:HealpixCap} = F(copy(first(fieldvalues(f))),    f.gradient_cache)


## derivatives

DerivBasis(::Type{<:HealpixS0Cap}) = Map
DerivBasis(::Type{<:HealpixS2Cap}) = QUMap

# picking out the right weight matrix for a given ∇ operator
get_W(∇::AbstractArray, gc::GradientCache) = get_W(first(∇), gc)
get_W(::∇i{<:Any,true},           gc) = gc.W_covariant
get_W(::∇i{<:Any,false},          gc) = gc.W_contravariant
get_W(::AdjOp{<:∇i{<:Any,true}},  gc) = gc.Wᵀ_covariant
get_W(::AdjOp{<:∇i{<:Any,false}}, gc) = gc.Wᵀ_contravariant
get_W(::Any, gc::Nothing) = error("Can't take gradients of this field, gradient cache not precomputed.")

function mul!(∇f::FieldVector{F}, ∇Op::∇Op, f::F) where {Nside,T,Nobs,F<:HealpixS0Cap{Nside,T,Nobs}}
    gc = f.gradient_cache
    W = get_W(∇Op, gc)
    @inbounds for i in eachindex(gc.neighbors)
        Ix = @view f.Ix[gc.neighbors[i]]
        ∇f[1].Ix[i], ∇f[2].Ix[i] = W[i] * Ix 
    end
    ∇f
end
function mul!(∇f::FieldVector, ∇Op::Union{∇Op,Adjoint{∇i,∇Op}}, f::HealpixS2Cap)
    gc = f.gradient_cache
    W = get_W(∇op, gc)
    @inbounds for i in eachindex(gc.neighbors)
        Qx = @view f.QUx[gc.neighbors[i],1]
        Ux = @view f.QUx[gc.neighbors[i],2]
        ∇f[1].Qx[i,1], ∇f[2].Qx[i,1] = W[i] * Qx
        ∇f[1].Ux[i,2], ∇f[2].Ux[i,2] = W[i] * Ux
    end
    imax = gc.neighbors[end][1] + 1
    ∇f[1].Qx[imax:end] .= ∇f[2].Qx[imax:end] .= ∇f[1].Ux[imax:end] .= ∇f[2].Ux[imax:end] .= NaN
    ∇f
end
*(∇Op::Union{∇Op,Adjoint{∇i,<:∇Op}}, f::HealpixCap) where {B} =  mul!(allocate_result(∇Op,f),∇Op,f)



*(∇Op::Adjoint{∇i,<:∇Op}, v::FieldVector{<:HealpixS0Cap}) = mul!(similar(v[1]), ∇Op, v)
function mul!(f′::F, ∇Op::Adjoint{∇i,<:∇Op}, v::FieldVector{F}, memf′::F=v[1]) where {Nside,T,Nobs,F<:HealpixS0Cap{Nside,T,Nobs}}
    gc = f′.gradient_cache
    W = get_W(∇Op, gc)
    for i in eachindex(gc.neighbors)
        f′.Ix[i] = -(  (W[i] * @view v[1].Ix[gc.neighbors[i]])[1] 
                     + (W[i] * @view v[2].Ix[gc.neighbors[i]])[2])
    end
    f′.Ix[Nobs+1:end] .= NaN
    f′
end

# individual components of the gradient (which is wasteful), but needed for the
# way `negδvelocityᴴ!` is currently written. todo: remove this and change the
# way that that's written instead.
function mul!(f′::F, ∇Op::AdjOp{<:∇i{component}}, f::F) where {component,Nside,T,Nobs,F<:HealpixS0Cap{Nside,T,Nobs}}
    if f′===f; f = copy(f); end # even more wasteful...
    gc = f′.gradient_cache
    W = get_W(∇Op, gc)
    for i in eachindex(gc.neighbors)
        f′.Ix[i] = -(W[i] * @view(f.Ix[gc.neighbors[i]]))[component+1]
    end
    f′
end



dot(a::H, b::H) where {Nside, H<:HealpixS0Cap{Nside}} = dot(nan2zero.(a.Ix), nan2zero.(b.Ix))  * hp.nside2pixarea(Nside)
dot(a::H, b::H) where {Nside, H<:HealpixS2Cap{Nside}} = dot(nan2zero.(a.QUx),nan2zero.(b.QUx)) * hp.nside2pixarea(Nside)


function plot(f::HealpixS0Cap, args...; cmap="RdBu_r", vlim=nothing, plot_type=(PyPlot.isinteractive() ? :mollzoom : :mollview), kwargs...)
    kwargs = Dict(kwargs...)
    cmap = get_cmap(cmap)
    if vlim!=nothing
        kwargs[:min], kwargs[:max] = -vlim, vlim
    end
    getproperty(hp,plot_type)(full(f), args...; cmap=cmap, kwargs...)
end

function plot(f::HealpixS2Cap, args...; kwargs...)
    plot(HealpixS0Cap(f.QUx[:,1], f.gradient_cache), args...; kwargs...)
    plot(HealpixS0Cap(f.QUx[:,2], f.gradient_cache), args...; kwargs...)
end
    
    
    
## conversion to flat sky maps
function azeqproj(f::HealpixS0Cap{<:Any,T}, θpix, Nside, lamb=false) where {T}
    pylab = pyimport("matplotlib.pylab")
    wasinteractive = pylab.isinteractive()
    try
        pylab.ioff()
        Ix = hp.azeqview(full(f), rot=(0,90), reso=θpix, xsize=Nside, lamb=lamb, return_projected_map=true)
        close()
        FlatS0Map{T,Flat{θpix,Nside,fourier∂}}(Ix)
    finally
        wasinteractive && pylab.ion()
    end
end
function azeqproj(f::HealpixS2Cap, θpix, Nside)
    FlatS2QUMap((azeqproj(HealpixS0Cap(getproperty(f,k), f.gradient_cache), θpix, Nside) for k in (:Qx,:Ux))...)
end
azeqproj(f::HealpixCap{Nside,T,Nobs}) where {Nside,T,Nobs} = azeqproj(f, round(600rad2deg(hp.nside2resol(Nside)))/10, Nside)

## broadcasting
broadcast_data(::Type{F}, f::F) where {F<:HealpixS0Cap} = (f.Ix,)
broadcast_data(::Type{F}, f::F) where {F<:HealpixS2Cap} = (f.QUx,)
metadata(::Type{F}, f::F) where {F<:HealpixCap} = (f.gradient_cache,)
metadata_reduce((m1,)::Tuple{GC}, (m2,)::Tuple{GC}) where {GC<:Union{Nothing,GradientCache}} = (m1,)
metadata_reduce((m1,)::Tuple,     (m2,)::Tuple) = error()

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
 
struct IsotropicHarmonicCov{Nside, T, Nobs, N, GC<:GradientCache{Nside, T, Nobs}} <: LinOp{Basis, Spin, HpxPix}
    Cℓ :: Array{T,N}
    gc :: GC
end
IsotropicHarmonicCov(Cℓ::Array,  gc::GradientCache{Nside,T}) where {Nside,T} = IsotropicHarmonicCov(T.(Cℓ), gc)

*(L::BandPassOp, f::HealpixCap{Nside, T}) where {Nside, T} = IsotropicHarmonicCov(T.(L.Wℓ), f.gradient_cache) * f
function mul!(f′::F, L::IsotropicHarmonicCov{Nside, T, Nobs}, f::F) where {Nside, T, Nobs, F<:HealpixCap{Nside, T, Nobs}}
    ℓmax = min(size(L.Cℓ,1)-1, 3Nside)
    aℓms = map2alm(f, ℓmax=ℓmax)
    @. aℓms *= L.Cℓ[1:ℓmax+1,:]'
    alm2map!(f′, aℓms)
end
function ldiv!(f′::F, L::IsotropicHarmonicCov{Nside, T, Nobs}, f::F) where {Nside, T, Nobs, F<:HealpixCap{Nside, T, Nobs}}
    ℓmax = min(size(L.Cℓ,1)-1, 3Nside)
    aℓms = map2alm(f, ℓmax=ℓmax)
    @. aℓms = nan2zero(L.Cℓ[1:ℓmax+1,:]' \ aℓms)
    alm2map!(f′, aℓms)
end

# one ring before the ring on which the first unobserved pixel is
get_zbounds(Nside, Nobs) = [ringinfo(Nside).cosθ[hp.pix2ring(Nside,[Nobs])[1]], 1] 

function map2alm(f::HealpixCap{Nside,T,Nobs}; ℓmax=2Nside) where {Nside,T,Nobs}
    zbounds = get_zbounds(Nside, Nobs)
    map2alm(first(fieldvalues(f)), Nside=Nside, ℓmax=ℓmax, zbounds=zbounds)
end

function alm2map!(f::HealpixCap{Nside,T,Nobs}, aℓms::Array{Complex{T},3}) where {Nside,T,Nobs}
    zbounds = get_zbounds(Nside, Nobs)
    alm2map!(first(fieldvalues(f)), aℓms, Nside=Nside, zbounds=zbounds)
    first(fieldvalues(f))[Nobs+1:end,:] .= NaN
    f
end


# todo: implement proper broadcasting
+(Σa::I, Σb::I) where {I<:IsotropicHarmonicCov} = IsotropicHarmonicCov(Σa.Cℓ.+Σb.Cℓ, Σa.gc)
*(Σa::I, Σb::I) where {I<:IsotropicHarmonicCov} = IsotropicHarmonicCov(Σa.Cℓ.*Σb.Cℓ, Σa.gc)
*(Σ::IsotropicHarmonicCov, α::Real) = IsotropicHarmonicCov(α*Σ.Cℓ, Σ.gc)
*(α::Real, Σ::IsotropicHarmonicCov) = IsotropicHarmonicCov(α*Σ.Cℓ, Σ.gc)
inv(Σ::IsotropicHarmonicCov) = IsotropicHarmonicCov(nan2zero.(inv.(Σ.Cℓ)), Σ.gc)
sqrt(Σ::IsotropicHarmonicCov) = IsotropicHarmonicCov(sqrt.(Σ.Cℓ), Σ.gc)
simulate(Σ::IsotropicHarmonicCov{Nside,T,Nobs,1}) where {Nside,T,Nobs} = 
    sqrt(Σ) * HealpixS0Cap(randn(T,Nobs)/T(hp.nside2resol(Nside)), Σ.gc)
simulate(Σ::IsotropicHarmonicCov{Nside,T,Nobs,2}) where {Nside,T,Nobs} = 
    sqrt(Σ) * HealpixS2Cap(randn(T,Nobs,2)/T(hp.nside2resol(Nside)), Σ.gc)
zero(Σ::IsotropicHarmonicCov{Nside,T,Nobs}) where {Nside,T,Nobs} = 
    HealpixS0Cap(zeros(T,Nobs), Σ.gc)


##
function HealpixCapMask(θmax, Δθapod, Nside)
    W(θ) = θ<θmax ? 1 : θmax<θ<(θmax+Δθapod) ? (cos(π*(θ-θmax)/Δθapod)+1)/2 : 0
    i,l = hp.ringinfo(Nside, collect(1:4Nside))[1:2]
    m = zeros(12*Nside^2)
    for (r,w) in zip(broadcast(:, i, i .+ l .- 1), W.(rad2deg.(acos.(hp.ringinfo(Nside, collect(1:4nside))[3]))))
        m[r.+1] .= w
    end
    m
end

##

# Healpy doesn't yet have the zbounds option, so call into the Healpix Fortran
# libraries directly. 

# needing to manually dlopen this is probably a bug in Healpix or gfortan, not
# sure which...
using Libdl
Libdl.dlopen("libgomp",Libdl.RTLD_GLOBAL)

@generated function map2alm(maps::Array{T,N}; Nside=hp.npix2nside(size(maps,1)), ℓmax=2Nside, mmax=ℓmax, zbounds=[-1,1]) where {N,T<:Union{Float32,Float64}}
    (spin, Tspin) = if (N==1)
        (), () 
    elseif (N==2)
        (2,), (Ref{Int32},)
    else
        error("maps should be Npix-×-1 or Npix-×-2")
    end
    fn_name = "__alm_tools_MOD_map2alm_$(N==1 ? "sc" : "spin")_$((T==Float32) ? "s" : "d")"
    quote
        aℓms = Array{Complex{T}}(undef, ($N,ℓmax+1,mmax+1))
        aℓms .= NaN
        ccall(
            ($fn_name, "libhealpix"), Nothing,
            (Ref{Int32}, Ref{Int32}, Ref{Int32}, $(Tspin...), Ref{T}, Ref{Complex{T}}, Ref{Float64}, Ref{Nothing}),
            Nside, ℓmax, mmax, $(spin...), maps, aℓms, Float64.(zbounds), C_NULL
        )
        aℓms
    end
end

@generated function alm2map!(maps::Array{T,N}, aℓms::Array{Complex{T},3}; Nside=hp.npix2nside(size(maps,1)), ℓmax=(size(aℓms,2)-1), mmax=(size(aℓms,3)-1), zbounds=[-1,1]) where {N,T<:Union{Float32,Float64}}
    (spin, Tspin) = if (N==1)
        (), () 
    elseif (N==2)
        (2,), (Ref{Int32},)
    else
        error("maps should be Npix-×-1 or Npix-×-2")
    end
    fn_name = "__alm_tools_MOD_alm2map_$(N==1 ? "sc_wrapper" : "spin")_$((T==Float32) ? "s" : "d")"
    quote
        ccall(
           ($fn_name, "libhealpix"), Nothing,
           (Ref{Int32}, Ref{Int32}, Ref{Int32}, $(Tspin...), Ref{Complex{T}}, Ref{T}, Ref{Float64}, Ref{Nothing}),
           Nside, ℓmax, mmax, $(spin...), aℓms, maps, Float64.(zbounds), C_NULL
        )
        maps
    end
end

function alm2cl(alm::Matrix{Complex{T}}; ℓmax=(size(alm,1)-1)) where {T}
    InterpolatedCℓs(0:ℓmax, [(abs2(alm[ℓ+1,1]) + 2sum(abs2.(alm[ℓ+1, 2:ℓ+1])))/(2ℓ+1) for ℓ=0:ℓmax])
end

function alm2cl(alm1::Matrix{Complex{T}}, alm2::Matrix{Complex{T}}; ℓmax=(size(alm,1)-1)) where {T}
    InterpolatedCℓs(0:ℓmax, real.([(dot(alm1[ℓ+1,1], alm2[ℓ+1,1]) + 2dot(alm1[ℓ+1,2:ℓ+1], alm2[ℓ+1,2:ℓ+1]))/(2ℓ+1) for ℓ=0:ℓmax]))
end

get_Cℓ(mp::HealpixS0Cap{Nside}; ℓmax=2Nside) where {Nside} = alm2cl(map2alm(mp, ℓmax=ℓmax))
get_Cℓ(mp1::HealpixS0Cap{Nside}, mp2::HealpixS0Cap{Nside}; ℓmax=2Nside) where {Nside} = alm2cl(map2alm(mp1, ℓmax=ℓmax), map2alm(mp2, ℓmax=ℓmax), ℓmax=ℓmax)
