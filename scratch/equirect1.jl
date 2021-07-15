
# Modules
# ==============================
using Tullio 
using PyPlot
using BenchmarkTools
using ProgressMeter
using LinearAlgebra
using FFTW
FFTW.set_num_threads(Threads.nthreads())

using CMBLensing
import CMBLensing: Map, AzFourier, QUAzFourier, QUMap
using CirculantCov: βcovSpin2, βcovSpin0, geoβ, 
multPP̄, multPP, periodize, Jperm # https://github.com/EthanAnderes/CirculantCov.jl

# LATER: remove LBblock dependence
using LBblocks: @sblock # https://github.com/EthanAnderes/LBblocks.jl

hide_plots = true



# Methods ...
# =======================================


function AzFourier(f::EquiRectMap)
    nφ = f.metadata.Nx
    EquiRectAzFourier(rfft(f.arr, 2) ./ √nφ, f.metadata)
end

function Map(f::EquiRectAzFourier)
    nφ = f.metadata.Nx
    EquiRectMap(irfft(f.arr, nφ, 2) .* √nφ, f.metadata)
end

function QUAzFourier(f::EquiRectQUMap)
    nθ = f.metadata.Ny
    nφ = f.metadata.Nx
    Uf = fft(f.arr, 2) ./ √nφ
    f▫ = similar(Uf, 2nθ, nφ÷2+1)
    for ℓ = 1:nφ÷2+1
        if (ℓ==1) | ((ℓ==nφ÷2+1) & iseven(nφ))
            f▫[1:nθ, ℓ]     .= Uf[:,ℓ]
            f▫[nθ+1:2nθ, ℓ] .= conj.(Uf[:,ℓ])
        else 
            f▫[1:nθ, ℓ]     .= Uf[:,ℓ]
            f▫[nθ+1:2nθ, ℓ] .= conj.(Uf[:,Jperm(ℓ,nφ)])
        end
    end
    EquiRectQUAzFourier(f▫, f.metadata)
end

function QUMap(f::EquiRectQUAzFourier)
    nθₓ2, nφ½₊1 = size(f.arr)
    nθ = f.metadata.Ny
    nφ = f.metadata.Nx
    @assert nφ½₊1 == nφ÷2+1
    @assert 2nθ   == nθₓ2

    pθk = similar(f.arr, nθ, nφ)
    for ℓ = 1:nφ½₊1
        if (ℓ==1) | ((ℓ==nφ½₊1) & iseven(nφ))
            pθk[:,ℓ] .= f.arr[1:nθ,ℓ] 
        else 
            pθk[:,ℓ]  .= f.arr[1:nθ,ℓ]     
            pθk[:,Jperm(ℓ,nφ)] .= conj.(f.arr[nθ+1:2nθ,ℓ])
        end
    end 
    EquiRectQUMap(ifft(pθk, 2) .* √nφ, f.metadata)
end

# Quick test

pj = CMBLensing.ProjEquiRect(;
    Ny=200, # nθ
    Nx=768, # nφ
    θspan  = (2.7, 2.9), 
    φspan  = (0.0, 2π/4), 
)

ϕ = EquiRectMap(randn(Float64, pj.Ny, pj.Nx), pj)
P = EquiRectQUMap(randn(ComplexF64, pj.Ny, pj.Nx), pj)

ϕ′ = AzFourier(ϕ)    
P′ = QUAzFourier(P)   # the printing of the size is wrong ...

# How do I get this to work?

2 * ϕ + ϕ′
2 * P + P′





# Spectral densities
# ==============================

ℓ, CEEℓ, CBBℓ, CΦΦℓ = @sblock let ℓmax = 11000
	ℓ    = 0:ℓmax
	Cℓ   = camb(;r=0.01, ℓmax);
	CBBℓ = Cℓ.tensor.BB(ℓ)
	CEEℓ = Cℓ.unlensed_scalar.EE(ℓ)
	CΦΦℓ = Cℓ.unlensed_scalar.ϕϕ(ℓ)
	for cl in (CEEℓ, CBBℓ, CΦΦℓ)
		cl[.!isfinite.(cl)] .= 0
	end

	return ℓ, CEEℓ, CBBℓ, CΦΦℓ
end

@sblock let hide_plots, ℓ, CEEℓ, CBBℓ, CΦΦℓ
	hide_plots && return 
	fig, ax = subplots(2)
	ax[1].plot(ℓ, @. ℓ^2*CEEℓ)
	ax[1].plot(ℓ, @. ℓ^2*CBBℓ)
	ax[2].plot(ℓ, @. ℓ^4*CΦΦℓ)
	ax[1].set_xscale("log")
	ax[2].set_xscale("log")
	ax[1].set_yscale("log")
	ax[2].set_yscale("log")
	return nothing
end

# Pixel grid
# ==============================


# pj = CMBLensing.ProjEquiRect(;
#     Ny=200, # nθ
#     Nx=768, # nφ
#     θspan  = (2.7, 2.9), 
#     φspan  = (0.0, 2π/4), 
# )
# 
# @show extrema(@. rad2deg(√pj.Ω)*60) 
# 
# # Plot √Ωpix over ring θ's 
# 
# @sblock let θ=pj.θ, φ=pj.φ, Ω=pj.Ω, Δθ=pj.Δθ, hide_plots
#     hide_plots && return
#     fig,ax = subplots(1)
#     ax.plot(θ, (@. rad2deg(√Ω)*60), label="sqrt pixel area (arcmin)")
#     ax.plot(θ, (@. rad2deg(Δθ)*60), label="Δθ (arcmin)")
#     ax.set_xlabel(L"polar coordinate $\theta$")
#     ax.legend()
#     return nothing
# end
# 

# Block diagonal cov matrices
# ==============================


# Now we don't have pj.freq_mult, pj.θ, ... etc
# What is the CMBLensing way to get grid features out of pj?
# Are there generic method names that I'm supposed to overload like pix(pj) ...?

EB▫, Phi▫  = @sblock let ℓ, CEEℓ, CBBℓ, CΦΦℓ, θ=pj.θ, φ=pj.φ, freq_mult=pj.freq_mult

    nθ, nφ = length(θ), length(φ)
    nφ2π  = nφ*freq_mult
    φ2π   = 2π*(0:nφ2π-1)/nφ2π |> collect

    covβEB   = βcovSpin2(ℓ, CEEℓ, CBBℓ, ngrid=50_000)
    covβPhi  = βcovSpin0(ℓ, CΦΦℓ,       ngrid=50_000)

    ptmW    = FFTW.plan_fft(Vector{ComplexF64}(undef, nφ)) 
    EBγⱼₖ   = zeros(ComplexF64, nφ)
    EBξⱼₖ   = zeros(ComplexF64, nφ)
    Phiγⱼₖ  = zeros(ComplexF64, nφ)

    T     = ComplexF64 # ComplexF32
    rT    = real(T)
    EB▫   = zeros(T,2nθ,2nθ,nφ÷2+1)
    Phi▫  = zeros(rT,nθ,nθ,nφ÷2+1)

    prgss = Progress(nθ, 1, "EB▫, Phi▫")
    for k = 1:nθ
        for j = 1:nθ
            θ1, θ2 = θ[j], θ[k]
            β      = geoβ.(θ1, θ2, φ2π[1], φ2π)

            covΦΦ̄  = covβPhi(β) 
            covPP̄, covPP = covβEB(β)  
            covPP̄ .*= multPP̄.(θ1, θ2, φ2π[1], φ2π)
            covPP .*= multPP.(θ1, θ2, φ2π[1], φ2π)
            
            ## periodize and restrict from φ2π to φ
            covΦΦ̄′   = periodize(covΦΦ̄, freq_mult)   
            covPP̄′   = periodize(covPP̄, freq_mult)       
            covPP′   = periodize(covPP, freq_mult)
  
            mul!(Phiγⱼₖ,  ptmW, covΦΦ̄′)
            mul!(EBγⱼₖ,   ptmW, covPP̄′)
            mul!(EBξⱼₖ,   ptmW, covPP′)

            @inbounds for ℓ = 1:nφ÷2+1
                Jℓ = Jperm(ℓ, nφ) # ℓ==1 ? 1 : nφ - ℓ + 2
                Phi▫[j,  k,    ℓ] = real(Phiγⱼₖ[ℓ])
                EB▫[j,   k,    ℓ] = EBγⱼₖ[ℓ]
                EB▫[j,   k+nθ, ℓ] = EBξⱼₖ[ℓ]
                EB▫[j+nθ,k,    ℓ] = conj(EBξⱼₖ[Jℓ])
                EB▫[j+nθ,k+nθ, ℓ] = conj(EBγⱼₖ[Jℓ])
            end
        end
        next!(prgss)
    end

    @show Base.summarysize(EB▫) / 1e9
    @show Base.summarysize(Phi▫)  / 1e9

    return EB▫, Phi▫
end;


# why the numerical non-positive def?

EB▫½, Phi▫½ = @sblock let EB▫, Phi▫ 
    EB▫½  = similar(EB▫)
    Phi▫½ = similar(Phi▫)
    for b in axes(EB▫,3)
        EB▫½[:,:,b]  .= sqrt(Hermitian(EB▫[:,:,b]))
        Phi▫½[:,:,b] .= sqrt(Symmetric(Phi▫[:,:,b]))
        # ... or ...
        ## EB▫½[:,:,b]  .= Array(cholesky(Hermitian(EB▫½[:,:,b])).L)
        ## Phi▫½[:,:,b] .= Array(cholesky(Symmetric(Phi▫½[:,:,b])).L)
    end
    EB▫½, Phi▫½
end


# Test simulation of ϕmap, Qmap, Umap
# =======================================

# Field sims

ϕ′ = BaseField{RingMapS0}(randn(Float64, pj.nθ, pj.nφ), pj)
P′ = BaseField{RingMapS2}(randn(ComplexF64, pj.nθ, pj.nφ), pj)

# Test conversion

@show BaseField{Ring▫S0}(ϕ′) isa BaseField{Ring▫S0}
@show BaseField{Ring▫S2}(P′) isa BaseField{Ring▫S2}

# How do I get this to work?

2 * BaseField{Ring▫S0}(ϕ′) + ϕ′
2 * BaseField{Ring▫S2}(P′) + P′


# TODO: figure out the right way to reduce the above code duple with a Spin 
# agnostic fourier ring constructor


ϕsim = tulliomult(Phi▫½,  ϕ′)
Psim = tulliomult(EB▫½, P′)

ϕmap = ϕsim[:]
Qmap = real.(Psim[:])
Umap = imag.(Psim[:])

ϕmap  |> matshow; colorbar()
Qmap  |> matshow; colorbar()
Umap  |> matshow; colorbar()



# TODO:
# =======================================
# • Figure out the m_fft transform stuff ... how to get unitary version
# • Need to make sure the sign of U matches CMBLensing
#   ... probably just need a negative spin 2 option in CirculantCov
# • Figure out how to get Block field operators and all the stuff to go with it 






# Methods ...
# =======================================

## Basis type

abstract type Ring <: Basis end

abstract type Ring▫   <: Ring  end
abstract type Ring▫S0 <: Ring▫ end
abstract type Ring▫S2 <: Ring▫ end

abstract type RingMap   <: Ring    end
abstract type RingMapS0 <: RingMap end
abstract type RingMapS2 <: RingMap end

## Pixel type 

struct AzEq{T<:Real} <: Proj
    nθ :: Int
    nφ :: Int
    θ  :: Vector{T}
    φ  :: Vector{T}
    Ω  :: Vector{T}
    Δθ :: Vector{T}
    freq_mult :: Int

    function AzEq(;
            nθnφ::Tuple{Int,Int}, 
            θspan::Tuple{T,T}, 
            φspan::Tuple{T,T}, 
            cosθEq::Bool = false, # default to θEq
        ) where T<:Real

        @assert Int(2π / φspan[2]) isa Int
        @assert θspan[1] < θspan[2]
        @assert φspan[1] < φspan[2]

        nθ, nφ  = nθnφ

        if cosθEq
            znorth = cos.(θspan[1])
            zsouth = cos.(θspan[2])
            θpix∂  = acos.(range(znorth, zsouth, length=nθ+1))
        else
            θpix∂   = T.(θspan[1] .+ (θspan[2] - θspan[1])*(0:nθ)/nθ)
        end
        Δθ = diff(θpix∂)
        θ  = θpix∂[2:end] .- Δθ/2    
        
        freq_mult = Int(2π / φspan[2])
        φ = T.(φspan[1] .+ (φspan[2] - φspan[1])*(0:nφ-1)/nφ) 
        
        Ω  = @. (φ[2] - φ[1]) * abs(cos(θpix∂[1:end-1]) - cos(θpix∂[2:end]))
        return new{T}(nθ, nφ, θ, φ, Ω, Δθ, freq_mult)
    end

end



## basis conversion

# Is there some guarentee/requirment that the conversion happens within the same Proj? 
# I guess I'm making it a requirement by the conversion definitions below.

# I'm partly confused how the BasisField type and the Proj type interact.
# For example, would there ever be a case when one would use RingMapS0 
# without the corresponding AzEq

# Working understanding:
# • The field type parameter AzEq <: Proj tells you what the metadata is
# • The field type parameter Ring <: Basis tells you what the dual Basis is

## TODO: incorperate planned ffts

function (::Type{BR▫})(fmap::BaseField{RingMapS0, <:AzEq}) where {BR▫<:BaseField{Ring▫S0}}
    nφ = fmap.metadata.nφ
    BR▫(rfft(fmap.arr, 2) ./ √nφ, fmap.metadata)
end

function (::Type{BRM})(f▫::BaseField{Ring▫S0, <:AzEq}) where {BRM<:BaseField{RingMapS0}}
    nφ = f▫.metadata.nφ
    BRM(irfft(f▫.arr, nφ, 2) .* √nφ, f▫.metadata)
end

function (::Type{BR▫})(pmap::BaseField{RingMapS2, <:AzEq}) where {BR▫<:BaseField{Ring▫S2}}
    nθ = pmap.metadata.nθ
    nφ = pmap.metadata.nφ
    Upmap = fft(pmap.arr, 2) ./ √nφ
    p▫    = similar(Upmap, 2nθ, nφ÷2+1)
    for ℓ = 1:nφ÷2+1
        if (ℓ==1) | ((ℓ==nφ÷2+1) & iseven(nφ))
            p▫[1:nθ, ℓ]     .= Upmap[:,ℓ]
            p▫[nθ+1:2nθ, ℓ] .= conj.(Upmap[:,ℓ])
        else 
            p▫[1:nθ, ℓ]     .= Upmap[:,ℓ]
            p▫[nθ+1:2nθ, ℓ] .= conj.(Upmap[:,Jperm(ℓ,nφ)])
        end
    end
    BR▫(p▫, pmap.metadata)
end

function (::Type{BRM})(p▫::BaseField{Ring▫S2, <:AzEq}) where {BRM<:BaseField{RingMapS2}}
    nθₓ2, nφ½₊1 = size(p▫.arr)
    nθ = p▫.metadata.nθ
    nφ = p▫.metadata.nφ
    @assert nφ½₊1 == nφ÷2+1
    @assert 2nθ   == nθₓ2

    pθk = similar(p▫, nθ, nφ)
    for ℓ = 1:nφ½₊1
        if (ℓ==1) | ((ℓ==nφ½₊1) & iseven(nφ))
            pθk[:,ℓ] .= p▫[1:nθ,ℓ] 
        else 
            pθk[:,ℓ]  .= p▫[1:nθ,ℓ]     
            pθk[:,Jperm(ℓ,nφ)] .= conj.(p▫[nθ+1:2nθ,ℓ])
        end
    end 
    BRM(ifft(pθk, 2) .* √nφ, p▫.metadata)
end

# Do I need these??
function (::Type{BR▫})(p▫::BaseField{<:Ring▫, <:AzEq}) where {R▫<:Ring▫, BR▫<:BaseField{R▫}}
    p▫
end

function (::Type{BRM})(pmap::BaseField{<:RingMap, <:AzEq}) where {RM<:RingMap, BRM<:BaseField{RM}}
    pmap
end

## f[:] -> map storage array... 
## This is in XFields.jl and I really like it.
# just adding it here temporarily 

Base.getindex(f::BaseField{RingMapS0}, ::Colon) = f.arr
Base.getindex(f::BaseField{RingMapS2}, ::Colon) = f.arr
Base.getindex(f::BaseField{Ring▫S0},   ::Colon) = BaseField{RingMapS0}(f).arr
Base.getindex(f::BaseField{Ring▫S2},   ::Colon) = BaseField{RingMapS2}(f).arr

Base.getindex(f::BaseField{RingMapS0}, ::typeof(!)) = BaseField{Ring▫S0}(f).arr
Base.getindex(f::BaseField{RingMapS2}, ::typeof(!)) = BaseField{Ring▫S2}(f).arr
Base.getindex(f::BaseField{Ring▫S0},   ::typeof(!)) = f.arr
Base.getindex(f::BaseField{Ring▫S2},   ::typeof(!)) = f.arr

#-

function tulliomult(M▫, f::BaseField{RS0}) where {RS0 <: Union{RingMapS0, Ring▫S0}}
    m▫ = BaseField{Ring▫S0}(f).arr
    @tullio n▫[i,m] :=  M▫[i,j,m] * m▫[j,m]
    BaseField{Ring▫S0}(n▫, f.metadata)
end

function tulliomult(M▫, f::BaseField{RS2}) where {RS2 <: Union{RingMapS2, Ring▫S2}}
    m▫ = BaseField{Ring▫S2}(f).arr
    @tullio n▫[i,m] :=  M▫[i,j,m] * m▫[j,m]
    BaseField{Ring▫S2}(n▫, f.metadata)
end

