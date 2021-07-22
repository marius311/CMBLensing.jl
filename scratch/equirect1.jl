
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
import CMBLensing: Map, AzFourier, QUAzFourier, QUMap, m_fft, m_rfft, m_irfft,  m_ifft
using CirculantCov: βcovSpin2, βcovSpin0, geoβ,
multPP̄, multPP, periodize, Jperm # https://github.com/EthanAnderes/CirculantCov.jl

## LATER: remove LBblock dependence
using LBblocks: @sblock # https://github.com/EthanAnderes/LBblocks.jl

hide_plots = true

using Test

# Methods ...
# =======================================

# EquiRectS0 ...
function tulliomult(M▫, f::Union{EquiRectAzFourier, EquiRectMap})
    m▫ = AzFourier(f).arr
    @tullio n▫[i,m] :=  M▫[i,j,m] * m▫[j,m]
    EquiRectAzFourier(n▫, f.metadata)
end
# EquiRectS2 
function tulliomult(M▫, f::Union{EquiRectQUAzFourier, EquiRectQUMap})
    m▫ = QUAzFourier(f).arr
    @tullio n▫[i,m] :=  M▫[i,j,m] * m▫[j,m]
    EquiRectQUAzFourier(n▫, f.metadata)
end

# Quick test
# ============================

pj = CMBLensing.ProjEquiRect(;
    Ny = 200, # nθ
    Nx = 768, # nφ
    θspan = (2.7, 2.9), 
    φspan = (0.0, 2π/4), 
)

#-

ϕ = EquiRectMap(randn(Float64, pj.Ny, pj.Nx), pj)
P = EquiRectQUMap(randn(ComplexF64, pj.Ny, pj.Nx), pj)

#-

ϕ′ = AzFourier(ϕ)    
P′ = QUAzFourier(P)   

# the printing of the size is wrong 

#-

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


# Now we don't have pj.freq_mult, pj.θ, ... etc
# What is the CMBLensing way to get grid features out of pj?
# Are there generic method names that I'm supposed to overload like pix(pj) ...?


# TODO: Put these into Proj ...
θ, φ, Ω, freq_mult, Δθ = @sblock let pj, T = Float64, cosθEq=false

    nθ, nφ = pj.Ny, pj.Nx
    θspan = pj.θspan
    φspan = pj.φspan

    @assert Int(2π / φspan[2]) isa Int
    @assert θspan[1] < θspan[2]
    @assert φspan[1] < φspan[2]

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
    θ, φ, Ω, freq_mult, Δθ
end;


@show extrema(@. rad2deg(√Ω)*60) 

# Plot √Ωpix over ring θ's 

@sblock let θ, φ, Ω, Δθ, hide_plots
    hide_plots && return
    fig,ax = subplots(1)
    ax.plot(θ, (@. rad2deg(√Ω)*60), label="sqrt pixel area (arcmin)")
    ax.plot(θ, (@. rad2deg(Δθ)*60), label="Δθ (arcmin)")
    ax.set_xlabel(L"polar coordinate $\theta$")
    ax.legend()
    return nothing
end


# Block diagonal cov matrices
# ==============================




EB▫, Phi▫  = @sblock let ℓ, CEEℓ, CBBℓ, CΦΦℓ, θ, φ, freq_mult

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

#-

EB▫½, Phi▫½ = @sblock let EB▫, Phi▫ 
    EB▫½  = similar(EB▫)
    Phi▫½ = similar(Phi▫)
    for b in axes(EB▫,3)
        ## EB▫½[:,:,b]  .= sqrt(Hermitian(EB▫[:,:,b]))
        ## Phi▫½[:,:,b] .= sqrt(Symmetric(Phi▫[:,:,b]))
        ## ... or ...
        try
            EB▫½[:,:,b]  .= Matrix(cholesky(Hermitian(EB▫[:,:,b])).L)
            Phi▫½[:,:,b] .= Matrix(cholesky(Symmetric(Phi▫[:,:,b])).L)
        catch
            @show b
            break
        end
    end
    EB▫½, Phi▫½
end;


# Test simulation of ϕmap, Qmap, Umap
# =======================================

# Field sim unit noise

ϕ′ = EquiRectMap(  randn(Float64,    pj.Ny, pj.Nx), pj)
P′ = EquiRectQUMap(randn(ComplexF64, pj.Ny, pj.Nx), pj)

# Test conversion

@test AzFourier(ϕ′) isa EquiRectAzFourier
@test QUAzFourier(P′) isa EquiRectQUAzFourier

@test all(Map(AzFourier(ϕ′)).arr .≈ ϕ′[:])
@test all(QUMap(QUAzFourier(P′)).arr .≈ P′[:])


# generate simulation 

ϕsim = tulliomult(Phi▫½,  ϕ′)
Psim = tulliomult(EB▫½, P′)

# plot maps of the simulated fields

@sblock let ϕsim, Psim, hide_plots=false
    hide_plots && return
    fig,ax = subplots(3)
    ϕsim[:] |> imshow(-, fig, ax[1])
    Psim[:] .|> real |> imshow(-, fig, ax[2]) # Qsim
    Psim[:] .|> imag |> imshow(-, fig, ax[3]) # Usim
    return nothing
end



#-

# gradient(x-> dot(Map(x), Map(x)), ϕsim)[1]


# TODO:
# =======================================
# • Need to make sure the sign of U matches CMBLensing ... probably just need a negative spin 2 option in CirculantCov 
# • Block field operators and all the stuff to go with it

