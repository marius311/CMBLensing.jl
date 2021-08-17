## Literate.notebook("equirect1.jl"; execute=false) 


# Modules
# ==============================
using Tullio 
using PyPlot
using BenchmarkTools
using ProgressMeter
using LinearAlgebra
using Random
using FFTW
FFTW.set_num_threads(Threads.nthreads())

using CMBLensing
using CirculantCov
import CirculantCov as CC # https://github.com/EthanAnderes/CirculantCov.jl

using Test

## LATER: remove LBblock dependence
using LBblocks: @sblock # https://github.com/EthanAnderes/LBblocks.jl

hide_plots = false


# Set the grid geometry
# ============================

pj = @sblock let 

    θ, θ∂ = CC.θ_grid(; θspan=(2.3,2.7), N=128, type=:equiθ)
    ## θ, θ∂ = CC.θ_grid(; θspan=(2.3,2.7), N=250, type=:equiθ)
    ## θ, θ∂ = CC.θ_grid(; θspan=(2.3,2.7), N=250, type=:equicosθ)
    ## θ, θ∂ = CC.θ_grid(; θspan=(2.3,2.7), N=2048, type=:healpix)

    φ, φ∂ = CC.φ_grid(; φspan=deg2rad.((-36, 36)), N=2*128-1)
    ## φ, φ∂ = CC.φ_grid(; φspan=deg2rad.((-60, 60)), N=1024)

    CMBLensing.ProjEquiRect(; θ, φ, θ∂, φ∂)
end;

#- 

# Testing: just to verify the formula for Ω
let θ∂ = [pi/2, 3pi/4, pi]
    Ω = 2π .* diff(.- cos.(θ∂))
    @test sum(Ω) ≈ 4pi/2
end

#-

@show extrema(@. rad2deg(√pj.Ω)*60) 

# Plot √Ωpix over ring θ's 

@sblock let pj, hide_plots
    hide_plots && return
    fig,ax = subplots(1)
    ax.plot(pj.θ, rad2deg.(.√pj.Ω)*60, label="sqrt pixel area (arcmin)")
    ax.plot(pj.θ, rad2deg.(diff(pj.θ∂))*60, label="Δθ (arcmin)")
    ax.set_xlabel(L"polar coordinate $\theta$")
    ax.legend()
    return fig
end;

# Spectral densities
# ==============================

ℓ, CEEℓ, CBBℓ, CΦΦℓ, CBeamℓ = @sblock let ℓmax = 11000, beamfwhm_arcmin = 30 
	ℓ    = 0:ℓmax
	Cℓ   = camb(;r=0.01, ℓmax);
	CBBℓ = Cℓ.tensor.BB(ℓ)
	CEEℓ = Cℓ.unlensed_scalar.EE(ℓ)
	CΦΦℓ = Cℓ.unlensed_scalar.ϕϕ(ℓ)
	for cl in (CEEℓ, CBBℓ, CΦΦℓ)
		cl[.!isfinite.(cl)] .= 0
	end

    beamfwhm_rad = beamfwhm_arcmin |> arcmin -> deg2rad(arcmin/60)
    σ²    = beamfwhm_rad^2 / 8 / log(2)
    CBeamℓ = @. exp( - σ²*ℓ*(ℓ+1) / 2)

	return ℓ, CEEℓ, CBBℓ, CΦΦℓ, CBeamℓ
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


# Block diagonal cov matrices
# ==============================

EB▪, Beam▪, Phi▪  = @sblock let ℓ, CEEℓ, CBBℓ, CΦΦℓ, CBeamℓ, pj

    θ, φ, Ω = pj.θ, pj.φ, pj.Ω
    nθ, nφ  = length(θ), length(φ)

    Γ_Phi  = CC.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, CΦΦℓ; ngrid=50_000)
    Γ_Beam = CC.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, CBeamℓ; ngrid=50_000)
    ΓC_EB  = CC.ΓCθ₁θ₂φ₁φ⃗_CMBpol(ℓ, CEEℓ, CBBℓ; ngrid=50_000)

    T     = ComplexF64 # ComplexF32
    rT    = real(T)
    EB▫   = zeros(T,2nθ,2nθ,nφ÷2+1)
    Beam▫ = zeros(rT,2nθ,2nθ,nφ÷2+1)
    Phi▫  = zeros(rT,nθ,nθ,nφ÷2+1)
    
    ptmW    = FFTW.plan_fft(Vector{ComplexF64}(undef, nφ)) 

    prgss = Progress(nθ, 1, "EB▫, Phi▫, Beam▫")
    for k = 1:nθ
        for j = 1:nθ
            Phiγⱼₖℓ⃗       = CC.γθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ_Phi,  ptmW)
            Beamγⱼₖℓ⃗      = CC.γθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ_Beam, ptmW)
            EBγⱼₖℓ⃗, EBξⱼₖℓ⃗ = CC.γθ₁θ₂ℓ⃗_ξθ₁θ₂ℓ⃗(θ[j], θ[k], φ, ΓC_EB..., ptmW)
            for ℓ = 1:nφ÷2+1
                Jℓ = CC.Jperm(ℓ, nφ)
                Phi▫[j,k, ℓ] = real(Phiγⱼₖℓ⃗[ℓ])

                Beam▫[j,   k   , ℓ] = real(Beamγⱼₖℓ⃗[ℓ])  * Ω[k]
                Beam▫[j+nθ,k+nθ, ℓ] = real(Beamγⱼₖℓ⃗[Jℓ]) * Ω[k]

                EB▫[j,   k   , ℓ]   = EBγⱼₖℓ⃗[ℓ]
                EB▫[j,   k+nθ, ℓ]   = EBξⱼₖℓ⃗[ℓ]
                EB▫[j+nθ,k   , ℓ]   = conj(EBξⱼₖℓ⃗[Jℓ])
                EB▫[j+nθ,k+nθ, ℓ]   = conj(EBγⱼₖℓ⃗[Jℓ])
            end
        end
        next!(prgss)
    end

    @show Base.summarysize(EB▫) / 1e9
    @show Base.summarysize(Phi▫)  / 1e9

    EB▪   = BlockDiagEquiRect{QUAzFourier}(EB▫, pj)
    Beam▪ = BlockDiagEquiRect{QUAzFourier}(Beam▫, pj)
    Phi▪  = BlockDiagEquiRect{AzFourier}(Phi▫, pj)

    return EB▪, Beam▪, Phi▪ 
end;


# Testing out indexing

@sblock let EB▪, Phi▪, idx = 2, hide_plots
    hide_plots && return
    fig,ax = subplots(1,2,figsize=(9,5))
    EB▪.blocks[:,:,idx]  .|> abs |> imshow(-, fig, ax[1])
    Phi▪.blocks[:,:,idx] .|> abs |> imshow(-, fig, ax[2])
    return fig
end;


# Testing out indexing

@sblock let EB▪, Phi▪, idx1 = 2, idx2 = 40, hide_plots
    hide_plots && return
    fig,ax = subplots(1,2,figsize=(9,5))
    ax[1].semilogy(eigen(Hermitian(EB▪.blocks[:,:,idx1])).values)
    ax[1].semilogy(eigen(Hermitian(EB▪.blocks[:,:,idx2])).values)
    ax[2].semilogy(eigen(Symmetric(Phi▪.blocks[:,:,idx1])).values)
    ax[2].semilogy(eigen(Symmetric(Phi▪.blocks[:,:,idx2])).values)
    return fig
end;



# Test simulation of ϕmap, Qmap, Umap
# =======================================

f0  = CMBLensing.simulate(Phi▪)
f2  = CMBLensing.simulate(EB▪)

f2′ = Beam▪ * f2

# plot maps of the simulated fields.

@sblock let f0, hide_plots
    hide_plots && return
    fig,ax = subplots(1,figsize=(8,4))
    f0[:]  |> imshow(-, fig, ax)
    ax.set_title("Phi sim") 
    return fig
end;


# plot maps of the simulated fields.

@sblock let f2, f2′, hide_plots
    hide_plots && return
    fig,ax = subplots(2,2, figsize=(12,8))
    f2[:] .|> real |> imshow(-, fig, ax[1,1])  # Qsim
    f2[:] .|> imag |> imshow(-, fig, ax[2,1])  # Usim
    f2′[:] .|> real |> imshow(-, fig, ax[1,2]) # Beamed Qsim
    f2′[:] .|> imag |> imshow(-, fig, ax[2,2]) # Beamed Usim
    ax[1,1].set_title("Q sim") 
    ax[2,1].set_title("U sim") 
    ax[1,2].set_title("Beamed Q sim") 
    ax[2,2].set_title("Beamed U sim") 
    return fig
end;

# Test for correct Fourier symmetry in monopole and nyquist f2 

let f2kk = f2[!], f2xx = f2[:]

    v = f2kk[1:end÷2,1]
    w = f2kk[end÷2+1:end,1]
    @test v ≈ conj.(w)

    if iseven(size(f2xx,2))
        v′ = f2kk[1:end÷2,end]
        w′ = f2kk[end÷2+1:end,end]
        @test v′ ≈ conj.(w′)
    end

end


# Simulation with pre-computed sqrt 
# =======================================


Phi▪½  = CMBLensing.mapblocks(Phi▪) do M 
    Matrix(sqrt(Hermitian(M)))
end;

EB▪½  = CMBLensing.mapblocks(EB▪) do M 
    Matrix(sqrt(Hermitian(M)))
end

# generate simulation 

g0 = Phi▪½ * EquiRectMap(randn(Float64,pj.Ny,pj.Nx),pj)
g2 = EB▪½ * EquiRectQUMap(randn(Float64,pj.Ny,pj.Nx), randn(Float64,pj.Ny,pj.Nx),pj);

# plot maps of the simulated fields

@sblock let g0, g2, hide_plots
    hide_plots && return
    fig,ax = subplots(3,figsize=(6,9))
    g0[:]  |> imshow(-, fig, ax[1])
    g2[:] .|> real |> imshow(-, fig, ax[2]) # Qsim
    g2[:] .|> imag |> imshow(-, fig, ax[3]) # Usim
    return fig
end;

# Tests
# =======================================

Cf0 = Phi▪
Cf2 = EB▪

# transform

@test AzFourier(f0)   ≈ f0
@test QUAzFourier(f2) ≈ f2
@test Map(f0)   ≈ f0
@test QUMap(f2) ≈ f2

@test AzFourier(f0)[:]   ≈ f0[:]
@test QUAzFourier(f2)[:] ≈ f2[:]
@test Map(f0)[!]   ≈ f0[!]
@test QUMap(f2)[!] ≈ f2[!]

@test AzFourier(g0)[:]   ≈ g0[:]
@test QUAzFourier(g2)[:] ≈ g2[:]
@test Map(g0)[!]   ≈ g0[!]
@test QUMap(g2)[!] ≈ g2[!]


# dot product independent of basis
@test dot(f0,f0) ≈ dot(Map(f0),Map(f0))     ≈ dot(AzFourier(f0), AzFourier(f0))
@test dot(f2,f2) ≈ dot(QUMap(f2),QUMap(f2)) ≈ dot(QUAzFourier(f2), QUAzFourier(f2))

# # creating block-diagonal covariance operators
# @test (Cf0 = Cℓ_to_Cov(:I, f0.proj, Cℓ.total.TT)) isa BlockDiagEquiRect
# @test (Cf2 = Cℓ_to_Cov(:P, f2.proj, Cℓ.total.EE, Cℓ.total.BB)) isa BlockDiagEquiRect

# sqrt
@test (sqrt(Cf0) * sqrt(Cf0) * f0) ≈ (Cf0 * f0)
@test (sqrt(Cf2) * sqrt(Cf2) * f2) ≈ (Cf2 * f2)

# simulation
@test simulate(Cf0) isa EquiRectS0
@test simulate(Cf2) isa EquiRectS2

# pinv
@test (pinv(Cf0) * Cf0 * f0) ≈ f0 rtol=1e-5
@test (pinv(Cf2) * Cf2 * f2) ≈ f2

# logdet
@test logdet(Cf0) ≈ logabsdet(Cf0)
@test logdet(Cf2) ≈ logabsdet(Cf2)

# adjoint
@test (f0' * (Cf0 * g0)) ≈ ((f0' * Cf0) * g0)
@test (f2' * (Cf2 * g2)) ≈ ((f2' * Cf2) * g2)


# gradients
@test_real_gradient(α -> (f0 + α * g0)' * pinv(Cf0) * (f0 + α * g0), 0)
@test_real_gradient(α -> (f2 + α * g2)' * pinv(Cf2) * (f2 + α * g2), 0)
@test_real_gradient(α -> logdet(α * Cf0), 0)
@test_real_gradient(α -> logdet(α * Cf2), 0)

###


θspan  = (π/2 .- deg2rad.((-40,-70)))
φspan  = deg2rad.((-60, 60))
φspanᵒ = deg2rad.((-50, 50))
Cℓ = camb()

# non-periodic
proj = ProjEquiRect(;Ny=128, Nx=128, θspan, φspan=φspanᵒ)
@test proj.Ny == proj.Nx == length(proj.θ) == length(proj.φ) == 128 

# constructor doesnt error
Nsides_big = [(128,128), (64,128), (128,64)]
Nside = Nsides_big[1]
f0 = EquiRectMap(rand(Nside...); θspan, φspan)
f2 = EquiRectQUMap(rand(ComplexF64, Nside...); θspan, φspan)

proj = ProjEquiRect(;Ny=Nside[1], Nx=Nside[2], θspan, φspan=φspanᵒ)
f0 = EquiRectMap(rand(Nside...), proj)
f2 = EquiRectQUMap(rand(ComplexF64, Nside...), proj)

@test f0 isa EquiRectMap
@test f2 isa EquiRectQUMap

# transform
@test Map(AzFourier(f0)) ≈ f0
@test QUMap(QUAzFourier(f2)) ≈ f2

# transform (testing equality independent of dot)
@test AzFourier(f0)[:]   ≈ f0[:]
@test QUAzFourier(f2)[:] ≈ f2[:]
@test Map(f0)[!]   ≈ f0[!]
@test QUMap(f2)[!] ≈ f2[!]

@test AzFourier(f0)[:]   ≈ f0[:]
@test QUAzFourier(f2)[:] ≈ f2[:]
@test Map(f0)[:]   ≈ f0[:]
@test QUMap(f2)[:] ≈ f2[:]



# TODO:
# =======================================
# • Need to make sure the sign of U matches CMBLensing 
#   ... probably just need a negative spin 2 option in CirculantCov 
# • add + and adjoint to the linear algebra methods for BlockDiagEquiRect's
# • get tests working, including incorperating CirculantCov.jl stuff with @ondemand



let Nside = (128,128), θspan  = (π/2 .- deg2rad.((-40,-70))), φspanᵒ = deg2rad.((-50, 50))


    proj = ProjEquiRect(;Ny=Nside[1], Nx=Nside[2], θspan, φspan=φspanᵒ)
    f0 = EquiRectMap(rand(Nside...), proj)
    f2 = EquiRectQUMap(rand(ComplexF64, Nside...), proj)

    @test f0 isa EquiRectMap
    @test f2 isa EquiRectQUMap

    # transform
    @test Map(AzFourier(f0)) ≈ f0
    @test QUMap(QUAzFourier(f2)) ≈ f2

end

