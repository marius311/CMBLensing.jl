
# Modules
# ==============================
using LinearAlgebra
using FFTW
using Tullio 
using PyPlot
using BenchmarkTools
using ProgressMeter
FFTW.set_num_threads(Threads.nthreads())
using LBblocks: @sblock # from github.com/EthanAnderes/LBblocks.jl

using CMBLensing
using CirculantCov: βcovSpin2, βcovSpin0, geoβ, 
multPP̄, multPP, periodize, Jperm

hide_plots = true

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

θ, φ, Ω, Δθ, nθ, nφ, freq_mult = @sblock let 
    freq_mult = 3
    nθ, nφ    = (200, 768)
    θnorth∂, θsouth∂ = (2.7, 2.9)
    φleft∂, φright∂  = (0.0, 2π/freq_mult)

    θpix∂   = θnorth∂ .+ (θsouth∂ - θnorth∂)*(0:nθ)/nθ  |> collect
    ## --- or -------
    ## znorth = cos.(θnorth∂)
    ## zsouth = cos.(θsouth∂)
    ## θpix∂ = acos.(range(znorth, zsouth, length=nθ+1))
    ## --------------
    Δθ = diff(θpix∂)
    θ = θpix∂[2:end] .- Δθ/2    
    φ  = φleft∂ .+ (φright∂ - φleft∂)*(0:nφ-1)/nφ  |> collect    
    Ω   = @. (φ[2] - φ[1]) * abs(cos(θpix∂[1:end-1]) - cos(θpix∂[2:end]))

    return θ, φ, Ω, Δθ, nθ, nφ, freq_mult
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


#  
# ==============================

EB▫, Phi▫  = @sblock let ℓ, CEEℓ, CBBℓ, CΦΦℓ, θ, φ, freq_mult

    nθ, nφ = length(θ), length(φ)
    nφ2π  = nφ*freq_mult
    φ2π   = 2π*(0:nφ2π-1)/nφ2π |> collect

    ## TODO: improve the adjustable resolution to help with periodicity
    covβEB   = βcovSpin2(ℓ, CEEℓ, CBBℓ)
    covβPhi  = βcovSpin0(ℓ, CΦΦℓ)

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

            covΦΦ̄   = covβPhi(β)  |> complex
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




# Methods for preping map arrays for ring transform mult
# =======================================

# Real spin0 maps have an implicit pairing with primal and dual frequency
# so we instead construct nφ÷2+1 vectors of length nθ 
spin0_to_▫(fmap::AbstractMatrix{<:Real}) = rfft(fmap, 2)    ./ √(size(fmap)[2])

▫_to_spin0(f▫::AbstractMatrix, nφ::Int)   = irfft(f▫, nφ, 2) .* √nφ

# Complex spin2 maps get frequency paired with dual frequency 
# to make nφ÷2+1 vectors of length 2nθ 
function spin2_to_▫(pmap::AbstractMatrix{<:Complex})
    nθ, nφ = size(pmap)
	Upmap  = fft(pmap, 2) ./ √nφ
    p▫     = similar(Upmap, 2nθ, nφ÷2+1)
    for ℓ = 1:nφ÷2+1
        if (ℓ==1) | ((ℓ==nφ÷2+1) & iseven(nφ))
            p▫[1:nθ, ℓ]     .= Upmap[:,ℓ]
            p▫[nθ+1:2nθ, ℓ] .= conj.(Upmap[:,ℓ])
        else 
            p▫[1:nθ, ℓ]     .= Upmap[:,ℓ]
            p▫[nθ+1:2nθ, ℓ] .= conj.(Upmap[:,Jperm(ℓ,nφ)])
        end
    end
    p▫
end

function ▫_to_spin2(p▫::AbstractMatrix, nφ::Int) where To 
    nθₓ2, nφ½₊1   = size(p▫)
    @assert nφ½₊1 == nφ÷2+1
    @assert iseven(nθₓ2)
    nθ  = nθₓ2÷2

    pθk = similar(p▫, nθ, nφ)
    for ℓ = 1:nφ½₊1
        if (ℓ==1) | ((ℓ==nφ½₊1) & iseven(nφ))
            pθk[:,ℓ] .= p▫[1:nθ,ℓ] 
        else 
            pθk[:,ℓ]  .= p▫[1:nθ,ℓ]     
            pθk[:,Jperm(ℓ,nφ)] .= conj.(p▫[nθ+1:2nθ,ℓ])
        end
    end 
    ifft(pθk, 2) .* √nφ
end


# Test simulation of ϕmap, Qmap, Umap
# =======================================


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

ϕ▫′ = spin0_to_▫(randn(Float64, nθ, nφ))
P▫′ = spin2_to_▫(randn(ComplexF64, nθ, nφ))

@tullio ϕ▫[i,m] :=  Phi▫½[i,j,m] * ϕ▫′[j,m]
@tullio P▫[i,m] :=  EB▫½[i,j,m]  * P▫′[j,m]

ϕmap = ▫_to_spin0(ϕ▫, nφ)
Pmap = ▫_to_spin2(P▫, nφ)

Qmap = real.(Pmap)
Umap = imag.(Pmap)

ϕmap |> matshow; colorbar()
Qmap |> matshow; colorbar()
Umap |> matshow; colorbar()



# TODO:
# =======================================
# • Need to make sure the sign of U matches CMBLensing
# • Get the array stuff into fields
# • Figure out the m_fft transform stuff ... how to get unitary version
# • Why are we fixing a theta gridding in ProjEquiRect{T}?
# • Figure out how to get Block field operators and all the stuff to go with it 

# 
# =======================================



# using Revise, CMBLensing, Images, TestImages, PyPlot

# arr = imresize(Images.gray.(float.(testimage("fabio_gray"))), (100,236));

# f = LambertMap(arr, θpix=10, rotator=(0,40,10))
# plot(f, vlim=[(0,1)]);

# h = project(f => ProjHealpix(512))
# plot(h);

# f′ = project(h => f.proj)
# plot([f′ f (f-f′)])

# f = EquiRectMap(arr, θspan = deg2rad.(180 .+ (-40,-70)), ϕspan = deg2rad.((-10,90)))
# imshow(f.arr); # no special plot function for EquiRectMaps implemented yet

# h = project(f => ProjHealpix(512))
# plot(h);

# f′ = project(h => f.proj)
# figure(figsize=(15,5))
# subplot(131).imshow(f′.arr)
# subplot(132).imshow(f.arr)
# subplot(133).imshow((f-f′).arr);
