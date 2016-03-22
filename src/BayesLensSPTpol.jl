module BayesLensSPTpol

# package code goes here

using PyCall, Dierckx

export	LensePrm,
		fftd,
		ifftd,
		ifftdr,
		class

FFTW.set_num_threads(CPU_CORES)

# This source file defines the scaled Fourier transforms used
include("gridfft.jl")



##########################################################
#=
Definition of the LensePrm Type.
Holds grid, model and planned FFT parameters for the quadratic estimate.
Allows easy argument passing.
=#
#############################################################

immutable LensePrm{dm, T1, T2}
	# grid parameters
	period::Float64
	nside::Int64
	deltx::Float64
	deltk::Float64
	nyq::Float64
	x::Array{Array{Float64,dm},1}
	k::Array{Array{Float64,dm},1}
	r::Array{Float64,dm}
	# parameters necessary for simulation
	cENNk::Array{Float64,dm}
	cBNNk::Array{Float64,dm}
	cϕϕk::Array{Float64,dm}
	cϕψk::Array{Float64,dm}
	cψψk::Array{Float64,dm}
	cEEk::Array{Float64,dm}
	cBBk::Array{Float64,dm}
	# parameters necessary for the quad est
	cEEobsk::Array{Float64,dm}
	cBBobsk::Array{Float64,dm}
	# saved plans for fast fft
	FFT::T1
	IFFT::T2            # this is the unnormalized version
	FFTconst::Float64   # these hold the normalization constants
	IFFTconst::Float64
end

"""
`LensePrm(dm, period, nside)` constructor for LensePrm{dm,T1,T2} type
"""
function LensePrm(dm, period, nside, cls, σEEarcmin,  σBBarcmin, beamFWHM)
	dm_nsides = fill(nside,dm)   # [nside,...,nside] <- dm times
	deltx     = period / nside
	deltk     = 2π / period
	nyq       = 2π / (2deltx)
	x         = [fill(NaN, dm_nsides...) for i = 1:dm]
	k         = [fill(NaN, dm_nsides...) for i = 1:dm]
	r         = fill(NaN, dm_nsides...)
	cENNk      = fill(NaN, dm_nsides...)
	cBNNk      = fill(NaN, dm_nsides...)
	cϕϕk      = fill(NaN, dm_nsides...)
	cϕψk      = fill(NaN, dm_nsides...)
	cψψk      = fill(NaN, dm_nsides...)
	cEEk      = fill(NaN, dm_nsides...)
	cBBk      = fill(NaN, dm_nsides...)
	cEEobsk	  = fill(NaN, dm_nsides...)
	cBBobsk   = fill(NaN, dm_nsides...)
	FFT       = plan_fft(rand(Complex{Float64},dm_nsides...); flags = FFTW.PATIENT, timelimit = 10)
	IFFT      = plan_bfft(rand(Complex{Float64},dm_nsides...); flags = FFTW.PATIENT, timelimit = 10)
	parms     = LensePrm{dm, typeof(FFT), typeof(IFFT)}(period, nside, deltx, deltk, nyq, x, k, r,
						cENNk, cBNNk, cϕϕk, cϕψk, cψψk,
						cEEk, cBBk, cEEobsk, cBBobsk,
						FFT, IFFT, (deltx / √(2π))^dm , (deltk / √(2π))^dm,
				)
	parms.x[:], parms.k[:] = getgrid(parms)
	parms.r[:]  =  √(sum([abs2(kdim) for kdim in parms.k]))
	parms.cENNk[:]      = cNNkgen(parms.r, parms.deltx; σunit=σEEarcmin, beamFWHM=beamFWHM)
	parms.cBNNk[:]      = cNNkgen(parms.r, parms.deltx; σunit=σBBarcmin, beamFWHM=beamFWHM)
	parms.cϕϕk[:]      = cls_to_cXXk(cls[:ell], cls[:ϕϕ], parms.r)
	parms.cϕψk[:]      = cls_to_cXXk(cls[:ell], cls[:ϕψ], parms.r)
	parms.cψψk[:]      = cls_to_cXXk(cls[:ell], cls[:ψψ], parms.r)
	parms.cEEk[:]      = cls_to_cXXk(cls[:ell], cls[:ee], parms.r)
	parms.cBBk[:]      = cls_to_cXXk(cls[:ell], cls[:bb], parms.r)
	parms.cEEobsk[:]   = parms.cEEk + parms.cENNk
	parms.cBBobsk[:]   = parms.cBNNk + parms.cBNNk
	return parms
end
function getxkside{dm,T1,T2}(g::LensePrm{dm,T1,T2})
	deltx    = g.period / g.nside
	deltk    = 2π / g.period
	xco_side = zeros(g.nside)
	kco_side = zeros(g.nside)
	for j in 0:(g.nside-1)
		xco_side[j+1] = (j < g.nside/2) ? (j*deltx) : (j*deltx - g.period)
		kco_side[j+1] = (j < g.nside/2) ? (j*deltk) : (j*deltk - 2*π*g.nside/g.period)
	end
	xco_side, kco_side
end
function getgrid{T1,T2}(g::LensePrm{1,T1,T2})
	xco_side, kco_side = getxkside(g)
	xco      = Array{Float64,1}[ xco_side ]
	kco      = Array{Float64,1}[ kco_side ]
	return xco, kco
end
function meshgrid(side_x,side_y)
    	nx = length(side_x)
    	ny = length(side_y)
    	xt = repmat(vec(side_x).', ny, 1)
    	yt = repmat(vec(side_y)  , 1 , nx)
    	return xt, yt
end
function getgrid{T1,T2}(g::LensePrm{2,T1,T2})
	xco_side, kco_side = getxkside(g)
	kco1, kco2 = meshgrid(kco_side, kco_side)
	xco1, xco2 = meshgrid(xco_side, xco_side)
	kco    = Array{Float64,2}[kco1, kco2]
	xco    = Array{Float64,2}[xco1, xco2]
	return xco, kco
end
function cls_to_cXXk{dm}(ell, cxxls, r::Array{Float64, dm})
	spl = Spline1D(ell, log(cxxls); k=1, bc="extrapolate", s=0.0)
	return squash(exp(map(spl, r)))::Array{Float64, dm}
end


import Base.show
function Base.show{dm, T1, T2}(io::IO, parms::LensePrm{dm, T1, T2})
	for vs in fieldnames(parms)
		(vs != :FFT) && (vs != :IFFT) && println(io, "$vs => $(getfield(parms,vs))")
		println("")
	end
end



# -------- converting from pixel noise std to noise per-unit pixel
σunit_to_σpixl(σunit, deltx, dm) = σunit / √(deltx ^ dm)
σpixl_to_σunit(σpixl, deltx, dm) = σpixl * √(deltx ^ dm)
function cNNkgen{dm}(r::Array{Float64,dm}, deltx; σunit=0.0, beamFWHM=0.0)
	beamSQ = exp(- (beamFWHM ^ 2) * (abs2(r) .^ 2) ./ (8 * log(2)) )
	return ones(size(r)) .* σunit .^ 2 ./ beamSQ
end


# -------- Simulate a mean zero Gaussian random field in the pixel domain given a spectral density.
function grf_sim_xk{dm, T1, T2}(cXXk::Array{Float64,dm}, p::LensePrm{dm, T1, T2})
	nsz = size(cXXk)
	dx  = p.deltx ^ dm
	zzk = √(cXXk) .* fftd(randn(nsz)./√(dx), p.deltx)
	return ifftdr(zzk, p.deltk), zzk
end



##########################################################
#=
wrap class code
=#
#############################################################
@pyimport classy
function class(;lmax = 6_000, r = 1.0, omega_b = 0.0224567, omega_cdm=0.118489, tau_reio = 0.128312, theta_s = 0.0104098, logA_s_1010 = 3.29056, n_s =  0.968602)
	cosmo = classy.Class()
	cosmo[:struct_cleanup]()
	cosmo[:empty]()
	params = Dict(
   		"output"        => "tCl, pCl, lCl",
   		"modes"         => "s,t",
   		"lensing"       => "yes",
		"l_max_scalars" => 7_500,
		"l_max_tensors" => 7_500,
        "omega_b"       => omega_b,
    	"omega_cdm"     => omega_cdm,
        "tau_reio"      => tau_reio,
        "100*theta_s"   => 100*theta_s,
        "ln10^{10}A_s"  => logA_s_1010,
        "n_s"           => n_s,
        "k_pivot"       => 0.05, ## this is k_star
		"r"             => r,
   		)
	cosmo[:set](params)
	cosmo[:compute]()
	cls_ln = cosmo[:lensed_cl](lmax)
	cls = cosmo[:raw_cl](lmax)
	rtn = Dict{Symbol, Array{Float64,1}}(
			:ell      => cls["ell"],
			:ln_tt  => cls_ln["tt"] * (10^6 * cosmo[:T_cmb]()) ^ 2,
			:ln_ee  => cls_ln["ee"] * (10^6 * cosmo[:T_cmb]()) ^ 2,
			:ln_bb  => cls_ln["bb"] * (10^6 * cosmo[:T_cmb]()) ^ 2,
			:ln_te  => cls_ln["te"] * (10^6 * cosmo[:T_cmb]()) ^ 2,
			:ln_tϕ  => cls_ln["tp"] * (10^6 * cosmo[:T_cmb]()),
			:tt 	=> cls["tt"] * (10^6 * cosmo[:T_cmb]()) ^ 2,
			:ee 	=> cls["ee"] * (10^6 * cosmo[:T_cmb]()) ^ 2,
			:bb 	=> cls["bb"] * (10^6 * cosmo[:T_cmb]()) ^ 2,
			:te 	=> cls["te"] * (10^6 * cosmo[:T_cmb]()) ^ 2,
			:tϕ 	=> cls["tp"] * (10^6 * cosmo[:T_cmb]()),
			:ϕϕ     => cls["pp"],
			:ϕψ     => 0.0.*cls["pp"],
			:ψψ     => 0.01.*cls["pp"],
		)
	return rtn
end





##########################################################
#=
Miscellaneous functions
=#
#############################################################
function radial_power{dm,T1,T2}(fk, smooth::Number, parms::LensePrm{dm,T1,T2})
	rtnk = Float64[]
	dk = parms.deltk
	kbins = collect((smooth*dk):(smooth*dk):(parms.nyq))
	for wavenumber in kbins
		indx = (wavenumber-smooth*dk) .< parms.r .<= (wavenumber+smooth*dk)
		push!(rtnk, sum(abs2(fk[indx]).* (dk.^dm)) / sum(indx))
	end
	return kbins, rtnk
end



squash{T<:Number}(x::T)         = isnan(x) ? zero(T) : isfinite(x) ? x : zero(T)
squash{T<:AbstractArray}(x::T)  = map(squash, x)::T
squash!{T<:AbstractArray}(x::T) = map!(squash, x)::T



end # module
