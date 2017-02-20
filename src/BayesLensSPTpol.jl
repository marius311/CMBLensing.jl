module BayesLensSPTpol

using PyCall, Dierckx, StatsBase

export
	# Custom types
	FFTgrid,
	MatrixCls,
	LenseDecomp,
	GibbsVariables,

	# Samplers
	hmc, rsampler, wf, gradupdate,
	wf_given_b, wf_given_e,

	# Lensing operations
	lense, invlense, lense_sc,

	# Other methods
	class,
	sim_xk,
	radial_power,
	qu2eb, eb2qu, sceb2qu, sceb2eb,
	squash, squash!


####### src code

FFTW.set_num_threads(Sys.CPU_CORES)

#=  To lint this file run:
using Lint
lintfile("src/BayesLensSPTpol.jl")
=#

#=##########################################################

Custom type Definitions

=##############################################################

immutable FFTgrid{dm, T}
	period::T
	nside::Int64
	Δx::T
	Δℓ::T
	nyq::T
	x::Array{T,1}
	k::Array{T,1}
	r::Array{T,dm}
	sincos2ϕ::Tuple{Array{T,dm},Array{T,dm}}
	FFT::FFTW.rFFTWPlan{T,-1,false,dm}  # saved plan for fast fft
end


#---- Holds the cls expanded out to the 2 d spectral matrices.
# ToDo: It might also be nice to hold σEEarcmin and σBBarcmin as well.
immutable MatrixCls{dm}
	cϕϕk::Array{Float64,dm}
	cϕψk::Array{Float64,dm}
	cψψk::Array{Float64,dm}
	cTTk::Array{Float64,dm}
	cTEk::Array{Float64,dm}
	cEEk::Array{Float64,dm}
	cBBk::Array{Float64,dm}
	cBB0k::Array{Float64,dm}
	cTTnoisek::Array{Float64,dm}
	cEEnoisek::Array{Float64,dm}
	cBBnoisek::Array{Float64,dm}
end


# ----- Holds the decomposition of the lensing displacements
immutable LenseDecomp
	indcol::Array{Int64,2}
	indrow::Array{Int64,2}
	rdisplx::Array{Float64,2}
	rdisply::Array{Float64,2}
	displx::Array{Float64,2}
	disply::Array{Float64,2}
	ϕk::Array{Complex{Float64},2}
	ψk::Array{Complex{Float64},2}
end



# ----- Holds the gibbs variables
type GibbsVariables
	ln_cex::Array{Float64,2}
	ln_sex::Array{Float64,2}
	ln_cb0x::Array{Float64,2}
	ln_sb0x::Array{Float64,2}
	invlen::LenseDecomp
	r::Float64
	r0::Float64  # this is fixed and doesn't change in each gibbs pass
end



##########################################################

# Type constructors

##############################################################

function FFTgrid{T<:Real}(::Type{T}, dm, period, nside; flags=FFTW.ESTIMATE, timelimit=5)
	Δx       = period/nside
	Δℓ       = 2π/period
	nyq      = 2π/(2Δx)
	x,k      = getxkside(Δx,Δℓ,period,nside)
	r        = sqrt.(.+((reshape(k.^2, (s=ones(Int,dm); s[i]=nside; tuple(s...))) for i=1:dm)...))
	ϕ        = angle.(k .+ im*k')
	sincos2ϕ = sin(2ϕ), cos(2ϕ)
	FFT      = (Δx/√(2π))^dm * plan_fft(rand(T,fill(nside,dm)...); flags=flags, timelimit=timelimit)
	FFTgrid{dm,T}(period, nside, Δx, Δℓ, nyq, x, k, r, sincos2ϕ, FFT)
end


function MatrixCls{dm,T}(g::FFTgrid{dm,T}, cls; σTTrad=0.0, σEErad=0.0,  σBBrad=0.0, beamFWHM=0.0)
	cϕϕk = cls_to_cXXk(cls[:ell], cls[:ϕϕ], g.r)
	cϕψk = cls_to_cXXk(cls[:ell], cls[:ϕψ], g.r)
	cψψk = cls_to_cXXk(cls[:ell], cls[:ψψ], g.r)
	cTTk = cls_to_cXXk(cls[:ell], cls[:tt], g.r)
	cTEk = cls_to_cXXk(cls[:ell], cls[:te], g.r)
	cEEk = cls_to_cXXk(cls[:ell], cls[:ee], g.r)
	cBBk = cls_to_cXXk(cls[:ell], cls[:bb], g.r)
	cBB0k= cls_to_cXXk(cls[:ell], cls[:bb0], g.r)
	cTTnoisek = cNNkgen(g.r; σrad=σTTrad, beamFWHM=beamFWHM)
	cEEnoisek = cNNkgen(g.r; σrad=σEErad, beamFWHM=beamFWHM)
	cBBnoisek = cNNkgen(g.r; σrad=σBBrad, beamFWHM=beamFWHM)
	MatrixCls{dm}(cϕϕk, cϕψk, cψψk, cTTk, cTEk, cEEk, cBBk, cBB0k, cTTnoisek, cEEnoisek, cBBnoisek)
end



function LenseDecomp(ϕk, ψk, g)
	#displx = real(g.FFT \ (im .* g.k[1] .* ϕk) +  g.FFT \ (im .* g.k[2] .* ψk))
	#disply = real(g.FFT \ (im .* g.k[2] .* ϕk) -  g.FFT \ (im .* g.k[1] .* ψk))
	displx, disply = LenseDecomp_helper1(ϕk, ψk, g)

	row, col  = size(g.x[1])
	indcol    = Array(Int64, row, col)
	indrow	  = Array(Int64, row, col)
	rdisplx   = Array(Float64, row, col)
	rdisply   = Array(Float64, row, col)
	@inbounds for j = 1:col, i = 1:row
		round_displx_deltx = round(Int64, displx[i,j]/g.Δx)
		round_disply_deltx = round(Int64, disply[i,j]/g.Δx)
	    indcol[i,j]  = indexwrap(j + round_displx_deltx, col)
	    indrow[i,j]  = indexwrap(i + round_disply_deltx, row)
		rdisplx[i,j] = displx[i,j] - g.Δx * round_displx_deltx
	    rdisply[i,j] = disply[i,j] - g.Δx * round_disply_deltx
	end
	return LenseDecomp(indcol, indrow, rdisplx, rdisply, displx, disply, copy(ϕk), copy(ψk))
end
function LenseDecomp_helper1(ϕk, ψk, g)
	tmpdxk = Array(Complex{Float64}, size(g.r))
	tmpdyk = Array(Complex{Float64}, size(g.r))
	@inbounds @simd for i in eachindex(tmpdxk, tmpdyk)
		tmpdxk[i] = complex(im * g.k[1][i] * ϕk[i] + im * g.k[2][i] * ψk[i])
		tmpdyk[i] = complex(im * g.k[2][i] * ϕk[i] - im * g.k[1][i] * ψk[i])
	end
	displx = real(g.FFT \ tmpdxk)
	disply = real(g.FFT \ tmpdyk)
	return displx, disply
end

LenseDecomp(len::LenseDecomp, g) = LenseDecomp(copy(len.ϕk), copy(len.ψk), g)

function GibbsVariables(g, r0,  r)
	n, m    = size(g.r)
	ln_cex  = zeros(Float64, n, m)
	ln_sex  = zeros(Float64, n, m)
	ln_cb0x = zeros(Float64, n, m)
	ln_sb0x = zeros(Float64, n, m)
	invlen  = LenseDecomp(zeros(Complex{Float64}, n, m), zeros(Complex{Float64}, n, m), g)
	return GibbsVariables(ln_cex, ln_sex, ln_cb0x, ln_sb0x, invlen, r, r0)
end



##########################################################

# Helper functions for the type constructors

##############################################################

indexwrap(ind::Int64, uplim)  = mod(ind - 1, uplim) + 1


function cls_to_cXXk{dm}(ell, cxxls, r::Array{Float64, dm})
		spl = Dierckx.Spline1D(ell, cxxls; k=1, bc="zero", s=0.0)
		rtn = squash(map(spl, r))::Array{Float64, dm}
		rtn[r.==0.0] = 0.0
		return squash(map(spl, r))::Array{Float64, dm}
end


function cNNkgen{dm}(r::Array{Float64,dm}; σrad=0.0, beamFWHM=0.0)
	beamSQ = exp(- (beamFWHM ^ 2) * abs2(r) ./ (8 * log(2)) )
	return ones(size(r)) .* abs2(σrad) ./ beamSQ
end



function getgrid{T}(g::FFTgrid{2,T})
	xco_side, kco_side = getxkside(g)
	kco1, kco2 = meshgrid(kco_side, kco_side)
	xco1, xco2 = meshgrid(xco_side, xco_side)
	kco    = Array{Float64,2}[kco1, kco2]
	xco    = Array{Float64,2}[xco1, xco2]
	return xco, kco
end


function getxkside(Δx,Δℓ,period,nside)
	xco_side, kco_side = zeros(nside), zeros(nside)
	for j in 0:(nside-1)
		xco_side[j+1] = (j < nside/2) ? (j*Δx) : (j*Δx - period)
		kco_side[j+1] = (j < nside/2) ? (j*Δℓ) : (j*Δℓ - 2π*nside/period)
	end
	xco_side, kco_side
end


function meshgrid(side_x,side_y)
	nx = length(side_x)
	ny = length(side_y)
	xt = repmat(vec(side_x).', ny, 1)
	yt = repmat(vec(side_y)  , 1 , nx)
	return xt, yt
end



##########################################################

# Lensing functions

##############################################################


""" Lense qx, ux:  `rqx, rux = lense(qx, ux, len, g, order=2, qk=g.FFT*qx, uk=g.FFT*ux)` """
function lense{T}(
			qx::Matrix{Float64},
			ux::Matrix{Float64},
			len,
			g::FFTgrid{2,T},
			order::Int64 = 2,
			qk::Matrix{Complex{Float64}} = g.FFT * qx,
			uk::Matrix{Complex{Float64}} = g.FFT * ux,
	)
	rqx, rux  = intlense(qx, ux, len)  # <--- return values
	@inbounds for n in 1:order, α₁ in 0:n
		# kα   = im ^ n .* g.k[1] .^ α₁ .* g.k[2] .^ (n - α₁)
		kα  = intlense_helper1(n, α₁, g.k[1], g.k[2])

		∂α_qx = real(g.FFT \ (kα .* qk))
		∂α_ux = real(g.FFT \ (kα .* uk))
		∂α_qx, ∂α_ux  = intlense(∂α_qx, ∂α_ux, len)

		# xα   = len.rdisplx .^ α₁ .* len.rdisply .^ (n - α₁) ./ factorial(α₁) ./ factorial(n - α₁)
		# rqx += xα .* ∂α_qx
		# rux += xα .* ∂α_ux
		intlense_helper2!(rqx, rux, n, α₁, len.rdisplx, len.rdisply, ∂α_qx, ∂α_ux)
    end
    return rqx, rux
end
function intlense_helper1(n, α₁, k1, k2)
	rtn  = Array(Complex{Float64}, size(k1))
	imn, nmα₁  = im ^ n, n - α₁
	@inbounds @simd for i in eachindex(rtn)
		rtn[i] = complex(imn * k1[i] ^ α₁ * k2[i] ^ nmα₁)
	end
	return rtn
end
function intlense_helper2!(rqx, rux, n, α₁, rx, ry, ∂qx, ∂ux)
	fα₁, fnmα₁, nmα₁ = factorial(α₁), factorial(n - α₁), n - α₁
	@inbounds @simd for i in eachindex(rqx, rux)
		xα      = rx[i] ^ α₁ * ry[i] ^ nmα₁ / fα₁ / fnmα₁
		rqx[i] += xα * ∂qx[i]
		rux[i] += xα * ∂ux[i]
	end
	return nothing
end



function intlense(qx, ux, len)
	rqx  = similar(qx)
	rux  = similar(ux)
    @inbounds for i in eachindex(rqx, rux)
            rqx[i] = qx[len.indrow[i], len.indcol[i]]
            rux[i] = ux[len.indrow[i], len.indcol[i]]
    end
    return rqx, rux
end

function invlense(len, g, order)
    # invϕx, _ = lense(real(g.FFT\(-len.ϕk)), zeros(len.displx), len, g, order)
    # invlen   = LenseDecomp(g.FFT * invϕx, zeros(len.displx), g)

	# the following just uses AntiLensing which appears to work better that the above
    invlen   = LenseDecomp(-len.ϕk, zeros(Complex{Float64}, size(len.displx)), g)
	return invlen
end



""" 'cex, sex, cbx, sbx  = lense_sc(ek, bk, len, g, order)' """
function lense_sc{T}(ek, bk, len, g::FFTgrid{2,T}, order::Int64 = 2)
		qk_ee, uk_ee, qx_ee, ux_ee = eb2qu(ek, zeros(size(bk)), g)
		qk_bb, uk_bb, qx_bb, ux_bb = eb2qu(zeros(size(bk)), bk, g)
		ln_qx_ee, ln_ux_ee = lense(qx_ee, ux_ee, len, g, order)
		ln_qx_bb, ln_ux_bb = lense(qx_bb, ux_bb, len, g, order)
		cex = -ln_qx_ee
		sex = -ln_ux_ee
		cbx = -ln_ux_bb
		sbx =  ln_qx_bb
		return cex, sex, cbx, sbx
end




##########################################################

# gibbs conditional samplers

##############################################################

include("gibbs_conditional_samplers.jl")




##########################################################

# Miscellaneous functions

##############################################################

@pyimport classy

function class(;ϕscale = 1.0, ψscale = 0.0, lmax = 6_000, r = 0.2, r0 = 100.0, omega_b = 0.0224567, omega_cdm=0.118489, tau_reio = 0.128312, theta_s = 0.0104098, logA_s_1010 = 3.29056, n_s =  0.968602)
	cosmo = classy.Class()
	cosmo[:struct_cleanup]()
	cosmo[:empty]()
	params = Dict(
   		"output"        => "tCl, pCl, lCl",
   		"modes"         => "s,t",
   		"lensing"       => "yes",
		"l_max_scalars" => lmax + 500,
		"l_max_tensors" => lmax + 500, #lmax + 500,
      	"omega_b"       => omega_b,
    	"omega_cdm"     => omega_cdm,
      	"tau_reio"      => tau_reio,
      	"100*theta_s"   => 100*theta_s,
      	"ln10^{10}A_s"  => logA_s_1010,
      	"n_s"           => n_s,
		"r"             => r,
        #"k_pivot"      => 0.05,
		#"k_step_trans" => 0.1, # 0.01 for super high resolution
   		#"l_linstep"    => 10, # 1 for super high resolution
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
			:ϕϕ   => ϕscale.*cls["pp"],
			:ϕψ   => 0.0.*cls["pp"],
			:ψψ   => ψscale.*cls["pp"],
			:bb0  => cls["bb"] * (10^6 * cosmo[:T_cmb]()) ^ 2 * (r0 / r),
		)
	return rtn
end



""" Convert qu to eb:  `ek, bk, ex, bx = qu2eb(qk, uk, g)` """
function qu2eb(qk, uk, g)
	φ2_l = 2angle(g.k[1] + im * g.k[2])
	ek   = - qk .* cos(φ2_l) - uk .* sin(φ2_l)
	bk   =   qk .* sin(φ2_l) - uk .* cos(φ2_l)
	ex   = real(g.FFT \ ek)
	bx   = real(g.FFT \ bk)
	return ek, bk, ex, bx
end


""" Convert eb to qu: `qk, uk, qx, ux = eb2qu(ek, bk, g)` """
function eb2qu(ek, bk, g)
	φ2_l = 2angle(g.k[1] + im * g.k[2])
	qk   = - ek .* cos(φ2_l) + bk .* sin(φ2_l)
	uk   = - ek .* sin(φ2_l) - bk .* cos(φ2_l)
    qx   = real(g.FFT \ qk)
	ux   = real(g.FFT \ uk)
	return qk, uk, qx, ux
end


""" Convert SE, CE, SB, CB to 'qx, ux = sceb2qu(SE, CE, SB, CB, g)' """
function sceb2qu(cex, sex, cbx, sbx, g)
	qx = - cex + sbx
	ux = - sex - cbx
	qk = g.FFT * qx
	uk = g.FFT * ux
	return qx, ux, qk, uk
end


function sceb2eb(cex, sex, cbx, sbx, g)
	qx = - cex + sbx
	ux = - sex - cbx
	qk = g.FFT * qx
	uk = g.FFT * ux
	ek, bk, ex, bx = qu2eb(qk, uk, g)
	return ex, bx, ek, bk
end

""" kbins, rtnk  = radial_power(fk, smooth, g) """
function radial_power{dm,T}(fk, smooth::Number, g::FFTgrid{dm,T})
	rtnk = Float64[]
	dk = g.Δℓ
	kbins = collect((smooth*dk):(smooth*dk):(g.nyq))
	for wavenumber in kbins
		indx = (wavenumber-smooth*dk) .< g.r .<= (wavenumber+smooth*dk)
		push!(rtnk, sum(abs2(fk[indx]).* (dk.^dm)) / sum(indx))
	end
	return kbins, rtnk
end


# -------- converting from pixel noise std to noise per-unit pixel
σunit_to_σpixl(σunit, Δx, dm) = σunit / √(Δx ^ dm)
σpixl_to_σunit(σpixl, Δx, dm) = σpixl * √(Δx ^ dm)

# -------- Simulate a mean zero Gaussian random field in the pixel domain given a spectral density.
function sim_xk{dm, T}(cXXk::Array{Float64,dm}, g::FFTgrid{dm, T})
	wx, wk = white_wx_wk(g)
	zk = √(cXXk) .* wk
	zx = real(g.FFT \ zk)
	return zx, zk
end

# ----- white noise
function white_wx_wk{dm, T}(g::FFTgrid{dm, T})
	dx  = g.Δx ^ dm
	wx = randn(size(g.r)) ./ √(dx)
	wk = g.FFT * wx
	return wx, wk
end


squash{T<:Number}(x::T)  = isnan(x) ? zero(T) : isfinite(x) ? x : zero(T)
function squash!{dm,T}(x::Array{T,dm}, mask::BitArray{dm}=trues(size(x)))
	@inbounds @simd for i in eachindex(x)
		if !mask[i] || !isfinite(x[i]) || isnan(x[i])
			x[i] = zero(T)
		end
	end
	return x
end
function squash{dm,T}(x::Array{T,dm}, mask::BitArray{dm}=trues(size(x)))
	y = copy(x)
	return squash!(y, mask)
end





end
