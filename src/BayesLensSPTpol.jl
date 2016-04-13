module BayesLensSPTpol

using PyCall, Dierckx

export 	FFTgrid,
		MatrixCls,
		LenseDecomp,
		lense,
		gradupdate,
		loglike,
		class,
		sim_xk,
		radial_power,
		qu2eb,
		eb2qu,
		squash,
		squash!

FFTW.set_num_threads(CPU_CORES)


#=  To lint this file run:
using Lint
lintfile("src/BayesLensSPTpol.jl")
=#

#=##########################################################

Custom type Definitions

=##############################################################

# ---- Holds grid, model and planned FFT parameters for the quadratic estimate.
immutable FFTgrid{dm, T}
	period::Float64
	nside::Int64
	deltx::Float64
	deltk::Float64
	nyq::Float64
	x::Array{Array{Float64,dm},1}
	k::Array{Array{Float64,dm},1}
	r::Array{Float64,dm}
	FFT::T  # saved plan for fast fft
end


#---- Holds the cls expanded out to the 2 d spectral matrices.
immutable MatrixCls{dm}
	cϕϕk::Array{Float64,dm}
	cϕψk::Array{Float64,dm}
	cψψk::Array{Float64,dm}
	cTTk::Array{Float64,dm}
	cTEk::Array{Float64,dm}
	cEEk::Array{Float64,dm}
	cBBk::Array{Float64,dm}
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



#=##########################################################

Type constructors

=##############################################################

function FFTgrid(dm, period, nside)
	dm_nsides = fill(nside,dm)   # [nside,...,nside] <- dm times
	deltx     = period / nside
	deltk     = 2π / period
	nyq       = 2π / (2deltx)
	x         = [fill(NaN, dm_nsides...) for i = 1:dm]
	k         = [fill(NaN, dm_nsides...) for i = 1:dm]
	r         =  fill(NaN, dm_nsides...)
	tmp       = rand(Complex{Float64},dm_nsides...)
	unnormalized_FFT = plan_fft(tmp; flags = FFTW.PATIENT, timelimit = 1)
	FFT = complex( (deltx / √(2π))^dm ) * unnormalized_FFT
	FFT \ tmp   # <-- initialize fast ifft
	g = FFTgrid{dm, typeof(FFT)}(period, nside, deltx, deltk, nyq, x, k, r, FFT)
	g.x[:], g.k[:] = getgrid(g)
	g.r[:]  =  √(sum([abs2(kdim) for kdim in g.k]))
	return g
end


function MatrixCls{dm,T}(g::FFTgrid{dm,T}, cls; σTTarcmin=0.0, σEEarcmin=0.0,  σBBarcmin=0.0, beamFWHM=0.0)
	cϕϕk = cls_to_cXXk(cls[:ell], cls[:ϕϕ], g.r)
	cϕψk = cls_to_cXXk(cls[:ell], cls[:ϕψ], g.r)
	cψψk = cls_to_cXXk(cls[:ell], cls[:ψψ], g.r)
	cTTk = cls_to_cXXk(cls[:ell], cls[:tt], g.r)
	cTEk = cls_to_cXXk(cls[:ell], cls[:te], g.r)
	cEEk = cls_to_cXXk(cls[:ell], cls[:ee], g.r)
	cBBk = cls_to_cXXk(cls[:ell], cls[:bb], g.r)
	cTTnoisek = cNNkgen(g.r; σunit=σTTarcmin, beamFWHM=beamFWHM)
	cEEnoisek = cNNkgen(g.r; σunit=σEEarcmin, beamFWHM=beamFWHM)
	cBBnoisek = cNNkgen(g.r; σunit=σBBarcmin, beamFWHM=beamFWHM)
	MatrixCls{dm}(cϕϕk, cϕψk, cψψk, cTTk, cTEk, cEEk, cBBk, cTTnoisek, cEEnoisek, cBBnoisek)
end


function LenseDecomp(ϕk, ψk, g)
	# the following is probably in the wrong order for the updates...
	displx = real(g.FFT \ (im .* g.k[1] .* ϕk) +  g.FFT \ (im .* g.k[2] .* ψk))
	disply = real(g.FFT \ (im .* g.k[2] .* ϕk) -  g.FFT \ (im .* g.k[1] .* ψk))
	row, col  = size(g.x[1])
	indcol    = Array(Int64, row, col)
	indrow	  = Array(Int64, row, col)
	rdisplx   = Array(Float64, row, col)
	rdisply   = Array(Float64, row, col)
	@inbounds for j = 1:col, i = 1:row
	    indcol[i,j]  = indexwrap(j + round(Int64, displx[i,j]/g.deltx), col)
	    indrow[i,j]  = indexwrap(i + round(Int64, disply[i,j]/g.deltx), row)
		rdisplx[i,j] = displx[i,j] - g.deltx * round(Int64, displx[i,j]/g.deltx)
	    rdisply[i,j] = disply[i,j] - g.deltx * round(Int64, disply[i,j]/g.deltx)
	end
	return LenseDecomp(indcol, indrow, rdisplx, rdisply, displx, disply, ϕk, ψk)
end



#=##########################################################

Helper functions for the type constructors

=##############################################################

indexwrap(ind::Int64, uplim)  = mod(ind - 1, uplim) + 1


function cls_to_cXXk{dm}(ell, cxxls, r::Array{Float64, dm})
	spl = Spline1D(ell, cxxls; k=1, bc="zero", s=0.0)
	return squash(map(spl, r))::Array{Float64, dm}
end


# !!!! check this one...
function cNNkgen{dm}(r::Array{Float64,dm}; σunit=0.0, beamFWHM=0.0)
	beamSQ = exp(- (beamFWHM ^ 2) * (abs2(r) .^ 2) ./ (8 * log(2)) )
	return ones(size(r)) .* σunit .^ 2 ./ beamSQ
end


function getgrid{T}(g::FFTgrid{2,T})
	xco_side, kco_side = getxkside(g)
	kco1, kco2 = meshgrid(kco_side, kco_side)
	xco1, xco2 = meshgrid(xco_side, xco_side)
	kco    = Array{Float64,2}[kco1, kco2]
	xco    = Array{Float64,2}[xco1, xco2]
	return xco, kco
end


function getxkside{dm,T}(g::FFTgrid{dm,T})
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


function meshgrid(side_x,side_y)
    	nx = length(side_x)
    	ny = length(side_y)
    	xt = repmat(vec(side_x).', ny, 1)
    	yt = repmat(vec(side_y)  , 1 , nx)
    	return xt, yt
end



#=##########################################################

Lensing functions

=##############################################################

""" Lense qx, ux:  `rqx, rux = lense(qx, ux, len, g, order = 2)` """
function lense{T}(
			qx::Matrix{Float64},
			ux::Matrix{Float64},
			len,
			g::FFTgrid{2,T},
			order::Int64 = 2,
			qk::Matrix{Complex{Float64}} = g.FFT * qx,
			uk::Matrix{Complex{Float64}} = g.FFT * ux
	)
	rqx, rux  = intlense(qx, ux, len)  # <--- return values
	@inbounds for n in 1:order, α₁ in 0:n
		kα   = im ^ n .* g.k[1] .^ α₁ .* g.k[2] .^ (n - α₁)
		∂α_qx = real(g.FFT \ (kα .* qk))
		∂α_ux = real(g.FFT \ (kα .* uk))
		∂α_qx, ∂α_ux  = intlense(∂α_qx, ∂α_ux, len)

		xα   = len.rdisplx .^ α₁ .* len.rdisply .^ (n - α₁)
		xα ./= factorial(α₁) * factorial(n - α₁)

		rqx += xα .* ∂α_qx
		rux += xα .* ∂α_ux
    end
    return rqx, rux
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





################################################
#
# Gradient update functions
#
################################################



function gradupdate{dm,T}(
			len, qx, ux, g::FFTgrid{dm, T}, mCls::MatrixCls{dm};
			maxitr::Int64 = 1,
			sg1::Float64 = 1e-8,
			sg2::Float64 = 1e-10,
			order::Int64 = 2,
			pmask::Int64 = 1000,
			ebmask::Int64 = 4000
			)
	φ2_l = 2angle(g.k[1] + im * g.k[2])
	Mq   = -0.5squash(abs2(cos(φ2_l)) ./ mCls.cEEk  + abs2(sin(φ2_l)) ./ mCls.cBBk )
	Mu   = -0.5squash(abs2(cos(φ2_l)) ./ mCls.cBBk  + abs2(sin(φ2_l)) ./ mCls.cEEk )
	Mqu  = -0.5squash(2cos(φ2_l) .* sin(φ2_l) ./ mCls.cEEk)
	Mqu -= -0.5squash(2cos(φ2_l) .* sin(φ2_l) ./ mCls.cBBk)
	Mq[g.r .>= ebmask]  = 0.0
	Mu[g.r .>= ebmask]  = 0.0
	Mqu[g.r .>= ebmask] = 0.0
	ϕcurrk, ψcurrk = copy(len.ϕk), copy(len.ψk)
    for cntr = 1:maxitr
        ϕgradk, ψgradk = ϕψgrad(len, qx, ux, g, Mq, Mu, Mqu, mCls, order)
        ϕcurrk[:] = ϕcurrk + ϕgradk .* sg1 .* mCls.cϕϕk .* (g.r .< pmask )
        ψcurrk[:] = ψcurrk + ψgradk .* sg2 .* mCls.cψψk .* (g.r .< pmask )
    end
	return LenseDecomp(ϕcurrk, ψcurrk, g)
end


function ϕψgrad(len, qx, ux, g, Mq, Mu, Mqu, mCls, order = 2)
	∂1ux = real(g.FFT \ (im * g.k[1] .* (g.FFT * ux)))
	∂2ux = real(g.FFT \ (im * g.k[2] .* (g.FFT * ux)))
	∂1qx = real(g.FFT \ (im * g.k[1] .* (g.FFT * qx)))
	∂2qx = real(g.FFT \ (im * g.k[2] .* (g.FFT * qx)))

	lqx, lux     = lense(qx, ux, len, g, order)
	l∂1qx, l∂1ux = lense(∂1qx, ∂1ux, len, g, order)
	l∂2qx, l∂2ux = lense(∂2qx, ∂2ux, len, g, order)

	lqk, luk     =  g.FFT*lqx, g.FFT*lux
	l∂1qk, l∂1uk =  g.FFT*l∂1qx, g.FFT*l∂1ux
	l∂2qk, l∂2uk =  g.FFT*l∂2qx, g.FFT*l∂2ux

	ϕ∇qqk, ψ∇qqk = ϕψgrad_terms(lqk, lqk, l∂1qk, l∂1qk, l∂2qk, l∂2qk, Mq, g)
    ϕ∇uuk, ψ∇uuk = ϕψgrad_terms(luk, luk, l∂1uk, l∂1uk, l∂2uk, l∂2uk, Mu, g)
    ϕ∇quk, ψ∇quk = ϕψgrad_terms(lqk, luk, l∂1qk, l∂1uk, l∂2qk, l∂2uk, Mqu, g)

	rtnϕk = ϕ∇qqk + ϕ∇uuk + ϕ∇quk - 2 * g.deltk ^ 2 * squash(len.ϕk ./ mCls.cϕϕk)
	rtnψk = ψ∇qqk + ψ∇uuk + ψ∇quk - 2 * g.deltk ^ 2 * squash(len.ψk ./ mCls.cψψk)
    return  rtnϕk, rtnψk
end


function ϕψgrad_terms(xk, yk, ∂1xk, ∂1yk, ∂2xk, ∂2yk, M, g)
    X₁YMx = (g.FFT \ ∂1xk) .* (g.FFT \ (yk .* M))
    X₂YMx = (g.FFT \ ∂2xk) .* (g.FFT \ (yk .* M))
    Y₁XMx = (g.FFT \ ∂1yk) .* (g.FFT \ (xk .* M))
    Y₂XMx = (g.FFT \ ∂2yk) .* (g.FFT \ (xk .* M))

	ϕgradk  = g.k[1] .* (g.FFT * X₁YMx)
	ϕgradk += g.k[2] .* (g.FFT * X₂YMx)
	ϕgradk += g.k[1] .* (g.FFT * Y₁XMx)
	ϕgradk += g.k[2] .* (g.FFT * Y₂XMx)
	ϕgradk *= - im * g.deltk ^ 2

	ψgradk  = g.k[1] .* (g.FFT * Y₂XMx)
	ψgradk -= g.k[2] .* (g.FFT * Y₁XMx)
	ψgradk += g.k[1] .* (g.FFT * X₂YMx)
	ψgradk -= g.k[2] .* (g.FFT * X₁YMx)
	ψgradk *= im * g.deltk ^ 2

    return ϕgradk, ψgradk
end





#=##########################################################

Miscellaneous functions

=##############################################################

@pyimport classy

function class(;ϕscale = 0.1, ψscale = 0.1, lmax = 6_000, r = 1.0, omega_b = 0.0224567, omega_cdm=0.118489, tau_reio = 0.128312, theta_s = 0.0104098, logA_s_1010 = 3.29056, n_s =  0.968602)
	cosmo = classy.Class()
	cosmo[:struct_cleanup]()
	cosmo[:empty]()
	params = Dict(
   		"output"        => "tCl, pCl, lCl",
   		"modes"         => "s,t",
   		"lensing"       => "yes",
		"l_max_scalars" => lmax + 500,
		"l_max_tensors" => 3_000, #lmax + 500,
        "omega_b"       => omega_b,
    	"omega_cdm"     => omega_cdm,
        "tau_reio"      => tau_reio,
        "100*theta_s"   => 100*theta_s,
        "ln10^{10}A_s"  => logA_s_1010,
        "n_s"           => n_s,
		"r"             => r,
        #"k_pivot"       => 0.05,
		#"k_step_trans"  => 0.1, # 0.01 for super high resolution
   		#"l_linstep"     => 10,  # 1 for super high resolution
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
			:ϕϕ     => ϕscale.*cls["pp"],
			:ϕψ     => 0.0.*cls["pp"],
			:ψψ     => ψscale.*cls["pp"],
		)
	return rtn
end


# --- compute loglike
function loglike(len, qx, ux, g, mCls; order::Int64 = 2, pmask::Int64 = 1000, ebmask::Int64 = 4000)
	ln_qx, ln_ux = lense(qx, ux, len, g, order)
	ln_ek, ln_bk, ln_ex, ln_bx = qu2eb(g.FFT*ln_qx, g.FFT*ln_ux, g)
	rloglike   = - 0.5 * sum(squash( abs2(ln_ek .* (g.r .<= ebmask)) ./ mCls.cEEk ))
	rloglike  += - 0.5 * sum(squash( abs2(ln_bk .* (g.r .<= ebmask)) ./ mCls.cBBk ))
	rloglike  += - 0.5 * sum(squash( abs2(len.ϕk .* (g.r .<= pmask)) ./ mCls.cϕϕk ))
	rloglike  += - 0.5 * sum(squash( abs2(len.ψk .* (g.r .<= pmask)) ./ mCls.cψψk ))
	rloglike  *= (g.deltk^2)
	return rloglike
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



function radial_power{dm,T}(fk, smooth::Number, g::FFTgrid{dm,T})
	rtnk = Float64[]
	dk = g.deltk
	kbins = collect((smooth*dk):(smooth*dk):(g.nyq))
	for wavenumber in kbins
		indx = (wavenumber-smooth*dk) .< g.r .<= (wavenumber+smooth*dk)
		push!(rtnk, sum(abs2(fk[indx]).* (dk.^dm)) / sum(indx))
	end
	return kbins, rtnk
end


# -------- converting from pixel noise std to noise per-unit pixel
σunit_to_σpixl(σunit, deltx, dm) = σunit / √(deltx ^ dm)
σpixl_to_σunit(σpixl, deltx, dm) = σpixl * √(deltx ^ dm)

# -------- Simulate a mean zero Gaussian random field in the pixel domain given a spectral density.
function sim_xk{dm, T}(cXXk::Array{Float64,dm}, g::FFTgrid{dm, T})
	wx, wk = white_wx_wk(g)
	zk = √(cXXk) .* wk
	zx = real(g.FFT \ zk)
	return zx, zk
end

# ----- white noise
function white_wx_wk{dm, T}(g::FFTgrid{dm, T})
	dx  = g.deltx ^ dm
	wx = randn(size(g.r)) ./ √(dx)
	wk = g.FFT * wx
	return wx, wk
end


squash{T<:Number}(x::T)  = isnan(x) ? zero(T) : isfinite(x) ? x : zero(T)
squash{T<:AbstractArray}(x::T)  = map(squash, x)::T
squash!{T<:AbstractArray}(x::T) = map!(squash, x)::T



end # module
