module BayesLensSPTpol

using PyCall, Dierckx

export FFTgrid, MatrixCls, QUpartials, LenseDecomp, class, sim_xk, squash, squash!

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


# --- Hold spatial QU derivatives for Taylor expansion
immutable QUpartials
    qx::Array{Float64,2}
    ux::Array{Float64,2}
    ∂1qx::Array{Float64,2}
    ∂2qx::Array{Float64,2}
    ∂1ux::Array{Float64,2}
    ∂2ux::Array{Float64,2}
    ∂11qx::Array{Float64,2}
    ∂12qx::Array{Float64,2}
    ∂22qx::Array{Float64,2}
    ∂11ux::Array{Float64,2}
    ∂12ux::Array{Float64,2}
    ∂22ux::Array{Float64,2}
	ex::Array{Float64,2}
	bx::Array{Float64,2}
	ek::Array{Complex{Float64},2}
	bk::Array{Complex{Float64},2}
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
	r         = fill(NaN, dm_nsides...)
	tmp       = rand(Complex{Float64},dm_nsides...)
	unnormalized_FFT = plan_fft(tmp; flags = FFTW.PATIENT, timelimit = 5)
	FFT = complex( (deltx / √(2π))^dm ) * unnormalized_FFT
	FFT \ tmp  # <---- activate fast ifft
	g = FFTgrid{dm, typeof(FFT)}(period, nside, deltx, deltk, nyq, x, k, r, FFT)
	g.x[:], g.k[:] = getgrid(g)
	g.r[:]  =  √(sum([abs2(kdim) for kdim in g.k]))
	return g
end


function QUpartials{T}(ek::Array{Complex{Float64},2}, bk::Array{Complex{Float64},2}, g::FFTgrid{2, T})
	φ2_l = 2.0 * angle(g.k[1] + im * g.k[2])
	qk   = - ek .* cos(φ2_l) + bk .* sin(φ2_l)
	uk   = - ek .* sin(φ2_l) - bk .* cos(φ2_l)
    qx   = real(g.FFT \ qk)
	ux   = real(g.FFT \ uk)
    return QUpartials(
        qx,
        ux,
    	real(g.FFT \ (im .* g.k[1] .* qk)),
    	real(g.FFT \ (im .* g.k[2] .* qk)),
    	real(g.FFT \ (im .* g.k[1] .* uk)),
    	real(g.FFT \ (im .* g.k[2] .* uk)),
    	real(g.FFT \ (im .* g.k[1] .* g.k[1] .* qk)),
    	real(g.FFT \ (im .* g.k[1] .* g.k[2] .* qk)),
    	real(g.FFT \ (im .* g.k[2] .* g.k[2] .* qk)),
    	real(g.FFT \ (im .* g.k[1] .* g.k[1] .* uk)),
    	real(g.FFT \ (im .* g.k[1] .* g.k[2] .* uk)),
    	real(g.FFT \ (im .* g.k[2] .* g.k[2] .* uk)),
		real(g.FFT \ ek),
		real(g.FFT \ bk),
		copy(ek),
		copy(bk),
        )
end


function QUpartials{T}(g::FFTgrid{2,T})
	row, col = size(g.x[1])
    return QUpartials(
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
		Array(Float64,row, col),
		Array(Float64,row, col),
		Array(Complex{Float64},row, col),
		Array(Complex{Float64},row, col),
        )
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


function LenseDecomp(g)
	row, col = size(g.x[1])
	indcol   = Array(Int64, row, col)
	indrow   = Array(Int64, row, col)
	displx   = Array(Float64, row, col)
	disply   = Array(Float64, row, col)
	rdisplx  = Array(Float64, row, col)
	rdisply  = Array(Float64, row, col)
	ϕk = zeros(Complex{Float64}, row, col)
	ψk = zeros(Complex{Float64}, row, col)
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

wrap class code

=##############################################################
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





#=##########################################################

Lensing functions

=##############################################################
include("lensing.jl")





#=##########################################################

Miscellaneous functions

=##############################################################


# Delta update for len.
function update!{T}(len::LenseDecomp, Δϕk, Δψk, g::FFTgrid{2,T})
	len.ϕk[:] = len.ϕk + Δϕk
	len.ψk[:] = len.ψk + Δψk
	len.displx[:] = real(g.FFT \ (im .* g.k[1] .* len.ϕk) +  g.FFT \ (im .* g.k[2] .* len.ψk))
	len.disply[:] = real(g.FFT \ (im .* g.k[2] .* len.ϕk) -  g.FFT \ (im .* g.k[1] .* len.ψk))
	row, col  = size(g.x[1])
	@inbounds for j = 1:col, i = 1:row
	    len.indcol[i,j]  = indexwrap(j + round(Int64, len.displx[i,j]/g.deltx), col)
	    len.indrow[i,j]  = indexwrap(i + round(Int64, len.disply[i,j]/g.deltx), row)
	    len.rdisplx[i,j] = len.displx[i,j] - g.deltx * round(Int64, len.displx[i,j]/g.deltx)
	    len.rdisply[i,j] = len.disply[i,j] - g.deltx * round(Int64, len.disply[i,j]/g.deltx)
	end
	return Void
end

# takes Qx and Ux in qu and updates all the remaining derivatives to match
function update!{T}(qu::QUpartials, g::FFTgrid{2, T})
    qk = g.FFT * qu.qx
	uk = g.FFT * qu.ux
    qu.∂1qx[:]  = real(g.FFT \ (im .* g.k[1] .* qk))
    qu.∂2qx[:]  = real(g.FFT \ (im .* g.k[2] .* qk))
    qu.∂1ux[:]  = real(g.FFT \ (im .* g.k[1] .* uk))
    qu.∂2ux[:]  = real(g.FFT \ (im .* g.k[2] .* uk))
    qu.∂11qx[:] = real(g.FFT \ (im .* g.k[1] .* g.k[1] .* qk))
    qu.∂12qx[:] = real(g.FFT \ (im .* g.k[1] .* g.k[2] .* qk))
    qu.∂22qx[:] = real(g.FFT \ (im .* g.k[2] .* g.k[2] .* qk))
    qu.∂11ux[:] = real(g.FFT \ (im .* g.k[1] .* g.k[1] .* uk))
    qu.∂12ux[:] = real(g.FFT \ (im .* g.k[1] .* g.k[2] .* uk))
    qu.∂22ux[:] = real(g.FFT \ (im .* g.k[2] .* g.k[2] .* uk))
	φ2_l     = 2.0 * angle(g.k[1] + im * g.k[2])
	qu.ek[:] = - qk .* cos(φ2_l) - uk .* sin(φ2_l)
	qu.bk[:] =   qk .* sin(φ2_l) - uk .* cos(φ2_l)
	qu.ex[:] = real(g.FFT \ qu.ek)
	qu.bx[:] = real(g.FFT \ qu.bk)
	return Void
end


# overwrite qu_sink with qu_source
function replace!(qu_sink::QUpartials, qu_source::QUpartials)
    qu_sink.qx[:]    = qu_source.qx
    qu_sink.ux[:]    = qu_source.ux
    qu_sink.∂1qx[:]  = qu_source.∂1qx
    qu_sink.∂2qx[:]  = qu_source.∂2qx
    qu_sink.∂1ux[:]  = qu_source.∂1ux
    qu_sink.∂2ux[:]  = qu_source.∂2ux
    qu_sink.∂11qx[:] = qu_source.∂11qx
    qu_sink.∂12qx[:] = qu_source.∂12qx
    qu_sink.∂22qx[:] = qu_source.∂22qx
    qu_sink.∂11ux[:] = qu_source.∂11ux
    qu_sink.∂12ux[:] = qu_source.∂12ux
    qu_sink.∂22ux[:] = qu_source.∂22ux
	qu_sink.ex[:]    = qu_source.ex
	qu_sink.bx[:]    = qu_source.bx
	qu_sink.ek[:]    = qu_source.ek
	qu_sink.bk[:]    = qu_source.bk
	return Void
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


squash{T<:Number}(x::T)         = isnan(x) ? zero(T) : isfinite(x) ? x : zero(T)
squash{T<:AbstractArray}(x::T)  = map(squash, x)::T
squash!{T<:AbstractArray}(x::T) = map!(squash, x)::T



end # module
