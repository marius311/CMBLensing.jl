
##########################################################
#=
Closures for generating ηkfun, Ckfun, C1kfun, C2kfun
=#
#############################################################

"""
Closure which generates a nonstationay phase model in ℝ^d by specifying matern C_k and tildeC_k.
"""
function gen_tangentMatern(; ν = 2.0, ρ = 1.0, σ  = 1.0, tild_ρ = 1.0, tild_ν = 2.5, t_0 = 1.0)
	Ckfun(kco) = maternk(kco; ν=ν, ρ=ρ, σ=σ)
	function ηkfun{dm}(kco::Array{Array{Float64,dm},1})
		d0 = 4ν / ρ / ρ
		dt = 4tild_ν / tild_ρ / tild_ρ
		r2  = zero(kco[1])
		for jj = 1:dm
			r2 += abs2(kco[jj])
		end
		X0  = Distributions.Beta(dm/2, ν)
		Xt  = Distributions.Beta(dm/2, tild_ν)
		F0        = Distributions.cdf(X0, r2 ./ (r2 + d0))
		invFtF0   = (1./Distributions.quantile(Xt, F0) - 1) .^ (-1/2)
		invFtF0 .*= √(dt)
		psiprime = Array{Float64,dm}[ squash( invFtF0.*kco[jj]./√(r2) ) for jj=1:dm]
		return Array{Float64,dm}[ (psiprime[jj] .- kco[jj]) ./ t_0 for jj=1:dm]
	end
	function C1kfun{dm}(kco::Array{Array{Float64,dm},1})
		local ηkval = ηkfun(kco)
		local Ckval = Ckfun(kco)
		local  rtnkimag = Array{Complex{Float64},dm}[ im .* ηkval[j] .* Ckval ./  (2π) ^ (dm/2) for j = 1:dm]
		# Note: the  1/((2π)^(dm/2)) makes C1kfun = fft of C1x
		return rtnkimag::Array{Array{Complex{Float64},dm},1}
	end
	function C2kfun{dm}(kco::Array{Array{Float64,dm},1})
		Ck  = Ckfun(kco)
		C2k = [zero(kco[1]) for i = 1:dm, j = 1:dm]
		for ii = 1:dm, jj = 1:dm
			C2k[ii,jj]   = ηkfun(kco)[ii] .* ηkfun(kco)[jj] .* Ck
			C2k[ii,jj] ./= - 2 * (2π) ^ (dm/2)
		end
		return C2k
	end

	return ηkfun::Function, Ckfun::Function, C1kfun::Function, C2kfun::Function
end



"""
Closure which generates a nonstationay phase model in ℝ^1 by specifying a band limited matern with tangent adjustment.
"""
function gen_bandlimited_tangentMatern(nyq; ν = 2.0, ρ = 1.0, σ  = 1.0, tild_ρ = 1.0, tild_ν = 2.5, t_0 = 1.0)
	function Ckfun{dm}(kco::Array{Array{Float64,dm},1})
		r       = √( sum( Array{Float64,dm}[ abs2(kco[jj]) for jj = 1:dm ] ) )
		tanr    = ((2nyq)/π) .* tan( r .* (π/(2nyq)) )
		sec2r   = abs2( sec( r .* (π/(2nyq)) ) )
		tanargs =  Array{Float64,dm}[ squash(tanr .* kco[jj] ./ r) for jj = 1:dm]
		rtnk    = maternk(tanargs; ν=ν, ρ=ρ, σ=σ)
		rtnk  .*= sec2r
		squash!(rtnk)
		return	rtnk
	end
	function ηkfun{dm}(kco::Array{Array{Float64,dm},1})
		d0 = 4ν / ρ / ρ
		dt = 4tild_ν / tild_ρ / tild_ρ
		X0  = Distributions.Beta(dm/2, ν)
		Xt  = Distributions.Beta(dm/2, tild_ν)
		r       = √( sum( Array{Float64,dm}[ abs2(kco[jj]) for jj = 1:dm ] ) )
		tanr    = ((2nyq)/π) .* tan( r .* (π/(2nyq)) )
		sec2r   = abs2( sec( r .* (π/(2nyq)) ) )
		F0        = Distributions.cdf(X0, abs2(tanr) ./ (abs2(tanr) + d0))
		invFtF0   = (1./Distributions.quantile(Xt, F0) - 1) .^ (-1/2)
		invFtF0 .*= √(dt)
		invFtF0   = ((2nyq)/π) .* atan( invFtF0 .* (π/(2nyq)) )
		psiprime = Array{Float64,dm}[ squash(invFtF0.*kco[jj]./r) for jj=1:dm]
		return Array{Float64,dm}[ (psiprime[jj] .- kco[jj]) ./ t_0 for jj=1:dm]
	end
	function C1kfun{dm}(kco::Array{Array{Float64,dm},1})
		local ηkval = ηkfun(kco)
		local Ckval = Ckfun(kco)
		local  rtnkimag = Array{Complex{Float64},dm}[ im .* ηkval[j] .* Ckval ./  (2π) ^ (dm/2) for j = 1:dm]
		# Note: the  1/((2π)^(dm/2)) makes C1kfun = fft of C1x
		return rtnkimag::Array{Array{Complex{Float64},dm},1}
	end
	function C2kfun{dm}(kco::Array{Array{Float64,dm},1})
		Ck  = Ckfun(kco)
		C2k = [zero(kco[1]) for i = 1:dm, j = 1:dm]
		for ii = 1:dm, jj = 1:dm
			C2k[ii,jj]   = ηkfun(kco)[ii] .* ηkfun(kco)[jj] .* Ck
			C2k[ii,jj] ./= - 2 * (2π) ^ (dm/2)
		end
		return C2k
	end
	return ηkfun::Function, Ckfun::Function, C1kfun::Function, C2kfun::Function
end




##########################################################
#=
The quadratic estimate, Aℓ, Cℓvar and Cℓbias
=#
#############################################################


"""
Computes the quadratic estimate *with* normalization Aℓ
"""
function estϕkfun{dm,T1,T2}(zk, parms::LensePrm{dm,T1,T2})
	rtnk   = unnormalized_estϕkfun(zk, parms)
    rtnk ./= invAℓfun(parms)
	squash!(rtnk)
    return rtnk
end



"""
Computes the quadratic estimate before normalization with Aℓ
"""
function unnormalized_estϕkfun{dm,T1,T2}(zk, parms::LensePrm{dm,T1,T2})
	deltx, deltk   = parms.deltx, parms.deltk
	zkinvCZZ       = similar(zk)
	zkinvCZZ_x_2im = similar(zk)
	for inx in eachindex(zkinvCZZ)
		tmp                 = zk[inx] / parms.CZZmobsk[inx]
		zkinvCZZ[inx]       = isnan(tmp) ? Complex(0.0) : isfinite(tmp) ? tmp : Complex(0.0)
		zkinvCZZ_x_2im[inx] = isnan(tmp) ? Complex(0.0) : isfinite(tmp) ? (2*im*tmp) : Complex(0.0)
	end
	Ax   =  parms.IFFT * (zkinvCZZ)
	Bpx  = zeros(Complex{Float64}, size(zk))
	rtnk = zeros(Complex{Float64}, size(zk))
	for p = 1:dm
		Bpx[:] = parms.IFFT * (zkinvCZZ_x_2im .* imag(parms.C1k[p]))
		rtnk  += conj(parms.ξk[p]) .* (parms.FFT * (Ax .* Bpx))
	end
	scale!(rtnk, (parms.IFFTconst)^2 * parms.FFTconst)
    return rtnk
end
function unnormalized_estϕkfun{dm,T1,T2}(xk, yk, parms::LensePrm{dm,T1,T2})
	deltx, deltk = parms.deltx, parms.deltk
	xkinvCZZ     = similar(xk)
	ykinvCZZ     = similar(yk)
	for inx in eachindex(xkinvCZZ, ykinvCZZ)
		xtmp          = xk[inx] / parms.CZZmobsk[inx]
		ytmp          = yk[inx] / parms.CZZmobsk[inx]
		xkinvCZZ[inx] = isnan(xtmp) ? Complex(0.0) : isfinite(xtmp) ? xtmp : Complex(0.0)
		ykinvCZZ[inx] = isnan(ytmp) ? Complex(0.0) : isfinite(ytmp) ? ytmp : Complex(0.0)
	end
	A1x  =  parms.IFFT * xkinvCZZ
	A2x  =  parms.IFFT * ykinvCZZ
	Dpx  = zeros(Complex{Float64}, size(xk))
	Cpx  = zeros(Complex{Float64}, size(xk))
	rtnk = zeros(Complex{Float64}, size(xk))
	for p = 1:dm
		Dpx[:] = parms.IFFT * (parms.C1k[p]  .* ykinvCZZ)
		Cpx[:] = parms.IFFT * (conj(parms.C1k[p]) .* xkinvCZZ)
		rtnk   +=  conj(parms.ξk[p]) .* (parms.FFT * (A1x .* Dpx - A2x .* Cpx))
	end
	scale!(rtnk, (parms.IFFTconst)^2 * parms.FFTconst)
    return rtnk
end



"""
Computes 1/Aℓ when CXXk == CZZmobsk
"""
function invAℓfun{dm,T1,T2}(CXXk, parms::LensePrm{dm,T1,T2})
	deltx, deltk = parms.deltx, parms.deltk
	CXXinvCZZ = CXXk ./ parms.CZZmobsk ./ parms.CZZmobsk
	squash!(CXXinvCZZ)
	ABCDx = zeros(Complex{Float64}, size(CXXk))
	rtnk  = zeros(Complex{Float64}, size(CXXk))
	for p = 1:dm   # diag terms
		Apqx = ifftd( parms.C1k[p] .* conj(parms.C1k[p]) .* CXXinvCZZ, deltk)
		Bx   = ifftd(                                       CXXinvCZZ, deltk)
		Cpx  = ifftd(                       parms.C1k[p] .* CXXinvCZZ, deltk)
		Dpx  = ifftd(                 conj(parms.C1k[p]) .* CXXinvCZZ, deltk)
		for ix in eachindex(ABCDx)
			ABCDx[ix] = 2 * Apqx[ix] * Bx[ix] - Cpx[ix] * Cpx[ix] - Dpx[ix] * Dpx[ix]
		end
		rtnk += abs2(parms.ξk[p]) .* fftd(ABCDx, deltx)
	end
	for p = 1:dm, q = (p+1):(dm)   # off diag terms
		Apqx = ifftd( parms.C1k[p] .* conj(parms.C1k[q]) .* CXXinvCZZ, deltk)
		Bx   = ifftd(                                       CXXinvCZZ, deltk)
		Cpx  = ifftd(                       parms.C1k[p] .* CXXinvCZZ, deltk)
		Cqx  = ifftd(                       parms.C1k[q] .* CXXinvCZZ, deltk)
		Dpx  = ifftd(                 conj(parms.C1k[p]) .* CXXinvCZZ, deltk)
		Dqx  = ifftd(                 conj(parms.C1k[q]) .* CXXinvCZZ, deltk)
		for ix in eachindex(ABCDx)
			ABCDx[ix] = 2 * Apqx[ix] * Bx[ix] - Cpx[ix] * Cqx[ix] - Dpx[ix] * Dqx[ix]
		end
		rtnk += 2 * parms.ξk[p] .* conj(parms.ξk[q]) .* fftd(ABCDx, deltx)
	end
	return rtnk::Array{Complex{Float64},dm}
end
function invAℓfun{dm,T1,T2}(parms::LensePrm{dm,T1,T2})
	CXXk = parms.CZZmk + parms.CNNk
	return invAℓfun(CXXk, parms)
end



"""
Computes Cℓvar, i.e. the variance spectral density of the quadratic estimate
"""
function Cℓvarfun{dm,T1,T2}(CXXk, parms::LensePrm{dm,T1,T2})
	invAℓ = invAℓfun(parms)
	rtnk  = invAℓfun(CXXk, parms) ./ invAℓ ./ invAℓ
	squash!(rtnk)
	scale!(rtnk, 2 * (2π) ^ (-dm/2) )
	return real(rtnk)
end
function Cℓvarfun{dm,T1,T2}(parms::LensePrm{dm,T1,T2})
	invAℓ = invAℓfun(parms)
	rtnk  = 1.0 ./ invAℓ
	squash!(rtnk)
	scale!(rtnk,  2 * (2π) ^ (-dm/2) )
	return real(rtnk)
end





function shiftfk_by_ω{dm,T1,T2}(fk, ω, parms::LensePrm{dm,T1,T2})
	fx    = parms.IFFT * fk
	ωdx  = sum([ω[jj] .* parms.x[jj] for jj = 1:dm ])
	rtnk  = parms.FFT * ( exp(-im .* ωdx) .* fx )
	scale!(rtnk, parms.IFFTconst * parms.FFTconst)
 	return rtnk
end





##########################################################
#=
The marginal spectral density, CZZmkfun
=#
#############################################################


"""
Computes the marginal spectral density
"""
function CZZmkfun{dm}(Cϕϕk::Array{Float64,dm}, Ck, ηk, ξk, x, k, deltx, deltk)
	abs2k = sum([abs2(kdim) for kdim in k])
	index_xeq0 = findmin(abs2k)[2]
	Σθ = [zero(x[1]) for i = 1:dm, j = 1:dm]
	for ii = 1:dm, jj = 1:dm
		Cθix_θjx   = ifftdr( ξk[ii] .* conj(ξk[jj]) .* Cϕϕk , deltk)
		Cθix_θjx .*= (2π) ^ (-dm/2)
		Σθx_ij     = Cθix_θjx[index_xeq0] .- Cθix_θjx
		Σθx_ij   .*= 2
		Σθ[ii,jj]  = copy(Σθx_ij)
	end
	imx    = im .* x
	CZmx   = zeros(Complex{Float64}, size(Cϕϕk))
	tmp    = zeros(Complex{Float64}, size(Cϕϕk))
	tmp1   = zeros(Float64, size(Cϕϕk))
	for yi in eachindex(CZmx)
		tmp[:] = Complex(0.0)
		tmp1[:] = 0.0
		for dims1 = 1:dm
			BLAS.axpy!(imx[dims1][yi], k[dims1], tmp)
			for dims2 = 1:dm
				myscaleadd!(-0.5*Σθ[dims1,dims2][yi], ηk[dims1], ηk[dims2], tmp1)
			end
			BLAS.axpy!(1.0, tmp1, tmp)
		end
		CZmx[yi] = fastsumXexpY(Ck,tmp)
	end
	scale!((deltk ^ dm) * ((2π) ^ (-dm)), CZmx)
	CZZmk   = fftd(CZmx, deltx)
	scale!((2π) ^ (dm/2) , CZZmk)
	return abs(real(CZZmk))
end
function myscaleadd!(number, mat1, mat2, storage)
	@inbounds for ind in eachindex(mat1, mat2, storage)
		storage[ind] = storage[ind] + number * mat1[ind] * mat2[ind]
	end
end


##########################################################
#=
Simulation
=#
#############################################################
"""
# Simulate nonstationary phase model.
"""
function simNPhaseGRF{dm,T1,T2}(ϕx, parms::LensePrm{dm,T1,T2}, parmsHR::LensePrm{dm,T1,T2})
	ϕk     = fftd(ϕx, parms.deltx)
	boldϕx = Array{Float64,dm}[ ifftdr(parms.ξk[j] .* ϕk, parms.deltk) for j = 1:dm ]
	imboldϕx = im .* boldϕx
	imx      = im .* parms.x
	hzx_noϕ  = grfsimx(parmsHR.Ck, parmsHR.deltx, parmsHR.deltk)
	hzk_noϕ  = fftd(hzx_noϕ, parmsHR.deltx)
	zx  = zeros(Complex{Float64}, size(ϕx))
	tmp = zeros(Complex{Float64}, size(hzk_noϕ))
	for yi in eachindex(zx)
		tmp[:] = Complex(0.0)
		for dims = 1:dm   # can you loop unroll this? would it help?
			BLAS.axpy!(imboldϕx[dims][yi], parmsHR.ηk[dims], tmp)
			BLAS.axpy!(imx[dims][yi], parmsHR.k[dims], tmp)
		end
		zx[yi] = fastsumXexpY(hzk_noϕ, tmp)
	end
	zx   .*= (parmsHR.deltk ^ dm) / (2π) ^ (dm/2)
	zk     = fftd(real(zx), parms.deltx)
	zx_noϕ = downsample(hzx_noϕ, Int64(parmsHR.nside/parms.nside))
	zkobs  = zk + fftd(grfsimx(parms.CNNk, parms.deltx, parms.deltk), parms.deltx)
	return zkobs, zk, real(zx), zx_noϕ, ϕk, ϕx
end
function simNPhaseGRF{dm,T1,T2}(parms::LensePrm{dm,T1,T2}, parmsHR::LensePrm{dm,T1,T2})
	ϕx   = grfsimx(parms.Cϕϕk, parms.deltx, parms.deltk)
	BLAS.axpy!(1.0, parms.Eϕx, ϕx)
	zkobs, zk, zx, zx_noϕ, ϕk, ϕx = simNPhaseGRF(ϕx, parms, parmsHR)
	return zkobs, zk, zx, zx_noϕ, ϕk, ϕx
end
function fastsumXexpY(hzk, tmp) # used in simNPhaseGRF
	rtnk = Complex(0.0)
	@inbounds for ind in eachindex(hzk,tmp)
		rtnk += hzk[ind] * exp(tmp[ind])
	end
	rtnk
end
downsample{T}(mat::Array{T,1}, hfcr::Int64) =  mat[1:hfcr:end]
downsample{T}(mat::Array{T,2}, hfcr::Int64) =  mat[1:hfcr:end, 1:hfcr:end]
downsample{T}(mat::Array{T,3}, hfcr::Int64) =  mat[1:hfcr:end, 1:hfcr:end, 1:hfcr:end]
# Simulate a mean zero Gaussian random field in the pixel domain given a spectral density.
function grfsimx{T,dm}(Ckvec::Array{T,dm}, deltx, deltk)
	nsz = size(Ckvec)
	dx  = deltx ^ dm
	zzk = √(Ckvec) .* fftd(randn(nsz)./√(dx), deltx)
	return ifftdr(zzk, deltk)::Array{Float64,dm}
end



#  converting from pixel noise std to noise per-unit pixel
σunit_to_σpixl(σunit, deltx, dm) = σunit / √(deltx ^ dm)
σpixl_to_σunit(σpixl, deltx, dm) = σpixl * √(deltx ^ dm)
function CNNkfun{dm}(k::Array{Array{Float64,dm},1}, deltx; σpixl=0.0, beamFWHM=0.0)
	local absk2  = mapreduce(abs2, +, k)::Array{Float64,dm}
	local beamSQ = exp(- (beamFWHM ^ 2) * (absk2 .^ 2) ./ (8 * log(2)) )
	return ones(size(k[1])) .* σpixl_to_σunit(σpixl, deltx, dm) .^ 2 ./ beamSQ
end


function maternk{dm}(kco::Array{Array{Float64,dm},1}; ν=1.1, ρ=1.0, σ=1.0)
    d1 = 4ν / ρ / ρ
    cu = ((2π) ^ dm) * (σ ^ 2) * gamma(ν + dm/2) * ((4ν) ^ ν)
	ed = (π ^ (dm/2)) * gamma(ν) * (ρ ^ (2ν))
	# note...the extra ((2π) ^ (dm)) is so the integral equals σ^2 = ∫ C_k dk/((2π) ^ (dm))
	absk2  = mapreduce(abs2, +, kco)::Array{Float64,dm}
	rtn = (cu / ed) ./ ((d1 +  absk2) .^ (ν + dm/2))
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
		push!(rtnk, sum(fk[indx]) / sum(indx))
	end
	return kbins, rtnk
end



squash{T<:Number}(x::T)         = isnan(x) ? zero(T) : isfinite(x) ? x : zero(T)
squash{T<:AbstractArray}(x::T)  = map(squash, x)::T
squash!{T<:AbstractArray}(x::T) = map!(squash, x)::T


function isgoodfreq{dm}(prop_of_nyq, deltx, kco::Array{Array{Float64,dm},1})
	# for generating a boolean mask
	magk = √(mapreduce(abs2, +, kco))::Array{Float64,dm}
	isgood = (magk .<= (prop_of_nyq * π / deltx)) & (magk .> 0.0)
	return isgood
end

function tent(x::Real, lend=π/2, uend=3π/2)
	# looks like this _/\_ with derivative ±1.`
	midd = lend + (uend - lend) / 2
	rtn  = 	(x ≤ lend) ? 0.0 :
			(x ≤ midd) ? ( x - lend) :
			(x ≤ uend) ? (-x + uend) : 0.0
	return rtn
end
function tent{T<:Real}(x::Array{T,1}, lend=π/2, uend=3π/2)
	return map(xr->tent(xr, lend, uend), x)
end


function Dθx_2_ϕx{dm,T1,T2}(Dθx, parms::LensePrm{dm,T1,T2})
	Dθk = fftd(Dθx, parms.deltx)
	θk  = squash(Dθk ./ (im .* parms.k[1]))
	ϕk  = squash(θk ./ parms.ξk[1])
	ϕx = ifftdr(ϕk, parms.deltk)
	return ϕx
end
function ϕx_2_Dθx{dm,T1,T2}(ϕx, parms::LensePrm{dm,T1,T2})
	ϕk = fftd(ϕx, parms.deltx)
	θk  = ϕk .* parms.ξk[1]
	Dθk  = θk .* (im .* parms.k[1])
	Dθx = ifftdr(Dθk, parms.deltk)
	return Dθx
end
function ϕk_2_Dθx{dm,T1,T2}(ϕk, parms::LensePrm{dm,T1,T2})
	θk  = ϕk .* parms.ξk[1]
	Dθk  = θk .* (im .* parms.k[1])
	Dθx = ifftdr(Dθk, parms.deltk)
	return Dθx
end
function ϕk_2_θx{dm,T1,T2}(ϕk, parms::LensePrm{dm,T1,T2})
	θk  = ϕk .* parms.ξk[1]
	θx = ifftdr(θk, parms.deltk)
	return θx
end




