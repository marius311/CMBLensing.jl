



#############################

#  wiener filtering + conditional simulation

#############################

function wf(dqx, dux, r0, r_curr, invlen_curr, g, mCls, order)
    # delense dqx, dux
    invdqx, invdux  = lense(dqx, dux, invlen_curr, g, order)
    invdek, invdbk, = qu2eb(g.FFT * invdqx, g.FFT * invdux, g)

    # conditionally simulate eksim and bksim
    eksim  = invdek .* mCls.cEEk ./ (mCls.cEEk + mCls.cEEnoisek)
    eksim += BayesLensSPTpol.white_wx_wk(g)[2] ./ √(1 ./ mCls.cEEk + 1 ./ mCls.cEEnoisek)
    squash!(eksim)

    cBBk = (r_curr/r0) * mCls.cBB0k
    bksim  = invdbk .* cBBk ./ (cBBk + mCls.cBBnoisek)
    bksim += BayesLensSPTpol.white_wx_wk(g)[2] ./ √(1 ./ cBBk + 1 ./ mCls.cBBnoisek)
    squash!(bksim)

    # now re-lense by generating fwdlen inverting invlen
    fwdlen = invlense(invlen_curr, g, order)
    simln_cex, simln_sex,  simln_cbx, simln_sbx = lense_sc(eksim, bksim, fwdlen, g, order)
    simln_cb0x = √(r0/r_curr) * simln_cbx
    simln_sb0x = √(r0/r_curr) * simln_sbx
    return simln_cex, simln_sex, simln_cb0x, simln_sb0x
end

wf(dqx, dux, gv::GibbsVariables, g, mCls, order) = wf(dqx, dux, gv.r0, gv.r, gv.invlen, g, mCls, order)



function wf_given_e(dqx, dux, r0, r, ln_cex, ln_sex, invlen, g, mCls, order)
    # delense dqx, dux after subtracting out ex
    invdqx, invdux  = lense(dqx + ln_cex, dux + ln_sex , invlen, g, order)
    invdek, invdbk, = qu2eb(g.FFT * invdqx, g.FFT * invdux, g)

    # conditionally simulate bksim
    cBBk = (r / r0) * mCls.cBB0k
    bksim  = invdbk .* cBBk ./ (cBBk + mCls.cBBnoisek)
    bksim += BayesLensSPTpol.white_wx_wk(g)[2] ./ √(1 ./ cBBk + 1 ./ mCls.cBBnoisek)
    squash!(bksim)

    # now re-lense by generating fwdlen inverting invlen
    fwdlen = invlense(invlen, g, order)
    tmp1, tmp2,  simln_cbx, simln_sbx = lense_sc(zeros(bksim), bksim, fwdlen, g, order)
    simln_cb0x = √(r0/r) * simln_cbx
    simln_sb0x = √(r0/r) * simln_sbx
    return simln_cb0x, simln_sb0x
end


function wf_given_b(dqx, dux, r0, r, ln_cb0x, ln_sb0x, invlen, g, mCls, order)
    # delense dqx, dux after subtracting out bx
    invdqx, invdux  = lense(dqx - √(r/r0) .* ln_sb0x,
                            dux + √(r/r0) .* ln_cb0x, invlen, g, order)
    invdek, invdbk, = qu2eb(g.FFT * invdqx, g.FFT * invdux, g)

    # conditionally simulate eksim
    eksim  = invdek .* mCls.cEEk ./ (mCls.cEEk + mCls.cEEnoisek)
    eksim += BayesLensSPTpol.white_wx_wk(g)[2] ./ √(1 ./ mCls.cEEk + 1 ./ mCls.cEEnoisek)
    squash!(eksim)

    # now re-lense by generating fwdlen inverting invlen
    fwdlen = invlense(invlen, g, order)
    simln_cex, simln_sex, = lense_sc(eksim, zeros(eksim), fwdlen, g, order)
    return simln_cex, simln_sex
end


#=###########################################

rsampler

=############################################



#function rsampler(gv::GibbsVariables, g, qx_obs, ux_obs, σEEarcmin)
#	return rsampler(gv.ln_cex, gv.ln_sex, gv.ln_cb0x, gv.ln_sb0x, gv.r0, g, qx_obs, ux_obs, σEEarcmin)
#end

# function rsampler{T}(
# 			ln_cex::Matrix{Float64},
# 			ln_sex::Matrix{Float64},
# 			ln_cb0x::Matrix{Float64},
# 			ln_sb0x::Matrix{Float64},
# 			r0,
# 			g::FFTgrid{2,T},
# 			qx_obs::Matrix{Float64},
# 			ux_obs::Matrix{Float64},
# 			σEEarcmin,
# 	)
# 	Nsmp	 = 500
# 	σpixel = (σEEarcmin/60./57.4)/g.deltx
# 	r_prop = linspace(0.00001, 0.5, Nsmp)
# 	chi2   = zeros(size(r_prop))
# 	for i = 1:Nsmp
# 			qx_rsd = qx_obs + ln_cex - √(r_prop[i]/r0) .* ln_sb0x
# 			ux_rsd = ux_obs + ln_sex + √(r_prop[i]/r0) .* ln_cb0x
# 			chi2[i]= (sumabs2(qx_rsd) + sumabs2(ux_rsd)) / σpixel^2
# 	end
# 	prob = exp( (minimum(chi2) - chi2) / 2)
# 	wv   = WeightVec(prob[:])
# 	return sample(r_prop, wv)
# end

function rsampler(gv, g, mCls, r0, dqx, dux, ebmask)
    dqk, duk         = g.FFT * dqx,     g.FFT * dux
    ln_cek, ln_sek   = g.FFT * gv.ln_cex,  g.FFT * gv.ln_sex
    ln_cb0k, ln_sb0k = g.FFT * gv.ln_cb0x, g.FFT * gv.ln_sb0x
	φ2_l = 2angle(g.k[1] + im * g.k[2])
	dek   = - dqk .* cos(φ2_l) - duk .* sin(φ2_l)
	dbk   =   dqk .* sin(φ2_l) - duk .* cos(φ2_l)
	Nsmp	 = 500
	r_prop = linspace(0.00001, 0.5, Nsmp)
	chi2   = zeros(size(r_prop))
	@inbounds for i = 1:Nsmp
			residqk = dqk + ln_cek - √(r_prop[i]/r0) .* ln_sb0k
			residuk = duk + ln_sek + √(r_prop[i]/r0) .* ln_cb0k
            residek = - residqk .* cos(φ2_l) - residuk .* sin(φ2_l)
	        residbk =   residqk .* sin(φ2_l) - residuk .* cos(φ2_l)
            chi2[i] =  0.5 * sumabs2(squash!(residek./√(mCls.cEEnoisek), ebmask)) * (g.deltk)^2
            chi2[i] += 0.5 * sumabs2(squash!(residbk./√(mCls.cBBnoisek), ebmask)) * (g.deltk)^2
	end
	prob =  exp(minimum(chi2) - chi2)
    prob .*= 1./r_prop # <--- added a Jeffreys prior
	wv   = WeightVec(prob[:])
	return sample(r_prop, wv)
end





###############################################

# Hamiltonian Markov Chain code

###############################################


function hmc(gv::GibbsVariables, g, mCls, order, pmask, ebmask; maxitr::Int64 = 25, ϵ::Float64 = 1.0e-5)
	#ln_qx0, ln_ux0 = sceb2qu(gv.ln_cex, gv.ln_sex, gv.ln_cb0x, gv.ln_sb0x, g)
    #invlen_update = hmc(gv.invlen, ln_qx0, ln_ux0, g, mCls, order=order, pmask=pmask, ebmask=ebmask)
    invlen_update = hmc(gv.invlen, - gv.ln_cex + gv.ln_sb0x, - gv.ln_sex - gv.ln_cb0x, g, mCls, order=order, pmask=pmask, ebmask=ebmask, maxitr=maxitr, ϵ=ϵ)
    return invlen_update
end



function hmc{T}(
			len_curr::LenseDecomp,
			qx::Matrix{Float64},
			ux::Matrix{Float64},
			g::FFTgrid{2,T},
			mCls::MatrixCls{2},
			qk::Matrix{Complex{Float64}} = g.FFT * qx,
			uk::Matrix{Complex{Float64}} = g.FFT * ux
			;
            maxitr::Int64 = 25,
            ϵ::Float64   = 1.0e-5,
			order::Int64 = 2,
			pmask::BitArray{2}  = trues(size(g.r)),
			ebmask::BitArray{2} = trues(size(g.r)),
	)
	#ϵ         = 1.0e-5*rand()

    # !!!! mk needs tuning
	# mk = g.deltk^2 * 1.0e-2 ./ mCls.cϕϕk
    mk  = g.deltk^2 * 1.0e-2 ./ mCls.cϕϕk ./ (0.75 + 0.25 * tanh((g.r-1500)./200))
    squash!(mk, pmask)

	pk        = (g.deltk / g.deltx) * (g.FFT * randn(size(g.r))) .* √(mk) # note that the variance of real(pk_init) and imag(pk_init) is mk/2
	loglk	  = loglike(len_curr, qx, ux, g, mCls, order=order, pmask=pmask, ebmask=ebmask)
	h_at_zero = 0.5 * sum( squash!( abs2(pk)./(2*mk/2), pmask) ) - loglk # the 0.5 is out front since only half the sum is unique
	# println("h_at_zero = $(round(h_at_zero)), loglk = $(round(loglk)), kinetic = $(round(h_at_zero+loglk))")

	loglk, len_prop, pk = lfrog(pk, ϵ, mk, maxitr, len_curr, g, qx, ux, qk, uk, mCls, order, pmask, ebmask)
	h_at_end = 0.5 * sum( squash!(abs2(pk)./(2*mk/2), pmask) ) - loglk # the 0.5 is out front since only half the sum is unique
	# println("h_at_end = $(round(h_at_end)), loglk = $(round(loglk)), kinetic = $(round(h_at_end+loglk))")

	prob_accept = minimum([1, exp(h_at_zero - h_at_end)])
	if rand() < prob_accept
	    println("Accept: prob_accept = $(round(prob_accept,4)), h_at_end = $(round(h_at_end)), h_at_zero = $(round(h_at_zero)), loglike = $(round(loglk))")
	    return LenseDecomp(len_prop, g) # ensures a copy is made
	else
	    println("Reject: prob_accept = $(round(prob_accept,4)), h_at_end = $(round(h_at_end)), h_at_zero = $(round(h_at_zero)), loglike = $(round(loglk))")
	    return LenseDecomp(len_curr, g) # ensures a copy is made
	end
	return nothing
end

function lfrog(pk, ϵ, mk, maxitr, len_curr, g, qx, ux, qk, uk, mCls, order, pmask, ebmask)
		φ2_l = 2angle(g.k[1] + im * g.k[2])
		Mq   = -0.5squash!(abs2(cos(φ2_l)) ./ mCls.cEEk  + abs2(sin(φ2_l)) ./ mCls.cBB0k, ebmask)
		Mu   = -0.5squash!(abs2(cos(φ2_l)) ./ mCls.cBB0k  + abs2(sin(φ2_l)) ./ mCls.cEEk, ebmask)
		Mqu  = -0.5squash!(2cos(φ2_l) .* sin(φ2_l) ./ mCls.cEEk, ebmask)
		Mqu -= -0.5squash!(2cos(φ2_l) .* sin(φ2_l) ./ mCls.cBB0k, ebmask)

		∂1uk = im * g.k[1] .* uk
		∂1qk = im * g.k[1] .* qk
		∂2uk = im * g.k[2] .* uk
		∂2qk = im * g.k[2] .* qk

		∂1ux = real(g.FFT \ ∂1uk)
		∂1qx = real(g.FFT \ ∂1qk)
		∂2ux = real(g.FFT \ ∂2uk)
		∂2qx = real(g.FFT \ ∂2qk)

		inv_mk = squash!(1./ (mk ./ 2.0), pmask)

		for i = 1:maxitr
			ϕgradk, ψgradk = ϕψgrad(len_curr, qx, ux, qk, uk, ∂1qx, ∂1ux, ∂1qk, ∂1uk, ∂2qx, ∂2ux, ∂2qk, ∂2uk, g, Mq, Mu, Mqu, mCls, order)
    		pk_halfstep =  pk + ϵ .* ϕgradk ./ 2.0
			ϕcurrk      = len_curr.ϕk + ϵ .* inv_mk .* pk_halfstep
			ψcurrk      = len_curr.ψk
			len_curr    = LenseDecomp(ϕcurrk, ψcurrk, g)
			ϕgradk, ψgradk = ϕψgrad(len_curr, qx, ux, qk, uk, ∂1qx, ∂1ux, ∂1qk, ∂1uk, ∂2qx, ∂2ux, ∂2qk, ∂2uk, g, Mq, Mu, Mqu, mCls, order)
			pk    = pk_halfstep + ϵ .* ϕgradk ./ 2.0
			loglk = loglike(len_curr, qx, ux, g, mCls, order=order, pmask=pmask, ebmask=ebmask)
			kintc = 0.5 * sum( squash!( abs2(pk)./(2*mk/2), pmask) )
			# println("h_at_$i = $(round(kintc-loglk)), loglk = $(round(loglk)), kinetic = $(round(kintc))")
		end

		loglk = loglike(len_curr, qx, ux, g, mCls, order=order, pmask=pmask, ebmask=ebmask)
		return loglk, len_curr, pk
end



# --- Gradient computations


function gradupdate(gv::GibbsVariables, g, mCls, order, pmask, ebmask; maxitr::Int64 = 1, ϵϕ::Float64 = 1e-8, ϵψ::Float64 = 1e-10)
    invlen_update = gradupdate(gv.invlen, - gv.ln_cex + gv.ln_sb0x, - gv.ln_sex - gv.ln_cb0x, g, mCls,
            order=order, pmask=pmask, ebmask=ebmask, maxitr=maxitr, sg1=ϵϕ, sg2=ϵψ)
    return invlen_update
end


function gradupdate{T}(
			len::LenseDecomp,
			qx::Matrix{Float64},
			ux::Matrix{Float64},
			g::FFTgrid{2,T},
			mCls::MatrixCls{2},
			qk::Matrix{Complex{Float64}} = g.FFT * qx,
			uk::Matrix{Complex{Float64}} = g.FFT * ux
			;
			maxitr::Int64 = 1,
			sg1::Float64 = 1e-8,
			sg2::Float64 = 1e-10,
			order::Int64 = 2,
			pmask::BitArray{2}  = trues(size(g.r)),
			ebmask::BitArray{2} = trues(size(g.r)),
	)
	φ2_l = 2angle(g.k[1] + im * g.k[2])
	Mq   = -0.5squash!(abs2(cos(φ2_l)) ./ mCls.cEEk  + abs2(sin(φ2_l)) ./ mCls.cBB0k, ebmask)
	Mu   = -0.5squash!(abs2(cos(φ2_l)) ./ mCls.cBB0k  + abs2(sin(φ2_l)) ./ mCls.cEEk, ebmask)
	Mqu  = -0.5squash!(2cos(φ2_l) .* sin(φ2_l) ./ mCls.cEEk, ebmask)
	Mqu -= -0.5squash!(2cos(φ2_l) .* sin(φ2_l) ./ mCls.cBB0k, ebmask)

	∂1uk = im * g.k[1] .* uk
	∂1qk = im * g.k[1] .* qk
	∂2uk = im * g.k[2] .* uk
	∂2qk = im * g.k[2] .* qk

	∂1ux = real(g.FFT \ ∂1uk)
	∂1qx = real(g.FFT \ ∂1qk)
	∂2ux = real(g.FFT \ ∂2uk)
	∂2qx = real(g.FFT \ ∂2qk)

	ϵ1 = squash!(sg1 .* mCls.cϕϕk, pmask)
	ϵ2 = squash!(sg2 .* mCls.cψψk, pmask)
	ϕcurrk, ψcurrk = copy(len.ϕk), copy(len.ψk)

    @inbounds for cntr = 1:maxitr
        ϕgradk, ψgradk = ϕψgrad(len, qx, ux, qk, uk, ∂1qx, ∂1ux, ∂1qk, ∂1uk, ∂2qx, ∂2ux, ∂2qk, ∂2uk, g, Mq, Mu, Mqu, mCls, order)
		ϕcurrk[:] = ϕcurrk + ϕgradk .* ϵ1
        ψcurrk[:] = ψcurrk + ψgradk .* ϵ2
		len = LenseDecomp(ϕcurrk, ψcurrk, g)
    end
	return len
end




function ϕψgrad(len, qx, ux, qk, uk, ∂1qx, ∂1ux, ∂1qk, ∂1uk, ∂2qx, ∂2ux, ∂2qk, ∂2uk, g, Mq, Mu, Mqu, mCls, order = 2)
	lqx, lux     = lense(qx, ux, len, g, order, qk, uk)
	l∂1qx, l∂1ux = lense(∂1qx, ∂1ux, len, g, order, ∂1qk, ∂1uk)
	l∂2qx, l∂2ux = lense(∂2qx, ∂2ux, len, g, order, ∂2qk, ∂2uk)

	lqk, luk     =  g.FFT*lqx, g.FFT*lux
	# l∂1qk, l∂1uk =  g.FFT*l∂1qx, g.FFT*l∂1ux
	# l∂2qk, l∂2uk =  g.FFT*l∂2qx, g.FFT*l∂2ux

	ϕ∇qqk, ψ∇qqk = ϕψgrad_terms(lqk, lqk, l∂1qx, l∂1qx, l∂2qx, l∂2qx, Mq, g)
    ϕ∇uuk, ψ∇uuk = ϕψgrad_terms(luk, luk, l∂1ux, l∂1ux, l∂2ux, l∂2ux, Mu, g)
    ϕ∇quk, ψ∇quk = ϕψgrad_terms(lqk, luk, l∂1qx, l∂1ux, l∂2qx, l∂2ux, Mqu, g)

	rtnϕk = ϕ∇qqk + ϕ∇uuk + ϕ∇quk - 2 * g.deltk ^ 2 * squash!(len.ϕk ./ mCls.cϕϕk)
	rtnψk = ψ∇qqk + ψ∇uuk + ψ∇quk - 2 * g.deltk ^ 2 * squash!(len.ψk ./ mCls.cψψk)
    return  rtnϕk, rtnψk
end


function ϕψgrad_terms(xk, yk, ∂1xx, ∂1yx, ∂2xx, ∂2yx, M, g)
    X₁YMx = ∂1xx .* (g.FFT \ (yk .* M))
    X₂YMx = ∂2xx .* (g.FFT \ (yk .* M))
    Y₁XMx = ∂1yx .* (g.FFT \ (xk .* M))
    Y₂XMx = ∂2yx .* (g.FFT \ (xk .* M))

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



# --- compute loglike ... note, this uses cBB0k
function loglike(len, qx, ux, g, mCls; order::Int64=2, pmask::BitArray{2}=trues(size(g.r)), ebmask::BitArray{2}=trues(size(g.r)) )
	ln_qx, ln_ux = lense(qx, ux, len, g, order) #<-- this is testing len as a delenser
	ln_ek, ln_bk, ln_ex, ln_bx = qu2eb(g.FFT*ln_qx, g.FFT*ln_ux, g)
	rloglike   = - 0.5 * sum( squash!( abs2(ln_ek)  ./ mCls.cEEk, ebmask) )
	rloglike  += - 0.5 * sum( squash!( abs2(ln_bk)  ./ mCls.cBB0k, ebmask) )
	rloglike  += - 0.5 * sum( squash!( abs2(len.ϕk) ./ mCls.cϕϕk, pmask) )
	rloglike  += - 0.5 * sum( squash!( abs2(len.ψk) ./ mCls.cψψk, pmask) )
	rloglike  *= (g.deltk^2)
	return rloglike
end
