#= ###################################

Test code for Wiener Filtering step

ToDo:

    - Merge with script1.jl
    - make an HMC file in src which defines the leapfrog operator and momentum flip
    - impliment HMC with new operators and include β ≂̸ 1
    - also impliment Look-ahead HMC

=# ####################################

# --- load modules
using BayesLensSPTpol
using PyPlot

# ----  set the seed
seedstart = rand(UInt64)
srand(seedstart)


# --- set grid geometry and cls
dm     = 2
nside  = nextprod([2,3,5,7], 256)
period = 5*nside*pi/(180*60) # nside*pi/(180*60) = 1 arcmin pixels
g      = FFTgrid(dm, period, nside)


# --- noise and signal cls
const r   = 0.2    # clbbk  has r value set to r
const r0  = 200.0  # clbb0k has r value set to r0
cls = class(r = r, r0 = r0)

fsky      = 1.0  # should this be (period)^2 / (4π)
σEEarcmin = √2 * 0.2 / √fsky
σBBarcmin = √2 * 0.2 / √fsky
mCls = MatrixCls(g, cls; σEEarcmin = σEEarcmin, σBBarcmin = σBBarcmin)


# --- lense
order  = 3 # Taylor lensing order
ϕx, ϕk = sim_xk(mCls.cϕϕk, g)
len    = LenseDecomp(ϕk, zeros(ϕk), g)
invlen = invlense(len, g, order)



################################

# Testing/debugging

#################################



# ---- test hmc + wf
len    = LenseDecomp(ϕk, zeros(ϕk), g)
ex, ek             = sim_xk(mCls.cEEk, g)
bx, bk             = sim_xk(mCls.cBBk, g)
qk, uk, qx, ux     = eb2qu(ek, bk, g)
ln_cex, ln_sex, ln_cbx, ln_sbx = lense_sc(ek, bk, len, g, order)
ln_sb0x = √r0 * (ln_sbx / √r)
ln_cb0x = √r0 * (ln_cbx / √r)
nex, nek           = sim_xk(mCls.cEEnoisek, g)
nbx, nbk           = sim_xk(mCls.cBBnoisek, g)
nqk, nuk, nqx, nux = eb2qu(nek, nbk, g)
dqx = - ln_cex + √(r/r0) * ln_sb0x + nqx
dux = - ln_sex - √(r/r0) * ln_cb0x + nux

# initialize gibbs variables
gv    = GibbsVariables(g, r0,  0.2)

# set to all the simulation truth values
gvtru = GibbsVariables(g, r0,  r)
gvtru.ln_cex  = copy(ln_cex)
gvtru.ln_sex  = copy(ln_sex)
gvtru.ln_cb0x = copy(ln_cb0x)
gvtru.ln_sb0x = copy(ln_sb0x)
gvtru.invlen  = LenseDecomp(invlen)

rs = Float64[gv.r]
for i = 1:50
    # ---- wf ----
    #gv.ln_cex, gv.ln_sex, gv.ln_cb0x, gv.ln_sb0x = wf(dqx, dux, gvtru, g, mCls, order) # <-- true invlen
    # gv.ln_cex, gv.ln_sex, gv.ln_cb0x, gv.ln_sb0x = wf(dqx, dux, gv, g, mCls, order)
    # Would it help to do an exact Sherman-Woodbury wf of the low-ell stuff?
    # Can you attempt to make a symetric proposal to update this with the previous wf result?
    # How about using wf to do a MH proposal?
    # Are you getting the right correlation among cek, sek, cb0k, sb0k?
    # Is it possible that HMC will work here?
    # Try fixing r to the true value here to see if it helps
    # Take a look at the spectral densities of E and B and see where the problem is
    # what if you break this into two steps... CEx, SEx | CB0x, lnSB0x...

    gv.ln_cb0x, gv.ln_sb0x = wf_given_e(dqx, dux, gv.r0, gv.r, gv.ln_cex, gv.ln_sex, gv.invlen, g, mCls, order)
    #gv.ln_cb0x, gv.ln_sb0x = wf_given_e(dqx, dux, gv.r0, gv.r, ln_cex, ln_sex, gv.invlen, g, mCls, order) #<-- true ln_cex, ln_sex
    #gv.ln_cb0x, gv.ln_sb0x = wf_given_e(dqx, dux, gv.r0, gv.r, ln_cex, ln_sex, invlen, g, mCls, order) #<-- true invlen, ln_cex, ln_sex

    gv.ln_cex, gv.ln_sex   = wf_given_b(dqx, dux, gv.r0, gv.r, gv.ln_cb0x, gv.ln_sb0x, gv.invlen, g, mCls, order)
    #gv.ln_cex, gv.ln_sex   = wf_given_b(dqx, dux, gv.r0, gv.r, ln_cb0x, ln_sb0x, gv.invlen, g, mCls, order) #<-- true ln_cb0x, ln_sb0x


    #gvtruelense = GibbsVariables(gv.ln_cex, gv.ln_sex, gv.ln_cb0x, gv.ln_sb0x, invlen, gv.r, gv.r0)
    #gv.ln_cex, gv.ln_sex   = wf_given_b(dqx, dux, gvtruelense, g, mCls, order)
    # !!! try first order de-lensing with ex as a template


    # --- hmc ---
    pmask  = g.r .< round(Int, g.nyq * 0.3)
    ebmask = g.r .< round(Int, g.nyq * 0.9)
    i<=50 && (@time gv.invlen = gradupdate(gv, g, mCls, order, pmask, ebmask; maxitr=25, ϵϕ=5e-8, ϵψ=0.0)) # initialize with a few gradient updates
    i>=51 && (@time gv.invlen  = hmc(gv, g, mCls, order, pmask, ebmask, maxitr=5, ϵ=2.0e-5))
    # Experiment with the mass vector (m1, m2, ...) (maybe it is proposing too large low ell modes?)
    # Program the β version with momentum flips
    # Program the look ahead version
    # Try a joint hmc on E, B, ϕ for the ancillary chain.

    # --- r ---
    ebmask_r = g.r .< 1500
    @show gvtru.r = gv.r = rsampler(gv, g, mCls, r0, dqx, dux, ebmask_r)
    push!(rs, gv.r)
    # Take a detailed look at the spectrums to see if you can spot where the excess r is coming from
    # Try replacing the r sampler with a MLE estimate.
end

figure(figsize = (17,4))
subplot(1,3,1)
imshow(real(g.FFT \ gv.invlen.ϕk));colorbar(format="%.0e")
subplot(1,3,2)
imshow(real(g.FFT \ invlen.ϕk));colorbar(format="%.0e")
subplot(1,3,3)
plt[:hist](rs, normed = true)


#figure()
#rs = [rsampler(gv, g, dqx, dux, σEEarcmin) for i=1:200]
#plt[:hist](rs, 25)


# take a look at the estimated B and E mode
test_ln_ex, test_ln_bx, test_ln_ek, test_ln_bk = sceb2eb(gv.ln_cex, gv.ln_sex, √(gv.r/gv.r0)*gv.ln_cb0x, √(gv.r/gv.r0)*gv.ln_sb0x, g)
test_cex, test_sex,  test_cbx, test_sbx = lense_sc(test_ln_ek, test_ln_bk, gv.invlen, g, order)
test_ex, test_bx, test_ek, test_bk = sceb2eb(test_cex, test_sex,  test_cbx, test_sbx , g)

#test_ln_ex, test_ln_bx, test_ln_ek, test_ln_bk = sceb2eb(ln_cex, ln_sex, ln_cbx, ln_sbx, g)
kbins, est_ln_cbbk = radial_power(test_ln_bk, 1, g)
kbins, est_ln_ceek = radial_power(test_ln_ek, 1, g)
kbins, est_cbbk = radial_power(test_bk, 1, g)
kbins, est_ceek = radial_power(test_ek, 1, g)
figure()
#loglog(kbins, kbins .* (kbins + 1) .* est_ln_cbbk / 2π, ".", label = L"est lense $C_l^{BB}$")
#loglog(kbins, kbins .* (kbins + 1) .* est_ln_ceek / 2π, ".", label = L"est lense $C_l^{EE}$")
loglog(kbins, kbins .* (kbins + 1) .* est_cbbk / 2π, ".", label = L"est $C_l^{BB}$")
loglog(kbins, kbins .* (kbins + 1) .* est_ceek / 2π, ".", label = L"est $C_l^{EE}$")
loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_bb] / 2π, label = L"lense $C_l^{BB}$")
loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_ee] / 2π, label = L"lense $C_l^{EE}$")
loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:bb] / 2π, label = L"primordial $C_l^{BB}$")
legend(loc = 3)

# --- look at the estimated lensing potential spectrum

#kbins, est_cϕϕk = radial_power(gv.invlen.ϕk, 1, g)
kbins, est_cϕϕk = radial_power(, 1, g)
kbins, cϕϕk = radial_power(ϕk, 1, g)
figure()
loglog(kbins, kbins .* (kbins + 1) .* est_cϕϕk / 2π, ".")
loglog(kbins, kbins .* (kbins + 1) .* cϕϕk / 2π, ".")
loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:ϕϕ] / 2π)
legend(loc = 3)



#
# #  ---- test wf_given_e and wf_given_b
#
# fsky      = 1.0  # this should actually be (period)^2 / (4π)
# σEEarcmin = √2 * 0.1 / √fsky
# σBBarcmin = √2 * 0.1 / √fsky
# mCls = MatrixCls(g, cls; σEEarcmin = σEEarcmin, σBBarcmin = σBBarcmin)
#
# order  = 4 # Taylor lensing order
# len    = LenseDecomp(ϕk, zeros(ϕk), g)
# ex, ek             = sim_xk(mCls.cEEk, g)
# bx, bk             = sim_xk(mCls.cBBk, g)
# qk, uk, qx, ux     = eb2qu(ek, bk, g)
# ln_cex, ln_sex, ln_cbx, ln_sbx = lense_sc(ek, bk, len, g, order)
# ln_sb0x = √r0 * (ln_sbx / √r)
# ln_cb0x = √r0 * (ln_cbx / √r)
# nex, nek           = sim_xk(mCls.cEEnoisek, g)
# nbx, nbk           = sim_xk(mCls.cBBnoisek, g)
# nqk, nuk, nqx, nux = eb2qu(nek, nbk, g)
# dqx = - ln_cex + √(r/r0) * ln_sb0x + nqx
# dux = - ln_sex - √(r/r0) * ln_cb0x + nux
# gv    = GibbsVariables(g, r0,  r)
# gv.ln_cex, gv.ln_sex, gv.ln_cb0x, gv.ln_sb0x = wf(dqx, dux, gv, g, mCls, order)
# gvtru = GibbsVariables(g, r0,  r)
# gvtru.ln_cex  = copy(ln_cex)
# gvtru.ln_sex  = copy(ln_sex)
# gvtru.ln_cb0x = copy(ln_cb0x)
# gvtru.ln_sb0x = copy(ln_sb0x)
# gvtru.invlen  = LenseDecomp(invlen)
#
# # it appears that errors in ln_sex and ln_cex have serious leakage into this step.
# test_ln_cb0x, test_ln_sb0x = wf_given_e(dqx, dux, r0, r, gv.ln_cex, gv.ln_sex, invlen, g, mCls, order)
# #test_ln_cb0x, test_ln_sb0x = wf_given_e(dqx, dux, r0, r, ln_cex, ln_sex, invlen, g, mCls, order)
#
# figure(figsize = (12,8))
# subplot(2,2,1)
# imshow(test_ln_cb0x);colorbar(format="%.0e")
# subplot(2,2,2)
# imshow(ln_cb0x);colorbar(format="%.0e")
# subplot(2,2,3)
# imshow(test_ln_sb0x);colorbar(format="%.0e")
# subplot(2,2,4)
# imshow(ln_sb0x);colorbar(format="%.0e")
#
#
# test_ln_cex, test_ln_sex = wf_given_b(dqx, dux, r0, r, gv.ln_cb0x, gv.ln_sb0x, invlen, g, mCls, order)
# #test_ln_cex, test_ln_sex = wf_given_b(dqx, dux, r0, r, ln_cb0x, ln_sb0x, invlen, g, mCls, order)
# figure(figsize = (12,8))
# subplot(2,2,1)
# imshow(test_ln_cex);colorbar(format="%.0e")
# subplot(2,2,2)
# imshow(ln_cex);colorbar(format="%.0e")
# subplot(2,2,3)
# imshow(test_ln_cex);colorbar(format="%.0e")
# subplot(2,2,4)
# imshow(test_ln_cex);colorbar(format="%.0e")



# # ✓ ---- test rsampler
# len    = LenseDecomp(ϕk, zeros(ϕk), g)
# ex, ek             = sim_xk(mCls.cEEk, g)
# bx, bk             = sim_xk(mCls.cBBk, g)
# qk, uk, qx, ux     = eb2qu(ek, bk, g)
# ln_cex, ln_sex, ln_cbx, ln_sbx = lense_sc(ek, bk, len, g, order)
# ln_sb0x = √r0 * (ln_sbx / √r)
# ln_cb0x = √r0 * (ln_cbx / √r)
# nex, nek           = sim_xk(mCls.cEEnoisek, g)
# nbx, nbk           = sim_xk(mCls.cBBnoisek, g)
# nqk, nuk, nqx, nux = eb2qu(nek, nbk, g)
# dqx = - ln_cex + √(r/r0) * ln_sb0x + nqx
# dux = - ln_sex - √(r/r0) * ln_cb0x + nux
# gv = GibbsVariables(g, r0,  0.1)
# gv.r = r
# gv.ln_cex = copy(ln_cex)
# gv.ln_sex = copy(ln_sex)
# gv.ln_cb0x = copy(ln_cb0x)
# gv.ln_sb0x = copy(ln_sb0x)
# for i = 1:20
#     @show gv.r = rsampler(gv, g, dqx, dux, σEEarcmin)
# end




#
# # ✓ ---- test hmc (also check for type instability in hmc and lfrog)
# len    = LenseDecomp(ϕk, zeros(ϕk), g)
# ex, ek             = sim_xk(mCls.cEEk, g)
# bx, bk             = sim_xk(mCls.cBBk, g)
# qk, uk, qx, ux     = eb2qu(ek, bk, g)
# ln_cex, ln_sex, ln_cbx, ln_sbx = lense_sc(ek, bk, len, g, order)
# ln_sb0x = √r0 * (ln_sbx / √r)
# ln_cb0x = √r0 * (ln_cbx / √r)
# gv = GibbsVariables(g, r0,  0.1)
# gv.r = r
# gv.ln_cex = copy(ln_cex)
# gv.ln_sex = copy(ln_sex)
# gv.ln_cb0x = copy(ln_cb0x)
# gv.ln_sb0x = copy(ln_sb0x)
# pmask  = trues(size(g.r))  #   g.r .< round(Int, g.nyq * 0.15)
# ebmask = trues(size(g.r))  #  g.r .< round(Int, g.nyq * 0.99)
# for i = 1:10
#     gv.invlen = hmc(gv, g, mCls, order, pmask, ebmask)
# end
#
# figure()
# subplot(2,1,1)
# imshow(real(g.FFT \ gv.invlen.ϕk));colorbar()
# subplot(2,1,2)
# imshow(real(g.FFT \ invlen.ϕk));colorbar()



# # ✓ ---- test lensing and delensing
# # It appears anti-lensing works as a better delenser.
# len    = LenseDecomp(ϕk, zeros(ϕk), g)
# invlen = invlense(len, g, order)
# ex, ek             = sim_xk(mCls.cEEk, g)
# bx, bk             = sim_xk(mCls.cBBk, g)
# qk, uk, qx, ux     = eb2qu(ek, bk, g)
#
# ln_qx, ln_ux    = lense(qx, ux, len, g, order)
# qxTest, uxTest  = lense(ln_qx, ln_ux, invlen, g, order)
#
# figure(figsize = (12,10))
# subplot(2,2,1)
# imshow(qx);title("qx");colorbar()
# subplot(2,2,2)
# imshow(ux);title("ux");colorbar()
# subplot(2,2,3)
# imshow(qxTest);title("qxTest");colorbar()
# subplot(2,2,4)
# imshow(uxTest);title("uxTest");colorbar()
#
# figure(figsize = (12,10))
# subplot(2,2,1)
# imshow(qx-ln_qx);title("qx-ln_qx");colorbar()
# subplot(2,2,2)
# imshow(ux-ln_ux);title("ux-ln_ux");colorbar()
# subplot(2,2,3)
# imshow(qx-qxTest);title("qx-qxTest");colorbar()
# subplot(2,2,4)
# imshow(ux-uxTest);title("ux-uxTest");colorbar()



# # ✓ ------ test the wf
# len    = LenseDecomp(ϕk, zeros(ϕk), g)
# invlen = invlense(len, g, order)
# ex, ek             = sim_xk(mCls.cEEk, g)
# bx, bk             = sim_xk(mCls.cBBk, g)
# qk, uk, qx, ux     = eb2qu(ek, bk, g)
# nex, nek           = sim_xk(mCls.cEEnoisek, g)
# nbx, nbk           = sim_xk(mCls.cBBnoisek, g)
# nqk, nuk, nqx, nux = eb2qu(nek, nbk, g)
# ln_cex, ln_sex, ln_cbx, ln_sbx = lense_sc(ek, bk, len, g, order)
# ln_sb0x = √r0 * (ln_sbx / √r)
# ln_cb0x = √r0 * (ln_cbx / √r)
# dqx = - ln_cex + √(r/r0) * ln_sb0x + nqx
# dux = - ln_sex - √(r/r0) * ln_cb0x + nux
#
# gv = GibbsVariables(g, r0,  0.1)
# gv.r = r
# gv.ln_cex = copy(ln_cex)
# gv.ln_sex = copy(ln_sex)
# gv.ln_cb0x = copy(ln_cb0x)
# gv.ln_sb0x = copy(ln_sb0x)
# gv.invlen  = LenseDecomp(invlen, g)
#
# ## ✓ ̌ test that we get copies of gv...
# # gv.invlen.ϕk[2,2]  # <--- non-zero
# # gv.invlen.ϕk[2,2] = 0
# # gv.invlen.ϕk[2,2]  # <--- now zero
# # invlen.ϕk[2,2]     # <--- should be non-zero
#
# wfln_cex, wfln_sex, wfln_cb0x, wfln_sb0x = wf(dqx, dux, gv, g, mCls, order)
#
# figure(figsize = (12,10))
# subplot(2,2,1)
# imshow(ln_cb0x);title("ln_cb0x");colorbar()
# subplot(2,2,2)
# imshow(wfln_cb0x);title("wfln_cb0x");colorbar()
# subplot(2,2,3)
# imshow(ln_sb0x);title("ln_sb0x");colorbar()
# subplot(2,2,4)
# imshow(wfln_sb0x);title("wfln_sb0x");colorbar()




#
# function sceb2qu(qx, ux, g)
#     φ2_l = 2angle(g.k[1] + im * g.k[2])
# 	# qk   = - ek .* cos(φ2_l) + bk .* sin(φ2_l)
# 	# uk   = - ek .* sin(φ2_l) - bk .* cos(φ2_l)
#     ek, bk, ex, bx = qu2eb(g.FFT * qk, g.FFT * ux, g)
# 	cex = real(g.FFT \ (ek .* cos(φ2_l)))
# 	sex = real(g.FFT \ (ek .* sin(φ2_l)))
#     cbx = real(g.FFT \ (bk .* cos(φ2_l)))
#     sbx = real(g.FFT \ (bk .* sin(φ2_l)))
# 	return cex, sex, cbx, sbx
# end





#############################

# Old stuff: constructing the covariance operators

#############################
#
# φ2_l = 2angle(g.k[1] + im * g.k[2])
# cov_cex = real(g.FFT \ ( mCls.cEEk .* cos(φ2_l).^2 ./ 2 / π))
# zspl  = fftshift(cov_cex)
# xspl  = fftshift(g.x[2])[:,1]
# yspl  = fftshift(g.x[1])[1,:][:]
# splcov_cex = Spline2D(xspl, yspl, fftshift(cov_cex); kx=3, ky=3, s=0.0)
#
# function cov_operator(splcov, v, len, g)
#     rtn = similar(v)
#     @inbounds for i in eachindex(v)
#         rtn[i] = sum(splcov.(g.x[2][i] + len.disply[i] - g.x[2] - len.disply,
#                              g.x[1][i] + len.displx[i] - g.x[1] - len.displx) .* v)
#     end
#     return rtn
# end
# function cov_operator2(splcov, v, len, g)
#
# end
#
# v = randn(size(g.x[1]))
# @time cv = cov_operator(splcov_cex, v, len, g)
#
