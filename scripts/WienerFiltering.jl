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
using PyPlot, Dierckx

# ----  set the seed
seedstart = rand(UInt64)
srand(seedstart)


# --- set grid geometry and cls
dm     = 2
nside  = nextprod([2,3,5,7], 512/2)
period = 1*nside*pi/(180*60) # nside*pi/(180*60) = 1 arcmin pixels
g      = FFTgrid(dm, period, nside)


# --- noise and signal cls
const r   = 0.3    # clbbk  has r value set to r
const r0  = 10.0  # clbb0k has r value set to r0
cls = class(r = r, r0 = r0)

fsky      = 0.5  # this should actually be (period)^2 / (4π)
σEEarcmin = √2 * 1.0 / √fsky
σBBarcmin = √2 * 1.0 / √fsky
mCls = MatrixCls(g, cls; σEEarcmin = σEEarcmin, σBBarcmin = σBBarcmin)


# --- lense
order  = 3 # Taylor lensing order
ϕx, ϕk = sim_xk(mCls.cϕϕk, g)
len    = LenseDecomp(ϕk, zeros(ϕk), g)
invlen = invlense(len, g, order)
# ToDo: test out len and invlen

# # ---- simulate the data
# ex, ek             = sim_xk(mCls.cEEk, g)
# bx, bk             = sim_xk(mCls.cBBk, g)
# qk, uk, qx, ux     = eb2qu(ek, bk, g)
#
# nex, nek           = sim_xk(mCls.cEEnoisek, g)
# nbx, nbk           = sim_xk(mCls.cBBnoisek, g)
# nqk, nuk, nqx, nux = eb2qu(nek, nbk, g)
#
# ln_cex, ln_sex, ln_cbx, ln_sbx = lense_sc(ek, bk, len, g, order)
# ln_sb0x = √r0 * (ln_sbx / √r)
# ln_cb0x = √r0 * (ln_cbx / √r)
#
# dqx = - ln_cex + √(r/r0) * ln_sb0x + nqx
# dux = - ln_sex - √(r/r0) * ln_cb0x + nux
#
#
# #############################
#
# # do a full Gibbs pass
#
# #############################
#
# pmask  = trues(size(g.r))  #   g.r .< round(Int, g.nyq * 0.15)
# ebmask = trues(size(g.r))   # ebmask =  g.r .< round(Int, g.nyq * 0.99)
#
# # --- initialize
# gv_curr = GibbsVariables(g, r0, 0.2)
#
# function gibbs(iterations, dqx, dux, gv_curr, g, mCls, order, pmask, ebmask, σEEarcmin)
#     gv_prop = deepcopy(gv_curr)
#     for i = 1:iterations
#         gv_prop.ln_cex, gv_prop.ln_sex, gv_prop.ln_cb0x, gv_prop.ln_sb0x = wf(dqx, dux, gv_prop, g, mCls, order)
#         gv_prop.r      = rsampler(gv_prop, g, dqx, dux, σEEarcmin)
#         @show gv_prop.r
#         gv_prop.invlen = hmc(gv_prop, g, mCls, order, pmask, ebmask)
#     end
#     return deepcopy(gv_prop)
# end
# # --- run the gibbs chain
# gv_curr = gibbs(10, dqx, dux, gv_curr, g, mCls, order, pmask, ebmask, σEEarcmin);
# # --- plot the results
# subplot(1,3,1)
# imshow(real(g.FFT \ gv_curr.invlen.ϕk))
# colorbar()
# subplot(1,3,2)
# imshow(-real(g.FFT \ len.ϕk))
# colorbar()
# subplot(1,3,3)
# imshow(real(g.FFT \ invlen.ϕk))
#





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

gv    = GibbsVariables(g, r0,  1.2r)
#gv.ln_cex  = copy(ln_cex)
#gv.ln_sex  = copy(ln_sex)
#gv.ln_cb0x = copy(ln_cb0x)
#gv.ln_sb0x = copy(ln_sb0x)
#gv.invlen  = LenseDecomp(invlen)

gvtru = GibbsVariables(g, r0,  r) # all true values except r
gvtru.ln_cex  = copy(ln_cex)
gvtru.ln_sex  = copy(ln_sex)
gvtru.ln_cb0x = copy(ln_cb0x)
gvtru.ln_sb0x = copy(ln_sb0x)
gvtru.invlen  = LenseDecomp(invlen)

pmask  = g.r .< round(Int, g.nyq * 0.5)
ebmask = g.r .< round(Int, g.nyq * 0.5)
rs = Float64[]
for i = 1:25
    gv.ln_cex, gv.ln_sex, gv.ln_cb0x, gv.ln_sb0x = wf(dqx, dux, gvtru, g, mCls, order) #<-- true invlen
    # gv.ln_cex, gv.ln_sex, gv.ln_cb0x, gv.ln_sb0x = wf(dqx, dux, gv, g, mCls, order)
    # Would it help to do an exact Sherman-Woodbury wf of the low-ell stuff?
    # Can you attempt to make a symetric proposal to update this with the previous wf result?
    # How about using wf to do a MH proposal?
    # Are you getting the right correlation among cek, sek, cb0k, sb0k?
    # Is it possible that HMC will work here?

    # gv.invlen  = gradupdate(gv, g, mCls, order, pmask, ebmask; maxitr=10, sg1=1e-8)
    gv.invlen  = hmc(gv, g, mCls, order, pmask, ebmask)
    # Experiment with the mass vector (m1, m2, ...) (maybe it is proposing too large low ell modes?)
    # Program the β version with momentum flips
    # Program the look ahead version

    # @show gvtru.r = gv.r = rsampler(gv, g, dqx, dux, σEEarcmin)
    @show gvtru.r = gv.r = rsampler(gv, g, mCls, r0, dqx, dux, ebmask)
    push!(rs, gv.r)
end

figure(figsize = (15,5))
subplot(1,3,1)
imshow(real(g.FFT \ gv.invlen.ϕk));colorbar()
subplot(1,3,2)
imshow(real(g.FFT \ invlen.ϕk));colorbar()
subplot(1,3,3)
plt[:hist](rs, 25)


#figure()
#rs = [rsampler(gv, g, dqx, dux, σEEarcmin) for i=1:200]
#plt[:hist](rs, 25)

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
