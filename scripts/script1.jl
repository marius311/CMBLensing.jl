#=########################################################

Run this script with the command `include("script1.jl")`

=##########################################################

# --- load modules
using BayesLensSPTpol
using PyPlot

# ----  set the seed
seedstart = rand(UInt64)
srand(seedstart)

# --- set grid geometry
dm     = 2
period = 0.2 # radians
nside  = nextprod([2,3,5,7], 400)
g      = FFTgrid(dm, period, nside)

# --- noise parameters
σTTarcmin  = 0.02
σEEarcmin  = √2 * σTTarcmin
σBBarcmin  = √2 * σTTarcmin
beamFWHM   = 0.0

# --- Taylor series lensing order
order = 2


#=###########################################

Generate the cls

=##########################################################

# --- cls
cls = class(
    ψscale      = 0.1,    # <-- cψψk = ψscale * baseline_cϕϕk
    ϕscale      = 1.0,  # <-- cϕϕk = ϕscale * baseline_cϕϕk
    lmax        = 6_000,
    r           = 0.2,
    omega_b     = 0.0224567,
    omega_cdm   = 0.118489,
    tau_reio    = 0.128312,
    theta_s     = 0.0104098,
    logA_s_1010 = 3.29056,
    n_s         = 0.968602
)

# ---- matrix form of the cls
mCls = MatrixCls(
    g, cls;
    σTTarcmin = σTTarcmin,
    σEEarcmin = σEEarcmin,
    σBBarcmin = σBBarcmin,
    beamFWHM  = beamFWHM
)


#=###########################################

Simulate E and B, ϕ, ψ to check it is working

=##########################################################

ex, ek         = sim_xk(mCls.cEEk, g)
bx, bk         = sim_xk(mCls.cBBk, g)
qk, uk, qx, ux = eb2qu(ek, bk, g)
ϕx, ϕk         = sim_xk(mCls.cϕϕk, g)
ψx, ψk         = sim_xk(mCls.cψψk, g)


#= --- Plot: check the ex and bx maps
figure(figsize = (12,5))
subplot(1,2,1)
imshow(fftshift(ex),
    interpolation = "nearest",
    origin="lower",
    extent=[minimum(g.x[1]), maximum(g.x[1]),minimum(g.x[2]), maximum(g.x[2])],
)
colorbar()
subplot(1,2,2)
imshow(fftshift(bx),
    interpolation = "nearest",
    origin="lower",
    extent=[minimum(g.x[1]), maximum(g.x[1]),minimum(g.x[2]), maximum(g.x[2])],
)
colorbar()
=#


#=###########################################

Lense

=##########################################################

# --- decompase lense
len = LenseDecomp(ϕk, ψk, g)

# --- lense qx and ux fields
@time ln_qx, ln_ux = lense(qx, ux, len, g, order)
# @time ln_qx, ln_ux = lense(qx, ux, len, g, order, qk, uk) # this one is a bit quicker


#= --- Plot: check the lensed fields have the right power.
ln_ek, ln_bk, ln_ex, ln_bx = qu2eb(g.FFT*ln_qx, g.FFT*ln_ux, g)
kbins, est_ln_cbbk = radial_power(ln_bk, 1, g)
kbins, est_ln_ceek = radial_power(ln_ek, 1, g)
figure()
loglog(kbins, kbins .* (kbins + 1) .* est_ln_cbbk / 2π, ".", label = L"est lense $C_l^{BB}$")
loglog(kbins, kbins .* (kbins + 1) .* est_ln_ceek / 2π, ".", label = L"est lense $C_l^{EE}$")
loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_bb] / 2π, label = L"lense $C_l^{BB}$")
loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_ee] / 2π, label = L"lense $C_l^{EE}$")
legend(loc = 3)
=#


#=###########################################

Impliment likelihood gradient ascent for `invlen`

=##########################################################

# --- initialize zero lense (actually this will estimate the inverse lense)
len_curr = LenseDecomp(zeros(ϕk), zeros(ψk), g)

pmask  = trues(size(g.r))  #   g.r .< round(Int, g.nyq * 0.15)
ebmask = trues(size(g.r))   # ebmask =  g.r .< round(Int, g.nyq * 0.99)
sg1    = 1e-10              # sg1    = 1e-10  # <-- size of gradient step for ϕ
sg2    = 1e-10              # sg2    = 1e-10  # <-- size of gradient step for ψ
@show loglike(len_curr, ln_qx, ln_ux, g,  mCls, order=order, pmask=pmask, ebmask=ebmask)

# for cntr = 1:5
#    @time len_curr = gradupdate(len_curr, ln_qx, ln_ux, g, mCls; maxitr=100, sg1=sg1,sg2=sg2,order=order,pmask=pmask,ebmask=ebmask)
#    @show loglike(len_curr, ln_qx, ln_ux, g, mCls, order=order, pmask=pmask, ebmask=ebmask)
# end

for cntr = 1:25
    @time len_curr = hmc(len_curr, ln_qx, ln_ux, g, mCls, order=order, pmask=pmask, ebmask=ebmask)
end


#= --- Plot: the estimated lensing potentials
figure()
subplot(2,2,1)
imshow(real(g.FFT \ (len_curr.ϕk.*(g.r .< Inf))) )#, vmin = minimum(-ϕx), vmax = maximum(-ϕx) )
title("estimated curl free potential: phi")
colorbar()
subplot(2,2,2)
imshow(-ϕx, vmin = minimum(-ϕx), vmax = maximum(-ϕx) )
title("simulation truth curl free potential: phi")
colorbar()
subplot(2,2,3)
imshow(real(g.FFT \ (len_curr.ψk.*(g.r .< Inf))) )#, vmin = minimum(-ψx), vmax = maximum(-ψx) )
title("estimated div free potential: psi")
colorbar()
subplot(2,2,4)
imshow(-ψx, vmin = minimum(-ψx), vmax = maximum(-ψx) )
title("simulation truth div free potential: psi")
colorbar()
=#


#= --- Plot: compare lensed and unlensed B-power
delensed_qx, delensed_ux = lense(ln_qx, ln_ux, len_curr, g, order) # last arg is the order of lensing
ln_ek, ln_bk, ln_ex, ln_bx = qu2eb(g.FFT*ln_qx, g.FFT*ln_ux, g)
delensed_ek, delensed_bk, delensed_ex, delensed_bx = qu2eb(g.FFT*delensed_qx, g.FFT*delensed_ux, g)
kbins, est_ln_cbbk = radial_power(ln_bk, 1, g)
kbins, est_delensed_cbbk = radial_power(delensed_bk, 1, g)
figure()
loglog(kbins, kbins .* (kbins + 1) .* est_ln_cbbk / 2π, ".", label = "lensed band power")
loglog(kbins, kbins .* (kbins + 1) .* est_delensed_cbbk / 2π, ".", label = "delensed band power")
loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_bb] / 2π, label = L"lense $C_l^{BB}$")
loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:bb] / 2π, label = L"unlensed $C_l^{BB}$")
legend(loc = 3)
=#




#=###########################################

Test the effect of lensing on white noise.
    > If lensing is a unitary operation (like a permutation)
      then it lensed white noise should be white noise.
    > Also, check to see if there is any lensing signature in lensed noise to see
      if we treat the de-lensed observations as unlensed plus white noise.
    > It appears that as I increase the lensing order...white noise gets more invariant to lensing

=##########################################################
#
#   # ---- lensed noise
#   order = 5
#   ϕx, ϕk = sim_xk(mCls.cϕϕk, g)
#   ψx, ψk = sim_xk(mCls.cψψk, g)
#   len     = LenseDecomp(ϕk, ψk, g)
#   mClsNoise          = deepcopy(mCls)
#   mClsNoise.cEEk[:]  = mCls.cEEnoisek
#   mClsNoise.cBBk[:]  = mCls.cBBnoisek
#   nex, nek           = sim_xk(mClsNoise.cEEk, g)
#   nbx, nbk           = sim_xk(mClsNoise.cBBk, g)
#   nqk, nuk, nqx, nux = eb2qu(nek, nbk, g)
#   ln_nqx, ln_nux     = lense(nqx, nux, len, g, order)
#   ln_nek, ln_nbk, ln_nex, ln_nbx = qu2eb(g.FFT*ln_nqx, g.FFT*ln_nux, g)
#
#
#   # ---- gradient ascent
#   len_curr = LenseDecomp(zeros(ϕk), zeros(ψk), g)
#   pmask  = trues(size(g.r))   # g.r .< round(Int, g.nyq * 0.5)
#   ebmask = trues(size(g.r))   # g.r .< round(Int, g.nyq * 0.99)
#   sg1    = 1e-8              # <-- size of gradient step for ϕ
#   sg2    = 1e-8               # <-- size of gradient step for ψ
#   @show loglike(len_curr, ln_nqx, ln_nux, g,  mClsNoise, order=order, pmask=pmask, ebmask=ebmask)
#   for cntr = 1:25
#       @time len_curr = gradupdate(len_curr, ln_nqx, ln_nux, g, mClsNoise; maxitr=100, sg1=sg1,sg2=sg2,order=order,pmask=pmask,ebmask=ebmask)
#       @show loglike(len_curr, ln_nqx, ln_nux, g, mClsNoise, order=order, pmask=pmask, ebmask=ebmask)
#   end
#
#
#
#   #= --- Plot: the lensed and unlensed noise band powers
#   kbins, ln_nb_bndpwrk = radial_power(ln_nbk, 1, g)
#   kbins,    nb_bndpwrk = radial_power(   nbk, 1, g)
#   kbins, ln_ne_bndpwrk = radial_power(ln_nek, 1, g)
#   kbins,    ne_bndpwrk = radial_power(   nek, 1, g)
#
#   figure(figsize = (15,5))
#   subplot(1,2,1)
#   plot(kbins, ln_nb_bndpwrk, ".", label = "lensed b noise")
#   plot(kbins,    nb_bndpwrk, ".", label = "b noise")
#   legend()
#   subplot(1,2,2)
#   plot(kbins, ln_ne_bndpwrk, ".", label = "lensed e noise")
#   plot(kbins,    ne_bndpwrk, ".", label = "e noise")
#   legend()
#   =#
#
#
#
#   #= --- Plot: the estimated lensing potentials
#   figure()
#   subplot(2,2,1)
#   imshow(real(g.FFT \ (len_curr.ϕk.*(g.r .< Inf))) )#, vmin = minimum(-ϕx), vmax = maximum(-ϕx) )
#   title("estimated curl free potential: phi")
#   colorbar()
#   subplot(2,2,2)
#   imshow(-ϕx, vmin = minimum(-ϕx), vmax = maximum(-ϕx) )
#   title("simulation truth curl free potential: phi")
#   colorbar()
#   subplot(2,2,3)
#   imshow(real(g.FFT \ (len_curr.ψk.*(g.r .< Inf))) )#, vmin = minimum(-ψx), vmax = maximum(-ψx) )
#   title("estimated div free potential: psi")
#   colorbar()
#   subplot(2,2,4)
#   imshow(-ψx, vmin = minimum(-ψx), vmax = maximum(-ψx) )
#   title("simulation truth div free potential: psi")
#   colorbar()
#   =#
#
#
#




#=###########################################

Try and run the gradient ascent directly on lensed + white noise

=##########################################################
#
#   # --- noise parameters
#   σTTarcmin  = 0.0008
#   σEEarcmin  = √2 * σTTarcmin
#   σBBarcmin  = √2 * σTTarcmin
#   beamFWHM   = 0.0
#
#   # ---- matrix form of the cls
#   mCls = MatrixCls(
#       g, cls;
#       σTTarcmin = σTTarcmin,
#       σEEarcmin = σEEarcmin,
#       σBBarcmin = σBBarcmin,
#       beamFWHM  = beamFWHM
#   )
#
#   # ---- for the gradient you need to say that the unlensed observations should behave like unlensed CMB + white
#   mClsNoise          = deepcopy(mCls)
#   mClsNoise.cEEk[:]  = mCls.cEEk + mCls.cEEnoisek
#   mClsNoise.cBBk[:]  = mCls.cBBk + mCls.cBBnoisek
#   #= --- check the signal to noise ratio
#   figure()
#   loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_bb] / 2π, label = L"lense $C_l^{BB}$")
#   loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_ee] / 2π, label = L"lense $C_l^{EE}$")
#   loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* mCls.cEEnoisek[1] / 2π, label = "noise level")
#   legend(loc = 3)
#   =#
#
#
#   # ---- lensed CMB + white noise
#   order = 5
#   ϕx, ϕk = sim_xk(mCls.cϕϕk, g)
#   ψx, ψk = sim_xk(mCls.cψψk, g)
#   len     = LenseDecomp(ϕk, ψk, g)
#
#   ex, ek         = sim_xk(mCls.cEEk, g)
#   bx, bk         = sim_xk(mCls.cBBk, g)
#   qk, uk, qx, ux = eb2qu(ek, bk, g)
#
#   nex, nek           = sim_xk(mCls.cEEnoisek, g)
#   nbx, nbk           = sim_xk(mCls.cBBnoisek, g)
#   nqk, nuk, nqx, nux = eb2qu(nek, nbk, g)
#
#   ln_nqx, ln_nux     = lense(qx + nqx, ux + nux, len, g, order)
#   ln_nek, ln_nbk, ln_nex, ln_nbx = qu2eb(g.FFT*ln_nqx, g.FFT*ln_nux, g)
#
#
#
#   # ---- gradient ascent
#   len_curr = LenseDecomp(zeros(ϕk), zeros(ψk), g)
#   pmask  = trues(size(g.r))   # g.r .< round(Int, g.nyq * 0.5)
#   ebmask = trues(size(g.r))   # g.r .< round(Int, g.nyq * 0.99)
#   sg1    = 1e-8              # <-- size of gradient step for ϕ
#   sg2    = 1e-8               # <-- size of gradient step for ψ
#   @show loglike(len_curr, ln_nqx, ln_nux, g,  mClsNoise, order=order, pmask=pmask, ebmask=ebmask)
#   for cntr = 1:100
#       @time len_curr = gradupdate(len_curr, ln_nqx, ln_nux, g, mClsNoise; maxitr=100, sg1=sg1,sg2=sg2,order=order,pmask=pmask,ebmask=ebmask)
#       @show loglike(len_curr, ln_nqx, ln_nux, g, mClsNoise, order=order, pmask=pmask, ebmask=ebmask)
#   end
#
#
#   #= --- Plot: the estimated lensing potentials
#   figure()
#   subplot(2,2,1)
#   imshow(real(g.FFT \ (len_curr.ϕk.*(g.r .< Inf))) )#, vmin = minimum(-ϕx), vmax = maximum(-ϕx) )
#   title("estimated curl free potential: phi")
#   colorbar()
#   subplot(2,2,2)
#   imshow(-ϕx, vmin = minimum(-ϕx), vmax = maximum(-ϕx) )
#   title("simulation truth curl free potential: phi")
#   colorbar()
#   subplot(2,2,3)
#   imshow(real(g.FFT \ (len_curr.ψk.*(g.r .< Inf))) )#, vmin = minimum(-ψx), vmax = maximum(-ψx) )
#   title("estimated div free potential: psi")
#   colorbar()
#   subplot(2,2,4)
#   imshow(-ψx, vmin = minimum(-ψx), vmax = maximum(-ψx) )
#   title("simulation truth div free potential: psi")
#   colorbar()
#   =#
#
#
#
#
#
#



#=###########################################

Impliment Gibbs (fixing r known) with HMC and exact conditional simulation

=##########################################################




#=###########################################

Include uncertainty in r in the Gibbs chain

=##########################################################
