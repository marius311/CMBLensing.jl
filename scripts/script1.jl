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
σTTarcmin  = 1.0
σEEarcmin  = √2 * σTTarcmin
σBBarcmin  = √2 * σTTarcmin
beamFWHM   = 0.0

# --- Taylor series lensing order
order = 3


#=###########################################

Generate the cls

=##########################################################

# --- cls
cls = class(
    ψscale      = 0.1,    # <-- cψψk = ψscale * baseline_cϕϕk
    ϕscale      = 1.0,  # <-- cϕϕk = ϕscale * baseline_cϕϕk
    lmax        = 6_000,
    r           = 0.3,
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

Experiment!! See how noise behaves under this lensing operation

=##########################################################

ntx, _  = sim_xk(mCls.cTTnoisek, g)
nbx, _  = sim_xk(mCls.cBBnoisek, g)
ln_ntx, ln_nbx = lense(ntx, nbx, len, g, order)

#= --- Plot: look at the lensed and unlensed noise band powers
kbins, ln_nb_bndpwr_k = radial_power(g.FFT*ln_nbx, 1, g)
kbins,    nb_bndpwr_k = radial_power(g.FFT*   nbx, 1, g)
kbins, ln_nt_bndpwr_k = radial_power(g.FFT*ln_ntx, 1, g)
kbins,    nt_bndpwr_k = radial_power(g.FFT*   ntx, 1, g)
figure()
plot(kbins, ln_nb_bndpwr_k, ".", label = "lensed b noise")
plot(kbins,   nb_bndpwr_k, ".", label = "b noise")
plot(kbins, ln_nt_bndpwr_k, ".", label = "lensed t noise")
plot(kbins,   nt_bndpwr_k, ".", label = "t noise")
legend()
=#



#=###########################################

Impliment likelihood gradient ascent for `invlen`

=##########################################################

# --- initialize zero lense (actually this will estimate the inverse lense)
len_curr = LenseDecomp(zeros(ϕk), zeros(ψk), g)

pmask  = round(Int, g.nyq * 0.5) # round(Int, 100 * g.deltk)
ebmask = round(Int, g.nyq * 0.95)
sg1    = 2e-10 # 2e-10  # <--- size of gradient step for ϕ
sg2    = 2e-10 # 2e-10  # <--- size of gradient step for ψ
@show loglike(len_curr, ln_qx, ln_ux, g,  mCls, order=order, pmask=pmask, ebmask=ebmask)
for cntr = 1:10
    @time len_curr = gradupdate(len_curr, ln_qx, ln_ux, g, mCls; maxitr=4, sg1=sg1,sg2=sg2,order=order,pmask=pmask,ebmask=ebmask)
    @show loglike(len_curr, ln_qx, ln_ux, g, mCls, order=order, pmask=pmask, ebmask=ebmask)
end


#= --- Plot: the estimated lensing potentials
figure()
subplot(2,2,1)
imshow(real(g.FFT \ (len_curr.ϕk.*(g.r .< Inf))), vmin = minimum(-ϕx), vmax = maximum(-ϕx) )
title("estimated curl free potential: phi")
colorbar()
subplot(2,2,2)
imshow(-ϕx, vmin = minimum(-ϕx), vmax = maximum(-ϕx) )
title("simulation truth curl free potential: phi")
colorbar()
subplot(2,2,3)
imshow(real(g.FFT \ (len_curr.ψk.*(g.r .< Inf))), vmin = minimum(-ψx), vmax = maximum(-ψx) )
title("estimated div free potential: psi")
colorbar()
subplot(2,2,4)
imshow(-ψx, vmin = minimum(-ψx), vmax = maximum(-ψx) )
title("simulation truth div free potential: psi")
colorbar()
=#



#= --- Plot: compare lensed and unlensed B-power
delensed_qx, delensed_ux = lense(qx, ux, len_curr, g, order) # last arg is the order of lensing
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

Impliment Gibbs (fixing r known) with HMC and exact conditional simulation

=##########################################################




#=###########################################

Include uncertainty in r in the Gibbs chain

=##########################################################
