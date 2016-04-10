#=########################################################

Run this script with the command `include("script1.jl")`

=##########################################################

# --- load modules
using BayesLensSPTpol
using PyPlot

# ----  set the seed
seedstart = rand(UInt64)
srand(seedstart)

# --- save figures to disk or not
savefigures = true

# --- set grid geometry
dm     = 2
period = 0.4 # radians
nside  = nextprod([2,3,5,7], 512)
g      = FFTgrid(dm, period, nside)

# --- cls
cls = class(
    lmax        = 6_000,
    r           = 0.2,
    omega_b     = 0.0224567,
    omega_cdm   = 0.118489,
    tau_reio    = 0.128312,
    theta_s     = 0.0104098,
    logA_s_1010 = 3.29056,
    n_s         = 0.968602
)

# --- noise parameters
σTTarcmin  = 1.0
σEEarcmin  = √2 * σTTarcmin
σBBarcmin  = √2 * σTTarcmin
beamFWHM   = 0.0

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


# check the ex and bx maps
#=
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
@time ln_qx, ln_ux = lense(qx, ux, len, g, 3) # last arg is the order of lensing

# # --- check the lensed fields have the right power.
# ln_ek, ln_bk, ln_ex, ln_bx = qu2eb(g.FFT*ln_qx, g.FFT*ln_ux, g)
# kbins, est_ln_cbbk = radial_power(ln_bk, 1, g)
# kbins, est_ln_ceek = radial_power(ln_ek, 1, g)
# figure()
# loglog(kbins, kbins .* (kbins + 1) .* est_ln_cbbk / 2π, ".", label = L"est lense $C_l^{BB}$")
# loglog(kbins, kbins .* (kbins + 1) .* est_ln_ceek / 2π, ".", label = L"est lense $C_l^{EE}$")
# loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_bb] / 2π, label = L"lense $C_l^{BB}$")
# loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_ee] / 2π, label = L"lense $C_l^{EE}$")
# legend(loc = 3)



#=###########################################

Impliment likelihood gradient ascent for `invlen`

=##########################################################

# --- initialize zero lense (actually estimating inverse lense)
len_curr = LenseDecomp(zeros(ϕk), zeros(ψk), g)

@show loglike(len_curr, ln_qx, ln_ux, g,  mCls, order = 3, pmask = 700, ebmask = round(Int,g.nyq * 0.7))
for cntr = 1:20
    len_curr = gradupdate(
            len_curr, ln_qx, ln_ux, g, mCls;
			maxitr = 2,
			sg1 = 1e-9,
			sg2 = 1e-10,
			order = 3,
			pmask = 500,
			ebmask = round(Int,g.nyq * 0.7)
            )
            @show loglike(len_curr, ln_qx, ln_ux, g, mCls, order = 3, pmask = 700, ebmask = round(Int,g.nyq * 0.7))
end

subplot(2,2,1)
imshow(real(g.FFT \ (len_curr.ϕk.*(g.r .< 700))))
colorbar()
subplot(2,2,2)
imshow(real(g.FFT \ (-ϕk.*(g.r .< 700))))
colorbar()
subplot(2,2,3)
imshow(real(g.FFT \ (len_curr.ψk.*(g.r .< 700))))
colorbar()
subplot(2,2,4)
imshow(real(g.FFT \ (-ψk.*(g.r .< 700))))
colorbar()


# φ2_l = 2angle(g.k[1] + im * g.k[2])
# Mq  = squash(- abs2(cos(φ2_l)) ./ mCls.cEEk  - abs2(sin(φ2_l)) ./ mCls.cBBk )
# Mu  = squash(- abs2(cos(φ2_l)) ./ mCls.cBBk  - abs2(sin(φ2_l)) ./ mCls.cEEk )
# Mqu =  squash(2 * cos(φ2_l) .* sin(φ2_l) ./ mCls.cBBk)
# Mqu -= squash(2 * cos(φ2_l) .* sin(φ2_l) ./ mCls.cEEk)
# Mq[g.r .>= 2000]  = 0.0
# Mu[g.r .>= 2000]  = 0.0
# Mqu[g.r .>= 2000] = 0.0
# ϕgradk, ψgradk = ϕψgrad(len_curr, qx, ux, g, Mq, Mu, Mqu, mCls, 2)
# subplot(2,2,1)
# imshow(real(g.FFT \ (ϕgradk.*mCls.cϕϕk.*(g.r .< 500))))
# subplot(2,2,2)
# imshow(real(g.FFT \ (ϕk.*(g.r .< 100))))
# subplot(2,2,3)
# imshow(real(g.FFT \ (ψgradk.*mCls.cψψk.*(g.r .< 500))))
# subplot(2,2,4)
# imshow(real(g.FFT \ (ψk.*(g.r .< 100))))


#=###########################################

Impliment Gibbs (fixing r known) with HMC and exact conditional simulation

=##########################################################




#=###########################################

Include uncertainty in r in the Gibbs chain

=##########################################################
