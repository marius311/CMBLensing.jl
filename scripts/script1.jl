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
period = 0.3 # radians
nside  = nextprod([2,3,5,7], 400)
g      = FFTgrid(dm, period, nside)

# --- cls
cls = class(r           = 1.0,
             omega_b     = 0.0224567,
             omega_cdm   = 0.118489,
             tau_reio    = 0.128312,
             theta_s     = 0.0104098,
             logA_s_1010 = 3.29056,
             n_s         = 0.968602 )

# --- noise parameters
σEEarcmin  = 1.0  # std noise per-unit pixel
σBBarcmin  = 1.0
σTTarcmin  = 1.0
beamFWHM   = 0.0

# ---- matrix form of the cls
matrixCls = MatrixCls(g, cls;
                σTTarcmin = σTTarcmin,
                σEEarcmin = σEEarcmin,
                σBBarcmin = σBBarcmin,
                beamFWHM  = beamFWHM )




#=#########################################################

Here are plots of the lensed and unlensed spectra

=##########################################################
figure()
loglog(cls[:ell] .* (cls[:ell] + 1) .* cls[:tt] / 2π, "r--", label = L"C_l^{TT}")
loglog(cls[:ell] .* (cls[:ell] + 1) .* cls[:ee] / 2π, "g--", label = L"C_l^{EE}")
loglog(cls[:ell] .* (cls[:ell] + 1) .* cls[:bb] / 2π, "b--", label = L"C_l^{BB}")

loglog(cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_tt] / 2π, "r-", label = L"\widetilde C_l^{TT}")
loglog(cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_ee] / 2π, "g-", label = L"\widetilde C_l^{EE}")
loglog(cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_bb] / 2π, "b-", label = L"\widetilde C_l^{BB}")
legend(loc = "best")



#=###########################################

Simulate E and B to check it is working

=##########################################################

ex, ek = sim_xk(matrixCls.cEEk, g)
bx, bk = sim_xk(matrixCls.cBBk, g)
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



# --- check that ek has the right radial power
kbins, est_ceek = BayesLensSPTpol.radial_power(ek, 1, g)
kbins, est_cbbk = BayesLensSPTpol.radial_power(bk, 1, g)
figure()
loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:ee] / 2π, "g", label = L"C_l^{EE}")
loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:bb] / 2π, "g", label = L"C_l^{BB}")
loglog(kbins, kbins .* (kbins + 1) .* est_ceek / 2π, ".", label = L"est $C_l^{EE}$")
loglog(kbins, kbins .* (kbins + 1) .* est_cbbk / 2π, ".", label = L"est $C_l^{BB}$")





#=###########################################

Get lensing simulations working

=##########################################################




#=###########################################

Impliment likelihood gradient ascent with respect to ϕk and ψk.
Use Blakes method for de-lensing

=##########################################################




#=###########################################

Impliment Gibbs (fixing r known) with HMC and exact conditional simulation

=##########################################################




#=###########################################

Include uncertainty in r in the Gibbs chain

=##########################################################
