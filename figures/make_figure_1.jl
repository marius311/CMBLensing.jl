#########################################################
#=

Run this script with the command `include("make_figure_1.jl")` entered into the Julia REPL in this directory.

=#
#########################################################

# --- load modules
using BayesLensSPTpol
using PyPlot

# ----  set the seed
#seedstart = rand(UInt64)
seedstart = 0xa65c9cf99fcbf681
srand(seedstart)

# --- save figures to disk or not
savefigures = true

# --- set grid geometry
const d_const = 2
const period_const = 0.3 # radians
const nside_const  = nextprod([2,3,5,7], 400)

# --- cls
cls       = class(r           = 1.0,
                 omega_b     = 0.0224567,
                 omega_cdm   = 0.118489,
                 tau_reio    = 0.128312,
                 theta_s     = 0.0104098,
                 logA_s_1010 = 3.29056,
                 n_s         = 0.968602
            )

# --- noise parameters
σEEarcmin    = 1.0  # std noise per-unit pixel
σBBarcmin    = 1.0
beamFWHM = 0.0




#########################################################
#=
Here are plots of the lensed and unlensed spectra
=#
#########################################################
figure()
loglog(cls[:ell] .* (cls[:ell] + 1) .* cls[:tt] / 2π, "r--", label = L"C_l^{TT}")
loglog(cls[:ell] .* (cls[:ell] + 1) .* cls[:ee] / 2π, "g--", label = L"C_l^{EE}")
loglog(cls[:ell] .* (cls[:ell] + 1) .* cls[:bb] / 2π, "b--", label = L"C_l^{BB}")

loglog(cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_tt] / 2π, "r-", label = L"\widetilde C_l^{TT}")
loglog(cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_ee] / 2π, "g-", label = L"\widetilde C_l^{EE}")
loglog(cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_bb] / 2π, "b-", label = L"\widetilde C_l^{BB}")
legend(loc = "best")





#########################################################
#=
define an instance of LensePrm which carries all the parameters.

=#
#########################################################

# --- define an instance of LensePrm which carries all the parameters
parms = LensePrm(d_const, period_const, nside_const, cls, σEEarcmin,  σBBarcmin, beamFWHM)



# #########################################################
#=
Simulate E to check it is working
=#
# #########################################################

ex, ek = BayesLensSPTpol.grf_sim_xk(parms.cEEk, parms)
figure()
imshow(ex,
    interpolation = "nearest",
    origin="lower",
    extent=[minimum(parms.x[1]), maximum(parms.x[1]),minimum(parms.x[2]), maximum(parms.x[2])],
)


# --- check that ek has the right radial power
kbins, est_ceek = BayesLensSPTpol.radial_power(ek, 1, parms)
figure()
loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:ee] / 2π, "g", label = L"C_l^{EE}")
loglog(kbins, kbins .* (kbins + 1) .* est_ceek / 2π, ".", label = L"C_l^{EE}")
