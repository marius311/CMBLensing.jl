"""
Compute the variance and normalizing constants for the B-mode quadratic delenser
=======================================================
ToDo:
	* Is there application in using this as a Null test?
	* What if you had a posterior distribution on ϕ and you did-delensing on each sample
	  then averaged over multiple realizations
	* I think the optimal weights are not quite right along the low variance directions. 
	  This is related to the variance of the individual terms is calculated
	* You should now try and update the optimal weights
	* Try QE with zero B mode and make a null test
	* Why in the heck does Blakes method estimate all order lensing better than first order lensing?
	* Why does first order lensing work for B but not for E?
	* Figure out the QE delenser for T. 
	* Compute the cross spectra of hatB and hatE.
	* Is the inflated effective noise the price to pay for unbiasedness? Can there possibly be another unbiased estimate?
	* Does this use the fact that the low ell power isn't lensed much?? 
	* - Find the best suboptimal weight function which can be computed with FFT
	  - Program in the actual B mode quadratic de-lenser
	  - Check that the simulations match the computed variance
	* compare with spectral delensing by subtracting out on the C_l scale
	* Notice that masking problems are not a problem for this QE 
	  ...since the induced mode coupling isn't where the estimate looks for signal. 
	  It may effect the noise variance and optimal weights tho.
"""





##################################################
# 
#  Preliminaries 
#
##################################################


""" ## Load modules, paths and functions """
srcpath  =  "/Users/ethananderes/Dropbox/BLimit/src/"
savepath =  "/Users/ethananderes/Dropbox/BLimit/paper/"
include(srcpath * "Interp.jl")
include(srcpath * "fft.jl")
include(srcpath * "funcs.jl")
using PyPlot, Interp


""" ## parameters of the simulation run  """
const n = 2 ^ 10
const pixl_wdth_arcmin = 1.0
const beamFWHM         = 1.0
const μKarcmin_noise   = 0.001
const ρϕhatϕ = 0.95  # corr btwn ϕ and hatϕ
const r      = 0.15


""" ## Grid generation """
const d = 2
const deltx   = pixl_wdth_arcmin * π / (180 * 60) #this is in radians
const period  = deltx * n
const deltk   = 2π / period  
const dk      = deltk ^ d
const dx      = deltx ^ d
const nyq     = 2π / (2deltx)
const x, y    = meshgrid([0:n-1] * deltx, [0:n-1] * deltx)
const k1, k2  = linspace(-nyq, nyq-deltk, int(n))  |> fftshift |> x->meshgrid(x,x)
const magk    = √(k1.^2 + k2.^2)


""" ## other constants """
const φ2_l    = 2.0 * angle(k1 + im * k2)
const nugget_at_each_pixel = abs2(μKarcmin_noise) / (pixl_wdth_arcmin ^ d)
const lmax   = nyq / 2

""" ## Spectra  """
using PyCall
@pyimport classy
cosmo = classy.Class()
cosmo[:struct_cleanup]() # important when running in a loop over different cosmologies
cosmo[:empty]() # important if you completely change cosmology
params = [
    "output"        => "tCl, pCl, lCl",
    "modes"         => "s,t",
    "lensing"       => "yes",
    "T_cmb"         => 2.726, # Kelvin units, subsequent scaling depends on this   
    "l_max_scalars" => 7_500,
    "l_max_tensors" => 7_500,
    "A_s"           => 2.3e-9,
    "n_s"           => 0.9624, 
    "h"             => 0.6711,
    "omega_b"       => 0.022068,
    "omega_cdm"     => 0.12029,
    "r"             => r ]
cosmo[:set](params)
cosmo[:compute]()
cls_ln = cosmo[:lensed_cl](7_000)
cls_ln["ee"] = cls_ln["ee"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls_ln["bb"] = cls_ln["bb"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls = cosmo[:raw_cl](7_000)
cls["ee"] = cls["ee"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls["bb"] = cls["bb"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
# now make-up the cross spectra




##  !!! see what things look like if ϕ is more peaked...
# cls["pp"]     = cls["pp"] .* (cls["ell"].^(1.5)) / 1000

# now make-up the cross spectra
# const ρϕhatϕ = 0.95  # corr btwn ϕ and hatϕ
# ρls = ρϕhatϕ + zeros(cls["pp"]) 
# cls["hatphatp"]     = cls["pp"] ./ abs2(ρls) 
# cls["hatp_cross_p"] = cls["pp"] 

##  different cross correlation structure
cls["hatnoise"]     = zeros(cls["pp"]) + cls["pp"][200] ./ 50000000000
cls["hatphatp"]     = cls["pp"] + cls["hatnoise"]
cls["hatp_cross_p"] = cls["pp"] 



""" # Make the spectral matrices  """
const cEE, cEEobs, cBB, cBBobs, cNN, cPP, cPhatPhat, cPhatP, cPhatNoise  = let
	sig    = (beamFWHM * (π / (180 * 60))) / (2 * √(2 * log(2)))
	beam   = exp(- (sig^2) * (magk.^2) / 2)
	beamSQ = abs2(beam)	
	cNN    = √(2) * nugget_at_each_pixel * dx ./ beamSQ
	
	index  = ceil(magk)
	index[find(index.==0)] = 1
	function makecXX(ells, clsXX, indexs)
		logCXX = linear_interp1(ells, log(clsXX), indexs)
		logCXX[find(logCXX .== 0)]  = -Inf
		logCXX[find(isnan(logCXX))] = -Inf
		return exp(logCXX)
	end
  cPhatNoise = makecXX(cls["ell"], cls["hatnoise"], index)
  cPhatP     = makecXX(cls["ell"], cls["hatp_cross_p"], index)
  cPhatPhat  = makecXX(cls["ell"], cls["hatphatp"], index)
	cPP       = makecXX(cls["ell"], cls["pp"], index)
	cBB       = makecXX(cls["ell"], cls["bb"], index)
	cEE       = makecXX(cls["ell"], cls["ee"], index)
	cBBobs    = cNN + makecXX(cls["ell"], cls_ln["bb"], index)
	cEEobs    = cNN + makecXX(cls["ell"], cls_ln["ee"], index)

  #!!! this is to avoid weight on the ell=0 term.
  cPhatNoise[magk .< magk[1,2]] = 0.0
  cPhatP[magk .< magk[1,2]] = 0.0    
  cPhatPhat[magk .< magk[1,2]] = 0.0 
  cPP[magk .< magk[1,2]] = 0.0       
  cBB[magk .< magk[1,2]] = 0.0       
  cEE[magk .< magk[1,2]] = 0.0       
  cBBobs[magk .< magk[1,2]] = 0.0    
  cEEobs[magk .< magk[1,2]] = 0.0    
  
	cEE, cEEobs, cBB, cBBobs, cNN, cPP, cPhatPhat, cPhatP, cPhatNoise 
end







##################################################
# 
#  Preliminaries are done. 
#  The following are snippets of code for testing and exploring.
#
##################################################



""" 
# Compare  N0  with  Aell_optimal
"""
#=

cos2kl   = cos(φ2_l).^2
sin2kl   = sin(φ2_l).^2
cmat     = ones(size(cBBobs)) ./ (magk .< lmax)
bmat     = ones(size(cBBobs))

Aell_Bop = Aell_optim(cEEobs, cBBobs, cPhatPhat, cPhatP, cPP; mode = :B)
amat     = (cos2kl .* cBBobs + sin2kl .* cEEobs) ./ (magk .< lmax)
N0B      = N0_fun(amat, bmat, cmat, cEEobs, cBBobs, cPhatPhat, cPhatP, cPP; mode = :B)

Aell_Eop = Aell_optim(cEEobs, cBBobs, cPhatPhat, cPhatP, cPP; mode = :E)
amat     = (sin2kl .* cBBobs + cos2kl .* cEEobs) ./ (magk .< lmax)
N0E      = N0_fun(amat, bmat, cmat, cEEobs, cBBobs, cPhatPhat, cPhatP, cPP; mode = :E)

specBlen_from_B = frst_ordr_spec_decomp(cEE, cBB, cPP; mode = :B, src = :B)
specBlen_from_E = frst_ordr_spec_decomp(cEE, cBB, cPP; mode = :B, src = :E)
specElen_from_B = frst_ordr_spec_decomp(cEE, cBB, cPP; mode = :E, src = :B)
specElen_from_E = frst_ordr_spec_decomp(cEE, cBB, cPP; mode = :E, src = :E)
specBlen_tot = specBlen_from_B + specBlen_from_E
specElen_tot = specElen_from_B + specElen_from_E

figure()
  # loglog(magk[1:275,1], magk[1:275,1].^2 .* cBB[1:275,1], label = "primordial BB")
  loglog(magk[1:275,1], magk[1:275,1].^2 .* cEE[1:275,1], label = "primordial EE")
  # loglog(magk[1:275,1], magk[1:275,1].^2 .* cNN[1:275,1], label = "noise level")
  # loglog(magk[1:275,1], magk[1:275,1].^2 .* (cBBobs - cNN)[1:275,1], label = "lensed B")
  # loglog(magk[1:275,1], magk[1:275,1].^2 .* (specBlen_from_B)[1:275,1], label = "specBlen_from_B")
  # loglog(magk[1:275,1], magk[1:275,1].^2 .* (specBlen_from_E)[1:275,1], label = "specBlen_from_E")
  # loglog(magk[1:275,1], magk[1:275,1].^2 .* (specBlen_tot)[1:275,1], label    = "specBlen_tot")
  # loglog(magk[1:275,1], magk[1:275,1].^2 .* (specElen_from_B)[1:275,1], label = "specElen_from_B")
  # loglog(magk[1:275,1], magk[1:275,1].^2 .* (specElen_from_E)[1:275,1], label = "specElen_from_E")
  # loglog(magk[1:275,1], magk[1:275,1].^2 .* (specElen_tot)[1:275,1], label    = "specElen_tot")
  # loglog(magk[1:275,1], magk[1:275,1].^2 .* (cEE + specElen_tot)[1:275,1], label    = "cEE + specElen_tot")
  # loglog(magk[1:275,1], magk[1:275,1].^2 .* (cBBobs - cBB - cNN)[1:275,1], label = "B lensing corruption")
  # loglog(magk[1:275,1], magk[1:275,1].^2 .* (cEEobs - cEE - cNN)[1:275,1], label = "E lensing corruption")
  loglog(magk[1:275,1], magk[1:275,1].^2 .* (cEEobs - cNN)[1:275,1], label = "E lensing total")
  # loglog(magk[1:275,1], magk[1:275,1].^2 .* Aell_Bop[1:275,1], ":", label = "Aell B op")
  # loglog(magk[1:275,1], magk[1:275,1].^2 .* Aell_Eop[1:275,1], ":", label = "Aell E op")
  loglog(magk[1:275,1], magk[1:275,1].^2 .* N0E[1:275,1] , ":", label = "N0E variance")
  loglog(magk[1:275,1], magk[1:275,1].^2 .* N0B[1:275,1] , ":", label = "N0B variance")
  legend(loc = "best")

=#





""" 
# Bin N0 and Aell_optimal to get estimation error bars
"""
#=

cos2kl   = cos(φ2_l).^2
sin2kl   = sin(φ2_l).^2
cmat     = ones(size(cBBobs)) ./ (magk .< lmax)
bmat     = ones(size(cBBobs))

Aell_Bop = Aell_optim(cEEobs, cBBobs, cPhatPhat, cPhatP, cPP; mode = :B)
amat     = (cos2kl .* cBBobs + sin2kl .* cEEobs) ./ (magk .< lmax)
N0B      = N0_fun(amat, bmat, cmat, cEEobs, cBBobs, cPhatPhat, cPhatP, cPP; mode = :B)

Aell_Eop = Aell_optim(cEEobs, cBBobs, cPhatPhat, cPhatP, cPP; mode = :E)
amat     = (sin2kl .* cBBobs + cos2kl .* cEEobs) ./ (magk .< lmax)
N0E      = N0_fun(amat, bmat, cmat, cEEobs, cBBobs, cPhatPhat, cPhatP, cPP; mode = :E)

#bin_edg    =  logspace(log(10,5*deltk), log(10,6000), 30)
bin_edg     =  (2 * deltk):(1 * deltk):6000
bin_cnt     =  bincount(magk, bin_edg) 


cBBvar_invvar_N0E   = 2./binsum( 1./abs2(N0E + cEE) , magk, bin_edg)
cBBvar_invvar_N0B   = 2./binsum( 1./abs2(N0B + cBB) , magk, bin_edg)
cBBvar_N0E   = binave( abs2(N0E + cEE), magk, bin_edg)
cBBvar_N0E ./= bin_cnt / 2
cBBvar_N0B   = binave( abs2(N0B + cBB), magk, bin_edg)
cBBvar_N0B ./= bin_cnt / 2
cBBvar_Aell_Eop   = binave( abs2(Aell_Eop + cEE), magk, bin_edg)
cBBvar_Aell_Eop ./= bin_cnt / 2
cBBvar_Aell_Bop   = binave( abs2(Aell_Bop + cBB), magk, bin_edg)
cBBvar_Aell_Bop ./= bin_cnt / 2
cBBvar_cBBobs    = binave(  abs2(cNN + cBB), magk, bin_edg)
cBBvar_cBBobs  ./= bin_cnt / 2
cBBvar_cEEobs    = binave(  abs2(cNN + cEE), magk, bin_edg)
cBBvar_cEEobs  ./= bin_cnt / 2
BBsignal         = binave( cBB, magk, bin_edg)
EEsignal         = binave( cEE, magk, bin_edg)


figure(figsize = (14,6))
subplot(1,2,1)
  loglog(bin_edg, √(cBBvar_N0E),   ".",   label = "suboptimal est std")
  loglog(bin_edg, √(cBBvar_invvar_N0E),   ".",   label = "suboptimal invvar std")
  loglog(bin_edg, √(cBBvar_Aell_Eop),".", label = "optimal est std")
  loglog(bin_edg, √(cBBvar_cEEobs),".", label  = "nominal est std")
  loglog(bin_edg, EEsignal, ".",label  = "binned CEE: signal to estimate")
  legend(loc = "best")
subplot(1,2,2)
 loglog(bin_edg, √(cBBvar_N0B),   ".",   label = "suboptimal est std")
 loglog(bin_edg, √(cBBvar_invvar_N0B),   ".",   label = "suboptimal invvar std")
 loglog(bin_edg, √(cBBvar_Aell_Bop),".", label = "optimal est std")
 loglog(bin_edg, √(cBBvar_cBBobs),".", label  = "nominal est std")
 loglog(bin_edg, BBsignal, ".",label  = "binned CBB: signal to estimate")
 legend(loc = "best")

=#


""" 
# Use simulation to approximate the true error variance in our estimate and Blakes estimate 
"""
include(srcpath * "lensing_sim.jl")
const E, B, tldE, tldB, Eobs, Bobs,  ϕ, hatϕ = scnd_ord_len_QU()
# const E, B, tldE, tldB, Eobs, Bobs,  ϕ, hatϕ = all_ord_len_QU()


# compute the quadratic estimates
amat_B   = (abs2(cos(φ2_l)) .* cBBobs + abs2(sin(φ2_l)) .* cEEobs) ./ (magk[1,3] .< magk .< lmax)
amat_E   = (abs2(sin(φ2_l)) .* cBBobs + abs2(cos(φ2_l)) .* cEEobs) ./ (magk[1,3] .< magk .< lmax)
cmat     = ones(size(cBBobs)) ./ (magk .< lmax)
bmat     = ones(size(cBBobs))

Aell_Bop = Aell_optim(cEEobs, cBBobs, cPhatPhat, cPhatP, cPP; mode = :B)
Aell_Eop = Aell_optim(cEEobs, cBBobs, cPhatPhat, cPhatP, cPP; mode = :E)
N0B      = N0_fun(amat_B, bmat, cmat, cEEobs, cBBobs, cPhatPhat, cPhatP, cPP; mode = :B)
N0E      = N0_fun(amat_E, bmat, cmat, cEEobs, cBBobs, cPhatPhat, cPhatP, cPP; mode = :E)
N0B_obs  = N0_obs_fun(amat_B, bmat, cmat, Eobs, Bobs, hatϕ, cPhatPhat, cPhatP, cPP; mode = :B)
N0E_obs  = N0_obs_fun(amat_E, bmat, cmat, Eobs, Bobs, hatϕ, cPhatPhat, cPhatP, cPP; mode = :E)
hatB     = hatEB(Eobs, Bobs, hatϕ, amat_B, bmat, cmat, cPhatPhat, cPhatP, cPP; mode = :B)
hatE     = hatEB(Eobs, Bobs, hatϕ, amat_E, bmat, cmat, cPhatPhat, cPhatP, cPP; mode = :E)

# now check that the power hatB and hatB match (cBB + N0B) and (cEE + N0E)
bin_edg      = (2 * deltk):(2 * deltk):4000
cBBln        = binave(cBBobs - cNN - cBB, magk, bin_edg)
NrsdBpwr     = binave(N0B, magk, bin_edg)
NrsdEpwr     = binave(N0E, magk, bin_edg)
NrsdBobspwr  = binave(N0B_obs, magk, bin_edg)
NrsdEobspwr  = binave(N0E_obs, magk, bin_edg)
ArsdBpwr     = binave(Aell_Bop, magk, bin_edg)
ArsdEpwr     = binave(Aell_Eop, magk, bin_edg)
cEEave       = binave(cEE, magk, bin_edg)
cBBave       = binave(cBB, magk, bin_edg)
hatBsq       = binave(abs2(hatB) .* dk, magk, bin_edg)
hatEsq       = binave(abs2(hatE) .* dk, magk, bin_edg)
rsdBpwr      = binave(abs2(B - hatB) .* dk, magk, bin_edg)
rsdEpwr      = binave(abs2(E - hatE) .* dk, magk, bin_edg)
Bsq          = binave(abs2(B) .* dk, magk, bin_edg)
Esq          = binave(abs2(E) .* dk, magk, bin_edg)

# check that my variances are about right...up to that non-Gaussian term
figure(figsize = (14,6))
subplot(1,2,1)
  #loglog(bin_edg, bin_edg .* hatEsq,    "b." , label = "hatEsq")
  loglog(bin_edg, bin_edg .* cEEave,     "b"  , label = "cEEave")
  #loglog(bin_edg, bin_edg .* Esq,       "g." , label = "Esq")
  loglog(bin_edg, bin_edg .* rsdEpwr,    "g." , label = "actual estimation error")
  loglog(bin_edg, bin_edg .* NrsdEpwr,   "g"  , label = "NE computed error spectrum")
  loglog(bin_edg, bin_edg .* NrsdEobspwr,"g:" , label = "NE obs computed error spectrum")
  loglog(bin_edg, bin_edg .* ArsdEpwr,   "r:" , label = "NE for optimal weights")
  legend(loc = "best")
subplot(1,2,2)
  # loglog(bin_edg, bin_edg .* hatBsq,   "b." , label = "hatBsq")
  loglog(bin_edg, bin_edg .* cBBave,     "b"  , label = "cBBave")
  # loglog(bin_edg, bin_edg .* Bsq,      "g." , label = "Bsq")
  loglog(bin_edg, bin_edg .* rsdBpwr,    "g." , label = "actual estimation error")
  loglog(bin_edg, bin_edg .* NrsdBpwr,   "g"  , label = "NB computed error spectrum")
  loglog(bin_edg, bin_edg .* NrsdBobspwr,"g:" , label = "NB obscomputed error spectrum")
  loglog(bin_edg, bin_edg .* ArsdBpwr,   "r:" , label = "NE for optimal weights")
  loglog(bin_edg, bin_edg .* cBBln,      "k"  , label = "B lensing forground")
  legend(loc = "best")



# now lets check Blakes method has the right variance
bin_edg     =  (2 * deltk):(2 * deltk):4000

cBBave      = binave(cBB, magk, bin_edg)
cEEave      = binave(cEE, magk, bin_edg)
cBBobsave   = binave(cBBobs, magk, bin_edg)
cBB_ln_ave  = binave(cBBobs - cNN, magk, bin_edg)
cBB_lnnoise_ave   = binave(cBBobs - cNN - cBB, magk, bin_edg)
cNN_ave           = binave(cNN, magk, bin_edg)

est_1st_lnBfromE  = frst_ordr_lnBfromE(Eobs, hatϕ, cEEobs, cEE, cPhatPhat, cPhatP)
tru_1st_lnBfromE  = frst_ordr_lnBfromE(E, ϕ)
Bplus1st          = binave(abs2(B    + tru_1st_lnBfromE) .* dk, magk, bin_edg)
tldBminus1st      = binave(abs2(tldB - tru_1st_lnBfromE) .* dk, magk, bin_edg)
Bobsminus_est_1st = binave(abs2(Bobs - est_1st_lnBfromE) .* dk, magk, bin_edg)
Bobsminus_est1st_B   = binave(abs2(Bobs - est_1st_lnBfromE - B) .* dk, magk, bin_edg)

c_error_BB_blake  = blakes_error_spec(cEEobs, cEE, cPhatPhat, cPhatP, cPP)
cBN_BN_Blake_ave               = binave(c_error_BB_blake, magk, bin_edg)
cBBnoise_after_ln_subtraction  = binave(cNN + c_error_BB_blake, magk, bin_edg)
allorderlen_minus_est_1st      = binave(abs2(tldB - B - est_1st_lnBfromE) .* dk, magk, bin_edg)
frtorderlen_minus_est_1st      = binave(abs2(tru_1st_lnBfromE - est_1st_lnBfromE) .* dk, magk, bin_edg)

#c_error_BB_blake_withmyunbiasedE = blakes_error_spec(cEE + Aell_Eop, cEE, cPhatPhat, cPhatP, cPP)
c_error_BB_blake_withmyunbiasedE = blakes_error_spec(cEE + N0E_obs, cEE, cPhatPhat, cPhatP, cPP)
cBN_BN_unbiasedBlake_ave  = binave(c_error_BB_blake_withmyunbiasedE, magk, bin_edg)
est_1st_lnBfromE_withmyunbiasedE = frst_ordr_lnBfromE(hatE, hatϕ, cEE + N0E_obs, cEE, cPhatPhat, cPhatP)
error_withmyunbiasedE     = binave(abs2(tldB - est_1st_lnBfromE_withmyunbiasedE - B) .* dk, magk, bin_edg)
first_order_error_withmyunbiasedE  = binave(abs2(tru_1st_lnBfromE - est_1st_lnBfromE_withmyunbiasedE) .* dk, magk, bin_edg)

cWFBBerror = binave(cBB .* N0B_obs ./ (cBB + N0B_obs), magk, bin_edg)



figure(figsize = (12,9))
  loglog(magk[1:275,1], magk[1:275,1] .* (abs2(hatB)[1,1:275])' * dk, "r." , label = "hatB")
  # loglog(bin_edg, bin_edg .* cBB_ln_ave, "b" , label = "cBB_ln_ave")
  #loglog(bin_edg, bin_edg .* cBBave,     "r" , label = "cBB")
  # loglog(bin_edg, bin_edg .* cEEave,   "r" , label = "cEE")
  # loglog(bin_edg, bin_edg .* cNN_ave,    "r" , label = "cNN")
  loglog(bin_edg, bin_edg .* cBB_lnnoise_ave, "k--" , label = "B lensing noise spectrum")
  # loglog(bin_edg, bin_edg .* cWFBBerror, "k--" , label = "B lensing WF noise spectrum")
  # loglog(magk[1:275], magk[1:275] .* (cBB.*N0B_obs./(cBB+N0B_obs))[1:275], "k--" , label = "B lensing WF noise spectrum")
  # loglog(bin_edg, bin_edg .* cBBobsave,  "g" , label = "cBBobsave")
  # loglog(bin_edg, bin_edg .* Bplus1st, "b." , label = "Bplus1st")
  # loglog(bin_edg, bin_edg .* tldBminus1st, "g." , label = "tldBminus1st")
  # loglog(bin_edg, bin_edg .* Bobsminus_est_1st, "r." , label = "Bobsminus_est_1st")
  # loglog(bin_edg, bin_edg .* cBBnoise_after_ln_subtraction, "b" , label = "cBBnoise_after_ln_subtraction")
  # loglog(bin_edg, bin_edg .* Bobsminus_est1st_B, "b." , label = "Bobsminus_est1st_B")

  loglog(bin_edg, bin_edg .* allorderlen_minus_est_1st, "g." , label = "all order Lensign forground subtraction error power")
  loglog(bin_edg, bin_edg .* frtorderlen_minus_est_1st, "g--" , label = "1st order lensing forground subtraction error power")
  loglog(bin_edg, bin_edg .* cBN_BN_Blake_ave,  "g" , label = "Blakes assumed approx spectra")
  # loglog(bin_edg, bin_edg .* (allorderlen_minus_est_1st - cBN_BN_Blake_ave),  "g:" , label = "allorderlen_minus_est_1st - cBN_BN_Blake_ave")
  # loglog(bin_edg, bin_edg .* (est_1st_minus_1st - cBN_BN_Blake_ave),  "g:" , label = "est_1st_minus_1st - cBN_BN_Blake_ave")
  loglog(bin_edg, bin_edg .* cBN_BN_unbiasedBlake_ave,  "k" , label = "Using my unbiased E for template subtraction")
  loglog(bin_edg, bin_edg .* first_order_error_withmyunbiasedE,  "k." , label = "actual error with my template")
  loglog(bin_edg, bin_edg .* rsdBpwr,    "g." , label = "actual estimation error")
legend(loc = "best")



figure()
wts_obs     = abs2(hatB) .* dk
wts_exp     = cBB

bin_wts_obs = binave(wts_obs, magk, bin_edg)
bin_wts_exp = binave(wts_exp, magk, bin_edg)

# average power looks about right
loglog(bin_edg, bin_edg .* bin_wts_obs, "b." , label = "bin_wts_obs")
loglog(bin_edg, bin_edg .* bin_wts_exp, "r" , label = "bin_wts_exp")

# what about power along the small directions
loglog(magk[1:275], magk[1:275] .* wts_obs[1:275,1], "g." , label = "bin_wts_obs")
loglog(magk[1:275], magk[1:275] .* wts_exp[1:275,1], "r" , label = "bin_wts_exp")
legend(loc = "best")




""" 
# Testing the total power 
	* See if the power is rotationally symmetric. Even though total power in an annulus could be good,
	  the important stuff for the estimate is when this total power is small (so that the inverse var weight is large).
"""
#=

bin_edg     =  (2 * deltk):(2 * deltk):4000
wts_obs     = dk .* abs2(cos(φ2_l) .* conj(Bobs) + sin(φ2_l) .* conj(Eobs)) ./ (magk .< lmax)
wts_exp     = (abs2(cos(φ2_l)) .* cBBobs + abs2(sin(φ2_l)) .* cEEobs) ./ (magk .< lmax)

bin_wts_obs = binave(wts_obs, magk, bin_edg)
bin_wts_exp = binave(wts_exp, magk, bin_edg)

# average power looks about right
loglog(bin_edg, bin_edg .* bin_wts_obs, "b." , label = "bin_wts_obs")
loglog(bin_edg, bin_edg .* bin_wts_exp, "r" , label = "bin_wts_exp")
legend(loc = "best")

# what about power along the small directions
plot(magk[1:275], magk[1:275] .* wts_obs[1:275,1], "b." , label = "bin_wts_obs")
plot(magk[1:275], magk[1:275] .* wts_exp[1:275,1], "r" , label = "bin_wts_exp")
legend(loc = "best")

=#














