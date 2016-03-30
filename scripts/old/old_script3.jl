
"""
Quadratic delenser for T
=======================================================
ToDo:
    * Re-check the computations. I think I had a conj mistake in the convolutions.
    * Try playing around with the power spectrum of T. In particular, I have a suspition that
      the reason it doesn't work is that the scale of ∇T is way to small compated with ∇ϕ and that is
      artificially inflating the variance. 
    * Try and see if something like 21cm emission has a better N0
    * What is the utility of this method...I guess it de-correlates the lensed T. Maybe a good application is when 
    one needs to get at a different correlation.
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
const n = 2 ^ 12
const pixl_wdth_arcmin = 1.0
const beamFWHM         = 1.0
const μKarcmin_noise   = 1.0


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
    "r"             => 0.15 ]
cosmo[:set](params)
cosmo[:compute]()
cls_ln = cosmo[:lensed_cl](7_000)
cls    = cosmo[:raw_cl](7_000)
cls_ln["tt"] = cls_ln["tt"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls["tt"]    = cls["tt"]    * (10^6 * cosmo[:T_cmb]()) ^ 2 


##  !!! see what things look like if ϕ is more peaked...
# cls["pp"]     = cls["pp"] .* (cls["ell"].^(1.8)) / 7_000
cls["pp"]     = cls["pp"] .* 2

# now make-up the cross spectra
# const ρϕhatϕ = 0.95  # corr btwn ϕ and hatϕ
# ρls = ρϕhatϕ + zeros(cls["pp"]) 
# cls["hatphatp"]     = cls["pp"] ./ abs2(ρls) 
# cls["hatp_cross_p"] = cls["pp"] 

##  different cross correlation structure
cls["hatnoise"]     = zeros(cls["pp"])  + cls["pp"][200] ./ 10
cls["hatphatp"]     = cls["pp"] + cls["hatnoise"]
cls["hatp_cross_p"] = cls["pp"] 




""" # Make the spectral matrices  """
function cTTln(cTT, cPP)  # you should check this ....
    τsq     = sum( (k1.^2) .* cPP .* dk ./ ((2π)^d) )
    kdic    = [1 => k1.*k1, 2 => k1.*k2, 3 => k2.*k2]
    tmpAx   = Array(Complex{Float64}, size(cTT))
    tmpBx   = Array(Complex{Float64}, size(cTT))
    rtn     = zeros(Complex{Float64}, size(cTT))
    for cntr = 1:3
        tmpAx[:,:]  = ifft2(kdic[cntr] .* cPP, deltk)
        tmpBx[:,:]  = ifft2(kdic[cntr] .* cTT, deltk)
        rtn[:,:]   += (cntr == 2?2:1) .* fft2(tmpAx .* conj(tmpBx), deltx)
    end
    rtn[:,:] ./= (2π)^(d/2)
    rtn[:,:] .+= (1 -  abs2(magk) .* τsq ) .* cTT
    return  real(rtn) 
end

const cTT, cTTobs, cNN, cPP, cPhatPhat, cPhatP, cPhatNoise = let
	sig    = (beamFWHM * (π / (180 * 60))) / (2 * √(2 * log(2)))
	beam   = exp(- (sig^2) * (magk.^2) / 2)
	beamSQ = abs2(beam)	
	cNN    = nugget_at_each_pixel * dx ./ beamSQ
	
	index  = ceil(magk)
	index[find(index.==0)] = 1
	function makecXX(ells, clsXX, indexs)
		logCXX = linear_interp1(ells, log(clsXX), indexs)
		logCXX[find(logCXX .== 0)]  = -Inf
		logCXX[find(isnan(logCXX))] = -Inf
		return exp(logCXX)
	end
	cPP        = makecXX(cls["ell"], cls["pp"], index)
    cPhatNoise = makecXX(cls["ell"], cls["hatnoise"], index)
	cPhatP     = makecXX(cls["ell"], cls["hatp_cross_p"], index)
    cPhatPhat  = makecXX(cls["ell"], cls["hatphatp"], index)
	cTT        = makecXX(cls["ell"], cls["tt"], index)
    # cTTobs    = cNN + makecXX(cls["ell"], cls_ln["tt"], index)
	cTTobs     = cNN + cTTln(cTT, cPP)

    #!!!!! make sure the weight on ell=0 is zero
    cPP[magk .< magk[1,2]] = 0.0
    cTT[magk .< magk[1,2]] = 0.0
    cTTobs[magk .< magk[1,2]] = 0.0
    cPhatP[magk .< magk[1,2]] = 0.0
    cPhatPhat[magk .< magk[1,2]] = 0.0
    cNN[magk .< magk[1,2]] = 0.0
    cPhatNoise[magk .< magk[1,2]] = 0.0

	cTT, cTTobs, cNN, cPP, cPhatPhat, cPhatP, cPhatNoise
end

#= Here we just check to make sure that my cTTlen is correct

semilogy(cls["ell"], cls["ell"].^2 .* cls_ln["tt"], label = "cTTlen")
semilogy(cls["ell"], cls["ell"].^2 .* cls["tt"], label = "cTT")
semilogy(magk[1:300,1], magk[1:300,1].^2 .* ((cTTobs - cNN)[1:300, 1]), label = "my lensing spec")

=#

""" # lensing simulation """
include(srcpath * "lensing_sim.jl")
# now choose if you want all order simulations or not
# const T, tldT, Tobs, ϕ, hatϕ = all_ord_len()
const T, tldT, Tobs, ϕ, hatϕ = scnd_ord_len()





##################################################
# 
#  Preliminaries are done. 
#  Make functions which compute Aell, N0T and hatT
#
##################################################


function Aell_forT(cTTobs, cPhatPhat, cPhatP, cPP)
    τsq     = sum( (k1.^2) .* cPP .* dk ./ ((2π)^d) )
    kdic    = [1 => k1.*k1, 2 => k1.*k2, 3 => k2.*k2]
    tmpAx   = Array(Complex{Float64}, size(cTTobs))
    preApqk = squash!(abs2(cPhatP) ./ cPhatPhat)
    tmpBx   = ifft2(squash!(1 ./ cTTobs), deltk)
    rtn     = zeros(Complex{Float64}, size(cTTobs))
    for cntr = 1:3
        tmpAx[:,:]  = ifft2(kdic[cntr] .* preApqk, deltk)
        rtn[:,:]   += (cntr == 2?2:1) .* kdic[cntr] .* fft2(tmpAx .* conj(tmpBx), deltx)
    end
    rtn[:,:] .*= exp(- abs2(magk) * τsq)
    rtn[:,:] ./= (2π)^(d/2)
    return  squash!(1 ./ real(rtn)) 
end


function N0obs_forT(Tobs, hatϕ, cTTobs, cPhatPhat, cPhatP, cPP)
    τsq     = sum( (k1.^2) .* cPP .* dk ./ ((2π)^d) )
    kdic    = [1 => k1.*k1, 2 => k1.*k2, 3 => k2.*k2]
    tmpAx   = Array(Complex{Float64}, size(cTTobs))
    preApqk = squash!(abs2(hatϕ*deltk) .* abs2(cPhatP) ./ abs2(cPhatPhat))
    tmpBx   = ifft2(squash!(abs2(Tobs*deltk) ./ abs2(cTTobs)),deltk)
    rtn     = zeros(Complex{Float64}, size(cTTobs))
    for cntr = 1:3
        tmpAx[:,:]  = ifft2(kdic[cntr] .* preApqk, deltk)
        rtn[:,:]   += (cntr == 2?2:1) .* kdic[cntr] .* fft2(tmpAx .* conj(tmpBx), deltx)
    end
    rtn[:,:] .*= exp(- abs2(magk) * τsq)
    rtn[:,:] .*= abs2(Aell_forT(cTTobs, cPhatPhat, cPhatP, cPP))
    rtn[:,:] ./= (2π)^(d/2)
    return  real(rtn) 
end


function N0_forT(cTTobs, cPhatPhat, cPhatP, cPP)
    τsq     = sum( (k1.^2) .* cPP .* dk ./ ((2π)^d) )
    kdic    = [1 => k1.*k1, 2 => k1.*k2, 3 => k2.*k2]
    tmpAx   = Array(Complex{Float64}, size(cTTobs))
    preApqk = squash!(abs2(cPhatP) ./ cPhatPhat)
    tmpBx   = ifft2(squash!(1 ./ cTTobs), deltk)
    rtn     = zeros(Complex{Float64}, size(cTTobs))
    for cntr = 1:3
        tmpAx[:,:]  = ifft2(kdic[cntr] .* preApqk, deltk)
        rtn[:,:]   += (cntr == 2?2:1) .* kdic[cntr] .* fft2(tmpAx .* conj(tmpBx), deltx)
    end
    rtn[:,:] .*= exp(- abs2(magk) * τsq)
    rtn[:,:] .*= abs2(Aell_forT(cTTobs, cPhatPhat, cPhatP, cPP))
    rtn[:,:] ./= (2π)^(d/2)
    return  real(rtn) 
end


function hatT_fun(Tobs, hatϕ, cTTobs, cPhatPhat, cPhatP, cPP)
    τsq      = sum( (k1.^2) .* cPP .* dk ./ ((2π)^d) )
    kdic     = [1 => k1, 2 => k2]
    Axdic    = [1 => ifft2(squash!(im .* k1 .* hatϕ .* cPhatP ./ cPhatPhat), deltk),
                2 => ifft2(squash!(im .* k2 .* hatϕ .* cPhatP ./ cPhatPhat), deltk)]
    Bx_term1 = ifft2(squash!(Tobs ./ cTTobs), deltk)
    rtn      = zeros(Complex{Float64}, size(cTTobs))
    for q = 1:2
        rtn[:,:]  +=  kdic[q] .* fft2(Axdic[q] .* conj(Bx_term1), deltx)
    end
    rtn[:,:] .*= Aell_forT(cTTobs, cPhatPhat, cPhatP, cPP)
    rtn[:,:] .*= - im .* exp(- abs2(magk) * τsq / 2)
    return  rtn 
end





##################################################
# 
#  Generate a new lensed T and phi pair for use with exploring the missing 4 point term
#
##################################################



Aell  = Aell_forT(cTTobs, cPhatPhat, cPhatP, cPP)
N0obs = N0obs_forT(Tobs, hatϕ, cTTobs, cPhatPhat, cPhatP, cPP)
# N0thy = N0_forT(cTTobs, cPhatPhat, cPhatP, cPP)
hatT  = hatT_fun(Tobs, hatϕ, cTTobs, cPhatPhat, cPhatP, cPP)

# _, _, Tobstest, _, _ = scnd_ord_len()
# N0ind = N0obs_forT(Tobstest, hatϕ, cTTobs, cPhatPhat, cPhatP, cPP)


function makeN1(bin_edg)
    N1 = zeros(Float64, length(bin_edg))
    numSim = 10
    for cnt = 1:numSim
        T1, tldT1, T1obs, ϕ1, hatϕ1 = scnd_ord_len()
        T2, tldT2, T2obs, ϕ2, hatϕ2 = scnd_ord_len()
        # T1, tldT1, T1obs, ϕ1, hatϕ1 = all_ord_len()
        # T2, tldT2, T2obs, ϕ2, hatϕ2 = all_ord_len()
        

        # --- this will give N1
        hatT12  = hatT_fun(tldT1, ϕ2, cTTobs, cPhatPhat, cPhatP, cPP)
        hatT21  = hatT_fun(tldT2, ϕ1, cTTobs, cPhatPhat, cPhatP, cPP)
        # --- this will give the signal.... doesn't work..
        #hatT12  = hatT_fun(tldT1, ϕ1, cTTobs, cPhatPhat, cPhatP, cPP)
        #hatT21  = hatT_fun(tldT2, ϕ2, cTTobs, cPhatPhat, cPhatP, cPP)
        # --- this will give N0
        # hatT12  = hatT_fun(T1obs, ϕ2, cTTobs, cPhatPhat, cPhatP, cPP)
        # hatT21  = hatT_fun(T1obs, ϕ2, cTTobs, cPhatPhat, cPhatP, cPP)

        N1     += binave(real(hatT12 .*  conj(hatT21) .* dk), magk, bin_edg)
    end
    N1 ./= numSim
    return N1
end


figure(1)
    bin_edg       = (4 * deltk):(4 * deltk):6000
    T_minus_hatT  = binave(abs2(T - hatT) .* dk, magk, bin_edg)
    T_minus_tldT  = binave(abs2(T - tldT) .* dk, magk, bin_edg)
    N0obs_bin     = binave(N0obs, magk, bin_edg)
    # N0thy_bin     = binave(N0thy, magk, bin_edg)
    # N0ind_bin     = binave(N0ind, magk, bin_edg)
    # N1            = makeN1(bin_edg)

    loglog(magk[1:300,1], magk[1:300,1] .* cTT[1:300,1], label = "primordial TT")
    loglog(magk[1:300,1], magk[1:300,1] .* cNN[1:300,1], label = "noise")
    loglog(magk[1:300,1], magk[1:300,1] .* Aell[1:300,1] , ":", label = "Aell variance")
    
    loglog(bin_edg, bin_edg .* T_minus_hatT,    "." , label = "actual estimation error")
    loglog(bin_edg, bin_edg .* T_minus_tldT,    "." , label = "nominal difference")
    loglog(bin_edg, bin_edg .* N0obs_bin ,      "." , label = "N0obs")
    # loglog(bin_edg, bin_edg .* N0thy_bin ,      "--" , label = "N0thy")
    # loglog(bin_edg, bin_edg .* N1 ,      "." , label = "N1")
    legend(loc = "best")


# this just shows the average displacement size. Useful when your playing around with different lensing models.
let
    d1 = ifft2r(im .* k1 .* ϕ)
    d2 = ifft2r(im .* k2 .* ϕ)
    println("Average displacement = $(mean(√(d1.^2 + d2.^2)) * (60 * 180 / π))")
end

figure(2)
    subplot(2,2,1)
    imshow(ifft2r(T))
    colorbar()
    subplot(2,2,2)
    imshow(ifft2r(hatT .*(2 .< magk.< 2400)))
    colorbar()
    subplot(2,2,3)
    imshow(ifft2r((hatT-T) .*(2 .< magk.< 2400)))
    colorbar()
    subplot(2,2,4)
    imshow(ifft2r((tldT-T) .*(2 .< magk.< 2400)))
    colorbar()



##################################################
# 
#  Test the unbiasedness
#
##################################################
#=

function testEst(T)
    estT = zeros(T)
    numSim = 5
    for cnt = 1:numSim
        T1, tldT1, T1obs, ϕ1, hatϕ1 = scnd_ord_len(T)
        estT += hatT_fun(T1obs, hatϕ1, cTTobs, cPhatPhat, cPhatP, cPP)
    end
    estT ./= numSim
    return estT
end
estT = testEst(T)



figure()
    bin_edg       = (4 * deltk):(4 * deltk):6000
    T_minus_hatT  = binave(abs2(T - hatT) .* dk, magk, bin_edg)
    T_minus_estT  = binave(abs2(T - estT) .* dk, magk, bin_edg)
    N0obs_bin     = binave(N0obs, magk, bin_edg)

    loglog(magk[1:300,1], magk[1:300,1] .* cTT[1:300,1], label = "primordial TT")
    loglog(magk[1:300,1], magk[1:300,1] .* cNN[1:300,1], label = "noise")
    loglog(bin_edg, bin_edg .* T_minus_hatT,    "." , label = "actual estimation error")
    loglog(bin_edg, bin_edg .* T_minus_estT,    "." , label = "T_minus_estT")
    loglog(bin_edg, bin_edg .* N0obs_bin ,      "." , label = "N0obs")
    legend(loc = "best")


=#




