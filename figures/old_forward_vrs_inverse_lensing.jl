""" 
# check out forward and inverse lensing potentials
To really explore the difference between forward and inverse lensing we 
need to using a high resolution ϕ and ψ. The main issue is that I want to quantify how much
ϕ power will leak into ψ power. The reason I needed this is to decide on what prior to use
on the inverse lensing potentials.  This might be quantifiable analytically by a similar method 
to that of how to compute lensed E and B spectral densities.  

Questions:

	* Compare the de-lensing by ϕ and ψ in the spectral domain.
		In particular, see the spectral power of the de-lensing B mode.

"""


""" # parameters of the simulation run  """
srcpath  =  "/Users/ethananderes/Dropbox/BLimit/src/"
include(srcpath * "Interp.jl")
include(srcpath * "fft.jl")
include(srcpath * "funcs.jl")
using PyPlot, Interp

""" # Grid generation """
const pixel_size_arcmin = 0.5
const n = 2^9
const d = 2
const deltx   = pixel_size_arcmin * pi / (180 * 60) #this is in radians
const period  = deltx * n
const deltk   = 2π / period  
const dk      = deltk ^ d
const dx      = deltx ^ d
const nyq     = 2π / (2deltx)
const x, y    = meshgrid([0:n-1] * deltx, [0:n-1] * deltx)
const k1, k2  = linspace(-nyq, nyq-deltk, int(n))  |> fftshift |> x->meshgrid(x,x)
const magk    = sqrt(k1.^2 .+ k2.^2)
const φ2_l    = 2.0 * angle(k1 + im * k2)

""" # Spectrums  """
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
    "l_max_scalars" => 9_000,
    "l_max_tensors" => 9_000,
    "A_s"           => 2.3e-9,
    "n_s"           => 0.9624, 
    "h"             => 0.6711,
    "omega_b"       => 0.022068,
    "omega_cdm"     => 0.12029,
    "r"             => 0.2 ]
cosmo[:set](params)
cosmo[:compute]()
cls = cosmo[:raw_cl](8_000)
cls["tt"] = cls["tt"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls["ee"] = cls["ee"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls["bb"] = cls["bb"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls["te"] = cls["te"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls["tp"] = cls["tp"] * (10^6 * cosmo[:T_cmb]()) 
cls["ps"] = 0.05 * cls["pp"]
 

cls_ln = cosmo[:lensed_cl](8_000)
cls_ln["tt"] = cls_ln["tt"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls_ln["ee"] = cls_ln["ee"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls_ln["bb"] = cls_ln["bb"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls_ln["te"] = cls_ln["te"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls_ln["tp"] = cls_ln["tp"] * (10^6 * cosmo[:T_cmb]()) 



""" # first make a very high resolution forward lens """
dispxhr, dispyhr, ϕxhr, ψxhr, ϕhr, ψhr, xhr, yhr = let
	nhr        = 4 * n
	deltxhr    = (1/4) * deltx
	periodhr   = deltxhr * nhr
	deltkhr    = 2π / periodhr  
	dkhr       = deltkhr ^ 2
	dxhr       = deltxhr ^ 2
	nyqhr      = 2π / (2deltxhr)
	xhr, yhr   = meshgrid([0:nhr-1] * deltxhr, [0:nhr-1] * deltxhr)
	k1hr, k2hr = linspace(-nyqhr, nyqhr-deltkhr, int(nhr))  |> fftshift |> x->meshgrid(x,x)
	magkhr     = sqrt(k1hr.^2 .+ k2hr.^2)

	# make the spectral matrices
	index  = ceil(magkhr)
	index[find(index.==0)] = 1

	logCPh = linear_interp1(cls["ell"], log(cls["pp"]), index)
	logCPh[find(logCPh .== 0)]  = -Inf
	logCPh[find(isnan(logCPh))] = -Inf
	cPh = exp(logCPh)
	
	logCPs = linear_interp1(cls["ell"], log(cls["ps"]), index)
	logCPs[find(logCPs .== 0)]  = -Inf
	logCPs[find(isnan(logCPs))] = -Inf
	cPs = exp(logCPs)

	ϕhr    =  √(cPh) .* fft2(randn(size(xhr))./ √(dxhr), deltxhr) 
	ψhr    =  √(cPs) .* fft2(randn(size(xhr))./ √(dxhr), deltxhr) 

	dispxhr   = ifft2r(im .* k1hr .* ϕhr, deltkhr)  + ifft2r(im .* k2hr .* ψhr, deltkhr)
	dispyhr   = ifft2r(im .* k2hr .* ϕhr, deltkhr)  - ifft2r(im .* k1hr .* ψhr, deltkhr)
	
	dispxhr, dispyhr, ifft2r(ϕhr, deltkhr),  ifft2r(ψhr, deltkhr), ϕhr, ψhr, xhr, yhr # this gets returned
end



""" # Now construct the inverse lensing displacement """
using PyCall
@pyimport scipy.interpolate as scii
function griddata(x::Matrix, y::Matrix, z::Matrix, xi::Matrix,yi::Matrix)
	xpd, ypd = Interp.perodic_padd_xy(x, y, 0.1)
	zpd = Interp.perodic_padd_z(z, 0.1)
	points = [xpd[:] ypd[:]]
	# grid = (vec(xi[1,:]), vec(yi[:,1]))
	grid = (xi, yi)
	zi = scii.griddata(points, zpd[:], grid, method = "cubic")
end
lensex = xhr + dispxhr
lensey = yhr + dispyhr
antix_dsp = griddata(lensex, lensey, -dispxhr, x, y)
antiy_dsp = griddata(lensex, lensey, -dispyhr, x, y)

function helmholtz(ax::Matrix, bx::Matrix)
	# (ax,bx) is the vector field defined in pixel space
	ak, bk = fft2(ax), fft2(bx)
	divk = squash!( (k1 .* ak + k2 .* bk) ./ (im * (k1.^2 + k2.^2)) )
	crlk = squash!( (ak - im .* k1 .* divk) ./ (im * k2) )
	divk[magk .<= 0.0] = 0.0
	crlk[magk .<= 0.0] = 0.0
	divx, crlx = ifft2r(divk), ifft2r(crlk)
	divx, divk, crlx, crlk
end
divx, divk, crlx, crlk = helmholtz(antix_dsp, antiy_dsp)



""" 
# plot the difference 
Question: the sign on the inverse lensing potential looks wrong. Maybe I defined it different in the Helmholtz decomp.
"""
subplot(2,2,1)
imshow(-ϕxhr);title("forward phi");colorbar(format="%.e")
subplot(2,2,2)
imshow(divx);title("inverse phi");colorbar(format="%.e")
subplot(2,2,3)
imshow(-ψxhr);title("forward psi");colorbar(format="%.e")
subplot(2,2,4)
imshow(crlx);title("inverse psi");colorbar(format="%.e")



""" # compare the spectral density of the inverse potentials with that of forward potentials """

loglog(cls["ell"], (cls["ell"].^4) .*  cls["pp"] / (2π), "b--", label = L"C_l^{\phi\phi}")
loglog(cls["ell"], (cls["ell"].^4) .*  cls["ps"] / (2π), "g--", label = L"C_l^{\psi\psi}")

loglog(110:100:9000, binave((magk .^ 4) .* abs2(divk*deltk) / (2π), magk, 110:100:9000), "b.", label = L"inv \phi")
loglog(110:100:9000, binave((magk .^ 4) .* abs2(crlk*deltk) / (2π), magk, 110:100:9000), "g.", label = L"inv \psi")
legend(loc = "best")



""" # Great. The forward and inverse stream functions look very close.  
There is a small bit of extra power in the stream function. This was expected. I bet we can come up 
with an analytical approximation to that added power. However, it is so small, that I'm guessing 
it is currently sub-dominant to the statistical uncertainty.
"""



""" # To further understand the difference, compare de-lensing with both the inverse lense and the forward lense """
unlensedQUdata, tldQUdata, ϕ, ψ = let
	nhr        = 4 * n
	deltxhr    = (1/4) * deltx
	periodhr   = deltxhr * nhr
	deltkhr    = 2π / periodhr  
	dkhr       = deltkhr ^ 2
	dxhr       = deltxhr ^ 2
	nyqhr      = 2π / (2deltxhr)
	xhr, yhr   = meshgrid([0:nhr-1] * deltxhr, [0:nhr-1] * deltxhr)
	k1hr, k2hr = linspace(-nyqhr, nyqhr-deltkhr, int(nhr))  |> fftshift |> x->meshgrid(x,x)
	magkhr     = sqrt(k1hr.^2 .+ k2hr.^2)
	φ2_lhr    = 2.0 * angle(k1hr + im * k2hr)

	# make the spectral matrices
	index  = ceil(magkhr)
	index[find(index.==0)] = 1

	logCBB = linear_interp1(cls["ell"],log(cls["bb"]), index)
	logCBB[find(logCBB .== 0)]  = -Inf 
	logCBB[find(isnan(logCBB))] = -Inf
	cBBhr = exp(logCBB);
	
	logCEE = linear_interp1(cls["ell"],log(cls["ee"]), index)
	logCEE[find(logCEE .== 0)]  = -Inf  
	logCEE[find(isnan(logCEE))] = -Inf
	cEEhr = exp(logCEE)

	# here we are using the high resolution lensing potentials simulated previously
	ϕx     = ifft2r(ϕhr, deltkhr)[1:4:end, 1:4:end]
	ψx     = ifft2r(ψhr, deltkhr)[1:4:end, 1:4:end]
	dispxhr   = ifft2r(im .* k1hr .* ϕhr, deltkhr)  + ifft2r(im .* k2hr .* ψhr, deltkhr)
	dispyhr   = ifft2r(im .* k2hr .* ϕhr, deltkhr)  - ifft2r(im .* k1hr .* ψhr, deltkhr)

    E    =  √(cEEhr) .* fft2(randn(size(xhr))./ √(dxhr), deltxhr) 
	B    =  √(cBBhr) .* fft2(randn(size(xhr))./ √(dxhr), deltxhr)  
	Q    = - E .* cos(φ2_lhr) + B .* sin(φ2_lhr)
	U    = - E .* sin(φ2_lhr) - B .* cos(φ2_lhr)
	Qx   = ifft2r(Q, deltkhr)
	Ux   = ifft2r(U, deltkhr)

	tldQx = spline_interp2(xhr, yhr, Qx, xhr +  dispxhr, yhr +  dispyhr)
	tldUx = spline_interp2(xhr, yhr, Ux, xhr +  dispxhr, yhr +  dispyhr)

    unlensedQUdata = QUandFriends(Qx[1:4:end,1:4:end],    Ux[1:4:end,1:4:end])	
    tldQUdata      = QUandFriends(tldQx[1:4:end,1:4:end], tldUx[1:4:end,1:4:end])	
	
	unlensedQUdata, tldQUdata, fft2(ϕx, deltx), fft2(ψx, deltx)
end # let
Ex    = - fft2(unlensedQUdata.Qx)    .* cos(φ2_l) - fft2(unlensedQUdata.Ux)    .* sin(φ2_l)  |> ifft2r
Bx    =   fft2(unlensedQUdata.Qx)    .* sin(φ2_l) - fft2(unlensedQUdata.Ux)    .* cos(φ2_l)  |> ifft2r
tldEx = - fft2(tldQUdata.Qx) .* cos(φ2_l) - fft2(tldQUdata.Ux) .* sin(φ2_l) |> ifft2r
tldBx =   fft2(tldQUdata.Qx) .* sin(φ2_l) - fft2(tldQUdata.Ux) .* cos(φ2_l) |> ifft2r


QϕψxAnti, UϕψxAnti, _, _, _, _   = easylense(tldQUdata, -ϕ, -ψ) 
QϕψxDel, UϕψxDel, _, _, _, _     = easylense(tldQUdata, fft2(divx), fft2(crlx)) 
est_BAnti   =   fft2(QϕψxAnti) .* sin(φ2_l) - fft2(UϕψxAnti) .* cos(φ2_l) 
est_BDel    =   fft2(QϕψxDel) .* sin(φ2_l) - fft2(UϕψxDel) .* cos(φ2_l) 
tldB        =   fft2(tldBx)

uplim = 6000
llim = 10

loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls_ln["ee"][llim:uplim]  / (2π), "g-", label = L"\widetilde C_l^{EE}")
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls["ee"][llim:uplim]  / (2π), "g--", label = L" C_l^{EE}")
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls_ln["bb"][llim:uplim]  / (2π), "b-", label = L"\widetilde C_l^{BB}")
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls["bb"][llim:uplim]  / (2π), "b--", label = L"C_l^{BB}")


loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(tldB*deltk) / (2π), magk, llim:deltk:uplim), "b.", label = L"tldB")
loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(est_BAnti*deltk) /(2π), magk, llim:deltk:uplim), "o", markeredgecolor = "blue",  mfc="none", label = L"est B Anti")
loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(est_BDel*deltk) /(2π), magk, llim:deltk:uplim), "o", markeredgecolor = "red",  mfc="none", label = L"est B Del")
loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(fft2(Bx)*deltk) /(2π), magk, llim:deltk:uplim), "*", markeredgecolor = "blue",  mfc="none", label = L"true B")
axis("tight")
legend(loc = "best")


""" 
# You should further test easylense etc... 
Basically, I don't know if:
	-- easylense is the problem, 
	-- or if divx and crlx are just not well approximated, 
	-- or if de-lensing is just really numeric
"""


""" 
To probe the difference between easylense and reallense let's 
look at the difference in lensed B-mode power from easylense and reallense. 
If it is less than the excess de-lense power then I think the above delensing descrepancy could be due
to something other than a difference between reallense and easylense
 """
easylensedQx, easylensedUx, _, _, _, _   = easylense(unlensedQUdata, ϕ, ψ)
# compare with tldQUdata

easylensedB   =   fft2(easylensedQx) .* sin(φ2_l) - fft2(easylensedUx) .* cos(φ2_l) 
tldB          =   fft2(tldBx)


uplim = 6000
llim = 10
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls_ln["ee"][llim:uplim]  / (2π), "g-")
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls["ee"][llim:uplim]  / (2π), "g--", label = L" C_l^{EE}")
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls_ln["bb"][llim:uplim]  / (2π), "b-", label = L"\widetilde C_l^{BB}")
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls["bb"][llim:uplim]  / (2π), "b--", label = L"C_l^{BB}")


# loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(tldB*deltk) / (2π), magk, llim:deltk:uplim), "b.", label = L"reallense B")
loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(easylensedB*deltk) /(2π), magk, llim:deltk:uplim), "o", markeredgecolor = "red",  mfc="none", label = L"easylense B")
loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(est_BAnti*deltk) /(2π), magk, llim:deltk:uplim), label = L"easy delense B with $\phi,\psi$")
loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(est_BDel*deltk) /(2π), magk, llim:deltk:uplim), label = L"easy delense B with $\phi^{inv},\psi^{inv}$ ")
loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(easylensedB*deltk - tldB*deltk) /(2π), magk, llim:deltk:uplim), label = "residual easy-real")


axis("tight")
legend(loc = "best")





