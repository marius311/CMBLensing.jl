"""
Test out B-mode delensing limit
=======================================================
ToDo:
	* Try a quadratic delenser in the Wiener filtering step...
	* To probe the determinant terms and the difference between inverse lensing etc, just 
		consider estimating a periodic geometric anisotropy using this anti-genometic idea.
	* Running B-mode delensing when C_l^BB = 0 and comparing to "perfect"-antilensing
	  makes me think the MLE estimates are estimating anti-lensing (since the spectra look the same).
		-- Is it possible that the determinant approximation is causing this? or is there a major 
		   conceptual problem I'm missing in the algorithm.
		-- Is it possible that the extra high frequency power of the inverse lense 
			is crucial to delensing the B (which would explain why I can't delense past antilense).
	* how sensitive is the algorithm to knowing the wrong r value. 
	* very sensitive to lmax on E, B. Could this be due to the taylor expansion...
	* update r with a gradient step
	* Program the Hamiltonian sampler of ϕinv and ψinv, with known r.
	* What is the role of the determinant term.
	* Maybe there is a linear form of the int+taylor lensing which can be applied to Wiener filtering
"""


""" ## Load modules, paths and functions """
# seed = Uint32[1461298913,3804461922,1461874617,4101139687]
# srand(seed)

#simdir   =  "/Users/ethananderes/Dropbox/BLimit/simulations_$(seed[1])/" # contains the simulation
srcpath  =  "/Users/ethananderes/Dropbox/BLimit/src/"
savepath =  "/Users/ethananderes/Dropbox/BLimit/paper/"

include(srcpath * "Interp.jl")
include(srcpath * "fft.jl")
include(srcpath * "funcs.jl")
using PyPlot, Interp




""" ## parameters of the simulation run  """
const streammultiplier  = 0.0000000000001  # change this eventually
const pixel_size_arcmin = 0.5
const beamFWHM = 2.0
const n = 2^10
const nugget_at_each_pixel = (4.0)^2
const r = 0.15



""" ## Grid generation """
const d = 2
const deltx   = pixel_size_arcmin * π / (180 * 60) #this is in radians
const period  = deltx * n
const deltk   = 2π / period  
const dk      = deltk ^ d
const dx      = deltx ^ d
const nyq     = 2π / (2deltx)
const x, y    = meshgrid([0:n-1] * deltx, [0:n-1] * deltx)
const k1, k2  = linspace(-nyq, nyq-deltk, int(n))  |> fftshift |> x->meshgrid(x,x)
const magk    = √(k1.^2 .+ k2.^2)
const φ2_l    = 2.0 * angle(k1 + im * k2)



""" ## set l max """
const maskupC = 2000
const maskupP = 3000  # l_max for for phi



""" ## Spectrums  """
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
cls_ln["tt"] = cls_ln["tt"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls_ln["ee"] = cls_ln["ee"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls_ln["bb"] = cls_ln["bb"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls_ln["te"] = cls_ln["te"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls_ln["tp"] = cls_ln["tp"] * (10^6 * cosmo[:T_cmb]()) 

cls = cosmo[:raw_cl](7_000)
cls["tt"] = cls["tt"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls["ee"] = cls["ee"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls["bb"] = cls["bb"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls["te"] = cls["te"] * (10^6 * cosmo[:T_cmb]()) ^ 2 
cls["tp"] = cls["tp"] * (10^6 * cosmo[:T_cmb]()) 

# add a field rotation
cls["ps"] = streammultiplier * cls["pp"]


""" # Here are plots of the lensed and unlensed spectra """ 
#=
loglog(cls["ell"] .* (cls["ell"] + 1) .* cls["tt"] / 2π, "r--", label = L"C_l^{TT}")
loglog(cls["ell"] .* (cls["ell"] + 1) .* cls["ee"] / 2π, "g--", label = L"C_l^{EE}")
loglog(cls["ell"] .* (cls["ell"] + 1) .* cls["bb"] / 2π, "b--", label = L"C_l^{BB}")

loglog(cls_ln["ell"] .* (cls_ln["ell"] + 1) .* cls_ln["tt"] / 2π, "r-", label = L"\widetilde C_l^{TT}")
loglog(cls_ln["ell"] .* (cls_ln["ell"] + 1) .* cls_ln["ee"] / 2π, "g-", label = L"\widetilde C_l^{EE}")
loglog(cls_ln["ell"] .* (cls_ln["ell"] + 1) .* cls_ln["bb"] / 2π, "b-", label = L"\widetilde C_l^{BB}")
legend(loc = "best")
=#


""" # Make the spectral matrices  """
const cEE, cBB, cPh, cPs, cNT, cNE = let
	sig    = (beamFWHM * (π / (180 * 60))))
	beamSQ   = exp(- (sig ^ 2) * (magk .^ 2) / (8*log(2)) )
	cNT    = nugget_at_each_pixel * dx ./ beamSQ
	cNE    = √(2) * nugget_at_each_pixel * dx ./ beamSQ
	
	index  = ceil(magk)
	index[find(index.==0)] = 1
	
	logCPh = linear_interp1(cls["ell"], log(cls["pp"]), index)
	logCPh[find(logCPh .== 0)]  = -Inf
	logCPh[find(isnan(logCPh))] = -Inf
	cPh = exp(logCPh)
	
	logCPs = linear_interp1(cls["ell"], log(cls["ps"]), index)
	logCPs[find(logCPs .== 0)]  = -Inf
	logCPs[find(isnan(logCPs))] = -Inf
	cPs = exp(logCPs)
	
	logCBB = linear_interp1(cls["ell"],log(cls["bb"]), index)
	logCBB[find(logCBB .== 0)]  = -Inf 
	logCBB[find(isnan(logCBB))] = -Inf
	cBB = exp(logCBB);
	
	logCEE = linear_interp1(cls["ell"],log(cls["ee"]), index)
	logCEE[find(logCEE .== 0)]  = -Inf  
	logCEE[find(isnan(logCEE))] = -Inf
	cEE = exp(logCEE)

	cEE, cBB, cPh, cPs, cNT, cNE
end


 
#= 
""" 
# Simulate with integer displacement and Taylor approximations
"""
unlensedQUdata, lQUdata, tldQUdata,  ϕ, ψ = let

	# simulate unlensed CMB
	E    =  √(cEE) .* fft2(randn(size(x))./ √(dx)) 
	B    =  √(cBB) .* fft2(randn(size(x))./ √(dx)) 
	Q    = - E .* cos(φ2_l) + B .* sin(φ2_l)
	U    = - E .* sin(φ2_l) - B .* cos(φ2_l)
	unlensedQUdata = QUandFriends(Q, U)
	
	# simulation lensing potentials
	ϕ    =  √(cPh) .* fft2(randn(size(x))./ √(dx)) 
	ψ    =  √(cPs) .* fft2(randn(size(x))./ √(dx)) 
	
	# convert to  displacements
	displx   = ifft2r(im .* k1 .* ϕ) + ifft2r(im .* k2 .* ψ)
	disply   = ifft2r(im .* k2 .* ϕ) - ifft2r(im .* k1 .* ψ)
	
	# Decompose the lensing displacements """
	row, col = size(x)
	rdisplx  = Array(Float64, row, col)
	rdisply  = Array(Float64, row, col)
	indcol   = Array(Int64, row, col)
	indrow   = Array(Int64, row, col)
	decomplense!(indcol, indrow, rdisplx, rdisply, x, y, displx, disply, deltx)
	
	# plot of subgrid displacements
	# plot(circshift(rdisplx[:,300],0)[1:90],".") 
	
	# do the integer lensing
	lQUdata = gridlense(unlensedQUdata, indcol, indrow)
	
	# do the taylor expansion lensing and put everything in a QU object
	tldQx  = lQUdata.Qx 
	tldQx += (lQUdata.∂1Qx .* rdisplx) 
	tldQx += (lQUdata.∂2Qx .* rdisply)
	tldQx += 0.5 * (rdisplx .* lQUdata.∂11Qx .* rdisplx ) 
	tldQx +=       (rdisplx .* lQUdata.∂12Qx .* rdisply ) 
	tldQx += 0.5 * (rdisply .* lQUdata.∂22Qx .* rdisply ) 
	
	tldUx  = lQUdata.Ux 
	tldUx += (lQUdata.∂1Ux .* rdisplx) 
	tldUx += (lQUdata.∂2Ux .* rdisply)
	tldUx += 0.5 * (rdisplx .* lQUdata.∂11Ux .* rdisplx ) 
	tldUx +=       (rdisplx .* lQUdata.∂12Ux .* rdisply ) 
	tldUx += 0.5 * (rdisply .* lQUdata.∂22Ux .* rdisply ) 
	
	tldQUdata = QUandFriends(tldQx, tldUx)

	unlensedQUdata, lQUdata, tldQUdata,  ϕ, ψ
end # let

# convert to E, B
Ex    = - fft2(unlensedQUdata.Qx)    .* cos(φ2_l) - fft2(unlensedQUdata.Ux)    .* sin(φ2_l)  |> ifft2r
Bx    =   fft2(unlensedQUdata.Qx)    .* sin(φ2_l) - fft2(unlensedQUdata.Ux)    .* cos(φ2_l)  |> ifft2r
lEx   = - fft2(lQUdata.Qx)   .* cos(φ2_l) - fft2(lQUdata.Ux)   .* sin(φ2_l) |> ifft2r
lBx   =   fft2(lQUdata.Qx)   .* sin(φ2_l) - fft2(lQUdata.Ux)   .* cos(φ2_l) |> ifft2r
tldEx = - fft2(tldQUdata.Qx) .* cos(φ2_l) - fft2(tldQUdata.Ux) .* sin(φ2_l) |> ifft2r
tldBx =   fft2(tldQUdata.Qx) .* sin(φ2_l) - fft2(tldQUdata.Ux) .* cos(φ2_l) |> ifft2r

=#

""" # This shows the difference between unlensed B, integer lensed B and integer + 2nd taylor lensed B """
#=
figure(figsize = (12, 3))
subplot(1,3,1)
imshow(Bx[1:100,1:100]); title("unlensed B"); clim(-1.8, 1.8); colorbar(); 
subplot(1,3,2)
imshow(lBx[1:100,1:100]); title("int lensed B"); clim(-1.8, 1.8); colorbar(); 
subplot(1,3,3)
imshow(tldBx[1:100,1:100]); title("int+taylor lensed B"); clim(-1.8, 1.8); colorbar(); 
=#




"""  # An alternative to Taylor lensing: high resolution lensing  """
unlensedQUdata, tldQUdata, ϕ, ψ = let
	nhr        = 4 * n
	deltxhr    = deltx / 4 
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

	logCPh = linear_interp1(cls["ell"], log(cls["pp"]), index)
	logCPh[find(logCPh .== 0)]  = -Inf
	logCPh[find(isnan(logCPh))] = -Inf
	cPhhr = exp(logCPh)
	
	logCPs = linear_interp1(cls["ell"], log(cls["ps"]), index)
	logCPs[find(logCPs .== 0)]  = -Inf
	logCPs[find(isnan(logCPs))] = -Inf
	cPshr = exp(logCPs)

	ϕhr    =  √(cPhhr) .* fft2(randn(size(xhr))./ √(dxhr), deltxhr) 
	ψhr    =  √(cPshr) .* fft2(randn(size(xhr))./ √(dxhr), deltxhr) 
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
	
	unlensedQUdata, tldQUdata, fft2(ϕx, deltx),  fft2(ψx, deltx)
end # let
Ex    = - fft2(unlensedQUdata.Qx)    .* cos(φ2_l) - fft2(unlensedQUdata.Ux)    .* sin(φ2_l)  |> ifft2r
Bx    =   fft2(unlensedQUdata.Qx)    .* sin(φ2_l) - fft2(unlensedQUdata.Ux)    .* cos(φ2_l)  |> ifft2r
tldEx = - fft2(tldQUdata.Qx) .* cos(φ2_l) - fft2(tldQUdata.Ux) .* sin(φ2_l) |> ifft2r
tldBx =   fft2(tldQUdata.Qx) .* sin(φ2_l) - fft2(tldQUdata.Ux) .* cos(φ2_l) |> ifft2r










""" ## Lets check that the spectral density of lensed B looks right """
#=

tldB = fft2(tldBx)
# lB = fft2(lBx)
B = fft2(Bx)

loglog(cls["ell"], cls_ln["ell"] .* (cls_ln["ell"] + 1) .* cls_ln["ee"] / (2π), "g-", label = L"\widetilde C_l^{EE}")
loglog(cls["ell"], cls_ln["ell"] .* (cls_ln["ell"] + 1) .* cls_ln["bb"] / (2π), "b-", label = L"\widetilde C_l^{BB}")
loglog(cls["ell"], cls["ell"] .* (cls["ell"] + 1) .* cls["bb"] / (2π), "b--", label = L"C_l^{BB}")
loglog(cls["ell"], cls["ell"] .* (cls["ell"] + 1) .* cls["bb"] / (2π), "b--", label = L"C_l^{BB}")

loglog(110:100:9000, binave(magk .* (magk + 1) .* abs2(tldB*deltk) / (2π), magk, 110:100:9000), ".", label = L"tldB")
# loglog(110:100:9000, binave(magk .* (magk + 1) .* abs2(lB*deltk)   / (2π), magk, 110:100:9000), ".", label = L"lB")
loglog(110:100:9000, binave(magk .* (magk + 1) .* abs2(B*deltk)    / (2π), magk, 110:100:9000), ".", label = L"B")
legend(loc = "best")

=#








""" # Gradient updates   """

# make the M weights
# this should eventually be pushed to a module or hardwired into the grad calculations
const Mq  = squash!(- abs2(cos(φ2_l)) ./ cEE  - abs2(sin(φ2_l)) ./ cBB ) 
const Mu  = squash!(- abs2(cos(φ2_l)) ./ cBB  - abs2(sin(φ2_l)) ./ cEE )
const Mqu = squash!(2 * cos(φ2_l) .* sin(φ2_l) ./ cBB) - squash!(2 * cos(φ2_l) .* sin(φ2_l) ./ cEE) 

Mq[magk .>= maskupC]  = 0.0 
Mu[magk .>= maskupC]  = 0.0 
Mqu[magk .>= maskupC] = 0.0 

ϕcurr = zero(ϕ)
ψcurr = zero(ψ)
@time gradupdate!(ϕcurr, ψcurr, tldQUdata, 5000, 1e-6, 1e-6)




""" # images, normalized to see the structure of the updates when small """
#=

subplot(2,2,2)
	ϕₓ = ifft2r(ϕ)
	imshow(
		ϕₓ, 
  		interpolation = "nearest", 
  		origin="lower", 
  		extent=(180/π)*[minimum(x), maximum(x),minimum(x), maximum(x)]
	) 
	colorbar(format="%.e")
subplot(2,2,1)
	ϕcurrₓ = ifft2r(ϕcurr)
	imshow(
		ϕcurrₓ, 
  		interpolation = "nearest", 
  		origin="lower", 
  		extent=(180/π)*[minimum(x), maximum(x),minimum(x), maximum(x)]
	) 
	colorbar(format="%.e")
subplot(2,2,4)
	ψₓ = ifft2r(ψ)
	imshow(
		ψₓ, 
  		interpolation = "nearest",
  		origin="lower", 
  		extent=(180/π)*[minimum(x), maximum(x),minimum(x), maximum(x)]
	) 
	colorbar(format="%.e")
subplot(2,2,3)
	ψcurrₓ = ifft2r(ψcurr)
	imshow(
		ψcurrₓ, 
  		interpolation = "nearest", 
  		origin="lower", 
  		extent=(180/π)*[minimum(x), maximum(x),minimum(x), maximum(x)]
	) 
	colorbar(format="%.e")

=#



""" # images, un-normalized """
#=

subplot(2,2,2)
	ϕₓ = ifft2r(ϕ)
	imshow(
		ϕₓ, 
  		interpolation = "nearest", 
  		vmin=minimum(ϕₓ),
  		vmax=maximum(ϕₓ), 
  		origin="lower", 
  		extent=(180/π)*[minimum(x), maximum(x),minimum(x), maximum(x)]
	) 
	colorbar(format="%.e")
subplot(2,2,1)
	ϕcurrₓ = ifft2r(ϕcurr)
	imshow(
		ϕcurrₓ, 
  		interpolation = "nearest", 
  		vmin=minimum(ϕₓ),
  		vmax=maximum(ϕₓ), 
  		origin="lower", 
  		extent=(180/π)*[minimum(x), maximum(x),minimum(x), maximum(x)]
	) 
subplot(2,2,4)
	ψₓ = ifft2r(ψ)
	imshow(
		ψₓ, 
  		interpolation = "nearest", 
  		vmin=minimum(ψₓ),
  		vmax=maximum(ψₓ), 
  		origin="lower", 
  		extent=(180/π)*[minimum(x), maximum(x),minimum(x), maximum(x)]
	) 
	colorbar(format="%.e")
subplot(2,2,3)
	ψcurrₓ = ifft2r(ψcurr)
	imshow(
		ψcurrₓ, 
  		interpolation = "nearest", 
  		vmin=minimum(ψₓ),
  		vmax=maximum(ψₓ), 
  		origin="lower", 
  		extent=(180/π)*[minimum(x), maximum(x),minimum(x), maximum(x)]
	) 

=#



""" ## Lets check the delensed spectral density of lensed B """
#=

Qϕψx, Uϕψx, ∂1Qϕψx, ∂1Uϕψx, ∂2Qϕψx, ∂2Uϕψx    = easylense(tldQUdata, -ϕcurr, -ψcurr) 
# Qϕψx, Uϕψx, ∂1Qϕψx, ∂1Uϕψx, ∂2Qϕψx, ∂2Uϕψx    = easylense(tldQUdata, -ϕ, -ψ) 
est_E   = - fft2(Qϕψx) .* cos(φ2_l) - fft2(Uϕψx) .* sin(φ2_l) 
est_B   =   fft2(Qϕψx) .* sin(φ2_l) - fft2(Uϕψx) .* cos(φ2_l) 
tldB    =   fft2(tldBx)

uplim = 6000
llim = 10

loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls_ln["ee"][llim:uplim]  / (2π), "g-", label = L"\widetilde C_l^{EE}")
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls["ee"][llim:uplim]  / (2π), "g--", label = L" C_l^{EE}")
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls_ln["bb"][llim:uplim]  / (2π), "b-", label = L"\widetilde C_l^{BB}")
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls["bb"][llim:uplim]  / (2π), "b--", label = L"C_l^{BB}")

loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(tldB*deltk) / (2π), magk, llim:deltk:uplim), "b.", label = L"tldB")
loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(est_B*deltk) /(2π), magk, llim:deltk:uplim), "o", markeredgecolor = "blue",  mfc="none", label = L"est B")
loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(B*deltk) /(2π), magk, llim:deltk:uplim), "*", markeredgecolor = "blue",  mfc="none", label = L"true B")
axis("tight")
legend(loc = "best")

=#

""" ## Here is plot that combines the above into one diagnostic """

figure(figsize = (9,12))
subplot2grid((3,2), (0,1), colspan=1)
	ϕₓ = ifft2r(ϕ)
	imshow(
		ϕₓ, 
  		interpolation = "nearest", 
  		vmin=minimum(ϕₓ),
  		vmax=maximum(ϕₓ), 
  		origin="lower", 
  		extent=(180/π)*[minimum(x), maximum(x),minimum(x), maximum(x)]
	) 
	colorbar(format="%.e")
subplot2grid((3,2), (0,0), colspan=1)
	ϕcurrₓ = ifft2r(ϕcurr)
	imshow(
		ϕcurrₓ, 
  		interpolation = "nearest", 
  		vmin=minimum(ϕₓ),
  		vmax=maximum(ϕₓ), 
  		origin="lower", 
  		extent=(180/π)*[minimum(x), maximum(x),minimum(x), maximum(x)]
	) 
subplot2grid((3,2), (1,1), colspan=1)
	ψₓ = ifft2r(ψ)
	imshow(
		ψₓ, 
  		interpolation = "nearest", 
  		vmin=minimum(ψₓ),
  		vmax=maximum(ψₓ), 
  		origin="lower", 
  		extent=(180/π)*[minimum(x), maximum(x),minimum(x), maximum(x)]
	) 
	colorbar(format="%.e")
subplot2grid((3,2), (1,0), colspan=1)
	ψcurrₓ = ifft2r(ψcurr)
	imshow(
		ψcurrₓ, 
  		interpolation = "nearest", 
  		vmin=minimum(ψₓ),
  		vmax=maximum(ψₓ), 
  		origin="lower", 
  		extent=(180/π)*[minimum(x), maximum(x),minimum(x), maximum(x)]
	) 
subplot2grid((3,2), (2,0), colspan=2)
Qϕψx, Uϕψx, ∂1Qϕψx, ∂1Uϕψx, ∂2Qϕψx, ∂2Uϕψx    = easylense(tldQUdata, -ϕcurr, -ψcurr) 
est_E   = - fft2(Qϕψx) .* cos(φ2_l) - fft2(Uϕψx) .* sin(φ2_l) 
est_B   =   fft2(Qϕψx) .* sin(φ2_l) - fft2(Uϕψx) .* cos(φ2_l) 
tldB    =   fft2(tldBx)
uplim = 6000
llim = 10
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls_ln["ee"][llim:uplim]  / (2π), "g-", label = L"\widetilde C_l^{EE}")
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls["ee"][llim:uplim]  / (2π), "g--", label = L" C_l^{EE}")
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls_ln["bb"][llim:uplim]  / (2π), "b-", label = L"\widetilde C_l^{BB}")
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls["bb"][llim:uplim]  / (2π), "b--", label = L"C_l^{BB}")

loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(tldB*deltk) / (2π), magk, llim:deltk:uplim), "b.", label = L"tldB")
loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(est_B*deltk) /(2π), magk, llim:deltk:uplim), "o", markeredgecolor = "blue",  mfc="none", label = L"est B")
loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(fft2(Bx)*deltk) /(2π), magk, llim:deltk:uplim), "*", markeredgecolor = "blue",  mfc="none", label = L"true B")
axis("tight")
legend(loc = "best")







""" # This shows the difference between estimated B, true primordial B """
#=

figure(figsize = (12, 3))
subplot(1,3,1)
imshow(Bx); title("unlensed B"); clim(-0.7, 0.7); colorbar(); 
subplot(1,3,2)
imshow(ifft2r(est_B)); title("estaimted B"); clim(-0.7, 0.7); colorbar(); 
subplot(1,3,3)
imshow(tldBx); title("lensed B"); clim(-0.7, 0.7); colorbar(); 

=#


""" # Lets look at the spectral power of the lensing estimates """
#=

uplim = 1000
llim = round(deltk)

semilogy(cls["ell"][llim:uplim], (cls["ell"][llim:uplim].^4) .* cls["pp"][llim:uplim]  / (2π), "b--", label = L"C_l^{\phi\phi}")
semilogy(cls["ell"][llim:uplim], (cls["ell"][llim:uplim].^4) .* cls["ps"][llim:uplim]  / (2π), "g--", label = L"C_l^{\psi\psi}")

semilogy(llim:deltk:uplim, binave((magk.^4) .* abs2(ϕcurr*deltk) / (2π), magk, llim:deltk:uplim), "o", markeredgecolor = "blue",  mfc="none", label = L"est phi")
semilogy(llim:deltk:uplim, binave((magk.^4) .* abs2(ϕ*deltk) / (2π), magk, llim:deltk:uplim), "b.", label = L"true phi")
semilogy(llim:deltk:uplim, binave((magk.^4) .* abs2(ψcurr*deltk) / (2π), magk, llim:deltk:uplim),  "o", markeredgecolor = "green",  mfc="none",  label = L"est psi")
semilogy(llim:deltk:uplim, binave((magk.^4) .* abs2(ψ*deltk) / (2π), magk, llim:deltk:uplim), "g.", label = L"true psi")
legend(loc = "best")


=#


""" # How about trying to probe the noise and bias by setting r= 0. """

unlensedQUdata, tldQUdata, ϕtrue, ψtrue = let
	nhr        = 4 * n
	deltxhr    = deltx / 4 
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

	logCEE = linear_interp1(cls["ell"],log(cls["ee"]), index)
	logCEE[find(logCEE .== 0)]  = -Inf  
	logCEE[find(isnan(logCEE))] = -Inf
	cEEhr = exp(logCEE)

	logCPh = linear_interp1(cls["ell"], log(cls["pp"]), index)
	logCPh[find(logCPh .== 0)]  = -Inf
	logCPh[find(isnan(logCPh))] = -Inf
	cPhhr = exp(logCPh)
	
	logCPs = linear_interp1(cls["ell"], log(cls["ps"]), index)
	logCPs[find(logCPs .== 0)]  = -Inf
	logCPs[find(isnan(logCPs))] = -Inf
	cPshr = exp(logCPs)

	ϕhr    =  √(cPhhr) .* fft2(randn(size(xhr))./ √(dxhr), deltxhr) 
	ψhr    =  √(cPshr) .* fft2(randn(size(xhr))./ √(dxhr), deltxhr) 
	ϕx     = ifft2r(ϕhr, deltkhr)[1:4:end, 1:4:end]
	ψx     = ifft2r(ψhr, deltkhr)[1:4:end, 1:4:end]

	dispxhr   = ifft2r(im .* k1hr .* ϕhr, deltkhr)  + ifft2r(im .* k2hr .* ψhr, deltkhr)
	dispyhr   = ifft2r(im .* k2hr .* ϕhr, deltkhr)  - ifft2r(im .* k1hr .* ψhr, deltkhr)

    E    =  √(cEEhr) .* fft2(randn(size(xhr))./ √(dxhr), deltxhr) 
	B    =  zero(E) 
	Q    = - E .* cos(φ2_lhr) + B .* sin(φ2_lhr)
	U    = - E .* sin(φ2_lhr) - B .* cos(φ2_lhr)
	Qx   = ifft2r(Q, deltkhr)
	Ux   = ifft2r(U, deltkhr)

	tldQx = spline_interp2(xhr, yhr, Qx, xhr +  dispxhr, yhr +  dispyhr)
	tldUx = spline_interp2(xhr, yhr, Ux, xhr +  dispxhr, yhr +  dispyhr)

    unlensedQUdata = QUandFriends(Qx[1:4:end,1:4:end],    Ux[1:4:end,1:4:end])	
    tldQUdata      = QUandFriends(tldQx[1:4:end,1:4:end], tldUx[1:4:end,1:4:end])	
	
	unlensedQUdata, tldQUdata, fft2(ϕx, deltx),  fft2(ψx, deltx)
end # let
Ex    = - fft2(unlensedQUdata.Qx)    .* cos(φ2_l) - fft2(unlensedQUdata.Ux)    .* sin(φ2_l)  |> ifft2r
Bx    =   fft2(unlensedQUdata.Qx)    .* sin(φ2_l) - fft2(unlensedQUdata.Ux)    .* cos(φ2_l)  |> ifft2r
tldEx = - fft2(tldQUdata.Qx) .* cos(φ2_l) - fft2(tldQUdata.Ux) .* sin(φ2_l) |> ifft2r
tldBx =   fft2(tldQUdata.Qx) .* sin(φ2_l) - fft2(tldQUdata.Ux) .* cos(φ2_l) |> ifft2r
tldB         = fft2(tldBx)
B            = fft2(Bx)


# make the M weights
# this should eventually be pushed to a module or hardwired into the grad calculations
const Mq  = squash!(- abs2(cos(φ2_l)) ./ cEE  - abs2(sin(φ2_l)) ./ cBB ) 
const Mu  = squash!(- abs2(cos(φ2_l)) ./ cBB  - abs2(sin(φ2_l)) ./ cEE )
const Mqu = squash!(2 * cos(φ2_l) .* sin(φ2_l) ./ cBB) - squash!(2 * cos(φ2_l) .* sin(φ2_l) ./ cEE) 

Mq[magk .>= maskupC]  = 0.0 
Mu[magk .>= maskupC]  = 0.0 
Mqu[magk .>= maskupC] = 0.0 

ϕcurr = zero(B)
ψcurr = zero(B)
@time gradupdate!(ϕcurr, ψcurr, tldQUdata, 2000, 1e-6, 1e-6)
@time gradupdate!(ϕcurr, ψcurr, tldQUdata, 2000, 3e-6, 3e-6)



Qϕψx, Uϕψx, _, _, _, _    = easylense(tldQUdata, -ϕcurr, -ψcurr) 
Qϕψxanti, Uϕψxanti, _, _, _, _    = easylense(tldQUdata, -ϕtrue, -ψtrue) 
est_B        = fft2(Qϕψx) .* sin(φ2_l) - fft2(Uϕψx) .* cos(φ2_l) 
est_B_anti   = fft2(Qϕψxanti) .* sin(φ2_l) - fft2(Uϕψxanti) .* cos(φ2_l) 

uplim = 6000
llim = 10

loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls_ln["ee"][llim:uplim]  / (2π), "g-", label = L"\widetilde C_l^{EE}")
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls["ee"][llim:uplim]  / (2π), "g--", label = L" C_l^{EE}")
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls_ln["bb"][llim:uplim]  / (2π), "b-", label = L"\widetilde C_l^{BB}")
loglog(cls["ell"][llim:uplim], cls["ell"][llim:uplim] .* (cls["ell"][llim:uplim] + 1) .* cls["bb"][llim:uplim]  / (2π), "b--", label = L"C_l^{BB}")

loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(tldB*deltk) / (2π), magk, llim:deltk:uplim), "b.", label = L"tldB")
loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(est_B*deltk) /(2π), magk, llim:deltk:uplim), ".", markeredgecolor = "red",  mfc="none", label = L"est B")
loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(est_B_anti*deltk) /(2π), magk, llim:deltk:uplim), ".", markeredgecolor = "black",  mfc="none", label = L"est B anti")
# loglog(llim:deltk:uplim, binave(magk .* (magk + 1) .* abs2(B*deltk) /(2π), magk, llim:deltk:uplim), ".", markeredgecolor = "black",  mfc="none", label = L"est B anti")
axis("tight")
legend(loc = "best")




