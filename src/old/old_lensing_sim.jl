###########################################
#
# T lensing code used in script3.jl
#
###########################################

function  all_ord_len()
	nhr        = 4n
	deltxhr    = deltx / 4
	periodhr   = deltxhr * nhr
	deltkhr    = 2π / periodhr
	dkhr       = deltkhr ^ 2
	dxhr       = deltxhr ^ 2
	nyqhr      = 2π / (2deltxhr)
	xhr, yhr   = meshgrid([0:nhr-1] * deltxhr, [0:nhr-1] * deltxhr)
	k1hr, k2hr = linspace(-nyqhr, nyqhr-deltkhr, int(nhr))  |> fftshift |> x->meshgrid(x,x)
	magkhr     = √(k1hr.^2 .+ k2hr.^2)

	# make the spectral matrices
	index  = ceil(magkhr)
	index[find(index.==0)] = 1

	logCTT = linear_interp1(cls["ell"],log(cls["tt"]), index)
	logCTT[find(logCTT .== 0)]  = -Inf
	logCTT[find(isnan(logCTT))] = -Inf
	cTThr = exp(logCTT);

	# simulation lensing potentials
	ϕ       =  √(cPP) .* fft2(randn(size(x))./ √(dx))
  # hatϕ    =  ϕ + √(cPP.*(1/(ρϕhatϕ^2) - 1)) .* fft2(randn(size(x))./ √(dx))
	hatϕ    =  ϕ + √(cPhatNoise) .* fft2(randn(size(x))./ √(dx))
	displx  = ifft2r(im .* k1 .* ϕ, deltk)
	disply  = ifft2r(im .* k2 .* ϕ, deltk)

  Thr    =  √(cTThr) .* fft2(randn(size(xhr))./ √(dxhr), deltxhr)
  Thrx   =  ifft2r(Thr, deltkhr)
  tldTx  = spline_interp2(xhr, yhr, Thrx, x + displx, y + disply)
  tldT   = fft2(tldTx, deltx)

  Tobs = tldT + √(cNN) .* fft2(randn(size(x))./ √(dx))

  fft2(Thrx[1:4:end,1:4:end], deltx), tldT, Tobs, ϕ, hatϕ
end # function


function  scnd_ord_len()
  # simulate unlensed CMB
  T    =  √(cTT) .* fft2(randn(size(x))./ √(dx))
  # simulation lensing potentials
  ϕ  =  √(cPP) .* fft2(randn(size(x))./ √(dx))
  # simulation estimated lensing
  # hatϕ  =  ϕ + √(cPP.*(1/(ρϕhatϕ^2) - 1)) .* fft2(randn(size(x))./ √(dx))
  hatϕ    =  ϕ + √(cPhatNoise) .* fft2(randn(size(x))./ √(dx))
  # convert to displacements
  displx   = ifft2r(im .* k1 .* ϕ, deltk)
  disply   = ifft2r(im .* k2 .* ϕ, deltk)
  # Decompose the lensing displacements """
  row, col = size(x)
  rdisplx  = Array(Float64, row, col)
  rdisply  = Array(Float64, row, col)
  indcol   = Array(Int64, row, col)
  indrow   = Array(Int64, row, col)
  decomplense!(indcol, indrow, rdisplx, rdisply, x, y, displx, disply, deltx)
  # do the integer lensing
  lTdata = gridlense(TandFriends(T), indcol, indrow)
  # do the taylor expansion lensing and put everything in a QU object
  tldTx  = lTdata.Tx
  tldTx += (lTdata.∂1Tx .* rdisplx)
  tldTx += (lTdata.∂2Tx .* rdisply)
  tldTx += 0.5 * (rdisplx .* lTdata.∂11Tx .* rdisplx )
  tldTx +=       (rdisplx .* lTdata.∂12Tx .* rdisply )
  tldTx += 0.5 * (rdisply .* lTdata.∂22Tx .* rdisply )

  tldT = fft2(tldTx, deltx)

  Tobs = tldT + √(cNN) .* fft2(randn(size(x))./ √(dx))

  T, tldT, Tobs, ϕ, hatϕ
end # function


function  scnd_ord_len(Tin)
  # simulation lensing potentials
  ϕ  =  √(cPP) .* fft2(randn(size(x))./ √(dx))
  # simulation estimated lensing
  # hatϕ  =  ϕ + √(cPP.*(1/(ρϕhatϕ^2) - 1)) .* fft2(randn(size(x))./ √(dx))
  hatϕ    =  ϕ + √(cPhatNoise) .* fft2(randn(size(x))./ √(dx))
  # convert to displacements
  displx   = ifft2r(im .* k1 .* ϕ, deltk)
  disply   = ifft2r(im .* k2 .* ϕ, deltk)
  # Decompose the lensing displacements """
  row, col = size(x)
  rdisplx  = Array(Float64, row, col)
  rdisply  = Array(Float64, row, col)
  indcol   = Array(Int64, row, col)
  indrow   = Array(Int64, row, col)
  decomplense!(indcol, indrow, rdisplx, rdisply, x, y, displx, disply, deltx)
  # do the integer lensing
  lTdata = gridlense(TandFriends(Tin), indcol, indrow)
  # do the taylor expansion lensing and put everything in a QU object
  tldTx  = lTdata.Tx
  tldTx += (lTdata.∂1Tx .* rdisplx)
  tldTx += (lTdata.∂2Tx .* rdisply)
  tldTx += 0.5 * (rdisplx .* lTdata.∂11Tx .* rdisplx )
  tldTx +=       (rdisplx .* lTdata.∂12Tx .* rdisply )
  tldTx += 0.5 * (rdisply .* lTdata.∂22Tx .* rdisply )

  tldT = fft2(tldTx, deltx)

  Tobs = tldT + √(cNN) .* fft2(randn(size(x))./ √(dx))

  Tin, tldT, Tobs, ϕ, hatϕ
end # function



###########################################
#
# QU lensing code used in script2.jl
#
###########################################




function scnd_ord_len_QU()
  # simulate unlensed CMB
  E    =  √(cEE) .* fft2(randn(size(x))./ √(dx))
  B    =  √(cBB) .* fft2(randn(size(x))./ √(dx))
  Q    = - E .* cos(φ2_l) + B .* sin(φ2_l)
  U    = - E .* sin(φ2_l) - B .* cos(φ2_l)
  unlensedQUdata = QUandFriends(Q, U)
  # Noise
  NE    =  √(cNN) .* fft2(randn(size(x))./ √(dx))
  NB    =  √(cNN) .* fft2(randn(size(x))./ √(dx))
  # simulation lensing potentials
  ϕ  =  √(cPP) .* fft2(randn(size(x))./ √(dx))
  # simulation estimated lensing
  # hatϕ  =  ϕ + √(cPP.*(1/(ρϕhatϕ^2) - 1)) .* fft2(randn(size(x))./ √(dx))
  hatϕ    =  ϕ + √(cPhatNoise) .* fft2(randn(size(x))./ √(dx))
  # convert to displacements
  displx   = ifft2r(im .* k1 .* ϕ)
  disply   = ifft2r(im .* k2 .* ϕ)
  # Decompose the lensing displacements """
  row, col = size(x)
  rdisplx  = Array(Float64, row, col)
  rdisply  = Array(Float64, row, col)
  indcol   = Array(Int64, row, col)
  indrow   = Array(Int64, row, col)
  decomplense!(indcol, indrow, rdisplx, rdisply, x, y, displx, disply, deltx)
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

  tldE = - fft2(tldQx, deltx) .* cos(φ2_l) - fft2(tldUx, deltx) .* sin(φ2_l)
  tldB =   fft2(tldQx, deltx) .* sin(φ2_l) - fft2(tldUx, deltx) .* cos(φ2_l)

  Eobs = tldE + NE
  Bobs = tldB + NB

  E, B, tldE, tldB, Eobs, Bobs, ϕ, hatϕ
end # let


function all_ord_len_QU()
  nhr        = 4n
  deltxhr    = deltx / 4
  periodhr   = deltxhr * nhr
  deltkhr    = 2π / periodhr
  dkhr       = deltkhr ^ 2
  dxhr       = deltxhr ^ 2
  nyqhr      = 2π / (2deltxhr)
  xhr, yhr   = meshgrid([0:nhr-1] * deltxhr, [0:nhr-1] * deltxhr)
  k1hr, k2hr = linspace(-nyqhr, nyqhr-deltkhr, int(nhr))  |> fftshift |> x->meshgrid(x,x)
  magkhr     = √(k1hr.^2 .+ k2hr.^2)
  φ2_lhr     = 2.0 * angle(k1hr + im * k2hr)

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

  # simulation lensing potentials
  ϕ       =  √(cPP) .* fft2(randn(size(x))./ √(dx))
  # hatϕ    =  ϕ + √(cPP.*(1/(ρϕhatϕ^2) - 1)) .* fft2(randn(size(x))./ √(dx))
  hatϕ    =  ϕ + √(cPhatNoise) .* fft2(randn(size(x))./ √(dx))
  displx  = ifft2r(im .* k1 .* ϕ, deltk)
  disply  = ifft2r(im .* k2 .* ϕ, deltk)

  Ehr    =  √(cEEhr) .* fft2(randn(size(xhr))./ √(dxhr), deltxhr)
  Bhr    =  √(cBBhr) .* fft2(randn(size(xhr))./ √(dxhr), deltxhr)
  Qhr    = - Ehr .* cos(φ2_lhr) + Bhr .* sin(φ2_lhr)
  Uhr    = - Ehr .* sin(φ2_lhr) - Bhr .* cos(φ2_lhr)
  Qhrx   = ifft2r(Qhr, deltkhr)
  Uhrx   = ifft2r(Uhr, deltkhr)

  tldQx  = spline_interp2(xhr, yhr, Qhrx, x + displx, y + disply)
  tldUx  = spline_interp2(xhr, yhr, Uhrx, x + displx, y + disply)

  tldE   = - fft2(tldQx, deltx) .* cos(φ2_l) - fft2(tldUx, deltx) .* sin(φ2_l)
  tldB   =   fft2(tldQx, deltx) .* sin(φ2_l) - fft2(tldUx, deltx) .* cos(φ2_l)

  E = - fft2(Qhrx[1:4:end,1:4:end], deltx) .* cos(φ2_l) - fft2(Uhrx[1:4:end,1:4:end], deltx) .* sin(φ2_l)
  B =   fft2(Qhrx[1:4:end,1:4:end], deltx) .* sin(φ2_l) - fft2(Uhrx[1:4:end,1:4:end], deltx) .* cos(φ2_l)

  Eobs = tldE + √(cNN) .* fft2(randn(size(x))./ √(dx))
  Bobs = tldB + √(cNN) .* fft2(randn(size(x))./ √(dx))

  E, B, tldE, tldB, Eobs, Bobs, ϕ, hatϕ
end # let
