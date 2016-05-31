#= ###################################

Test code for Local Likelihood

=# ####################################


# ---- number of workers
addprocs(5)

# --- load modules
using BayesLensSPTpol
using PyPlot, Dierckx

# --- set grid geometry
@everywhere dm     = 2
@everywhere nside  = nextprod([2,3,5,7], 1024/2)
@everywhere period = 1*nside*pi/(180*60) # nside*pi/(180*60) = 1 arcmin pixels
@everywhere g      = BayesLensSPTpol.FFTgrid(dm, period, nside)

# --- noise and signal cls
@everywhere const r   = 0.1 # clbbk has r value set to r
@everywhere fsky      = 1.0 # should this be (period)^2 / (4π)
@everywhere σEEarcmin = √2 * 0.2 / √fsky
@everywhere σBBarcmin = √2 * 0.2 / √fsky
@everywhere σEErad    = σEEarcmin * (π / 180 / 60)
@everywhere σBBrad    = σBBarcmin * (π / 180 / 60)
@everywhere cls  = BayesLensSPTpol.class(r = r, r0 = 1.0)
@everywhere mCls = BayesLensSPTpol.MatrixCls(g, cls; σEEarcmin = σEEarcmin, σBBarcmin = σBBarcmin)

# --- lense
order  = 4  # Taylor lensing order
ϕx, ϕk = sim_xk(mCls.cϕϕk, g)
len     = LenseDecomp(ϕk, zeros(ϕk), g)

# --- generate the data
ex, ek             = sim_xk(mCls.cEEk, g)
bx, bk             = sim_xk(mCls.cBBk, g)
qk, uk, qx, ux     = eb2qu(ek, bk, g)
ln_qx, ln_ux       = lense(qx, ux, len, g, order)
nex, nek           = sim_xk(mCls.cEEnoisek, g)
nbx, nbk           = sim_xk(mCls.cBBnoisek, g)
nqk, nuk, nqx, nux = eb2qu(nek, nbk, g)
dqx = ln_qx + nqx
dux = ln_ux + nux
dek, dbk, dex, dbx = qu2eb(g.FFT * dqx, g.FFT * dux, g)
ln_ek, ln_bk, ln_ex, ln_bx = qu2eb(g.FFT * ln_qx, g.FFT * ln_ux, g)


# ---- plot power
kbins, nbk_pwr = radial_power(nbk, 1, g)
kbins, dbk_pwr = radial_power(dbk, 1, g)
kbins, ln_bk_pwr = radial_power(ln_bk, 1, g)
figure()
loglog(kbins, kbins .* (kbins + 1) .* nbk_pwr / 2π, ".", label = L"b noise pwr")
loglog(kbins, kbins .* (kbins + 1) .* dbk_pwr  / 2π, ".", label = L"b data pwr")
loglog(kbins, kbins .* (kbins + 1) .* ln_bk_pwr  / 2π, ".", label = L"b lense pwr")
loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_bb] / 2π, label = L"lense $C_l^{BB}$")
loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:ln_ee] / 2π, label = L"lense $C_l^{EE}$")
loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* cls[:bb] / 2π, label = L"primordial $C_l^{BB}$")
loglog(cls[:ell], cls[:ell] .* (cls[:ell] + 1) .* abs2(σBBrad) / 2π, label = L"b noise")
legend(loc = 3)




# ---- spline covariance matrices
@everywhere const SplCEx, SplSEx, SplSCEx, SplCB0x, SplSB0x, SplSCB0x = let
    ghr      = BayesLensSPTpol.FFTgrid(2, period, 4*nside)
    mClshr   = BayesLensSPTpol.MatrixCls(ghr, BayesLensSPTpol.class(r = r, r0 = 1.0))
    φ2_l     = 2angle(ghr.k[1] + im * ghr.k[2])
    cos²φ    = abs2(cos(φ2_l))
    sin²φ    = abs2(sin(φ2_l))
    sincosφ  = cos(φ2_l).*sin(φ2_l)
    cCEk     = cos²φ.* mClshr.cEEk
    cSEk     = sin²φ.* mClshr.cEEk
    cSCEk    = sincosφ.* mClshr.cEEk
    cCB0k    = cos²φ.* mClshr.cBB0k
    cSB0k    = sin²φ.* mClshr.cBB0k
    cSCB0k   = -sincosφ.* mClshr.cBB0k
    covCEx   = real(ghr.FFT \ cCEk) ./ (2π)
    covSEx   = real(ghr.FFT \ cSEk) ./ (2π)
    covSCEx  = real(ghr.FFT \ cSCEk) ./ (2π)
    covCB0x  = real(ghr.FFT \ cCB0k) ./ (2π)
    covSB0x  = real(ghr.FFT \ cSB0k) ./ (2π)
    covSCB0x = real(ghr.FFT \ cSCB0k) ./ (2π)
    xspl     = fftshift(ghr.x[2])[:,1]
    yspl     = fftshift(ghr.x[1])[1,:][:]
    SplCEx   = Dierckx.Spline2D(xspl, yspl, fftshift(covCEx);   kx=3, ky=3, s=0.0)
    SplSEx   = Dierckx.Spline2D(xspl, yspl, fftshift(covSEx);   kx=3, ky=3, s=0.0)
    SplSCEx  = Dierckx.Spline2D(xspl, yspl, fftshift(covSCEx);  kx=3, ky=3, s=0.0)
    SplCB0x  = Dierckx.Spline2D(xspl, yspl, fftshift(covCB0x);  kx=3, ky=3, s=0.0)
    SplSB0x  = Dierckx.Spline2D(xspl, yspl, fftshift(covSB0x);  kx=3, ky=3, s=0.0)
    SplSCB0x = Dierckx.Spline2D(xspl, yspl, fftshift(covSCB0x); kx=3, ky=3, s=0.0)
    SplCEx, SplSEx, SplSCEx, SplCB0x, SplSB0x, SplSCB0x # last line is returned
end

funnames = [:covfunCEx, :covfunSEx, :covfunSCEx, :covfunCB0x, :covfunSB0x, :covfunSCB0x]
Splnames = [:SplCEx, :SplSEx, :SplSCEx, :SplCB0x, :SplSB0x, :SplSCB0x]
for fun in zip(funnames, Splnames)
    quote
    @everywhere function $(fun[1]){T,d}(x::Array{T,d},y::Array{T,d})
        rtn = $(fun[2])(vec(x), vec(y))::Array{Float64,1}
        return reshape(rtn, size(x))
    end # function
    end |> eval
end # for


# ---- define the loglikelihood functions
@everywhere function loglike(r, VLd, D, hlogdet)
    logdetΣ = hlogdet + sum(log(D+r))
    rout    = - 0.5 * dot(VLd, VLd./(D+r)) - 0.5 * logdetΣ
    return rout
end
@everywhere function loglike_profile(rng, xdiffI, ydiffI, dqxblkI, duxblkI, σEErad, deltx)
    qudata = vcat(dqxblkI, duxblkI)
    ΣceI   = covfunCEx(xdiffI, ydiffI)
    ΣseI   = covfunSEx(xdiffI, ydiffI)
    ΣcbI   = covfunCB0x(xdiffI, ydiffI)
    ΣsbI   = covfunSB0x(xdiffI, ydiffI)
    ΣsceI  = covfunSCEx(xdiffI, ydiffI)
    ΣscbI  = covfunSCB0x(xdiffI, ydiffI)
    ΣeI     = [ΣceI ΣsceI; ΣsceI.' ΣseI]
    ΣbI     = [ΣsbI ΣscbI; ΣscbI.' ΣcbI]
    ΣnoiseI = abs2(σEErad/deltx) * eye(length(qudata))
    L       = chol(ΣbI, Val{:L})
    A       = L \ (ΣeI + ΣnoiseI)
    B       = transpose(L \ transpose(A))
    D,V     = eig(Symmetric(B))
    tmp     =  L \ qudata
    VLd     =  At_mul_B(V, tmp)
    hlogdet = 2sum(log(diag(L)))
    rout    = Array{Float64}(length(rng))
    for i=1:length(rng)
        @inbounds rout[i] = loglike(rng[i], VLd, D, hlogdet)
    end
    return rout
end
# function loglike(r, Σe, Σb, Σnoise, qudata)
#     Σ    = Σe + r .* Σb + Σnoise
#     rout = - 0.5 * dot(qudata, Σ \ qudata) - 0.5 * logdet(Σ)
#     return rout
# end
# function loglike_profile(rng, xdiffI, ydiffI, dqxblkI, duxblkI, σEErad, g)
#     qudata = vcat(dqxblkI, duxblkI)
#     ΣceI   = covfunCEx(xdiffI, ydiffI)
#     ΣseI   = covfunSEx(xdiffI, ydiffI)
#     ΣcbI   = covfunCB0x(xdiffI, ydiffI)
#     ΣsbI   = covfunSB0x(xdiffI, ydiffI)
#     ΣsceI  = covfunSCEx(xdiffI, ydiffI)
#     ΣscbI  = covfunSCB0x(xdiffI, ydiffI)
#     ΣeI = [ΣceI ΣsceI; ΣsceI.' ΣseI]
#     ΣbI = [ΣsbI ΣscbI; ΣscbI.' ΣcbI]
#     ΣnoiseI = abs2(σEErad/g.deltx) * eye(length(qudata))
#     rout = Array{Float64}(length(rng))
#     for i=1:length(rng)
#         rout[i] = loglike(rng[i], ΣeI, ΣbI, ΣnoiseI, qudata)
#     end
#     return rout
# end

#= --- test the speed of loglike
rng = 0.01:0.001:.2
dqxblkI, duxblkI = dqxblks[1], duxblks[1]
ydiffI = lnx1blks[1] .- lnx1blks[1].'
xdiffI = lnx2blks[1] .- lnx2blks[1].'
@time loglike_profile(rng, xdiffI, ydiffI, dqxblkI, duxblkI, σEErad, g)
@time loglike_profile_test(rng, xdiffI, ydiffI, dqxblkI, duxblkI, σEErad, g)
@code_warntype loglike_profile_test(rng, xdiffI, ydiffI, dqxblkI, duxblkI, σEErad, g)

Profile.clear()
@profile loglike_profile(rng, xdiffI, ydiffI, dqxblkI, duxblkI, σEErad, g)

qudata = vcat(dqxblkI, duxblkI)
ΣceI   = covfunCEx(xdiffI, ydiffI)
ΣseI   = covfunSEx(xdiffI, ydiffI)
ΣcbI   = covfunCB0x(xdiffI, ydiffI)
ΣsbI   = covfunSB0x(xdiffI, ydiffI)
ΣsceI  = covfunSCEx(xdiffI, ydiffI)
ΣscbI  = covfunSCB0x(xdiffI, ydiffI)
ΣeI = [ΣceI ΣsceI; ΣsceI.' ΣseI]
ΣbI = [ΣsbI ΣscbI; ΣscbI.' ΣcbI]
ΣnoiseI = abs2(σEErad/g.deltx) * eye(length(qudata))
@time L   = chol(ΣbI, Val{:L})
@time A = (L \ (ΣeI + ΣnoiseI))
@time B = transpose(L \ transpose(A))
@time D,V = eig(B)


@time covfunCEx(xdiffI, ydiffI)
@code_warntype covfunCEx(xdiffI, ydiffI)
xv = vec(xdiffI[:]); yv = vec(ydiffI[:])
@time reshape(SplCEx(xv, yv), size(xdiffI))
=#


# ---- Extract patches
#sizbl = (48, 48)
sizbl = (32, 32)
#sizbl = (48, 48)
#sizbl = (64, 32)
lengbl = sizbl[1]*sizbl[2]
nb  = nside ÷ sizbl[1]
mb  = nside ÷ sizbl[2]
dqxblks  = Vector{Float64}[]
duxblks  = Vector{Float64}[]
lnx1blks = Vector{Float64}[]
lnx2blks = Vector{Float64}[]
dqx_fs  = fftshift(dqx)
dux_fs  = fftshift(dux)
lnx1_fs = fftshift(g.x[1] + len.displx)
lnx2_fs = fftshift(g.x[2] + len.disply)
for ib in 1:nb, jb in 1:mb
    push!(dqxblks,   dqx_fs[(ib-1)*nb + (1:sizbl[1]), (jb-1)*mb + (1:sizbl[2])][:])
    push!(duxblks,   dux_fs[(ib-1)*nb + (1:sizbl[1]), (jb-1)*mb + (1:sizbl[2])][:])
    push!(lnx1blks, lnx1_fs[(ib-1)*nb + (1:sizbl[1]), (jb-1)*mb + (1:sizbl[2])][:])
    push!(lnx2blks, lnx2_fs[(ib-1)*nb + (1:sizbl[1]), (jb-1)*mb + (1:sizbl[2])][:])
end


# ---- the local profiles
rng = 0.0:0.0005:0.3
numblks = length(dqxblks)
tic()
rll = @parallel (hcat) for blkI in 1:numblks
    dqxblkI, duxblkI = dqxblks[blkI], duxblks[blkI]
    ydiffI = lnx1blks[blkI] .- lnx1blks[blkI].'
    xdiffI = lnx2blks[blkI] .- lnx2blks[blkI].'
    loglike_profile(rng, xdiffI, ydiffI, dqxblkI, duxblkI, σEErad, g.deltx)
end
toc()
plot(rng, 2*mean(rll, 2))
plot(rng, maximum(2*mean(rll, 2)) - 3*√2 + 0.*rng, ":", label="chisq 3 sigma detection region")
# plot(rng, rll[:,5])

# tic()
# rll = @parallel (hcat) for blk in zip(dqxblks, duxblks, lnx1blks, lnx2blks)
#     dqxblkI, duxblkI = blk[1], blk[2]
#     ydiffI = blk[3] .- transpose(blk[3])
#     xdiffI = blk[4] .- transpose(blk[4])
#     loglike_profile(rng, xdiffI, ydiffI, dqxblkI, duxblkI, σEErad, deltx)
# end
# toc()
#
# @everywhere function loglike_profile(rng, blk, σEErad, deltx)
#     dqxblkI, duxblkI = blk[1], blk[2]
#     ydiffI = blk[3] .- transpose(blk[3])
#     xdiffI = blk[4] .- transpose(blk[4])
#     return loglike_profile(rng, xdiffI, ydiffI, dqxblkI, duxblkI, σEErad, deltx)
# end
# rll = pmap(blk->loglike_profile(rng, blk, σEErad, deltx), zip(dqxblks, duxblks, lnx1blks, lnx2blks))
# rll
#plot(rng, rllIave,"k")


# when combining over different ϕ you need to average on the Likelihood scale..
