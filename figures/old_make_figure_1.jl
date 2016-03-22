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
const period_const = 2π
const nside_const  = nextprod([2,3,5,7], 400)

# --- cls 
ν      = 1.5
tild_ν = 1.7
ρ      = 0.015*period_const
tild_ρ = 0.014*period_const
t_0    = 1.5

# --- noise parameters
σpixl    = 0.0
beamFWHM = 0.0

# --- prior parameters
νΦ = 5.0
ρΦ = 0.15*period_const
σΦ = 0.5*(period_const/2π)^2




#########################################################
#=
Generate the first set of simulations
=#
#########################################################

# --- define an instance of PhaseParms which carries all the parameters
parms = PhaseParms(d_const, period_const, nside_const)
ηkfun, Ckfun, C1kfun, C2kfun = gen_tangentMatern(ν=ν, tild_ν=tild_ν, ρ=ρ, tild_ρ=tild_ρ, t_0=t_0)
NonstationaryPhase.CNNkfun{dm,T1,T2}(p::PhaseParms{dm,T1,T2})   = CNNkfun(p.k, p.deltx; σpixl=σpixl, beamFWHM=beamFWHM)
Cϕϕkfun{dm,T1,T2}(p::PhaseParms{dm,T1,T2})  = maternk(p.k; ν=νΦ, ρ=ρΦ, σ=σΦ)
parms.ηk[:]       =  ηkfun(parms.k)
parms.Ck[:]       =  Ckfun(parms.k)
parms.C1k[:]      =  C1kfun(parms.k)
parms.C2k[:]      =  C2kfun(parms.k)
parms.CNNk[:]     =  CNNkfun(parms)
parms.Cϕϕk[:]     = Cϕϕkfun(parms)
parms.CZZmk[:]    =  CZZmkfun(parms.Cϕϕk, parms.Ck, parms.ηk, parms.ξk, parms.x, parms.k, parms.deltx, parms.deltk)
parms.CZZmobsk[:] = parms.CZZmk + parms.CNNk
parms.CZZmobsk[parms.r .< 1]              = Inf
parms.CZZmobsk[parms.r .>= 0.9*parms.nyq] = Inf
parms.ξk[:] = Array{Complex{Float64},2}[ im.*parms.k[2], -im.*parms.k[1] ]  # new

# Now a High resolution version of PhaseParms for simulation
parmsHR = PhaseParms(d_const, period_const, nside_const)
parmsHR.ηk[:]   =  ηkfun(parmsHR.k)
parmsHR.Ck[:]   =  Ckfun(parmsHR.k)
parmsHR.CNNk[:] =  CNNkfun(parmsHR)
parmsHR.Cϕϕk[:] =  Cϕϕkfun(parmsHR)
parms.ξk[:] = Array{Complex{Float64},2}[ im.*parmsHR.k[2], -im.*parmsHR.k[1] ]  # new


# ---- generate the  simulation and other plotting quantities
zkobs, zk, zx, zx_noϕ, ϕk, ϕx = simNPhaseGRF(parms, parmsHR)
Dθx          = NonstationaryPhase.ϕx_2_Dθx(ϕx, parms)
@time Cℓvar        = Cℓvarfun(parms)
@time approxCℓbias = approxCℓbiasfun(parms)
@time estϕk  = estϕkfun(zkobs, parms)
estϕx  = ifftdr(estϕk, parms.deltk)




# #########################################################
# #=
# Now plot
# =#
# #########################################################


# ------ figure ---- show the quadratic estimate
function makeplots()
    global fig1 = figure(figsize = (10,10))

    # ----- figure: plot Z(x)
    ax1 = subplot2grid((2,2), (1,1))
    imshow(zx,
        interpolation = "nearest",
        origin="lower",
        extent=[minimum(parms.x[1]), maximum(parms.x[1]),minimum(parms.x[2]), maximum(parms.x[2])],
    )
    title("Nonstationary phase simulation")

    # ----- figure: plot Cℓvarfun, Cℓbias and the signal
    ax2 = subplot2grid((2,2), (1,0))
    δ0 = 1 / parms.deltk ^ d_const
    kbins, MSE_radpower = radial_power(abs2(parms.r.*(estϕk-ϕk))./δ0, 1, parms)
    semilogy(parms.r[1:25,1], abs2(parms.r[1:25,1]).*parms.Cϕϕk[1:25,1], "k:", label = L"\ell^2 C^{\phi\phi}_\ell", linewidth = 3.0)
    semilogy(parms.r[1:25,1], abs2(parms.r[1:25,1]).*approxCℓbias[1:25,1], "b--", label = L"approx $\ell^2 C^{bias\, \hat\phi}_\ell$", linewidth = 2.0)
    semilogy(parms.r[1:25,1], abs2(parms.r[1:25,1]).*Cℓvar[1:25,1], "g-", label = L"\ell^2C^{var\, \hat\phi}_\ell", linewidth = 2.0)
    semilogy(kbins[1:25],MSE_radpower[1:25], "go", label = L"radial $\ell^2 |\phi_\ell - \hat\phi_\ell|^2\delta_0$", linewidth = 2.0)
    #ylabel("spectral density")
    #xlabel(L"frequency $\ell$")
    title("Variance, bias and empirical MSE")
    #axis("tight")
    legend(loc="best",fontsize = 10)


    # --- figure: plot est Φ(x)
    ax4 = subplot2grid((2,2), (0,0))
    imshow(estϕx,
        interpolation = "nearest",
        origin="lower",
        extent=[minimum(parms.x[1]), maximum(parms.x[1]),minimum(parms.x[2]), maximum(parms.x[2])],
        )
    title(L"Quadratic estimate $\hat\phi(x)$")


    # --- figure: plot true Φ(x)
    ax3 = subplot2grid((2,2), (0,1))
    imshow(ϕx,
        interpolation = "nearest",
        origin="lower",
        extent=[minimum(parms.x[1]), maximum(parms.x[1]),minimum(parms.x[2]), maximum(parms.x[2])],
        vmin = minimum(estϕx), vmax = maximum(estϕx)
        )
    title(L"Simulation truth $\phi(x)$")

    fig1[:subplots_adjust](wspace=0.1)

end


# --- now evaluate the above `script'
makeplots()

# --- save the figures to disk
if savefigures
    fig1[:savefig]("figure4.pdf", dpi=300, bbox_inches="tight", transparent=true)
    plt[:close](fig1)
end
