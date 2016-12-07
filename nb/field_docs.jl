
push!(LOAD_PATH, pwd()*"/../src")
using BayesLensSPTpol
using CMBFields
using PyPlot
using PyCall


#can comment this out if you don't care about adjusting plot looks:
@pyimport seaborn as sns
sns.set_context("notebook",font_scale=1.4,rc=Dict("lines.markeredgewidth"=>1))
sns.set_palette("colorblind")
# sns.set_style("darkgrid")
sns.set_style("ticks",Dict("xtick.direction"=>"in","ytick.direction"=>"in","image.cmap"=>"jet"))

# --- set grid geometry and cls
dm     = 2
nside  = nextprod([2,3,5,7], 256)
period = nside*π/(180*60)   # nside*pi/(180*60) = 1 arcmin pixels
Ωpix   = deg2rad(1/60)^2    # this is pixel area in radians: (π/(180*60))^2 == deg2rad(1/60)^2
g      = FFTgrid(dm, period, nside)

# --- noise and signal cls
const r   = 0.1    # this will be the simulation truth r value used to generate the data
const r0  = 1.0    # baseline r0 value used to rescale B mode in the algorithm
cls = class(r = r, r0 = r0);


#############################################
#  maps
#############################################
n = FlatS0Map(randn(nside,nside),g)
s = FlatS0Fourier(complex(randn(nside,nside)),g);
# what basis is is stored

@show typeof(n+n)
@show typeof(n+s)
@show typeof(s+s);

#----EA. Question: how will this decide if d <: FlatS0Map or d <: FlatS0Fourier
d = n+s
matshow(d[:Tx])
matshow(abs2(s[:Tl]))

y = 2*n + s/5 + 1;
s[:Tx] |> matshow


#############################################
#  covariance
#############################################
S = Cℓ_to_cov(FlatS0FourierDiagCov, cls[:ell], cls[:tt], g);
matshow(log(abs2(S.Cl)))

S.Cl[S.Cl.==0] = 1e-8 # threshold S so we can safely do S^-1
# ----- EA. Note: It would be nice to extrapolate beyond lmax with log linear extrapolation
# ----- Is there a natural fit from LCDM parameters?


# add a "mask" by blowing up the noise in the center of the map
μKarcminT = 5
N = FlatS0MapDiagCov(fill(μKarcminT^2 * Ωpix, (nside,nside)),g);
mi,me = (round(Int,(0.5+x*0.25)*nside) for x=[-1,1])
# ---- EA. Not sure I understand what mi and me is?
N.Cx[mi:me,mi:me] *= 1000000;

s = simulate(S)
n = simulate(N) / √Ωpix # <-- TODO: fix needing this factor
d = s + n;

# ------- checking the noise variance and S covariance etc....
Ωk      =  (2π / g.period) ^ 2 # pixel area in Fourier grid
σTTrad  = μKarcminT * √(Ωpix)  # == μKarcminT * (π / 180 / 60)
wx      = randn(size(g.r)) ./ √(Ωpix)     # white noise
wk      = g.FFT * wx
#wk  ∼ randn(size(g.r)) ./ √(Ωk)     # white noise..but needs Hermitan symetries
#Cwwl = 1
# ⟨|wk|^2⟩ = Cwwl / Ωk
cTTk = BayesLensSPTpol.cls_to_cXXk(cls[:ell], cls[:tt], g.r)
nx  =  σTTrad * wx
@show std(nx), σTTrad / √(Ωpix)
zk = √(cTTk) .* wk
zx = real(g.FFT \ zk)

# here is the covariance function
covCTk = cTTk ./ (2π)
CTx = real(g.FFT \ covCTk)
# -----------------------





matshow(d[:Tx], vmin=-300, vmax=300)

(S*n)[:Tx] |> matshow

A = 1+N*S^(-1)
matshow((A*s)[:Tx], vmin=-300, vmax=300)

s[:]

A[~s] * s[:]

typeof((A[~s] * s[:])[~s])

matshow((A[~s] * s[:])[~s][:Tx], vmin=-300, vmax=300)

using IterativeSolvers

# the form in which we apply the Wiener filter
A = (1+S^(1/2)*N^(-1)*S^(1/2))
b = S^(1/2)*N^(-1)*d

# note the use of the ~ notation
x = cg(A[~b], b[:], tol=1e-3)[~b]
swf = S^(1/2)*x;

matshow(d[:Tx], vmin=-300, vmax=300); xlabel("Data")
matshow(swf[:Tx], vmin=-300, vmax=300); xlabel("WF Solution")
