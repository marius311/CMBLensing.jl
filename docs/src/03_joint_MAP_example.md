---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Julia 1.4.0
    language: julia
    name: julia-1.4
---

# MAP estimation


Here, we give an example of how to compute the joint maximum a posteriori (MAP) estimate of the CMB temperature and polarization fields, $f$, and the lensing potential, $\phi$.

```julia
using CMBLensing, PyPlot
```

## Compute spectra


First, we compute the fiducial CMB power spectra which generate our simulated data,

```julia
Cℓ = camb(r=0.05);
```

Next, we chose the noise power-spectra:

```julia
Cℓn = noiseCℓs(μKarcminT=1, ℓknee=100);
```

Plot these up for reference,

```julia
loglog(Cℓ.total.BB,c="C0")
loglog(Cℓ.unlensed_total.BB,"--",c="C0")
loglog(Cℓ.total.EE,c="C1")
loglog(Cℓ.unlensed_total.EE,"--",c="C1")
loglog(Cℓn.BB,"k:")
legend(["lensed B","unlensed B","lensed E","unlensed E", "noise (beam not deconvolved)"]);
```

## Configure the type of data


These describe the setup of the simulated data we are going to work with (and can be changed in this notebook),

```julia
θpix  = 3        # pixel size in arcmin
Nside = 128      # number of pixels per side in the map
pol   = :P       # type of data to use (can be :T, :P, or :TP)
T     = Float32  # data type (Float32 is ~2 as fast as Float64);
```

## Generate simulated data


With these defined, the following generates the simulated data and returns the true unlensed and lensed CMB fields, `f` and `f̃` ,and the true lensing potential, `ϕ`, as well as a number of other quantities stored in the "DataSet" object `ds`. 

```julia
@unpack f, f̃, ϕ, ds = load_sim_dataset(
    seed = 3,
    Cℓ = Cℓ,
    Cℓn = Cℓn,
    θpix = θpix,
    T = T,
    Nside = Nside,
    pol = pol,
)

@unpack Cf, Cϕ = ds;
```

## Examine simulated data


The true $\phi$ map,

```julia
plot(ϕ, title = raw"true $\phi$");
```

The "true" unlensed field, $f$,

```julia
plot(f, title = "true unlensed " .* ["E" "B"]);
```

And the "true" lensed field,

```julia
plot(LenseFlow(ϕ)*f, title = "true lensed " .* ["E" "B"]);
```

The data (stored in the `ds` object) is basically `f̃` with a beam applied plus a sample of the noise,

```julia
plot(ds.d, title = "data " .* ["E" "B"]);
```

# Run the optimizer


Now we compute the maximum of the joint posterior, $\mathcal{P}\big(f, \phi \,\big|\,d\big)$

```julia
@time fbf, ϕbf, tr = MAP_joint(ds, nsteps=30, progress=:verbose, αmax=0.1);
```

# Examine results


The expected value of the final best-fit $\chi^2 (=-2\log \mathcal{P}$) is given by the number degrees of freedom in the data, i.e. the total number of pixels in T and/or EB.

```julia
χ² = -2tr[end][:lnPcur]
```

```julia
dof = length(Map(f)[:])
```

Here's how far away our final $\chi^2$ is from this expectation, in units of $\sigma$. We expect this should be somewhere in the range (-3,3) for about 99.7% of simulated datasets.

```julia
(χ² - dof)/sqrt(2dof)
```

Here's the best-fit $\phi$ relative to the truth,

```julia
plot(10^6*[ϕ ϕbf], title=["true" "best-fit"] .* raw" $\phi$", vlim=17);
```

Here is the difference in terms of the power spectra. Note the best-fit has high-$\ell$ power suppressed, like a Wiener filter solution (in fact what we're doing here is akin to a non-linear Wiener filter). In the high S/N region ($\ell\lesssim1000$), the difference is approixmately equal to the noise, which you can see is almost two orders of magnitude below the signal.

```julia
loglog(ℓ⁴ * Cℓ.total.ϕϕ, "k")
loglog(get_ℓ⁴Cℓ(ϕ))
loglog(get_ℓ⁴Cℓ(ϕbf))
loglog(get_ℓ⁴Cℓ(ϕbf-ϕ))
xlim(80,3000)
ylim(5e-9,2e-6)
legend(["theory",raw"true $\phi$", raw"best-fit $\phi$", "difference"])
xlabel(raw"$\ell$")
ylabel(raw"$\ell^4 C_\ell$");
```

The best-fit unlensed fields relative to truth,

```julia
plot([f,fbf], title = ["true", "best-fit"] .* " unlensed " .* ["E" "B"]);
```

The best-fit lensed field (bottom row) relative to truth (top row),

```julia
plot([f̃, LenseFlow(ϕbf)*fbf], title = ["true", "best-fit"] .* " lensed " .* ["E" "B"]);
```
