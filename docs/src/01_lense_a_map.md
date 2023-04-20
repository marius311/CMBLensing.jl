---
jupyter:
  jupytext:
    formats: ipynb,md
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Julia 1.9.0-rc2
    language: julia
    name: julia-1.9
  language_info:
    file_extension: .jl
    mimetype: application/julia
    name: julia
    version: 1.9.0
---

# Lensing a flat-sky map

```julia
using CMBLensing, PythonPlot
```

First we load a simulated unlensed field, $f$, and lensing potential, $\phi$,

```julia
(;ds, f, ϕ) = load_sim(
    θpix  = 2,       # size of the pixels in arcmin
    Nside = 256,     # number of pixels per side in the map
    T     = Float32, # Float32 or Float64 (former is ~twice as fast)
    pol   = :I,       # :I for Intensity, :P for polarization, or :IP for both=
);
```

We can lense the map with LenseFlow,

```julia
f̃ = LenseFlow(ϕ) * f;
```

And flip between lensed and unlensed maps,

```julia
animate([f,f̃], fps=1)
```

The difference between lensed and unlensed,

```julia
plot(f-f̃);
```

## Loading your own data


CMBLensing flat-sky `Field` objects like `f` or `ϕ`  are just thin wrappers around arrays. You can get the underlying data arrays for $I(\mathbf{x})$, $Q(\mathbf{x})$, and $U(\mathbf{x})$ with `f[:Ix]`, `f[:Qx]`, and `f[:Ux]` respectively, or the Fourier coefficients, $I(\mathbf{l})$, $Q(\mathbf{l})$, and $U(\mathbf{l})$ with `f[:Il]`, `f[:Ql]`, and `f[:Ul]`,

```julia
mapdata = f[:Ix]
```

If you have your own map data in an array you'd like to load into a CMBLensing `Field` object, you can construct it as follows:

```julia
FlatMap(mapdata, θpix=3)
```

For more info on `Field` objects, see [Field Basics](../05_field_basics/).


## Inverse lensing


You can inverse lense a map with the `\` operator (which does `A \ b ≡ inv(A) * b`):

```julia
LenseFlow(ϕ) \ f;
```

Note that this is true inverse lensing, rather than lensing by the negative deflection (which is often called "anti-lensing"). This means that lensing then inverse lensing a map should get us back the original map. Lets check that this is the case:

```julia
Ns = [7 10 20]
plot([f - (LenseFlow(ϕ,N) \ (LenseFlow(ϕ,N) * f)) for N in Ns],
    title=["ODE steps = $N" for N in Ns]);
```

A cool feature of LenseFlow is that inverse lensing is trivially done by running the LenseFlow ODE in reverse. Note that as we crank up the number of ODE steps above, we recover the original map to higher and higher precision.


## Other lensing algorithms


We can also lense via:
* `PowerLens`: the standard Taylor series expansion to any order:
$$ f(x+\nabla x) \approx f(x) + (\nabla f)(\nabla \phi) + \frac{1}{2} (\nabla \nabla f) (\nabla \phi)^2 + ... $$

* `TayLens` ([Næss&Louis 2013](https://arxiv.org/abs/1307.0719)): like `PowerLens`, but first a nearest-pixel permute step, then a Taylor expansion around the now-smaller residual displacement

```julia
plot([(PowerLens(ϕ,2)*f - f̃) (Taylens(ϕ,2)*f - f̃)], 
    title=["PowerLens - LenseFlow" "TayLens - LenseFlow"]);
```

## Benchmarking


LenseFlow is highly optimized code since it appears on the inner-most loop of our analysis algorithms. To benchmark LenseFlow, note that there is first a precomputation step, which caches some data in preparation for applying it to a field of a given type. This was done automatically when evaluating `LenseFlow(ϕ) * f` but we can benchmark it separately since in many cases this only needs to be done once for a given $\phi$, e.g. when Wiener filtering at fixed $\phi$,

```julia
using BenchmarkTools
```

```julia
@benchmark precompute!!(LenseFlow(ϕ),f)
```

Once cached, it's faster and less memory intensive to repeatedly apply the operator:

```julia
@benchmark Lϕ * f setup=(Lϕ=precompute!!(LenseFlow(ϕ),f))
```

Note that this documentation is generated on limited-performance cloud servers. Actual benchmarks are likely much faster locally or on a cluster, and yet (much) faster on [GPU](../06_gpu/).
