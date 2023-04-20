# API 

```@contents
Pages = ["api.md"]
```

```@index
Pages = ["api.md"]
```

## Simulation
```@docs
load_sim
simulate
```

## Lensing estimation

```@docs
MAP_joint
MAP_marg
sample_joint
argmaxf_logpdf
quadratic_estimate
```

## Lensing operators

```@docs
LenseFlow
BilinearLens
Taylens
PowerLens
CMBLensing.antilensing
```

## Configuration options

```@docs
CMBLensing.FFTW_NUM_THREADS
CMBLensing.FFTW_TIMELIMIT
```

## Other

```@autodocs
Modules = [CMBLensing]
Order   = [:function, :type, :macro, :constant]
Filter  = x -> !(x in [
    load_sim,
    simulate,
    MAP_joint,
    MAP_marg,
    sample_joint,
    argmaxf_logpdf,
    quadratic_estimate,
    LenseFlow,
    BilinearLens,
    Taylens,
    PowerLens,
    CMBLensing.antilensing,
    CMBLensing.FFTW_NUM_THREADS,
    CMBLensing.FFTW_TIMELIMIT,
])
```
