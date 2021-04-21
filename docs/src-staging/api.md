# API 

```@contents
Pages = ["api.md"]
```

## Simulation
```@docs
load_sim
resimulate
resimulate!
```

## Lensing estimation

```@docs
MAP_joint
MAP_marg
sample_joint
argmaxf_lnP
quadratic_estimate
```

## Field constructors

```@docs
FlatMap
FlatFourier
FlatQUMap
FlatQUFourier
FlatEBMap
FlatEBFourier
FlatIQUMap
FlatIQUFourier
FlatIEBMap
FlatIEBFourier
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
    resimulate,
    resimulate!,
    MAP_joint,
    MAP_marg,
    sample_joint,
    argmaxf_lnP,
    quadratic_estimate,
    FlatMap,
    FlatFourier,
    FlatQUMap,
    FlatQUFourier,
    FlatEBMap,
    FlatEBFourier,
    FlatIQUMap,
    FlatIQUFourier,
    FlatIEBMap,
    FlatIEBFourier,
    LenseFlow,
    BilinearLens,
    Taylens,
    PowerLens,
    CMBLensing.antilensing,
    CMBLensing.FFTW_NUM_THREADS,
    CMBLensing.FFTW_TIMELIMIT,
])
```
