# Startup

If using Julia 1.9 or above, you can get large speedups (about 10X) in startup time by using the native code caching feature. The idea is that when precompiling CMBLensing, several typically-used functions are run and precompiled, making them much faster to use in subsequent sessions. 

Because this can make precompilation of CMBLensing take somewhat longer (a few minutes), its disabled by default. But you can enable it by running, 

```julia
julia> using CMBLensing

julia> CMBLensing.set_preferences!(CMBLensing, "precompile" => true)
```

The setting will be stored in a file called `LocalPreferences.toml` in your active environment (which you are free to edit by hand). The next time you start Julia it will precompile CMBLensing (which will now take a few minutes), but after that startup will be much faster. The precompilation calls the following functions:

* `load_sim`
* `logpdf`
* `gradient` of `logpdf`

with `Float32` and `Float64` CPU arrays for a dataset with `pol=:I`, `pol=:P`, and `pol=:IP`. Those funtions and any called by those functions will be _much_ faster on first call after precompilation is enabled. If you don't need all those combinations, its also possible to specify just a subset of them, e.g.:

```julia
julia> CMBLensing.set_preferences!(CMBLensing, "precompile" => "[(:P, Float32, Array))]")
```

For even more speedups and control over what to precompile, you can create a "Startup" package following the instruction here: [PrecompileTools.jl#Startup](https://julialang.github.io/PrecompileTools.jl/stable/#Tutorial:-local-%22Startup%22-packages).