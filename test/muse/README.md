# Optional MUSE integration tests

From the repository root, create an isolated environment for the optional
MuseInference integration and run its focused tests:

```sh
julia --project=test/muse -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=test/muse test/muse_mixed_parameters.jl
julia --project=test/muse test/muse_mixed_simulation.jl
```

This environment is separate from the main test suite because MuseInference's
current dependency bounds select older ComponentArrays and GPUArrays versions.
