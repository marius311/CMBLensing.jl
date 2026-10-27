# Optional MUSE integration tests

From the repository root, create an isolated environment for the optional
MuseInference integration and run its focused tests:

```sh
julia --project=test/muse -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=test/muse test/muse_mixed_parameters.jl
```

This environment was tested with Julia 1.8.5 and 1.10.10 and MuseInference
0.2.4; it has not been verified on Julia 1.13. It is separate from the main
test suite because MuseInference's current dependency bounds select older
ComponentArrays and GPUArrays versions.
