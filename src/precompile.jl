
for pol in (:I, :P, :IP)
    for T in (Float32, Float64)
        for storage in (Array,)
            (;f, ϕ, ds) = load_sim(;pol, T, storage, Nside=16, θpix=1, seed=0)
            logpdf(ds; f, ϕ)
            (;f°, ϕ°) = mix(ds; f, ϕ)
            logpdf(Mixed(ds); f°, ϕ°)
            gradient((f, ϕ) -> logpdf(ds; f, ϕ), f, ϕ)
            MAP_joint(ds, nsteps=2, progress=false)
        end
    end
end

Memoization.empty_all_caches!()
