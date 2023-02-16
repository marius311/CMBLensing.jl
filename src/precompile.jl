
let 
    
    precompile = @load_preference("precompile", default=false)

    if precompile != false

        if precompile == true
            precompile_type_matrix = Iterators.product((:I,:P,:IP), (Float32,Float64), (Array,))
        else
            precompile_type_matrix = Iterators.product(eval(Meta.parse(precompile))...)
        end

        for (pol, T, storage) in precompile_type_matrix
            (;f, ϕ, ds) = load_sim(;pol, T, storage, Nside=16, θpix=1, seed=0)
            logpdf(ds; f, ϕ)
            (;f°, ϕ°) = mix(ds; f, ϕ)
            logpdf(Mixed(ds); f°, ϕ°)
            gradient((f, ϕ) -> logpdf(ds; f, ϕ), f, ϕ)
            MAP_joint(ds, nsteps=2, progress=false)
        end

        Memoization.empty_all_caches!()

    end

end