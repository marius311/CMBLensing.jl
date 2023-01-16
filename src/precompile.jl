
for pol in (:I, :P, :IP)
    for T in (Float32, Float64)
        for storage in (Array,)
            load_sim(;pol, T, storage, Nside=4, θpix=1)
        end
    end
end

Memoization.empty_all_caches!()
