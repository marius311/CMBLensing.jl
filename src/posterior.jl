
### generic DataSet

"""
    mix(f, ϕ,    ds::DataSet)
    mix(f, ϕ, θ, ds::DataSet)
    
Compute the mixed `(f°, ϕ°)` from the unlensed field `f` and lensing potential
`ϕ`, given the definition of the mixing matrices in `ds` evaluated at parameters
`θ` (or at fiducial values if no `θ` provided).
"""
mix(f, ϕ, ds::DataSet) = mix(f,ϕ,(;),ds)
function mix(f, ϕ, θ, ds::DataSet)
    @unpack D,G,L = ds(θ)
    L(ϕ)*D*f, G*ϕ
end


"""
    unmix(f°, ϕ°,    ds::DataSet)
    unmix(f°, ϕ°, θ, ds::DataSet)

Compute the unmixed/unlensed `(f, ϕ)` from the mixed field `f°` and mixed
lensing potential `ϕ°`, given the definition of the mixing matrices in `ds`
evaluated at parameters `θ` (or at fiducial values if no `θ` provided). 
"""
unmix(f°, ϕ°, ds::DataSet) = unmix(f°,ϕ°,(;),ds)
function unmix(f°, ϕ°, θ, ds::DataSet)
    @unpack D,G,L = ds(θ)
    ϕ = G\ϕ°
    D\(L(ϕ)\f°), ϕ
end

# maybe keep this
lnPriorθ(θ, ds::DataSet) = 0
