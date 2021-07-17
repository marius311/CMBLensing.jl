
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


@doc doc"""
    lnP(t, fₜ, ϕₜ,    ds::DataSet)
    lnP(t, fₜ, ϕₜ, θ, ds::DataSet)

Compute the log posterior probability in the joint parameterization as a
function of the field, $f_t$, the lensing potential, $\phi_t$, and possibly some
cosmological parameters, $\theta$. The subscript $t$ can refer to either a
"time", e.g. passing `t=0` corresponds to the unlensed parametrization and `t=1`
to the lensed one, or can be `:mix` correpsonding to the mixed parametrization.
In all cases, the arguments `fₜ` and `ϕₜ` should then be $f$ and $\phi$ in
that particular parametrization.

If any parameters $\theta$ are provided, we also include the determinant terms for
covariances which depend on $\theta$. In the mixed parametrization, we
also include any Jacobian determinant terms that depend on $\theta$. 

The argument `ds` should be a `DataSet` and stores the masks, data, etc...
needed to construct the posterior. 
"""
lnP(t, fₜ, ϕₜ,    ds::DataSet) = lnP(Val(t), fₜ, ϕₜ, (;), ds)
lnP(t, fₜ, ϕₜ, θ, ds::DataSet) = lnP(Val(t), fₜ, ϕₜ,  θ,  ds)

function lnP(::Val{t}, fₜ, ϕ, θ, ds::DataSet) where {t}
    
    @unpack Cn,Cf,Cϕ,L,M,B,d = ds
    
    f,f̃ = t==0 ? (fₜ, L(ϕ)*fₜ) : (L(ϕ)\fₜ, fₜ)
    Δ = d - M(θ)*B(θ)*f̃ - nonCMB_data_components(θ,ds)
    (
        -(1//2) * (
            Δ'*pinv(Cn(θ))*Δ + logdet(Cn,θ) +
            f'*pinv(Cf(θ))*f + logdet(Cf,θ) +
            ϕ'*pinv(Cϕ(θ))*ϕ + logdet(Cϕ,θ)
        ) 
        + lnPriorθ(θ,ds)
    )

end

function lnP(::Val{:mix}, f°, ϕ°, θ, ds::DataSet)
    lnP(Val(0), unmix(f°,ϕ°,θ,ds)..., θ, ds) - logdet(ds.D,θ) - logdet(ds.G,θ)
end

# can be specialized for specific DataSet types:
nonCMB_data_components(θ, ds::DataSet) = 0
lnPriorθ(θ, ds::DataSet) = 0


## NoLensingDataSet

lnP(   f,    ds::NoLensingDataSet) = lnP(f, (;), ds)
lnP(_, f, θ, ds::NoLensingDataSet) = lnP(f, θ,   ds)

function lnP(f, θ, ds::NoLensingDataSet)
    
    @unpack Cn,Cf,M,B,d = ds
    
    Δ = d - M(θ)*B(θ)*f - nonCMB_data_components(θ,ds)
    (
        -(1//2) * (
            Δ'*pinv(Cn(θ))*Δ + logdet(Cn,θ) +
            f'*pinv(Cf(θ))*f + logdet(Cf,θ)
        ) 
        + lnPriorθ(θ,ds)
    )

end
