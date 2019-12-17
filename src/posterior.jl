
"""
    mix(f, ϕ,                ds::DataSet)
    mix(f, ϕ, θ::NamedTuple, ds::DataSet)
    
Compute the mixed `(f°, ϕ°)` from the unlensed field `f` and lensing potential
`ϕ`, given the definition of the mixing matrices in `ds` evaluated at parameters
`θ` (or at fiducial values if no `θ` provided).
"""
mix(f, ϕ, ds::DataSet) = mix(f,ϕ,NamedTuple(),ds)
function mix(f, ϕ, θ::NamedTuple, ds::DataSet)
    @unpack D,G,QL,MÐ,MŁ = ds(;θ...)
    QL(ϕ,MÐ,MŁ)*D*f, G*ϕ
end


"""
    unmix(f°, ϕ°,                ds::DataSet)
    unmix(f°, ϕ°, θ::NamedTuple, ds::DataSet)

Compute the unmixed/unlensed `(f, ϕ)` from the mixed field `f°` and mixed
lensing potential `ϕ°`, given the definition of the mixing matrices in `ds`
evaluated at parameters `θ` (or at fiducial values if no `θ` provided). 
"""
unmix(f°, ϕ°, ds::DataSet) = unmix(f°,ϕ°,NamedTuple(),ds)
function unmix(f°, ϕ°, θ::NamedTuple, ds::DataSet)
    @unpack D,G,QL,MÐ,MŁ = ds(;θ...)
    ϕ = G\ϕ°
    D\(QL(ϕ,MÐ,MŁ)\f°), ϕ
end


@doc doc"""
    lnP(t, fₜ, ϕₜ,                ds::DataSet)
    lnP(t, fₜ, ϕₜ, θ::NamedTuple, ds::DataSet)

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
lnP(t, fₜ, ϕₜ,                ds::DataSet) = lnP(Val(t), fₜ, ϕₜ, NamedTuple(), ds)
lnP(t, fₜ, ϕₜ, θ::NamedTuple, ds::DataSet) = lnP(Val(t), fₜ, ϕₜ, θ,            ds)

function lnP(::Val{t}, fₜ, ϕ, θ::NamedTuple, ds::DataSet) where {t}
    
    @unpack Cn,Cf,Cϕ,M,P,B,L,d = ds(;θ...)
    
    f,f̃ = t==0 ? (fₜ, L(ϕ)*fₜ) : (L(ϕ)\fₜ, fₜ)
    Δ = d - M*P*B*f̃
    -1/2f0 * (
        Δ'*pinv(Cn)*Δ + logdet(ds.Cn,θ) +
        f'*pinv(Cf)*f + logdet(ds.Cf,θ) +
        ϕ'*pinv(Cϕ)*ϕ + logdet(ds.Cϕ,θ)
    )

end

function lnP(::Val{:mix}, f°, ϕ°, θ::NamedTuple, ds::DataSet)
    lnP(Val(0), unmix(f°,ϕ°,θ,ds)..., θ, ds) - logdet(ds.D,θ) - logdet(ds.G,θ)
end



### marginal posterior gradients

δlnP_δϕ(ϕ, ds, ::Type{L}=LenseFlow; kwargs...) where {L} = δlnP_δϕ(L(ϕ), ds; kwargs...)

function δlnP_δϕ(L::LenseOp, ds; Nmc_det=100, progress=false, return_sims=false)
    
    @unpack d,P,M,B,Cn,Cf,Cn̂,G = ds
    
    if G!=IdentityOp; @warn "δlnP_δϕ does not currently handle the G mixing matrix"; end

    function gQD(L, ds)
        y = B' * M' * P' * (Σ(L, ds) \ ds.d)
        y * δLf_δϕ(Cf*(L'*y), L)
    end

    det_sims = @showprogress pmap(1:Nmc_det) do i gQD(L, resimulate(ds, f̃=L*simulate(ds.Cf))) end

    g = gQD(L, ds) - mean(det_sims)

    return_sims ? (g, det_sims) : g 

end
