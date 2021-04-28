
"""
    mix(f, ϕ,    ds::DataSet)
    mix(f, ϕ, θ, ds::DataSet)
    
Compute the mixed `(f°, ϕ°)` from the unlensed field `f` and lensing potential
`ϕ`, given the definition of the mixing matrices in `ds` evaluated at parameters
`θ` (or at fiducial values if no `θ` provided).
"""
mix(f, ϕ, ds::DataSet) = mix(f,ϕ,NamedTuple(),ds)
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
unmix(f°, ϕ°, ds::DataSet) = unmix(f°,ϕ°,NamedTuple(),ds)
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
lnP(t, fₜ, ϕₜ,    ds::DataSet) = lnP(Val(t), fₜ, ϕₜ, NamedTuple(), ds)
lnP(t, fₜ, ϕₜ, θ, ds::DataSet) = lnP(Val(t), fₜ, ϕₜ, θ,            ds)

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


### marginal posterior gradients

function δlnP_δϕ(
    ϕ, θ, ds;
    previous_state = nothing,
    use_previous_MF = false,
    Nsims = (isnothing(previous_state) ? 50 : previous_state.Nsims), 
    Nbatch = 1,
    weights = :unlensed,
    return_state = false,
    progress = false,
    conjgrad_kwargs = (tol=1e-1,nsteps=500),
    aggressive_gc = fieldinfo(ϕ).Nside>=512
)

    dsθ = ds(θ)
    set_distributed_dataset(dsθ)
    @unpack d,L,G,Cf,Cf̃,Cn,Cϕ = dsθ
    (d.Nbatch == Nbatch == 1) || error("δlnP_δϕ for batched fields not implemented")
    (G == I) || error("δlnP_δϕ with G!=I not implemented")
    Lϕ = L(ϕ)

    if isnothing(previous_state)
        f_sims = [simulate(Cf; Nbatch) for i=1:Nsims÷Nbatch]
        n_sims = [simulate(Cn; Nbatch) for i=1:Nsims÷Nbatch]
        f_wf_sims_prev = fill(nothing, Nsims÷Nbatch)
        f_wf_prev = nothing
    else
        @unpack f_sims, n_sims, f_wf_sims_prev, f_wf_prev = previous_state
    end

    # generate simulated data for the current ϕ
    d_sims = map(f_sims, n_sims) do f,n
        resimulate(dsθ, f̃=Lϕ*f, n=n).d
    end

    W = (weights == :unlensed) ? I : (Cf̃ * pinv(Cf))

    # gradient of the quadratic (in d) piece of the likelihood
    function get_gQD(Lϕ, ds, f_wf_prev)
        @unpack Cf = ds
        f_wf, history = argmaxf_lnP(
            Lϕ, θ, ds;
            which = :wf,
            fstart = (isnothing(f_wf_prev) ? 0d : f_wf_prev),
            conjgrad_kwargs = (history_keys=(:i,:res), conjgrad_kwargs...)
        )
        aggressive_gc && cuda_gc()
        v = Lϕ' \ (Cf \ f_wf)
        w = W * f_wf
        g = gradient(ϕ -> v' * (Lϕ(ϕ) * w), getϕ(Lϕ))[1]
        (;g, f_wf, history)
    end

    if use_previous_MF
        gQD = get_gQD(Lϕ, dsθ, f_wf_prev)
        @unpack gQD_sims, ḡ = previous_state
    else
        gQD, gQD_sims = peel(pmap([dsθ.d, d_sims...], [f_wf_prev, f_wf_sims_prev...]) do d, f_wf_prev
            get_gQD(Lϕ, @set(get_distributed_dataset().d=d), f_wf_prev)
        end)
        ḡ = mean(map(gQD_sims) do gQD
            mean(unbatch(gQD.g))
        end)
    end
    
    # final total posterior gradient, including gradient of the prior
    g = gQD.g - ḡ - Cϕ\ϕ

    if return_state
        g, (;g, f_sims, n_sims, Nsims, ḡ, gQD, gQD_sims, f_wf_prev=gQD.f_wf, f_wf_sims_prev=getindex.(gQD_sims,:f_wf))
    else
        g
    end

end
