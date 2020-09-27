
"""
    mix(f, ϕ,                ds::DataSet)
    mix(f, ϕ, θ::NamedTuple, ds::DataSet)
    
Compute the mixed `(f°, ϕ°)` from the unlensed field `f` and lensing potential
`ϕ`, given the definition of the mixing matrices in `ds` evaluated at parameters
`θ` (or at fiducial values if no `θ` provided).
"""
mix(f, ϕ, ds::DataSet) = mix(f,ϕ,NamedTuple(),ds)
function mix(f, ϕ, θ::NamedTuple, ds::DataSet)
    @unpack D,G,L = ds(θ)
    L(ϕ)*D*f, G*ϕ
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
    @unpack D,G,L = ds(θ)
    ϕ = G\ϕ°
    D\(L(ϕ)\f°), ϕ
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
    
    @unpack Cn,Cf,Cϕ,L,M,B,d = ds
    @unpack T = fieldinfo(d)
    
    f,f̃ = t==0 ? (fₜ, L(ϕ)*fₜ) : (L(ϕ)\fₜ, fₜ)
    Δ = d - M(θ)*B(θ)*f̃ - nonCMB_data_components(θ,ds)
    (
        -T(1/2) * (
            Δ'*pinv(Cn(θ))*Δ + logdet(Cn,θ) +
            f'*pinv(Cf(θ))*f + logdet(Cf,θ) +
            ϕ'*pinv(Cϕ(θ))*ϕ + logdet(Cϕ,θ)
        ) 
        + T(lnPriorθ(θ,ds))
    )

end

function lnP(::Val{:mix}, f°, ϕ°, θ::NamedTuple, ds::DataSet)
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
    Nsims = (previous_state==nothing ? 50 : previous_state.Nsims), 
    Nbatch = 1,
    weights = :unlensed,
    return_state = false,
    progress = false,
    conjgrad_kwargs = (tol=1e-1,nsteps=500),
    aggressive_gc = fieldinfo(ϕ).Nside>=512
)

    @unpack d,P,M,B,Cn,Cf,Cf̃,Cϕ,Cn̂,G,L = ds(θ)
    Lϕ = L(ϕ)
    Lϕ_batch = Nbatch==1 ? Lϕ : cache(LenseFlow(identity.(batch(ϕ,Nbatch))),identity.(batch(d,Nbatch)))
    
    if G!=1
        @warn "δlnP_δϕ does not currently handle the G mixing matrix"
    end
    
    if (previous_state == nothing)
        f_sims = [simulate(batch(Cf,Nbatch)) for i=1:Nsims÷Nbatch]
        n_sims = [simulate(batch(Cn,Nbatch)) for i=1:Nsims÷Nbatch]
        f_wf_sims_guesses = fill(nothing, Nsims÷Nbatch)
        f_wf_guess = nothing
    else
        @unpack f_sims, n_sims, f_wf_sims_guesses, f_wf_guess = previous_state
    end

    # generate simulated datasets for the current ϕ
    ds_sims = map(f_sims, n_sims) do f,n
        resimulate(ds, f̃=Lϕ_batch*f, n=n).ds
    end

    W = (weights == :unlensed) ? 1 : (Cf̃ * pinv(Cf))

    # gradient of the quadratic (in ϕ) piece of the likelihood
    function get_gQD(Lϕ, ds, f_wf_guess)
        f_wf, hist = argmaxf_lnP(
            Lϕ, θ, ds;
            which = :wf,
            guess = (f_wf_guess==nothing ? 0d : f_wf_guess),
            conjgrad_kwargs = (hist=(:i,:res), conjgrad_kwargs...)
        )
        aggressive_gc && gc()
        v = Lϕ' \ (Cf \ f_wf)
        w = W * f_wf
        g = gradient(ϕ -> v' * (Lϕ(ϕ) * w), getϕ(Lϕ))[1]
        @namedtuple(g, f_wf, hist)
    end

    # gQD for the real data
    gQD = get_gQD(Lϕ, ds, f_wf_guess)

    # gQD for several simulated datasets
    if use_previous_MF
        @unpack gQD_sims, ḡ = previous_state
    else
        gQD_sims = pmap(ds_sims, f_wf_sims_guesses) do ds_sim, f_wf_guess
            get_gQD(Lϕ_batch, ds_sim, f_wf_guess)
        end
        ḡ = mean(mapreduce(vcat, gQD_sims) do gQD
            [batchindex(gQD.g,i) for i=1:batchsize(gQD.g)]
        end)
    end
    
    # final total posterior gradient, including gradient of the prior
    g = gQD.g - ḡ - Cϕ\ϕ

    if return_state
        g, @namedtuple(g, f_sims, n_sims, ḡ, gQD, gQD_sims, f_wf_guess=gQD.f_wf, f_wf_sims_guesses=getindex.(gQD_sims,:f_wf))
    else
        g
    end

end
