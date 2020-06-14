
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

function signal_model(f, f̃, ϕ, θ, ds::DataSet)
    @unpack M,B = ds
    M(θ) * B(θ) * f̃
end

function lnP(::Val{t}, fₜ, ϕ, θ::NamedTuple, ds::DataSet) where {t}
    
    @unpack Cn,Cf,Cϕ,L,d = ds
    
    f,f̃ = t==0 ? (fₜ, L(ϕ)*fₜ) : (L(ϕ)\fₜ, fₜ)
    Δ = d - signal_model(f,f̃,ϕ,θ,ds)
    -1/2f0 * (
        Δ'*pinv(Cn(θ))*Δ + logdet(Cn,θ) +
        f'*pinv(Cf(θ))*f + logdet(Cf,θ) +
        ϕ'*pinv(Cϕ(θ))*ϕ + logdet(Cϕ,θ)
    )

end

function lnP(::Val{:mix}, f°, ϕ°, θ::NamedTuple, ds::DataSet)
    lnP(Val(0), unmix(f°,ϕ°,θ,ds)..., θ, ds) - logdet(ds.D,θ) - logdet(ds.G,θ)
end



### marginal posterior gradients

function δlnP_δϕ(
    ϕ, ds; 
    previous_state=nothing,
    Nsims=(previous_state==nothing ? 50 : previous_state.Nsims), 
    weights=:unlensed,
    return_state=false,
    progress=false,
    conjgrad_kwargs=()
    )

    @unpack d,P,M,B,Cn,Cf,Cf̃,Cϕ,Cn̂,G,L = ds
    Lϕ = L(ϕ)
    
    if !in(G,(1,IdentityOp))
        @warn "δlnP_δϕ does not currently handle the G mixing matrix"
    end
    
    if (previous_state == nothing)
        f_sims = [simulate(ds.Cf) for i=1:Nsims]
        n_sims = [simulate(ds.Cn) for i=1:Nsims]
        f_wf_sims_guesses = fill(nothing, Nsims)
        f_wf_guess = nothing
    else
        @unpack f_sims, n_sims, f_wf_sims_guesses, f_wf_guess = previous_state
    end

    # generate simulated datasets for the current ϕ
    ds_sims = map(f_sims, n_sims) do f,n
        resimulate(ds, f=f, ϕ=ϕ, n=n).ds
    end

    W = (weights == :unlensed) ? 1 : (Cf̃ * pinv(Cf))

    # gradient of the quadratic piece of the likelihood
    function get_gQD(Lϕ, ds, f_wf_guess)
        f_wf, hist = argmaxf_lnP(
            Lϕ, ds;
            which = :wf, 
            guess = (f_wf_guess==nothing ? 0d : f_wf_guess), 
            conjgrad_kwargs = (hist=(:i,:res), conjgrad_kwargs...)
        )
        v = Lϕ' \ (Cf \ f_wf)
        w = W * f_wf
        g = gradient(ϕ -> v' * (L(ϕ) * w), ϕ)[1]
        @namedtuple(g, f_wf, hist)
    end

    # gQD for the real data
    gQD = get_gQD(Lϕ, ds, f_wf_guess)

    # gQD for several simulated datasets, used to compute the gradient of the
    # logdet term via Monte-Carlo
    gQD_sims = @showprogress (progress ? 1 : Inf) "Hutchinson's method: " pmap(ds_sims, f_wf_sims_guesses) do ds, f_wf_guess
        get_gQD(Lϕ, ds, f_wf_guess)
    end

    # final total gradient, including gradient of the prior
    g = gQD.g - mean(getindex.(gQD_sims,:g)) - Cϕ\ϕ

    if return_state
        g, @namedtuple(f_sims, n_sims, gQD, gQD_sims, f_wf_guess=gQD.f_wf, f_wf_sims_guesses=getindex.(gQD_sims,:f_wf))
    else
        g
    end

end
