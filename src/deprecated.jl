
### argmaxf_lnP

@doc doc"""
    argmaxf_lnP(ϕ,                ds::DataSet; kwargs...)
    argmaxf_lnP(ϕ, θ, ds::DataSet; kwargs...)
    argmaxf_lnP(Lϕ,               ds::DataSet; kwargs...)
    
Computes either the Wiener filter at fixed $\phi$, or a sample from this slice
along the posterior.

Keyword arguments: 

* `which` — `:wf`, `:sample`, or `fluctuation` to compute 1) the Wiener filter,
  i.e. the best-fit of $\mathcal{P}(f\,|\,\phi,d)$, 2) a sample from
  $\mathcal{P}(f\,|\,\phi,d)$, or 3) a sample minus the Wiener filter, i.e. the
  fluctuation on top of the mean.
* `fstart` — starting guess for `f` for the conjugate gradient solver
* `conjgrad_kwargs` — Passed to the inner call to [`conjugate_gradient`](@ref)

"""
argmaxf_lnP(ϕ::Field,    ds::DataSet; kwargs...) = argmaxf_lnP(cache(ds.L(ϕ),ds.d), (;), ds; kwargs...)
argmaxf_lnP(ϕ::Field, θ, ds::DataSet; kwargs...) = argmaxf_lnP(cache(ds.L(ϕ),ds.d),   θ, ds; kwargs...)

function argmaxf_lnP(
    Lϕ, 
    θ,
    ds::DataSet; 
    which = :wf, 
    fstart = nothing, 
    preconditioner = :diag, 
    conjgrad_kwargs = (tol=1e-1,nsteps=500)
)
    
    @unpack d, Cn, Cn̂, Cf, M, M̂, B, B̂ = ds(θ)
    
    Δ = d - nonCMB_data_components(θ,ds)
    b = 0
    if (which in (:wf, :sample))
        b += Lϕ'*B'*M'*(Cn\Δ)
    end
    if (which in (:fluctuation, :sample))
        ξf = simulate(Cf; d.Nbatch)
        ξn = simulate(Cn; d.Nbatch)
        b += Cf\ξf + Lϕ'*B'*M'*(Cn\ξn)
    end
    
    A_diag  = pinv(Cf) +     B̂'*M̂'*pinv(Cn̂)*M̂*B̂
    A_zeroϕ = pinv(Cf) +     B'*M'*pinv(Cn̂)*M*B
    A       = pinv(Cf) + Lϕ'*B'*M'*pinv(Cn)*M*B*Lϕ
    
    A_preconditioner = @match preconditioner begin
        :diag  => A_diag
        :zeroϕ => FuncOp(op⁻¹ = (b -> (conjugate_gradient(A_diag, A_zeroϕ, b, zero(b); conjgrad_kwargs.tol))))
        _      => error("Unrecognized preconditioner='$preconditioner'")
    end
    
    conjugate_gradient(A_preconditioner, A, b, (isnothing(fstart) ? zero(b) : fstart); conjgrad_kwargs...)
    
end


### Σ (which really was never used)

@doc doc"""
    Σ(ϕ::Field,  ds; [conjgrad_kwargs])
    Σ(Lϕ,        ds; [conjgrad_kwargs])
    
An operator for the data covariance, `Cn + M*B*L*Cf*L'*B'*M'`, which can
applied and inverted. `conjgrad_kwargs` are passed to the underlying call to
`conjugate_gradient`.
"""
Σ(ϕ::Field, ds; kwargs...) = Σ(ds.L(ϕ), ds; kwargs...)
function Σ(Lϕ, ds; conjgrad_kwargs=(tol=1e-1,nsteps=500))
    @unpack d,M,B,Cn,Cf,Cn̂,B̂,M̂ = ds
    SymmetricFuncOp(
        op   = x -> (Cn + M*B*Lϕ*Cf*Lϕ'*B'*M')*x,
        op⁻¹ = x -> conjugate_gradient((Cn̂ .+ M̂*B̂*Cf*B̂'*M̂'), Σ(Lϕ, ds), x; conjgrad_kwargs...)
    )
end


### lnP

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
    Δ = d - M(θ)*(B(θ)*f̃) - nonCMB_data_components(θ,ds)
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


### resimulate

@doc doc"""
    resimulate(ds::DataSet; [f, ϕ, n, f̃, rng, seed])

Make a new DataSet with the data replaced by a simulation. Keyword
argument fields will be used instead of new simulations, if they are
provided. 

Returns a named tuple of `(;ds, f, ϕ, n, f̃)`.
"""
resimulate(ds::DataSet; kwargs...) = resimulate!(copy(ds); kwargs...)

@doc doc"""
    resimulate!(ds::DataSet; [f, ϕ, n, f̃, rng, seed])

Replace the data in this DataSet in-place with a simulation. Keyword
argument fields will be used instead of new simulations, if they are
provided. 

Returns a named tuple of `(;ds, f, ϕ, n, f̃)`.
"""
function resimulate!(
    ds::DataSet; 
    f=nothing, ϕ=nothing, n=nothing, f̃=nothing,
    Nbatch=(isnothing(ds.d) ? nothing : ds.d.Nbatch),
    rng=global_rng_for(ds.d), seed=nothing
)

    @unpack M,B,L,Cϕ,Cf,Cn,d = ds()
    
    if isnothing(f̃)
        if isnothing(ϕ)
            ϕ = simulate(Cϕ; Nbatch, rng, seed)
        end
        if isnothing(f)
            f = simulate(Cf; Nbatch, rng, seed = (isnothing(seed) ? nothing : seed+1))
        end
        f̃ = L(ϕ)*f
    else
        f = ϕ = nothing
    end
    if isnothing(n)
        n = simulate(Cn; Nbatch, rng, seed = (isnothing(seed) ? nothing : seed+2))
    end

    ds.d = d = M*B*f̃ + n
    
    (;ds,f,ϕ,n,f̃,d)
    
end


function resimulate!(
    ds::NoLensingDataSet; 
    f=nothing, n=nothing,
    Nbatch=(isnothing(ds.d) ? nothing : ds.d.Nbatch),
    rng=global_rng_for(ds.d), seed=nothing
)

    @unpack M,B,Cf,Cn,d = ds()
    
    if isnothing(f)
        f = simulate(Cf; Nbatch, rng, seed = (isnothing(seed) ? nothing : seed+1))
    end
    if isnothing(n)
        n = simulate(Cn; Nbatch, rng, seed = (isnothing(seed) ? nothing : seed+2))
    end

    ds.d = d = M*B*f + n
    
    (;ds,f,n,d)
    
end

