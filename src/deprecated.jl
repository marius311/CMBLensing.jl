
### argmaxf_lnP

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

    if which == :wf
        Base.depwarn("`argmaxf_lnP(...; which=:wf)` is deprecated, use `argmaxf_logpdf` instead.", :argmaxf_lnP, force=true)
        argmaxf_logpdf(@set(ds.L=Lϕ), (;ϕ=getϕ(Lϕ), θ); fstart, preconditioner, conjgrad_kwargs)
    elseif which == :sample
        Base.depwarn("`argmaxf_lnP(...; which=:sample)` is deprecated, use `sample_f` instead.", :sample_f, force=true)
        sample_f(@set(ds.L=Lϕ), (;ϕ=getϕ(Lϕ), θ); fstart, preconditioner, conjgrad_kwargs)
    else
        error("argmaxf_lnP(...; which=:fluctuation) has been removed.")
    end

end


### Σ (which really was never used)

Σ(ϕ::Field, ds; kwargs...) = Σ(ds.L(ϕ), ds; kwargs...)
function Σ(Lϕ, ds; conjgrad_kwargs=(tol=1e-1,nsteps=500))
    Base.depwarn("`Σ` is deprecated and will be removed in a future version.", :Σ, force=true)
    @unpack d,M,B,Cn,Cf,Cn̂,B̂,M̂ = ds
    SymmetricFuncOp(
        op   = x -> (Cn + M*B*Lϕ*Cf*Lϕ'*B'*M')*x,
        op⁻¹ = x -> conjugate_gradient((Cn̂ .+ M̂*B̂*Cf*B̂'*M̂'), Σ(Lϕ, ds), x; conjgrad_kwargs...)
    )
end


### lnP

lnP(t, fₜ, ϕₜ,    ds::DataSet) = lnP(Val(t), fₜ, ϕₜ, (;), ds)
lnP(t, fₜ, ϕₜ, θ, ds::DataSet) = lnP(Val(t), fₜ, ϕₜ,  θ,  ds)
function lnP(::Val{t}, fₜ, ϕ, θ, ds::DataSet) where {t}
    if t == 0
        Base.depwarn("`lnP(0, f, ϕ, θ, ds)` is deprecated, use `logpdf(ds; f, ϕ, θ)` instead.", :lnP0, force=true)
        logpdf(ds; f=fₜ, ϕ, θ)
    else
        error("lnP(1, f, ϕ, θ, ds) has been removed.")
    end
end
function lnP(::Val{:mix}, f°, ϕ°, θ, ds::DataSet)
    Base.depwarn("`lnP(:mix, f°, ϕ°, θ, ds)` is deprecated, use `logpdf(Mixed(ds); f°, ϕ°, θ)` instead.", :lnPmix, force=true)
    logpdf(Mixed(ds); f°, ϕ°, θ)
end
function nonCMB_data_components(θ, ds::DataSet)
    error("`nonCMB_data_components` is removed in favor of defining a custom `@fwdmodel`.")
end


## NoLensingDataSet
lnP(   f,    ds::NoLensingDataSet) = lnP(f, (;), ds)
lnP(_, f, θ, ds::NoLensingDataSet) = lnP(f, θ,   ds)
function lnP(f, θ, ds::NoLensingDataSet)
    Base.depwarn("`lnP(f, ϕ, θ, ds)` is deprecated, use `logpdf(ds; f, ϕ, θ)` instead.", :lnP0, force=true)
    logpdf(ds; f, θ)
end

### mixing
unmix(f°, ϕ°, ds::DataSet) = unmix(f°,ϕ°,(;),ds)
function unmix(f°, ϕ°, θ, ds::DataSet)
    Base.depwarn("`unmix(f°, ϕ°, θ, ds)` is deprecated, use `unmix(ds; f°, ϕ°, θ)` instead.", :unmix, force=true)
    unmix(ds; f°, ϕ°, θ)
end
mix(f, ϕ, ds::DataSet) = mix(f,ϕ,(;),ds)
function mix(f, ϕ, θ, ds::DataSet)
    Base.depwarn("`mix(f, ϕ, θ, ds)` is deprecated, use `mix(ds; f, ϕ, θ)` instead.", :mix, force=true)
    mix(ds; f, ϕ, θ)
end


### resimulate

resimulate(ds::DataSet; kwargs...) = resimulate!(copy(ds); kwargs...)
function resimulate!(
    ds::DataSet; 
    f=nothing, ϕ=nothing, n=nothing, f̃=nothing,
    Nbatch=(isnothing(ds.d) ? nothing : ds.d.Nbatch),
    rng=Random.default_rng(), seed=nothing
)

    (f̃ != nothing) || (n != nothing) && error("Passing f̃ or n to `resimulate!` has been removed.")
    Base.depwarn("`resimulate(ds)` is deprecated, use `simulate(ds)` instead.", :resimulate, force=true)
    @unpack f, f̃, ϕ, d = simulate(rng, ds; f = (isnothing(f) ? missing : f), ϕ = (isnothing(ϕ) ? missing : ϕ))
    ds.d = d
    (;ds, f, f̃, ϕ, d)
    
end


Base.@deprecate white_noise(f::Field, rng::AbstractRNG) randn!(rng, f)
