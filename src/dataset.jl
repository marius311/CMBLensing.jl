
# Stores variables needed to construct the posterior
@kwdef mutable struct DataSet{F}
    d :: F           # data
    Cϕ               # ϕ covariance
    Cf               # unlensed field covariance
    Cf̃ = nothing     # lensed field covariance (not always needed)
    Cn               # noise covariance
    Cn̂ = Cn          # approximate noise covariance, diagonal in same basis as Cf
    M  = 1           # user mask
    M̂  = M           # approximate user mask, diagonal in same basis as Cf
    B  = 1           # beam and instrumental transfer functions
    B̂  = B           # approximate beam and instrumental transfer functions, diagonal in same basis as Cf
    D  = 1           # mixing matrix for mixed parametrization
    G  = 1           # reparametrization for ϕ
    P  = 1           # pixelization operator (if estimating field on higher res than data)
    L  = alloc_cache(LenseFlow(similar(diag(Cϕ))),d) # a CachedLenseFlow which will be reused for memory
end

function subblock(ds::DataSet, block)
    DataSet(map(collect(pairs(fields(ds)))) do (k,v)
        @match (k,v) begin
            ((:Cϕ || :G || :L), v)              => v
            (_, L::Union{Nothing,FuncOp,Real})  => L
            (_, L)                              => getindex(L,block)
        end
    end...)
end

function (ds::DataSet)(;θ...)
    DataSet(map(fieldvalues(ds)) do v
        (v isa ParamDependentOp) ? v(;θ...) : v
    end...)
end

function check_hat_operators(ds::DataSet)
    @unpack B̂, M̂, Cn̂, Cf = ds()
    @assert(all([(L isa Scalar) || (L isa typeof(Cf)) || (Cf isa FlatIEBCov && L isa DiagOp{<:FlatIEBFourier}) for L in [B̂,M̂,Cn̂]]),
            "B̂, M̂, Cn̂ should be scalars or the same type as Cf")
end

adapt_structure(to, ds::DataSet) = DataSet(adapt(to, fieldvalues(ds))...)


@doc doc"""
    resimulate(ds::DataSet; [f, ϕ, n])
    
Make a new DataSet replacing the data with a simulation, potentially given a
fixed f, ϕ, or n, if any are provided. 

Returns a named tuple of `(ds, f, ϕ, n, f̃)`
"""
function resimulate(
    ds::DataSet{F}; 
    f=nothing, ϕ=nothing, n=nothing, 
    rng=global_rng_for(F), seed=nothing) where {F}
    
    if (ϕ == nothing)
        ϕ = simulate(ds.Cϕ, rng=rng, seed=seed)
    end
    if (f == nothing)
        f = simulate(ds.Cf, rng=rng, seed=(seed==nothing ? nothing : seed+1))
    end
    if (n == nothing)
        n = simulate(ds.Cn, rng=rng, seed=(seed==nothing ? nothing : seed+2))
    end

    @unpack M,P,B,L = ds
    f̃ = L(ϕ)*f
    d = M*P*B*f̃ + n
    ds = (@set ds.d = d)
    
    @namedtuple(ds,f,ϕ,n,f̃)
end

@doc doc"""
    resimulate!(ds::DataSet; [f, ϕ, n])
    
Replace the data in this DataSet in-place with a simulation, potentially given a
fixed f, ϕ, or n, if any are provided. 
    
Returns a named tuple of `(ds, f, ϕ, n, f̃)`
"""
function resimulate!(ds::DataSet; kwargs...)
    ds′ = ds
    @unpack ds,f,ϕ,n,f̃ = resimulate(ds; kwargs...)
    ds′.d = ds.d
    @namedtuple(ds=ds′,f,ϕ,n,f̃)
end



@doc doc"""
    load_sim_dataset
    
Create a `DataSet` object with some simulated data. E.g.

```julia
@unpack f,ϕ,ds = load_sim_dataset(;
    θpix  = 2,
    Nside = 128,
    pol   = :I,
    T     = Float32
);
```

"""
function load_sim_dataset(;
    
    # basic configuration
    θpix,
    θpix_data = θpix,
    Nside,
    pol,
    T = Float32,
    storage = Array,
    
    # noise parameters, or set Cℓn or even Cn directly
    μKarcminT = 3,
    ℓknee = 100,
    αknee = 3,
    Cℓn = nothing,
    Cn = nothing,
    
    # beam parameters, or set B directly
    beamFWHM = 0,
    B = nothing,
    
    # mask parameters, or set M directly
    pixel_mask_kwargs = nothing,
    bandpass_mask = LowPass(3000),
    M = nothing, M̂ = nothing,
    
    # theory
    Cℓ = nothing,
    fiducial_θ = NamedTuple(),
    rfid = nothing,
    
    seed = nothing,
    D = nothing,
    G = nothing,
    Nϕ_fac = 2,
    ϕ=nothing, f=nothing, f̃=nothing, Bf̃=nothing, n=nothing, d=nothing, # can override any of these simulated fields
    L = LenseFlow,
    ∂mode = fourier∂
    )
    
    # the biggest ℓ on the 2D fourier grid
    ℓmax = round(Int,ceil(√2*fieldinfo(Flat(θpix=θpix,Nside=Nside)).nyq)+1)
    
    # CMB Cℓs
    if rfid != nothing
        @warn "`rfid` will be removed in a future version. Use `fiducial_θ=(r=...,)` instead."
        fiducial_θ = merge(fiducial_θ,(r=rfid,))
    end
    Aϕ₀ = get(fiducial_θ, :Aϕ, 1)
    fiducial_θ = Base.structdiff(fiducial_θ, NamedTuple{(:Aϕ,)}) # remove Aϕ key if present
    if Cℓ==nothing
        Cℓ = camb(;fiducial_θ..., ℓmax=ℓmax)
    else
        if !isempty(fiducial_θ)
            error("Can't pass both `Cℓ` and `fiducial_θ` parameters which affect `Cℓ`, choose one or the other.")
        elseif maximum(Cℓ.total.TT.ℓ) < ℓmax
            error("ℓmax of `Cℓ` argument should be higher than $ℓmax for this configuration.")
        end
    end
    r₀ = Cℓ.params.r
    
    # noise Cℓs (these are non-debeamed, hence beamFWHM=0 below; the beam comes in via the B operator)
    if (Cℓn == nothing)
        Cℓn = noiseCℓs(μKarcminT=μKarcminT, beamFWHM=0, ℓknee=ℓknee, αknee=αknee, ℓmax=ℓmax)
    end
    
    # some things which depend on whether we chose :I, :P, or :IP
    pol = Symbol(pol)
    S,ks,F,F̂,nF = @match pol begin
        :I  => (S0,  (:TT,),            FlatMap,    FlatFourier,    1)
        :P  => (S2,  (:EE,:BB),         FlatQUMap,  FlatEBFourier,  2)
        :IP => (S02, (:TT,:EE,:BB,:TE), FlatIQUMap, FlatIEBFourier, 3)
        _   => throw(ArgumentError("`pol` should be one of :I, :P, or :IP"))
    end
    
    # pixelization
    Pix = Flat(Nside=Nside, θpix=θpix, ∂mode=∂mode)
    if (θpix_data == θpix)
        Pix_data = Pix
        P = Identity
    else
        Pix_data = Flat(Nside=Nside÷(θpix_data÷θpix), θpix=θpix_data, ∂mode=∂mode)
        P = FuncOp(
            op  = f -> ud_grade(f, θpix_data, deconv_pixwin=false, anti_aliasing=false),
            opᴴ = f -> ud_grade(f, θpix,      deconv_pixwin=false, anti_aliasing=false)
        )
    end
    
    # covariances
    Cϕ₀ = adapt(storage, Cℓ_to_Cov(Pix,      T, S0, (Cℓ.total.ϕϕ)))
    Cfs = adapt(storage, Cℓ_to_Cov(Pix,      T, S,  (Cℓ.unlensed_scalar[k] for k in ks)...))
    Cft = adapt(storage, Cℓ_to_Cov(Pix,      T, S,  (Cℓ.tensor[k]          for k in ks)...))
    Cf̃  = adapt(storage, Cℓ_to_Cov(Pix,      T, S,  (Cℓ.total[k]           for k in ks)...))
    Cn̂  = adapt(storage, Cℓ_to_Cov(Pix_data, T, S,  (Cℓn[k]                for k in ks)...))
    if (Cn == nothing); Cn = Cn̂; end
    Cf = ParamDependentOp((;r=r₀,   _...)->(Cfs + T(r/r₀)*Cft))
    Cϕ = ParamDependentOp((;Aϕ=Aϕ₀, _...)->(T(Aϕ) .* Cϕ₀))
    
    # data mask
    if (M == nothing)
        M̂ = M = adapt(storage, Cℓ_to_Cov(Pix_data, T, S, ((k==:TE ? 0 : 1) * bandpass_mask.diag.Wℓ for k in ks)...; units=1))
        if (pixel_mask_kwargs != nothing)
            M = M * adapt(storage, Diagonal(F{Pix_data}(repeated(T.(make_mask(Nside÷(θpix_data÷θpix),θpix_data; pixel_mask_kwargs...).Ix),nF)...)))
        end
    end
    if diag(M̂) isa BandPass
        M̂ = Diagonal(M̂ * one(diag(Cf)))
    end
    
    # beam
    if (B == nothing)
        B̂ = B = adapt(storage, Cℓ_to_Cov(Pix, T, S, ((k==:TE ? 0 : 1) * sqrt(beamCℓs(beamFWHM=beamFWHM)) for k=ks)..., units=1))
    end
    
    # simulate data
    seed_for_storage!(storage, seed)
    if (ϕ  == nothing); ϕ  = simulate(Cϕ); end
    if (f  == nothing); f  = simulate(Cf); end
    if (n  == nothing); n  = simulate(Cn); end
    Lϕ = cache(L(ϕ),f)
    if (f̃  == nothing); f̃  = Lϕ*f;         end
    if (Bf̃ == nothing); Bf̃ = B*f̃;          end
    if (d  == nothing); d  = M*P*Bf̃ + n;   end
    
    # put everything in DataSet
    ds = DataSet(;@namedtuple(d, Cn, Cn̂, Cf, Cf̃, Cϕ, M, M̂, B, B̂, D, P, L=Lϕ)...)
    
    
    # with the DataSet created, we now create the mixing matrices D and G, which
    # will close-over `ds` and use `ds.Cf` and `ds.Cϕ`, so if these are later
    # changed the mixing matrices will remain consistent. we have to wrap the
    # closed-over `ds` in a Ref to prevent a circular dependency since our
    # recursive `adapt` function doesn't check for loops, but doesn't recurse
    # into `Ref`s. 
    
    if (G == nothing)
        Nϕ = quadratic_estimate(ds,(pol in (:P,:IP) ? :EB : :TT)).Nϕ / Nϕ_fac
        G₀ = @. nan2zero(sqrt(1 + 2/($Cϕ()/Nϕ)))
        ds.G = ParamDependentOp((;Aϕ=Aϕ₀, _...)->(@. nan2zero(sqrt(1 + 2/(($Cϕ(;Aϕ=Aϕ)/Nϕ)))/G₀)))
    end
    
    if (D == nothing)
        σ²len = T(deg2rad(5/60)^2)
        ds.D = ParamDependentOp(
            function (;r=r₀, _...)
                Cfr = Cf(;r=r)
                sqrt(Diagonal(diag(Cfr) .+ σ²len .+ 2*diag(Cn̂)) * pinv(Cfr))
            end,
        )
    end
    
    return adapt(storage, @namedtuple(f, f̃, ϕ, n, ds, ds₀=ds(), T, P=Pix, Cℓ, L))
    
end
