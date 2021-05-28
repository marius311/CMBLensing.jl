

abstract type DataSet end

copy(ds::DS) where {DS<:DataSet} = DS(fields(ds)...)

hash(ds::DataSet, h::UInt64) = foldr(hash, (typeof(ds), fieldvalues(ds)...), init=h)

function show(io::IO, ds::DataSet)
    println(io, typeof(ds), ": ")
    ds_dict = OrderedDict(k => getproperty(ds,k) for k in propertynames(ds) if k!=Symbol("_super"))
    for line in split(sprint(show, MIME"text/plain"(), ds_dict, context = (:limit => true)), "\n")[2:end]
        println(io, line)
    end
end


# util for distributing a singleton global dataset to workers
"""
    set_distributed_dataset(ds)
    get_distributed_dataset()

Sometimes it's more performant to distribute a DataSet object to
parallel workers just once, and have them refer to it from the global
state, rather than having it get automatically but repeatedly sent as
part of closures. This provides that functionality. Use
`set_distributed_dataset(ds)` from the master process to set the
global DataSet and `get_distributed_dataset()` from any process to
retrieve it. Repeated calls will not resend `ds` if it hasn't changed
(based on `hash(ds)`) and if no new workers have been added since the
last send.
"""
function set_distributed_dataset(ds)
    h = hash((procs(), ds))
    if h != _distributed_dataset_hash
        @everywhere @eval CMBLensing _distributed_dataset = $ds
        global _distributed_dataset_hash = h
    end
    nothing
end
get_distributed_dataset() = _distributed_dataset
_distributed_dataset = nothing
_distributed_dataset_hash = nothing


# Stores variables needed to construct the posterior
@kwdef mutable struct BaseDataSet <: DataSet
    d                # data
    Cϕ               # ϕ covariance
    Cf               # unlensed field covariance
    Cf̃ = nothing     # lensed field covariance (not always needed)
    Cn               # noise covariance
    Cn̂ = Cn          # approximate noise covariance, diagonal in same basis as Cf
    M  = I           # user mask
    M̂  = M           # approximate user mask, diagonal in same basis as Cf
    B  = I           # beam and instrumental transfer functions
    B̂  = B           # approximate beam and instrumental transfer functions, diagonal in same basis as Cf
    D  = I           # mixing matrix for mixed parametrization
    G  = I           # reparametrization for ϕ
    L  = LenseFlow   # lensing operator, possibly cached for memory reuse
end

function subblock(ds::DS, block) where {DS<:DataSet}
    DS(map(collect(pairs(fields(ds)))) do (k,v)
        @match (k,v) begin
            ((:Cϕ || :G || :L), v)              => v
            (_, L::Union{Nothing,FuncOp,Real})  => L
            (_, L)                              => getindex(L,block)
        end
    end...)
end

function (ds::DataSet)(θ) 
    DS = typeof(ds)
    DS(map(fieldvalues(ds)) do v
        (v isa Union{ParamDependentOp,DataSet}) ? v(θ) : v
    end...)
end
(ds::DataSet)(;θ...) = ds((;θ...))


adapt_structure(to, ds::DS) where {DS <: DataSet} = DS(adapt(to, fieldvalues(ds))...)


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


@doc doc"""

    load_sim(;kwargs...)

The starting point for many typical sessions. Creates a `BaseDataSet`
object with some simulated data, returing the DataSet and simulated
truths, which can then be passed to other maximization / sampling
functions. E.g.:

```julia
@unpack f,ϕ,ds = load_sim(;
    θpix  = 2,
    Nside = 128,
    pol   = :P,
    T     = Float32
)
```

Keyword arguments: 

* `θpix` — Angular resolution, in arcmin. 
* `Nside` — Number of pixels in the map as an `(Ny,Nx)` tuple, or a
  single number for square maps. 
* `pol` — One of `:I`, `:P`, or `:IP` to select intensity,
  polarization, or both. 
* `T = Float32` — Precision, either `Float32` or `Float64`.
* `storage = Array` — Set to `CuArray` to use GPU.
* `Nbatch = 1` — Number of batches of data in this dataset.
* `μKarcminT = 3` — Noise level in temperature in μK-arcmin.
* `ℓknee = 100` — 1/f noise knee.
* `αknee = 3` — 1/f noise slope.
* `beamFWHM = 0` — Beam full-width-half-max in arcmin.
* `pixel_mask_kwargs = (;)` — NamedTuple of keyword arguments to
  pass to `make_mask` to create the pixel mask.
* `bandpass_mask = LowPass(3000)` — Operator which performs
  Fourier-space masking.
* `fiducial_θ = (;)` — NamedTuple of keyword arguments passed to
  `camb()` for the fiducial model.
* `seed = nothing` — Specific seed for the simulation.
* `L = LenseFlow` — Lensing operator.

Returns a named tuple of `(;f, f̃, ϕ, n, ds, Cℓ, proj)`.


"""
function load_sim(;
    
    # basic configuration
    θpix,
    Nside,
    pol,
    T = Float32,
    storage = Array,
    Nbatch = 1,
    
    # noise parameters, or set Cℓn or even Cn directly
    μKarcminT = 3,
    ℓknee = 100,
    αknee = 3,
    Cℓn = nothing,
    Cn = nothing,
    
    # beam parameters, or set B directly
    beamFWHM = 0,
    B = nothing, B̂ = nothing,
    
    # mask parameters, or set M directly
    pixel_mask_kwargs = nothing,
    bandpass_mask = LowPass(3000),
    M = nothing, M̂ = nothing,

    # theory
    Cℓ = nothing,
    fiducial_θ = NamedTuple(),
    rfid = nothing,
    
    rng = global_rng_for(storage),
    seed = nothing,
    D = nothing,
    G = nothing,
    Nϕ_fac = 2,
    L = LenseFlow,

)
    
    # projection
    Ny, Nx = Nside .* (1,1)
    proj = ProjLambert(;Ny, Nx, θpix, T, storage)

    # the biggest ℓ on the 2D fourier grid
    ℓmax = round(Int,ceil(√2*proj.nyquist)+1)
    
    # CMB Cℓs
    if (rfid != nothing)
        @warn "`rfid` will be removed in a future version. Use `fiducial_θ=(r=...,)` instead."
        fiducial_θ = merge(fiducial_θ,(r=rfid,))
    end
    Aϕ₀ = T(get(fiducial_θ, :Aϕ, 1))
    fiducial_θ = Base.structdiff(fiducial_θ, NamedTuple{(:Aϕ,)}) # remove Aϕ key if present
    if (Cℓ == nothing)
        Cℓ = camb(;fiducial_θ..., ℓmax=ℓmax)
    else
        if !isempty(fiducial_θ)
            error("Can't pass both `Cℓ` and `fiducial_θ` parameters which affect `Cℓ`, choose one or the other.")
        elseif maximum(Cℓ.total.TT.ℓ) < ℓmax
            error("ℓmax of `Cℓ` argument should be higher than $ℓmax for this configuration.")
        end
    end
    r₀ = T(Cℓ.params.r)
    
    # noise Cℓs (these are non-debeamed, hence beamFWHM=0 below; the beam comes in via the B operator)
    if (Cℓn == nothing)
        Cℓn = noiseCℓs(μKarcminT=μKarcminT, beamFWHM=0, ℓknee=ℓknee, αknee=αknee, ℓmax=ℓmax)
    end
    
    # some things which depend on whether we chose :I, :P, or :IP
    pol = Symbol(pol)
    ks,F,F̂,nF = @match pol begin
        :I  => ((:TT,),            FlatMap,    FlatFourier,    1)
        :P  => ((:EE,:BB),         FlatQUMap,  FlatEBFourier,  2)
        :IP => ((:TT,:EE,:BB,:TE), FlatIQUMap, FlatIEBFourier, 3)
        _   => throw(ArgumentError("`pol` should be one of :I, :P, or :IP"))
    end
    
    # covariances
    Cϕ₀ = Cℓ_to_Cov(:I,  proj, (Cℓ.total.ϕϕ))
    Cfs = Cℓ_to_Cov(pol, proj, (Cℓ.unlensed_scalar[k] for k in ks)...)
    Cft = Cℓ_to_Cov(pol, proj, (Cℓ.tensor[k]          for k in ks)...)
    Cf̃  = Cℓ_to_Cov(pol, proj, (Cℓ.total[k]           for k in ks)...)
    Cn̂  = Cℓ_to_Cov(pol, proj, (Cℓn[k]                for k in ks)...)
    if (Cn == nothing); Cn = Cn̂; end
    Cf = ParamDependentOp((;r=r₀,   _...)->(Cfs + (T(r)/r₀)*Cft))
    Cϕ = ParamDependentOp((;Aϕ=Aϕ₀, _...)->(T(Aϕ) * Cϕ₀))
    
    # data mask
    if (M == nothing)
        Mfourier = Cℓ_to_Cov(pol, proj, ((k==:TE ? 0 : 1) * bandpass_mask.diag.Wℓ for k in ks)...; units=1)
        if (pixel_mask_kwargs != nothing)
            Mpix = adapt(storage, Diagonal(F(repeated(T.(make_mask(Nside,θpix; pixel_mask_kwargs...).Ix),nF)..., proj)))
        else
            Mpix = I
        end
        M = Mfourier * Mpix
        if (M̂ == nothing)
            M̂ = Mfourier
        end
    else
        if (M̂ == nothing)
            M̂ = M
        end
    end
    if (M̂ isa DiagOp{<:BandPass})
        M̂ = Diagonal(M̂ * one(diag(Cf)))
    end
    
    # beam
    if (B == nothing)
        B = Cℓ_to_Cov(pol, proj, ((k==:TE ? 0 : 1) * sqrt(beamCℓs(beamFWHM=beamFWHM)) for k=ks)..., units=1)
    end
    if (B̂ == nothing)
        B̂ = B
    end
    
    # creating lensing operator cache
    Lϕ = alloc_cache(L(diag(Cϕ)),diag(Cf))

    # put everything in DataSet
    ds = BaseDataSet(;d=nothing, Cn, Cn̂, Cf, Cf̃, Cϕ, M, M̂, B, B̂, D, L=Lϕ)
    
    # simulate data
    @unpack ds,f,f̃,ϕ,n = resimulate(ds; rng, seed)


    # with the DataSet created, we now more conveniently create the mixing matrices D and G
    if (G == nothing)
        Nϕ = quadratic_estimate(ds).Nϕ / Nϕ_fac
        G₀ = sqrt(I + Nϕ * pinv(Cϕ()))
        ds.G = ParamDependentOp((;Aϕ=Aϕ₀, _...)->(pinv(G₀) * sqrt(I + 2 * Nϕ * pinv(Cϕ(Aϕ=Aϕ)))))
    end
    if (D == nothing)
        σ²len = T(deg2rad(5/60)^2)
        ds.D = ParamDependentOp(
            function (;r=r₀, _...)
                Cfr = Cf(;r=r)
                sqrt((Cfr + I*σ²len + 2*Cn̂) * pinv(Cfr))
            end,
        )
    end

    if Nbatch > 1
        ds.d *= batch(ones(Int,Nbatch))
        ds.L = alloc_cache(L(ϕ*batch(ones(Int,Nbatch))), ds.d)
    end
    
    return (;f, f̃, ϕ, n, ds, ds₀=ds(), Cℓ, proj)
    
end

@deprecate load_sim_dataset(args...; kwargs...) load_sim(args...; kwargs...)
