

abstract type DataSet{DS} end

getproperty(ds::DS, k::Symbol) where {DS<:DataSet{<:DataSet}} = 
    hasfield(DS, k) ? getfield(ds, k) : getproperty(getfield(ds, :_super), k)
setproperty!(ds::DS, k::Symbol, v) where {DS<:DataSet{<:DataSet}} = 
    hasfield(DS, k) ? setfield!(ds, k, v) : setproperty!(getfield(ds, :_super), k, v)
propertynames(ds::DS) where {DS′<:DataSet, DS<:DataSet{DS′}} = 
    union(fieldnames(DS), fieldnames(DS′))

function new_dataset(::Type{DS}; kwargs...) where {DS′<:DataSet, DS<:DataSet{DS′}}
    kw  = filter(((k,_),)->  k in fieldnames(DS),  kwargs)
    kw′ = filter(((k,_),)->!(k in fieldnames(DS)), kwargs)
    DS(_super=DS′(;kw′...); kw...)
end

copy(ds::DS) where {DS<:DataSet} = 
    DS(((k==:_super ? copy(v) : v) for (k,v) in pairs(fields(ds)))...)

# needed until fix to https://github.com/FluxML/Zygote.jl/issues/685
Zygote.grad_mut(ds::DataSet) = Ref{Any}((;(propertynames(ds) .=> nothing)...))


# Stores variables needed to construct the posterior
@kwdef mutable struct BaseDataSet <: DataSet{Nothing}
    d                # data
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

function (ds::DataSet)(θ::NamedTuple) 
    DS = typeof(ds)
    DS(map(fieldvalues(ds)) do v
        (v isa Union{ParamDependentOp,DataSet}) ? v(θ) : v
    end...)
end
(ds::DataSet)(;θ...) = ds((;θ...))

function check_hat_operators(ds::DataSet)
    @unpack B̂, M̂, Cn̂, Cf = ds()
    @assert(all([(L isa Scalar) || (basis(diag(L))==basis(diag(Cf))) for L in [B̂,M̂,Cn̂]]),
            "B̂, M̂, Cn̂ are used for preconditioning and should be scalars or in the same basis as Cf")
end

adapt_structure(to, ds::DS) where {DS <: DataSet} = DS(adapt(to, fieldvalues(ds))...)


@doc doc"""
    resimulate(ds::DataSet; [f, ϕ, n])
    
Make a new DataSet with the data replaced by a simulation, potentially given a
fixed f, ϕ, or n, if any are provided. 

Returns a named tuple of `(ds, f, ϕ, n, f̃)`
"""
resimulate(ds::DataSet; kwargs...) = resimulate!(copy(ds); kwargs...)

@doc doc"""
    resimulate!(ds::DataSet; [f, ϕ, n])
    
Replace the data in this DataSet in-place with a simulation, potentially given a
fixed f, ϕ, or n, if any are provided. 
    
Returns a named tuple of `(ds, f, ϕ, n, f̃)`
"""
function resimulate!(
    ds::DataSet; 
    f=nothing, ϕ=nothing, n=nothing, f̃=nothing,
    rng=global_rng_for(ds.d), seed=nothing
)

    @unpack M,P,B,L = ds
    
    if (f̃ == nothing)
        if (ϕ == nothing)
            ϕ = simulate(ds.Cϕ, rng=rng, seed=seed)
        end
        if (f == nothing)
            f = simulate(ds.Cf, rng=rng, seed=(seed==nothing ? nothing : seed+1))
        end
        f̃ = L(ϕ)*f
    else
        f = ϕ = nothing
    end
    if (n == nothing)
        n = simulate(ds.Cn, rng=rng, seed=(seed==nothing ? nothing : seed+2))
    end

    ds.d = d = M*P*B*f̃ + n
    
    @namedtuple(ds,f,ϕ,n,f̃,d)
    
end


@doc doc"""
    load_sim
    
Create a `BaseDataSet` object with some simulated data, returing the DataSet
and simulated truths. E.g.

```julia
@unpack f,ϕ,ds = load_sim(;
    θpix  = 2,
    Nside = 128,
    pol   = :I,
    T     = Float32
);
```

"""
function load_sim(;
    
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
    ∂mode = fourier∂

)
    

    # the biggest ℓ on the 2D fourier grid
    ℓmax = round(Int,ceil(√2*fieldinfo(Flat(θpix=θpix,Nside=Nside)).nyq)+1)
    
    # CMB Cℓs
    if (rfid != nothing)
        @warn "`rfid` will be removed in a future version. Use `fiducial_θ=(r=...,)` instead."
        fiducial_θ = merge(fiducial_θ,(r=rfid,))
    end
    Aϕ₀ = get(fiducial_θ, :Aϕ, 1)
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
        P = 1
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
    Cϕ = ParamDependentOp((;Aϕ=Aϕ₀, _...)->(T(Aϕ) * Cϕ₀))
    
    # data mask
    if (M == nothing)
        Mfourier = adapt(storage, Cℓ_to_Cov(Pix_data, T, S, ((k==:TE ? 0 : 1) * bandpass_mask.diag.Wℓ for k in ks)...; units=1))
        if (pixel_mask_kwargs != nothing)
            Mpix = adapt(storage, Diagonal(F{Pix_data}(repeated(T.(make_mask(Nside÷(θpix_data÷θpix),θpix_data; pixel_mask_kwargs...).Ix),nF)...)))
        else
            Mpix = 1
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
        B = adapt(storage, Cℓ_to_Cov(Pix, T, S, ((k==:TE ? 0 : 1) * sqrt(beamCℓs(beamFWHM=beamFWHM)) for k=ks)..., units=1))
    end
    if (B̂ == nothing)
        B̂ = B
    end
    
    # creating lensing operator cache
    Lϕ = alloc_cache(L(diag(Cϕ)),diag(Cf))

    # put everything in DataSet
    ds = BaseDataSet(;@namedtuple(d=nothing, Cn, Cn̂, Cf, Cf̃, Cϕ, M, M̂, B, B̂, D, P, L=Lϕ)...)
    
    # simulate data
    @unpack ds,f,f̃,ϕ,n = resimulate(ds, rng=rng, seed=seed)


    # with the DataSet created, we now more conveniently create the mixing matrices D and G
    if (G == nothing)
        Nϕ = quadratic_estimate(ds,(pol in (:P,:IP) ? :EB : :TT)).Nϕ / Nϕ_fac
        G₀ = @. nan2zero(sqrt(1 + 2/($Cϕ()/Nϕ)))
        ds.G = ParamDependentOp((;Aϕ=Aϕ₀, _...)->(pinv(G₀) * sqrt(I + 2 * Nϕ * pinv(Cϕ(Aϕ=Aϕ)))))
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

@deprecate load_sim_dataset(args...; kwargs...) load_sim(args...; kwargs...)
