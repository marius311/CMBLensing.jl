
# Stores variables needed to construct the posterior
@kwdef struct DataSet{Td,TCn,TCf,TCf̃,TCϕ,TCn̂,TM,TM̂,TB,TB̂,TD,TG,TP,TL}
    d  :: Td                # data
    Cϕ :: TCϕ               # ϕ covariance
    Cf :: TCf               # unlensed field covariance
    Cf̃ :: TCf̃ = nothing     # lensed field covariance (not always needed)
    Cn :: TCn               # noise covariance
    Cn̂ :: TCn̂ = Cn          # approximate noise covariance, diagonal in same basis as Cf
    M  :: TM  = 1           # user mask
    M̂  :: TM̂  = M           # approximate user mask, diagonal in same basis as Cf
    B  :: TB  = 1           # beam and instrumental transfer functions
    B̂  :: TB̂  = B           # approximate beam and instrumental transfer functions, diagonal in same basis as Cf
    D  :: TD  = IdentityOp  # mixing matrix for mixed parametrization
    G  :: TG  = IdentityOp  # reparametrization for ϕ
    P  :: TP  = 1           # pixelization operator (if estimating field on higher res than data)
    L  :: TL  = alloc_cache(LenseFlow(similar(diag(Cϕ))),d) # a CachedLenseFlow which will be reused for memory
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
    resimulate(ds::DataSet; f=..., ϕ=...)
    
Resimulate the data in a given dataset, potentially at a fixed f and/or ϕ (both
are resimulated if not provided)
"""
function resimulate(ds::DataSet; f=simulate(ds.Cf), ϕ=simulate(ds.Cϕ), n=simulate(ds.Cn), f̃=ds.L(ϕ)*f, return_truths=false)
    @unpack M,P,B = ds
    @set! ds.d = M*P*B*f̃ + n
    return_truths ? @namedtuple(ds,f,ϕ,n,f̃) : ds
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
    rfid = 0.05,
    Cℓ = nothing,
    
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
    if Cℓ == nothing
        Cℓ = camb(r=rfid, ℓmax=ℓmax)
    end
    
    # noise Cℓs (these are non-debeamed, hence beamFWHM=0 below; the beam comes in via the B operator)
    if (Cℓn == nothing)
        Cℓn = noiseCℓs(μKarcminT=μKarcminT, beamFWHM=0, ℓknee=ℓknee, αknee=αknee, ℓmax=ℓmax)
    end
    
    # some things which depend on whether we chose :I, :P, or :IP
    pol = Symbol(pol)
    S,ks,F,F̂,nF = @match pol begin
        :I  => (S0,  (:TT,),            FlatMap,    FlatFourier,    1)
        :P  => (S2,  (:EE,:BB),         FlatQUMap,  FlatEBFourier,  2)
        :IP => (S02, (:TT,:EE,:BB,:TE), FlatIQUMap, FlatTEBFourier, 3)
        _   => throw(ArgumentError("`pol` should be one of :I, :P, or :IP"))
    end
    
    # pixelization
    Pix = Flat(Nside=Nside, θpix=θpix, ∂mode=∂mode)
    if (θpix_data == θpix)
        Pix_data = Pix
        P = IdentityOp
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
    Cf = ParamDependentOp((mem; r=rfid, _...)->(mem .= Cfs + T(r/rfid)*Cft), similar(Cfs))
    Cϕ = ParamDependentOp((mem; Aϕ=1  , _...)->(mem .= T(Aϕ) .* Cϕ₀), similar(Cϕ₀))
    
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
    
    # D mixing matrix
    if (D == nothing)
        σ²len = T(deg2rad(5/60)^2)
        D = ParamDependentOp(
            function (mem;r=rfid,_...)
                Cfr = Cf(mem,r=r)
                mem .= sqrt(Diagonal(diag(Cfr) .+ σ²len .+ 2*diag(Cn̂)) * pinv(Cfr))
            end,
            similar(Cf())
        )
    end
      
    # simulate data
    if (seed != nothing); seed!(seed); end
    if (ϕ  == nothing); ϕ  = simulate(Cϕ); end
    if (f  == nothing); f  = simulate(Cf); end
    if (n  == nothing); n  = simulate(Cn); end
    Lϕ = cache(L(ϕ),f)
    if (f̃  == nothing); f̃  = Lϕ*f;         end
    if (Bf̃ == nothing); Bf̃ = B*f̃;          end
    if (d  == nothing); d  = M*P*Bf̃ + n;   end
    
    # put everything in DataSet
    ds = DataSet(;@namedtuple(d, Cn, Cn̂, Cf, Cf̃, Cϕ, M, M̂, B, B̂, D, P, L=Lϕ)...)
    
    # with the DataSet created, we can now more conveniently call the quadratic
    # estimate to compute Nϕ if needed for the G mixing matrix
    if (G == nothing)
        Nϕ = quadratic_estimate(ds,(pol in (:P,:IP) ? :EB : :TT)).Nϕ / Nϕ_fac
        G₀ = @. nan2zero(sqrt(1 + 2/($Cϕ()/Nϕ)))
        G = ParamDependentOp((;Aϕ=1,_...)->(@. nan2zero(sqrt(1 + 2/(($(Cϕ(Aϕ=Aϕ))/Nϕ)))/G₀)))
    end
    @set! ds.G = G
   
    
    return adapt(storage, @namedtuple(f, f̃, ϕ, n, ds, ds₀=ds(), T, P=Pix, Cℓ, L))
    
end


###


function load_healpix_sim_dataset(;
    Nside,
    pol,
    gradient_cache,
    T = Float32,
    μKarcminT = 3,
    ℓknee = 100,
    αknee = 3,
    ℓmax_ops = 2Nside,
    ℓmax_data = 3000,
    beamFWHM = 0,
    rfid = 0.05,
    Cℓ = camb(r=rfid, ℓmax=ℓmax_ops),
    Cℓn = nothing,
    Cn = nothing,
    seed = nothing,
    M = nothing,
    B = nothing,
    D = nothing,
    G = nothing,
    ϕ = nothing,
    f = nothing,
    mask_kwargs = nothing,
    L = LenseFlow)
    
    @assert pol==:T 
    
    # Cℓs
    if (Cℓn == nothing)
        Cℓn = noiseCℓs(μKarcminT, beamFWHM=0, ℓknee=ℓknee, αknee=αknee, ℓmax=ℓmax_ops)
    end
    Cℓf, Cℓf̃ = Cℓ[:f], Cℓ[:f̃]

    Cf = IsotropicHarmonicCov(T.(nan2zero.(Cℓf[:TT][0:ℓmax_ops])), gradient_cache)
    Cf̃ = IsotropicHarmonicCov(T.(nan2zero.(Cℓf̃[:TT][0:ℓmax_ops])), gradient_cache)
    Cn = IsotropicHarmonicCov(T.(nan2zero.(Cℓn[:TT][0:ℓmax_ops])), gradient_cache)
    Cϕ = IsotropicHarmonicCov(T.(nan2zero.(Cℓf[:ϕϕ][0:ℓmax_ops])), gradient_cache)
    
    P=B=1 #for now
    
    if (seed != nothing); seed!(seed); end
    if (ϕ==nothing); ϕ = simulate(Cϕ); end
    if (f==nothing); f = simulate(Cf); end
    f̃ = L(ϕ)*f
    n = simulate(Cn)
    d = M*P*B*f̃ + n

    
    # put everything in DataSet
    ds = DataSet(;@namedtuple(d, Cn, Cf, Cf̃, Cϕ, M)...)

    
    return @namedtuple(f, f̃, ϕ, n, ds, ds₀=ds())


end
