
# mixing matrix for mixed parametrization
D_mix(Cf::LinOp; rfid=0.1, σ²len=deg2rad(5/60)^2) =
     ParamDependentOp((;r=rfid, _...)->(nan2zero.(sqrt.((Diagonal(evaluate(Cf,r=rfid))+σ²len) ./ Diagonal(evaluate(Cf,r=r))))))

# Stores variables needed to construct the likelihood
@with_kw struct DataSet{Td,TCn,TCf,TCf̃,TCϕ,TCn̂,TB̂,TM,TB,TD,TG,TP}
    d  :: Td                # data
    Cn :: TCn               # noise covariance
    Cϕ :: TCϕ               # ϕ covariance
    Cf :: TCf               # unlensed field covariance
    Cf̃ :: TCf̃ = nothing     # lensed field covariance (not always needed)
    Cn̂ :: TCn̂ = Cn          # approximate noise covariance, diagonal in same basis as Cf
    M  :: TM  = 1           # user mask
    B  :: TB  = 1           # beam and instrumental transfer functions
    B̂  :: TB̂  = B           # approximate beam and instrumental transfer functions, diagonal in same basis as Cf
    D  :: TD  = IdentityOp  # mixing matrix for mixed parametrization
    G  :: TG  = IdentityOp  # reparametrization for ϕ
    P  :: TP  = 1           # pixelization operator to estimate field on higher res than data
end

function subblock(ds::DataSet, block)
    DataSet(map(collect(pairs(fields(ds)))) do (k,v)
        @match (k,v) begin
            ((:Cϕ || :G), v)                    => v
            (_, L::Union{Nothing,FuncOp,Real})  => L
            (_, L)                              => getproperty(L,block)
        end
    end...)
end

function (ds::DataSet)(;θ...)
    DataSet(map(fieldvalues(ds)) do v
        (v isa ParamDependentOp) ? v(;θ...) : v
    end...)
end

    
@doc doc"""
    resimulate(ds::DataSet; f=..., ϕ=...)
    
Resimulate the data in a given dataset, potentially at a fixed f and/or ϕ (both
are resimulated if not provided)
"""
function resimulate(ds::DataSet; f=simulate(ds.Cf), ϕ=simulate(ds.Cϕ), n=simulate(ds.Cn), f̃=LenseFlow(ϕ)*f)
    @unpack M,P,B = ds
    DataSet(ds, d = M*P*B*f̃ + n)
end





@doc doc"""
    load_sim_dataset
    
Create a `DataSet` object with some simulated data. 

"""
function load_sim_dataset(;
    θpix,
    θpix_data = θpix,
    Nside,
    use,
    T = Float32,
    μKarcminT = 3,
    ℓknee = 100,
    αknee = 3,
    ℓmax_data = 3000,
    beamFWHM = 0,
    rfid = 0.05,
    Cℓ = camb(r=rfid),
    Cℓn = nothing,
    Cn = nothing,
    seed = nothing,
    M = nothing,
    B = nothing, B̂ = nothing,
    D = nothing,
    G = nothing,
    ϕ=nothing, f=nothing, f̃=nothing, Bf̃=nothing, n=nothing, d=nothing, # override any of these simulated fields
    mask_kwargs = nothing,
    L = LenseFlow,
    ∂mode = fourier∂
    )
    
    # the biggest ℓ on the 2D fourier grid
    ℓmax = round(Int,ceil(√2*FFTgrid(T,Flat{θpix,Nside}).nyq))
    
    # Cℓs
    if (Cℓn == nothing)
        Cℓn = noisecls(μKarcminT, beamFWHM=0, ℓknee=ℓknee, αknee=αknee, ℓmax=ℓmax)
    end
    Cℓf, Cℓf̃ = Cℓ[:f], Cℓ[:f̃]
    
    # types which depend on whether T/E/B
    use = Symbol(use)
    if (use == :EB)
        @warn("switch to use=:P")
        use = :P
    elseif (use == :TEB)
        @warn("switch to use=:TP")
        use = :TP
    end
    SS,ks = Dict(:TP=>((S0,S2),(:TT,:EE,:BB,:TE)), :P=>((S2,),(:EE,:BB)), :T=>((S0,),(:TT,)))[use]
    F,F̂,nF = Dict(:TP=>(FlatIQUMap,FlatTEBFourier,3), :P=>(FlatS2QUMap,FlatS2EBFourier,2), :T=>(FlatS0Map,FlatS0Fourier,1))[use]
    
    # pixelization
    P = (θpix_data == θpix) ? 1 : FuncOp(
        op  = f -> ud_grade(f, θpix_data, deconv_pixwin=false, anti_aliasing=false),
        opᴴ = f -> ud_grade(f, θpix,      deconv_pixwin=false, anti_aliasing=false)
    )
    Pix      = Flat{θpix,Nside,∂mode}
    Pix_data = Flat{θpix_data,Nside÷(θpix_data÷θpix),∂mode}
    
    # covariances
    Cϕ₀            =  Cℓ_to_Cov(T,Pix,     S0,    Cℓf[:ϕϕ])
    Cfs,Cft,Cf̃,Cn̂  = (Cℓ_to_Cov(T,Pix,     SS..., (Cℓx[k] for k=ks)...) for Cℓx in (Cℓ[:fs],Cℓ[:ft],Cℓf̃,Cℓn))
    if (Cn == nothing)
        Cn         =  Cℓ_to_Cov(T,Pix_data,SS..., (Cℓn[k] for k=ks)...)
    end
    Cf = ParamDependentOp((;r=rfid, _...)->(@. Cfs + (r/rfid)*Cft))
    Cϕ = ParamDependentOp((;Aϕ=1,   _...)->(@. Aϕ*Cϕ₀))
    
    # data mask
    if (M == nothing) && (mask_kwargs != nothing)
        M = LowPass(ℓmax_data) * FullDiagOp(F{T,Pix_data}(repeated(T.(sptlike_mask(Nside÷(θpix_data÷θpix),θpix_data; mask_kwargs...)),nF)...))
    elseif (M == nothing)
        M = LowPass(ℓmax_data)
    end
    
    # beam
    if (B == nothing)
        B = let ℓ=0:ℓmax; Cℓ_to_Cov(T,Pix,SS..., (InterpolatedCℓs(ℓ, (k==:TE ? zero(ℓ) : @.(exp(-ℓ^2*deg2rad(beamFWHM/60)^2/(8*log(2))/2)))) for k=ks)...); end;
    end
    if (B̂ == nothing)
        B̂ = B
    end
    
    # mixing matrices
    if (D == nothing); D = D_mix(Cf,rfid=rfid); end
    if (G == nothing); G = IdentityOp; end
    
    # simulate data
    if (seed != nothing); seed!(seed); end
    if (ϕ  == nothing); ϕ  = simulate(Cϕ); end
    if (f  == nothing); f  = simulate(Cf); end
    if (n  == nothing); n  = simulate(Cn); end
    if (f̃  == nothing); f̃  = L(ϕ)*f;       end
    if (Bf̃ == nothing); Bf̃ = B*f̃;          end
    if (d  == nothing); d  = M*P*Bf̃ + n;   end
    
    # put everything in DataSet
    ds = DataSet(;(@ntpack d Cn Cn̂ Cf Cf̃ Cϕ M B B̂ D G P)...)
    
    return @ntpack f f̃ ϕ n ds ds₀=>ds() T P=>Pix Cℓ
    
end


###


function load_healpix_sim_dataset(;
    Nside,
    use,
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
    
    @assert use==:T 
    
    # Cℓs
    if (Cℓn == nothing)
        Cℓn = noisecls(μKarcminT, beamFWHM=0, ℓknee=ℓknee, αknee=αknee, ℓmax=ℓmax_ops)
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
    ds = DataSet(;(@ntpack d Cn Cf Cf̃ Cϕ M)...)

    
    return @ntpack f f̃ ϕ n ds ds₀=>ds()


end
