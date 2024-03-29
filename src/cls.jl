

### AbstractCℓs

# Data types to hold Cℓs that lets us do some convenient arithmetic with the Cℓs
# while not worrying about carrying around the ℓ labels, as well as automatically
# interpolating to any ℓ.

abstract type AbstractCℓs end

struct Cℓs{TCℓ,Tℓ} <: AbstractCℓs
    ℓ :: Tℓ
    Cℓ :: TCℓ
    concrete :: Bool
    itp
end
Cℓs(Cℓ; ℓstart=1, kwargs...) = Cℓs(ℓstart:(ℓstart+length(Cℓ)-1),Cℓ; kwargs...)
function Cℓs(ℓ, Cℓ; concrete=true)
    mask = (!isnan).(Cℓ)
    if !all(mask)
        ℓ = ℓ[mask]
        Cℓ = Cℓ[mask]
    end
    if ℓ == ℓ[1]:ℓ[end]
        ℓ = ℓ[1]:ℓ[end]
    end
    itp = LinearInterpolation(ℓ, identity.(Cℓ), extrapolation_bc=NaN)
    Cℓs(ℓ, Cℓ, concrete, itp)
end
function new_ℓs(ic1::Cℓs, ic2::Cℓs)
    sort!((!ic1.concrete && !ic2.concrete) ? union(ic1.ℓ,ic2.ℓ) : union((ic.ℓ for ic in (ic1,ic2) if ic.concrete)...))
end
getindex(ic::Cℓs, idx) = ic.itp.(idx)
getindex(ic::Cℓs, ::Colon) = ic.Cℓ
(ic::Cℓs)(idx) = ic.itp.(idx)
# custom JLD2 serialization bc the default for LinearInterpolation
# creates a closure which isnt future-proof when deserializing
function JLD2.writeas(::Type{<:Cℓs{TCℓ,Tℓ}}) where {TCℓ,Tℓ}
    Tuple{Val{Cℓs},NamedTuple{(:ℓ,:Cℓ,:concrete),Tuple{Tℓ,TCℓ,Bool}}}
end
function JLD2.wconvert(::Type{<:Tuple{Val{Cℓs},NamedTuple}}, ic::Cℓs)
    (Val(Cℓs), (;ic.ℓ, ic.Cℓ, ic.concrete))
end
function JLD2.rconvert(::Type{<:Cℓs}, (_,s)::Tuple{Val{Cℓs},NamedTuple})
    Cℓs(s.ℓ, s.Cℓ; s.concrete)
end
hash(ic::Cℓs, h::UInt) = foldr(hash, (ic.ℓ, ic.Cℓ, ic.concrete), init=h)


struct FuncCℓs{F<:Function} <: AbstractCℓs
    f :: F
    concrete :: Bool
    FuncCℓs(f::F) where {F<:Function} = new{F}(f,false)
end
getindex(fc::FuncCℓs, idx) = fc.f.(idx)
broadcastable(fc::FuncCℓs) = Ref(fc)
ℓ² = FuncCℓs(ℓ -> ℓ^2)
ℓ⁴ = FuncCℓs(ℓ -> ℓ^4)
toDℓ = FuncCℓs(ℓ -> ℓ*(ℓ+1)/2π)
toCℓ = FuncCℓs(ℓ -> 2π/(ℓ*(ℓ+1)))

# algebra with Cℓs and FuncCℓs
promote(ic::Cℓs, fc::FuncCℓs) = (ic, Cℓs(ic.ℓ, fc[ic.ℓ], concrete=fc.concrete))
promote(fc::FuncCℓs, ic::Cℓs) = reverse(promote(ic,fc))
# todo: may make sense to hook into the broadcasting API and make much of
# the below more succinct:
for op in (:*, :/, :+, :-, :±)
    @eval ($op)(ac1::AbstractCℓs, ac2::AbstractCℓs) = ($op)(promote(ac1,ac2)...)
    @eval ($op)(ic1::Cℓs, ic2::Cℓs) = (ℓ = new_ℓs(ic1,ic2); Cℓs(ℓ, broadcast($op,ic1[ℓ],ic2[ℓ]), concrete=(ic1.concrete||ic2.concrete)))
    @eval ($op)(x::Real, ic::Cℓs) = Cℓs(ic.ℓ, broadcast($op,x,ic.Cℓ), concrete=ic.concrete)
    @eval ($op)(ic::Cℓs, x::Real) = Cℓs(ic.ℓ, broadcast($op,ic.Cℓ,x), concrete=ic.concrete)
    @eval ($op)(x::Real, fc::FuncCℓs) = FuncCℓs(ℓ -> ($op)(x,fc.f(ℓ)))
    @eval ($op)(fc::FuncCℓs, x::Real) = FuncCℓs(ℓ -> ($op)(fc.f(ℓ),x))
    @eval ($op)(fc1::FuncCℓs, fc2::FuncCℓs) = FuncCℓs(ℓ -> ($op)(fc1.f(ℓ),fc2.f(ℓ)))
    @eval ($op)(fc::FuncCℓs, v::AbstractArray) = broadcast($op, fc, v)
    @eval ($op)(v::AbstractArray, fc::FuncCℓs) = broadcast($op, v, fc)
end
for op in (:^, :sqrt, :abs)
    @eval ($op)(ic::Cℓs, args...) = Cℓs(ic.ℓ, broadcast($op, ic.Cℓ, args...), concrete=ic.concrete)
end
std(x::Vector{<:Cℓs}) = sqrt(abs(mean(x.^2) - mean(x)^2))
shiftℓ(Δℓ, Cℓ; factor=false) = Cℓs(factor ? Cℓ.ℓ .* Δℓ : Cℓ.ℓ .+ Δℓ, Cℓ.Cℓ)


function get_Cℓ end
get_Dℓ(args...; kwargs...) = ℓ² * get_Cℓ(args...; kwargs...) / 2π
get_ℓ⁴Cℓ(args...; kwargs...) = ℓ⁴ * get_Cℓ(args...; kwargs...)
function get_ρℓ(f; which, kwargs...)
    a,b = Symbol.(split(string(which),""))
    get_ρℓ(f[a], f[b]; kwargs...)
end
function get_ρℓ(f1,f2; kwargs...)
    Cℓ1 = get_Cℓ(f1; kwargs...)
    Cℓ2 = get_Cℓ(f2; kwargs...)
    Cℓx = get_Cℓ(f1,f2; kwargs...)
    Cℓs(Cℓ1.ℓ, @. Cℓx.Cℓ/sqrt(Cℓ1.Cℓ*Cℓ2.Cℓ))
end


# used to powerlaw extrapolate Cℓs at very high-ℓ (usually ℓ>6000) just so we
# don't have deal with zeros / infinities there
function extrapolate_Cℓs(ℓ_out, ℓ_in, Cℓ_in::AbstractVector{T}) where {T}
    if all(Cℓ_in .> 0)
        itp = LinearInterpolation(log.(ℓ_in), log.(Cℓ_in), extrapolation_bc = :line)
        Cℓ_out = @. T.(exp(itp(log(ℓ_out))))
    else
        # probably a cross-spectra, don't do anything fancy here
        Cℓ_out = T.(LinearInterpolation(ℓ_in, Cℓ_in, extrapolation_bc = 0).(ℓ_out))
    end
    Cℓs(ℓ_out, Cℓ_out, concrete=false)
end

function smooth(Cℓ::Cℓs; newℓs=minimum(Cℓ.ℓ):maximum(Cℓ.ℓ), xscale=:linear, yscale=:linear, smoothing=0.75)
    (fx, fx⁻¹) = @match xscale begin
        :linear => (identity, identity)
        :log    => (log,      exp)
        _ => throw(ArgumentError("'xscale' should be :log or :linear"))
    end
    (fy, fy⁻¹) = @match yscale begin
        :linear => (identity, identity)
        :log    => (log,      exp)
        _ => throw(ArgumentError("'xscale' should be :log or :linear"))
    end
    
    Cℓs(newℓs, fy⁻¹.(Loess.predict(loess(fx.(Cℓ.ℓ),fy.(Cℓ.Cℓ),span=smoothing),fx.(newℓs))), concrete=Cℓ.concrete)
end


### camb interface (via Python/pycamb)

const _default_Cℓs_path = joinpath(@__DIR__, "../dat/default_camb_Cls.jld2")
const _default_Cℓs_params = isfile(_default_Cℓs_path) ? Dict(pairs(load(_default_Cℓs_path, "params"))) : Dict()
@memoize _default_Cℓs() = load(_default_Cℓs_path, "Cℓ")

@memoize function camb(;
    ℓmax = 6000, 
    r = 0.2, ωb = 0.0224567, ωc = 0.118489, τ = 0.055, Σmν = 0.06,
    θs = 0.0104098, H0 = nothing, logA = 3.043, nₛ = 0.968602, nₜ = -r/8,
    AL = 1,
    k_pivot = 0.002
)
    
    params = Base.@locals
    pop!.((params,), (:cache, :value), nothing) # why is :value ending up in here???
    if (
        params[:ℓmax] <= get(_default_Cℓs_params, :ℓmax, 0) &&
        keys(params) == keys(_default_Cℓs_params) &&
        all(params[k] == _default_Cℓs_params[k] for k in keys(params) if k != :ℓmax)
    )
        return _default_Cℓs()
    end

    camb = pyimport("camb")
    
    ℓmax′ = min(5000, ℓmax)
    cp = camb.set_params(
        ombh2 = ωb,
        omch2 = ωc,
        tau = τ,
        mnu = Σmν,
        cosmomc_theta = θs,
        H0 = H0,
        ns = nₛ,
        nt = nₜ,
        As = exp(logA) * 1e-10,
        pivot_scalar = k_pivot,
        pivot_tensor = k_pivot,
        lmax = ℓmax′,
        r = r,
        Alens = AL,
    )
    cp.max_l_tensor = 2ℓmax′
    cp.max_eta_k_tensor = 4ℓmax′
    cp.WantScalars = true
    cp.WantTensors = true
    cp.DoLensing = true
    cp.set_nonlinear_lensing(true)
    
    res = camb.get_results(cp)
    
    ℓ  = collect(2:ℓmax -1)
    ℓ′ = collect(2:ℓmax′-1)
    α = (10^6 * cp.TCMB)^2
    toCℓ′ = @. 1/(ℓ′*(ℓ′+1)/(2π))

    Cℓϕϕ = extrapolate_Cℓs(ℓ, ℓ′, 2π * PyArray(res.get_lens_potential_cls(ℓmax′))[3:ℓmax′,1] ./ ℓ′.^4)

    return (;
        map(["unlensed_scalar","lensed_scalar","tensor","unlensed_total","total"]) do k
            Symbol(k) => (;
                map(enumerate([:TT,:EE,:BB,:TE])) do (i,x)
                    Symbol(x) => extrapolate_Cℓs(ℓ, ℓ′, PyArray(α * res.get_cmb_power_spectra()[k])[3:ℓmax′,i] .* toCℓ′)
                end..., 
                ϕϕ = Cℓϕϕ
            )
        end...,
        params = (;params...)
    )

end


@doc """
    load_camb_Cℓs(;path_prefix, custom_tensor_params=nothing, 
        unlensed_scalar_postfix, unlensed_tensor_postfix, lensed_scalar_postfix, lenspotential_postfix)
    
Load some Cℓs from CAMB files. 

`path_prefix` specifies the prefix for the files, which are then expected to
have the normal CAMB postfixes: `scalCls.dat`, `tensCls.dat`, `lensedCls.dat`,
`lenspotentialCls.dat`, unless otherwise specified via the other keyword
arguments. `custom_tensor_params` can be used to call CAMB directly for the
`unlensed_tensors`, rather than reading them from a file (since alot of times this file
doesn't get saved). The value should be a Dict/NamedTuple which will be passed
to a call to `camb`, e.g. `custom_tensor_params=(r=0,)` for zero
tensors. 

"""
function load_camb_Cℓs(;
    path_prefix,
    ℓmax = nothing,
    custom_tensor_params = nothing,
    unlensed_scalar_postfix = "scalCls.dat",
    unlensed_tensor_postfix = "tensCls.dat",
    lensed_scalar_postfix   = "lensedCls.dat",
    lenspotential_postfix   = "lenspotentialCls.dat")
    
    unlensed_scalar_filename = path_prefix*unlensed_scalar_postfix
    unlensed_tensor_filename = path_prefix*unlensed_tensor_postfix
    lensed_scalar_filename   = path_prefix*lensed_scalar_postfix
    lenspotential_filename   = path_prefix*lenspotential_postfix
    
    _extrapolateCℓs(ℓ,Cℓ) = ℓmax == nothing ? Cℓs(ℓ,Cℓ,concrete=false) : extrapolate_Cℓs(2:ℓmax,ℓ,Cℓ)
    
    ℓ,Cℓϕϕ = collect.(eachcol(readdlm(lenspotential_filename,skipstart=1)[1:end,[1,6]]))
    @. Cℓϕϕ /= (ℓ*(ℓ+1))^2/2π
    Cℓϕϕ = _extrapolateCℓs(ℓ, Cℓϕϕ)
    
    unlensed_scalar = Dict([:ℓ,:TT,:EE,:TE,:ϕϕ] .=> collect.(eachcol(readdlm(unlensed_scalar_filename,skipstart=1)[1:end,1:5])))
    ℓ = pop!(unlensed_scalar,:ℓ)
    for x in [:TT,:EE,:TE]
        @. unlensed_scalar[x] /= ℓ*(ℓ+1)/(2π)
    end
    unlensed_scalar[:BB] = 0ℓ
    unlensed_scalar = (;(k=>_extrapolateCℓs(ℓ,Cℓ) for (k,Cℓ) in unlensed_scalar)...)


    lensed_scalar = Dict([:ℓ,:TT,:EE,:BB,:TE] .=> collect.(eachcol(readdlm(lensed_scalar_filename,skipstart=1)[1:end,1:5])))
    ℓ = pop!(lensed_scalar,:ℓ)
    for x in [:TT,:EE,:BB,:TE]
        @. lensed_scalar[x] /= ℓ*(ℓ+1)/(2π)
    end
    lensed_scalar = (;(k=>_extrapolateCℓs(ℓ,Cℓ) for (k,Cℓ) in lensed_scalar)...)

    if custom_tensor_params != nothing
        if ℓmax != nothing
            custom_tensor_params = merge(custom_tensor_params, (ℓmax=ℓmax,))
        end
        tensor = camb(;custom_tensor_params...).tensor
        params = custom_tensor_params
    else
        tensor = Dict([:ℓ,:TT,:EE,:BB,:TE] .=> collect.(eachcol(readdlm(unlensed_tensor_filename,skipstart=1)[2:end,1:5])))
        ℓ = pop!(tensor,:ℓ)
        for x in [:TT,:EE,:BB,:TE]
            @. tensor[x] /= ℓ*(ℓ+1)/(2π)
        end
        tensor = (;(k=>_extrapolateCℓs(ℓ,Cℓ) for (k,Cℓ) in tensor)...)
        params = (;)
    end
    
    unlensed_total = (;(k=>unlensed_scalar[k]+tensor[k] for k in [:TT,:EE,:BB,:TE])..., ϕϕ=Cℓϕϕ)
    total          = (;(k=>lensed_scalar[k]+tensor[k]   for k in [:TT,:EE,:BB,:TE])..., ϕϕ=Cℓϕϕ)
    
    (;unlensed_scalar,tensor,lensed_scalar,unlensed_total,total,params)
    
end


### noise

@doc doc"""
    noiseCℓs(;μKarcminT, beamFWHM=0, ℓmax=8000, ℓknee=100, αknee=3)
    
Compute the (:TT,:EE,:BB,:TE) noise power spectra given white noise + 1/f.
Polarization noise is scaled by $\sqrt{2}$ relative to `μKarcminT`. `beamFWHM` is
in arcmin.
"""
function noiseCℓs(;μKarcminT, beamFWHM=0, ℓmax=8000, ℓknee=100, αknee=3)
    ℓ = 2:ℓmax
    Bℓ = beamCℓs(beamFWHM=beamFWHM, ℓmax=ℓmax)[ℓ]
    Nℓ1f = @. 1 + (ℓknee/ℓ)^αknee

    return (;
        map([:TT,:EE,:BB]) do x
            x => Cℓs(ℓ, fill((x==:TT ? 1 : 2)*(deg2rad(μKarcminT/60))^2,ℓmax-1) ./ Bℓ .* Nℓ1f)
        end...,
        TE = Cℓs(ℓ, zeros(ℓmax-1))
    )
end

@doc doc"""
    beamCℓs(;beamFWHM, ℓmax=8000)
    
Compute the beam power spectrum, often called $W_\ell$. A map should be
multiplied by the square root of this.
"""
function beamCℓs(;beamFWHM, ℓmax=8000)
    Cℓs(2:ℓmax, @. exp(-(2:ℓmax)^2*deg2rad(beamFWHM/60)^2/(8*log(2))))
end
