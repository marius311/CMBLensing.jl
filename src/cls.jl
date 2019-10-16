

### AbstractCℓs

# Data types to hold Cℓs that lets us do some convenient arithmetic with the Cℓs
# while not worrying about carrying around the ℓ labels, as well as automatically
# interpolating to any ℓ.

abstract type AbstractCℓs end

struct InterpolatedCℓs{I} <: AbstractCℓs
    etp :: I
    concrete :: Bool
end
InterpolatedCℓs(ℓ, Cℓ; concrete=true) = InterpolatedCℓs(LinearInterpolation(ℓ[(!isnan).(Cℓ)], filter(!isnan,Cℓ), extrapolation_bc=NaN), concrete)
getproperty(ic::InterpolatedCℓs, s::Symbol) = getproperty(ic,Val(s))
getproperty(ic::InterpolatedCℓs, ::Val{:ℓ}) = first(ic.etp.itp.knots)
getproperty(ic::InterpolatedCℓs, ::Val{:Cℓ}) = ic.etp.itp.coefs
getproperty(ic::InterpolatedCℓs, ::Val{s}) where {s} = getfield(ic,s)
propertynames(ic::IC) where {IC<:InterpolatedCℓs} = (:ℓ, :Cℓ, fieldnames(IC)...)
new_ℓs(ic1::InterpolatedCℓs, ic2::InterpolatedCℓs) = 
    sort!((!ic1.concrete && !ic2.concrete) ? union(ic1.ℓ,ic2.ℓ) : union((ic.ℓ for ic in (ic1,ic2) if ic.concrete)...))
for plot in (:plot, :loglog, :semilogx, :semilogy)
    @eval function ($plot)(ic::InterpolatedCℓs, args...; kwargs...)
		($plot)(ic.ℓ, ic.Cℓ, args...; kwargs...)
	end
	@eval function ($plot)(ic::InterpolatedCℓs{<:AbstractExtrapolation{<:Measurement}}, args...; kwargs...)
		errorbar(ic.ℓ, Measurements.value.(ic.Cℓ), Measurements.uncertainty.(ic.Cℓ), args...; marker=".", ls="", capsize=2, kwargs...)
		($plot) in [:loglog,:semilogx] && xscale("log")
		($plot) in [:loglog,:semilogy] && yscale("log")
	end
end

getindex(ic::InterpolatedCℓs, idx) = ic.etp(idx)
(ic::InterpolatedCℓs)(idx) = ic.etp(idx)


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

# algebra with InterpolatedCℓs and FuncCℓs
promote(ic::InterpolatedCℓs, fc::FuncCℓs) = (ic, InterpolatedCℓs(ic.ℓ, fc[ic.ℓ], concrete=fc.concrete))
promote(fc::FuncCℓs, ic::InterpolatedCℓs) = reverse(promote(ic,fc))
# todo: may make sense to hook into the broadcasting API and make much of
# the below more succinct:
for op in (:*, :/, :+, :-)
	@eval ($op)(ac1::AbstractCℓs, ac2::AbstractCℓs) = ($op)(promote(ac1,ac2)...)
    @eval ($op)(ic1::InterpolatedCℓs, ic2::InterpolatedCℓs) = (ℓ = new_ℓs(ic1,ic2); InterpolatedCℓs(ℓ, broadcast($op,ic1[ℓ],ic2[ℓ]), concrete=(ic1.concrete||ic2.concrete)))
    @eval ($op)(x::Real, ic::InterpolatedCℓs) = InterpolatedCℓs(ic.ℓ, broadcast($op,x,ic.Cℓ), concrete=ic.concrete)
    @eval ($op)(ic::InterpolatedCℓs, x::Real) = InterpolatedCℓs(ic.ℓ, broadcast($op,ic.Cℓ,x), concrete=ic.concrete)
	@eval ($op)(x::Real, fc::FuncCℓs) = FuncCℓs(ℓ -> ($op)(x,fc.f(ℓ)))
	@eval ($op)(fc::FuncCℓs, x::Real) = FuncCℓs(ℓ -> ($op)(fc.f(ℓ),x))
	@eval ($op)(fc1::FuncCℓs, fc2::FuncCℓs) = FuncCℓs(ℓ -> ($op)(fc1.f(ℓ),fc2.f(ℓ)))
	@eval ($op)(fc::FuncCℓs, v::AbstractArray) = broadcast($op, fc, v)
	@eval ($op)(v::AbstractArray, fc::FuncCℓs) = broadcast($op, v, fc)
end
for op in (:^, :sqrt)
	@eval ($op)(ic::InterpolatedCℓs, args...) = InterpolatedCℓs(ic.ℓ, broadcast($op, ic.Cℓ, args...), concrete=ic.concrete)
end



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
    InterpolatedCℓs(Cℓ1.ℓ, @. Cℓx.Cℓ/sqrt(Cℓ1.Cℓ*Cℓ2.Cℓ))
end


# used to powerlaw extrapolate Cℓs at very high-ℓ (usually ℓ>6000) just so we
# don't have deal with zeros / infinities there
function extrapolate_Cℓs(ℓout, ℓin, Cℓ)
	InterpolatedCℓs(ℓout, 
	    if all(Cℓ .> 0)
	        itp = LinearInterpolation(log.(ℓin), log.(Cℓ), extrapolation_bc = Interpolations.Line())
	        @. (exp(itp(log(ℓout))))
	    else
	        LinearInterpolation(ℓin, Cℓ, extrapolation_bc = 0).(ℓout)
	    end,
		concrete=false
	)
end

function smooth(Cℓ::InterpolatedCℓs; newℓs=minimum(Cℓ.ℓ):maximum(Cℓ.ℓ), xscale=:linear, yscale=:linear, smoothing=0.75)
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
    
    InterpolatedCℓs(newℓs, fy⁻¹(Loess.predict(loess(fx.(Cℓ.ℓ),fy.(Cℓ.Cℓ),span=smoothing),fx.(newℓs))), concrete=Cℓ.concrete)
end


### camb interface (via Python/pycamb)

function camb(;
    ℓmax = 6000, 
    r = 0.2, ωb = 0.0224567, ωc = 0.118489, τ = 0.055, Σmν = 0.06,
    Θs = 0.0104098, logA = 3.043, nₛ = 0.968602, nₜ = -r/8,
    Aϕ = 1,
    k_pivot = 0.002)
	
	params = Base.@locals
    
    camb = pyimport(:camb)
    ℓmax′ = min(5000,ℓmax)
    cp = camb.set_params(
        ombh2 = ωb,
        omch2 = ωc,
        tau = τ,
        mnu = Σmν,
        cosmomc_theta = Θs,
        H0 = nothing,
        ns = nₛ,
        nt = nₜ,
        As = exp(logA)*1e-10,
        pivot_scalar = k_pivot,
        pivot_tensor = k_pivot,
        lmax = ℓmax′,
        r = r
    )
    cp.max_l_tensor = ℓmax′
    cp.max_eta_k_tensor = 2ℓmax′
    cp.WantScalars = true
    cp.WantTensors = true
    cp.DoLensing = true
    
    res = camb.get_results(cp)
    
    
    ℓ  = collect(2:ℓmax -1)
    ℓ′ = collect(2:ℓmax′-1)
    α = (10^6*cp.TCMB)^2
    toCℓ′ = @. 1/(ℓ′*(ℓ′+1)/(2π))
	
    Cℓϕϕ = extrapolate_Cℓs(ℓ,ℓ′,Aϕ*2π*res.get_lens_potential_cls(ℓmax′)[3:ℓmax′,1]./ℓ′.^4)
	
    return (;
		map(["unlensed_scalar","lensed_scalar","tensor","unlensed_total","total"]) do k
			Symbol(k) => (;
				map(enumerate([:TT,:EE,:BB,:TE])) do (i,x)
					Symbol(x) => extrapolate_Cℓs(ℓ,ℓ′,res.get_cmb_power_spectra()[k][3:ℓmax′,i].*toCℓ′.*α)
				end..., 
				ϕϕ = Cℓϕϕ
			)
		end...,
		params = (;params...)
	)

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
			x => InterpolatedCℓs(ℓ, fill((x==:TT ? 1 : 2)*(deg2rad(μKarcminT/60))^2,ℓmax-1) ./ Bℓ .* Nℓ1f)
		end...,
		TE = InterpolatedCℓs(ℓ, zeros(ℓmax-1))
	)
end

@doc doc"""
	beamCℓs(;beamFWHM, ℓmax=8000)
	
Compute the beam power spectrum, often called $W_\ell$. A map should be
multiplied by the square root of this.
"""
function beamCℓs(;beamFWHM, ℓmax=8000)
	InterpolatedCℓs(2:ℓmax, @. exp(-(2:ℓmax)^2*deg2rad(beamFWHM/60)^2/(8*log(2))))
end
