export class, noisecls, camb



function camb(;lmax = 6000, 
                r = 0.2, ωb = 0.0224567, ωc=0.118489, τ = 0.055, 
                Θs = 0.0104098, logA = 3.043, nₛ = 0.968602, nₜ = -r/8,
                k_pivot = 0.002)

    camb = pyimport(:camb)
    cp = camb[:set_params](
        ombh2 = ωb,
        omch2 = ωc,
        tau = τ,
        cosmomc_theta = Θs,
        H0 = nothing,
        ns = nₛ,
        nt = nₜ,
        As = exp(logA)*1e-10,
        pivot_scalar = k_pivot,
        pivot_tensor = k_pivot,
        lmax = lmax,
        r = r
    )
    cp[:max_l_tensor] = lmax
    cp[:max_eta_k_tensor] = 2lmax
    cp[:WantScalars] = true
    cp[:WantTensors] = true
    cp[:DoLensing] = true
    
    res = camb[:get_results](cp)
    
    
    ℓ = collect(1:lmax-1)
    α = (10^6*cp[:TCMB])^2
    toCℓ = @. 1/(ℓ*(ℓ+1)/(2π))
    Cℓϕ = Dict{Symbol,Vector{Float64}}(:ℓ=>ℓ, :ϕϕ=>2π*res[:get_lens_potential_cls](lmax)[2:lmax,1]./ℓ.^4)
    Cℓs = Dict(k=>merge(Cℓϕ,Dict(x=>res[:get_cmb_power_spectra]()[v][2:lmax,i].*toCℓ.*α 
                                 for (i,x) in enumerate([:TT,:EE,:BB,:TE])))
           for (k,v) in Dict(:fs=>"unlensed_scalar",:f̃s=>"lensed_scalar",:ft=>"tensor",:f=>"unlensed_total",:f̃=>"total"))
               
end

function class(;lmax = 8000, 
                r = 0.2, ωb = 0.0224567, ωc=0.118489, τ = 0.055, 
                Θs = 0.0104098, logA = 3.043, nₛ = 0.968602,
                k_pivot = 0.002, modes = "s,t")


    classy = pyimport("classy")
	cosmo = classy[:Class]()
	cosmo[:struct_cleanup]()
	cosmo[:empty]()
	params = Dict(
       		"output"        => contains(modes,"s") ? "tCl, pCl, lCl" : "tCl, pCl",
       		"modes"         => modes,
       		"lensing"       => contains(modes,"s") ? "yes" : "no",
          	"omega_b"       => ωb,
        	"omega_cdm"     => ωc,
          	"tau_reio"      => τ,
          	"100*theta_s"   => 100*Θs,
            "k_pivot"       => k_pivot
	)
    if contains(modes,"s") 
        merge!(params,Dict(
          	"ln10^{10}A_s"  => logA, # submit CLASS Issue for this
            "l_max_scalars" => lmax+500,
          	"n_s"           => nₛ
        ))
    end
    if contains(modes,"t")
        merge!(params,Dict(
            "l_max_tensors" => lmax+500,
            "r" => r
        ))
    end
	cosmo[:set](params)
	cosmo[:compute]()
    
    α = 10^6 * cosmo[:T_cmb]()
    tCl = Dict(k=>v[2:end] for (k,v) in cosmo[:raw_cl](lmax))
	Cℓ = Dict{Symbol,Vector{Float64}}(
            :ℓ  => tCl["ell"],
            :TT => tCl["tt"] * α^2,
            :EE => tCl["ee"] * α^2,
            :BB => tCl["bb"] * α^2,
            :TE => tCl["te"] * α^2,
            :Tϕ => tCl["tp"] * α,
            :ϕϕ => tCl["pp"])
    if contains(modes,"s")
        lCl = Dict(k=>v[2:end] for (k,v) in cosmo[:lensed_cl](lmax))
        C̃ℓ = Dict{Symbol,Vector{Float64}}(
    			:ℓ  => lCl["ell"],
    			:TT => lCl["tt"] * α^2,
    			:EE => lCl["ee"] * α^2,
    			:BB => lCl["bb"] * α^2,
    			:TE => lCl["te"] * α^2,
    			:Tϕ => tCl["tp"] * α,
    			:ϕϕ => tCl["pp"])
        Dict(:f=>Cℓ,:f̃=>C̃ℓ)
    else
        Dict(:ft=>Cℓ)
    end
end





"""
* `μKarcminT`: temperature noise in μK-arcmin
* `beamFWHM`: beam-FWHM in arcmin
"""
function noisecls(μKarcminT;beamFWHM=6,ℓmax=8000,ℓknee=100,αknee=3)
    ℓ = 1:ℓmax
    Bℓ = @. exp(ℓ^2*deg2rad(beamFWHM/60)^2/(8*log(2)))
    Nℓ1f = @. 1 + (ℓknee/ℓ)^αknee
    Cℓs = Dict{Symbol,Any}(:ℓ=>ℓ)
    for x in [:TT,:EE,:BB]
        Cℓs[x]=fill((x==:TT?1:2)*(deg2rad(μKarcminT/60))^2,ℓmax) .* Bℓ .* Nℓ1f
    end
    Cℓs[:TE]=zeros(ℓmax)
    Cℓs
end
