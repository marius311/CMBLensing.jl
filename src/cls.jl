
function class(;lmax = 6000, 
                r = 0.2, ωb = 0.0224567, ωc=0.118489, τ = 0.128312, 
                Θs = 0.0104098, logA = 3.29056, nₛ = 0.968602,
                ϕscale = 1.0, ψscale = 0.0, r₀ = 100.0)

	cosmo = classy[:Class]()
	cosmo[:struct_cleanup]()
	cosmo[:empty]()
	params = Dict(
   		"output"        => "tCl, pCl, lCl",
   		"modes"         => "s,t",
   		"lensing"       => "yes",
		"l_max_scalars" => lmax + 500,
		"l_max_tensors" => lmax + 500,
      	"omega_b"       => ωb,
    	"omega_cdm"     => ωc,
      	"tau_reio"      => τ,
      	"100*theta_s"   => 100*Θs,
      	"ln10^{10}A_s"  => logA,
      	"n_s"           => nₛ,
		"r"             => r,
        #"k_pivot"      => 0.05,
		#"k_step_trans" => 0.1, # 0.01 for super high resolution
   		#"l_linstep"    => 10, # 1 for super high resolution
   		)
	cosmo[:set](params)
	cosmo[:compute]()
	C̃ℓ,Cℓ = (Dict([(k,v[2:end]) for (k,v) in cosmo[x](lmax)]) for x in (:lensed_cl,:raw_cl))
    α = 10^6 * cosmo[:T_cmb](); α² = α^2
	rtn = Dict{Symbol, Vector{Float64}}(
			:ℓ      => Cℓ["ell"],
			:ln_tt  => C̃ℓ["tt"] * α²,
			:ln_ee  => C̃ℓ["ee"] * α²,
			:ln_bb  => C̃ℓ["bb"] * α²,
			:ln_te  => C̃ℓ["te"] * α²,
			:ln_tϕ  => C̃ℓ["tp"] * α,
			:tt 	=> Cℓ["tt"] * α²,
			:ee 	=> Cℓ["ee"] * α²,
			:bb 	=> Cℓ["bb"] * α²,
			:te 	=> Cℓ["te"] * α²,
			:tϕ 	=> Cℓ["tp"] * α,
			:ϕϕ   => ϕscale.*Cℓ["pp"],
			:ϕψ   => 0.0.*Cℓ["pp"],
			:ψψ   => ψscale.*Cℓ["pp"],
			:bb0  => Cℓ["bb"] * α² * (r₀ / r),
		)
	return rtn
end

"""
Get the Cℓ as a 2D flat sky covariance
"""
Cℓ_2D(ℓ, Cℓ, r) = extrapolate(interpolate((ℓ,),Cℓ,Gridded(Linear())),0)[r]
