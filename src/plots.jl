begin
	@recipe function f(m::FlatMap)
	    seriestype   :=  :heatmap
	    seriescolor  --> :vik
	    aspect_ratio --> :equal
	    axis --> false
	    grid --> false
	    size --> (500, 450)
	    clims --> (-maximum(abs.(m)), maximum(abs.(m)))
	    m[:Ix]
	end
	@recipe f(m::FlatFourier) = Map(m)
	
	@recipe function f(Cℓ::InterpolatedCℓs, ℓrange::AbstractVector = 1:5000)
	    xguide --> "ℓ"
	    ℓrange, Cℓ.(ℓrange)
	end
end