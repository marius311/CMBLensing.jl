"""
    symplectic_integrate(x₀, p₀, Λ, U, δUδx N=50, ϵ=0.1, progress=true)
    

"""
function symplectic_integrate(x₀, p₀, Λ, U, δUδx; N=50, ϵ=0.1, progress=false)
    
    xᵢ, pᵢ = x₀, p₀
    δUδxᵢ = δUδx(xᵢ)

    @showprogress (progress ? 1 : Inf) for i=1:N
        xᵢ₊₁    = xᵢ - ϵ * (Λ \ (pᵢ - ϵ/2 * δUδxᵢ))
        δUδx₊₁  = δUδx(xᵢ₊₁)
        pᵢ₊₁    = pᵢ - ϵ/2 * (δUδx₊₁ + δUδxᵢ)
        xᵢ, pᵢ, δUδxᵢ = xᵢ₊₁, pᵢ₊₁, δUδx₊₁
    end

    H(x,p) = U(x) - p⋅(Λ\p)/2
    ΔH = H(xᵢ,pᵢ) - H(x₀,p₀)
    
    return  ΔH, xᵢ, pᵢ
    
end
