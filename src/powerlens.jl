# 
# This file defines a lensing operator that works on any generic Field as long
# as ∂x and ∂y are defined for that field. 
# 
# The gradient of the lensed field with respect to the unlensed field and to ϕ
# can also be computed. 
# 
# This just does the standard taylor series expansion around ∇ϕ to arbitrary
# order (without a pixel permute step), but since the name "Taylens" is already
# taken, this is called "PowerLens"
# 

export PowerLens, δPδf̃_df̃dfϕ

struct PowerLens{F<:Field{<:Any,<:S0}} <: LenseOp
    order::Int
    ∂xϕⁱ::Dict{Int,Union{Int,F}}
    ∂yϕⁱ::Dict{Int,Union{Int,F}}
end

function PowerLens(ϕ; order=4)
    ∂xϕ, ∂yϕ = Ł(∂x*ϕ), Ł(∂y*ϕ)
    PowerLens{typeof(∂xϕ)}(order,(Dict(i=>(i==0?1:∂ϕ.^i) for i=0:order) for ∂ϕ=(∂xϕ,∂yϕ))...)
end

function *(L::PowerLens, f::Field)
    f̂ = Ð(f)
    f̃ = 1Ł(f)
    for n in 1:L.order, (a,b) in zip(0:n,n:-1:0)
        @. f̃ += L.∂xϕⁱ[a] * L.∂yϕⁱ[b] * $Ł(∂x^a * ∂y^b * f̂) / factorial(a) / factorial(b)
    end
    f̃
end


*(J::δf̃_δfϕᵀ{<:PowerLens},δPδf̃::Field) = (δf̃δfᵀ(J.L,δPδf̃), δf̃δϕᵀ(J.L,J.f,δPδf̃))

""" Compute (δf̃(f)/δf)ᵀ * δP/δf̃ """
function δf̃δfᵀ(L::PowerLens, δPδf̃::Field)
    ŁδPδf̃ = Ł(δPδf̃)
    r = 1Ð(δPδf̃)
    for n in 1:L.order, (a,b) in zip(0:n,n:-1:0)
        @. r += (-1)^n * ∂x^a * ∂y^b * $Ð(@. L.∂xϕⁱ[a] * L.∂yϕⁱ[b] * ŁδPδf̃) / factorial(a) / factorial(b)
    end
    r
end

""" Compute (δf̃(f)/δϕ)ᵀ * δP/δf̃ """
function δf̃δϕᵀ(L::PowerLens{F}, f::Field, δPδf̃::Field) where {F}
    ŁδPδf̃ = Ł(δPδf̃)
    Ðf = Ð(f)
    r = Ð(zero(F))
    ∂ⁿfᵀ_δPδf̃ = Ł(zero(F))
    for n in 1:L.order, (a,b) in zip(0:n,n:-1:0)
        ∂ⁿfᵀ_δPδf̃ .= (Ł(@. ∂x^a * ∂y^b * Ðf)' * ŁδPδf̃) ./ factorial(a) ./ factorial(b)
        @. r += -(  ((a==0) ? 0 : (∂x * $Ð(@. a * L.∂xϕⁱ[a-1] * L.∂yϕⁱ[b] * ∂ⁿfᵀ_δPδf̃)))
                  + ((b==0) ? 0 : (∂y * $Ð(@. b * L.∂xϕⁱ[a] * L.∂yϕⁱ[b-1] * ∂ⁿfᵀ_δPδf̃))))
    end
    r
end
