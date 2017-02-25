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

immutable PowerLens{F<:Field} <: LinearOp
    order::Int
    ∂xϕⁱ::Dict{Int,Union{F,Int}}
    ∂yϕⁱ::Dict{Int,Union{F,Int}}
end

Ł = LenseBasis

function PowerLens{F<:FlatS0}(ϕ::F; order=4)
    PowerLens{F}(order,(Dict(i=>(i==0?1:(Ł(∂*ϕ))^i) for i=0:order) for ∂=(∂x,∂y))...)
end

function *{F<:Field}(L::PowerLens, f::F)
    f̃ = f
    for n in 1:L.order, (a,b) in zip(0:n,n:-1:0)
        f̃ += L.∂xϕⁱ[a] * L.∂yϕⁱ[b] * Ł(∂x^a * ∂y^b * f) / factorial(a) / factorial(b)
    end
    f̃
end

""" Compute (df̃(f)/df)ᵀ ⋅ δf̃ """
function df̃dfᵀ{F<:Field}(L::PowerLens, δf::F)
    r = δf
    for n in 1:L.order, (a,b) in zip(0:n,n:-1:0)
        r += (-1)^n * ∂x^a * ∂y^b * (L.∂xϕⁱ[a] * L.∂yϕⁱ[b] * Ł(δf)) / factorial(a) / factorial(b)
    end
    r
end

""" Compute (df̃(f)/dϕ)ᵀ ⋅ δf̃ """
function df̃dϕᵀ{F<:Field}(L::PowerLens{F}, f::Field, δf̃::Field)
    r = zero(F)
    for n in 1:L.order, (a,b) in zip(0:n,n:-1:0)
        ∂ⁿfᵀ_δf̃ = Ł(∂x^a * ∂y^b * f)' * Ł(δf̃) / factorial(a) / factorial(b)
        r += -(  ((a==0) ? 0 : (∂x * (a * L.∂xϕⁱ[a-1] * L.∂yϕⁱ[b] * ∂ⁿfᵀ_δf̃)))
               + ((b==0) ? 0 : (∂y * (b * L.∂xϕⁱ[a] * L.∂yϕⁱ[b-1] * ∂ⁿfᵀ_δf̃))))
    end
    r
end
