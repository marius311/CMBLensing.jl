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
    ∂xⁱϕ::Dict{Int,Union{F,Int}}
    ∂yⁱϕ::Dict{Int,Union{F,Int}}
end

Ł = LenseBasis

function PowerLens{F<:FlatS0}(ϕ::F; order=4)
    PowerLens{F}(order,(Dict(i=>(i==0?1:Ł(∂*ϕ)^i) for i=0:order) for ∂=(∂x,∂y))...)
end

function *{F<:Field}(L::PowerLens, f::F)
    f̃ = f
    for n in 1:L.order, (a,b) in zip(0:n,n:-1:0)
        f̃ += L.∂xⁱϕ[a] * L.∂yⁱϕ[b] * Ł(∂x^a * ∂y^b * f) / factorial(a*b)
    end
    f̃
end

""" Compute df̃(f)/dfᵀ ⋅ δf """
function df̃dfᵀ{F<:Field}(L::PowerLens, δf::F)
    r = δf
    for n in 1:L.order, (a,b) in zip(0:n,n:-1:0)
        r += (-1)^n * ∂x^a * ∂y^b * (L.∂xⁱϕ[a] * L.∂yⁱϕ[b] * Ł(δf)) / factorial(a*b)
    end
    r
end

""" Compute df̃(f)/dϕᵀ ⋅ δϕ """
function df̃dϕᵀ{F<:Field}(L::PowerLens{F}, f::Field, δϕ::FlatS0)
    r = 0f
    for n in 1:L.order, (a,b) in zip(0:n,n:-1:0)
        ∂ⁿf = Ł(∂x^a * ∂y^b * f) / factorial(a*b)
        r += -(  ((a==0) ? 0 : (∂x * (a * L.∂xⁱϕ[a-1] * L.∂yⁱϕ[b] * Ł(δϕ) * ∂ⁿf)))
               + ((b==0) ? 0 : (∂y * (b * L.∂xⁱϕ[a] * L.∂yⁱϕ[b-1] * Ł(δϕ) * ∂ⁿf))))
    end
    r
end
