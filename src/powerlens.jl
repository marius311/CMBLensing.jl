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

struct PowerLens{F<:Field} <: LinOp{Pix,Spin,Basis}
    order::Int
    ∂xϕⁱ::Dict{Int,Union{F,Int}}
    ∂yϕⁱ::Dict{Int,Union{F,Int}}
end

function PowerLens{F<:FlatS0}(ϕ::F; order=4)
    PowerLens{F}(order,(Dict(i=>(i==0?1:(Ł(∂*ϕ)).^i) for i=0:order) for ∂=(∂x,∂y))...)
end

function *{F<:Field}(L::PowerLens, f::F)
    f̂ = 1Ð(f)
    f̃ = 1Ł(f)
    for n in 1:L.order, (a,b) in zip(0:n,n:-1:0)
        @. f̃ += L.∂xϕⁱ[a] * L.∂yϕⁱ[b] * $Ł(∂x^a * ∂y^b * f̂) / factorial(a) / factorial(b)
    end
    f̃
end

dLdf̃_df̃dfϕ(L::PowerLens, f::Field, dLdf̃::Field) = [df̃dfᵀ(L,dLdf̃), df̃dϕᵀ(L,f,dLdf̃)]

""" Compute (df̃(f)/df)ᵀ * dLdf̃ """
function df̃dfᵀ{F<:Field}(L::PowerLens, dLdf̃::F)
    r = dLdf̃
    for n in 1:L.order, (a,b) in zip(0:n,n:-1:0)
        r += @. (-1)^n * ∂x^a * ∂y^b * $Ð(L.∂xϕⁱ[a] * L.∂yϕⁱ[b] * $Ł(dLdf̃)) / factorial(a) / factorial(b)
    end
    r
end

""" Compute (df̃(f)/dϕ)ᵀ * dLdf̃ """
function df̃dϕᵀ{F<:Field}(L::PowerLens{F}, f::Field, dLdf̃::Field)
    f̂ = Ð(f)
    r = zero(F)
    for n in 1:L.order, (a,b) in zip(0:n,n:-1:0)
        ∂ⁿfᵀ_dLdf̃ = Ł(@. ∂x^a * ∂y^b * f̂)' * Ł(dLdf̃) ./ factorial(a) ./ factorial(b)
        r += -(  ((a==0) ? 0 : (∂x * (a * L.∂xϕⁱ[a-1] * L.∂yϕⁱ[b] * ∂ⁿfᵀ_dLdf̃)))
               + ((b==0) ? 0 : (∂y * (b * L.∂xϕⁱ[a] * L.∂yϕⁱ[b-1] * ∂ⁿfᵀ_dLdf̃))))
    end
    r
end
