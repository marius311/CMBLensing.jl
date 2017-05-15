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

export PowerLens

@∷ struct PowerLens{N,F<:Field{∷,<:S0}} <: LenseOp
    ∂xϕⁱ::Dict{Int,Union{Int,F}}
    ∂yϕⁱ::Dict{Int,Union{Int,F}}
end

function PowerLens{N}(ϕ) where {N}
    ∂xϕ, ∂yϕ = Ł(∂x*ϕ), Ł(∂y*ϕ)
    PowerLens{N,typeof(∂xϕ)}((Dict(i=>(i==0?1:∂ϕ.^i) for i=0:N) for ∂ϕ=(∂xϕ,∂yϕ))...)
end

""" Create from an existing PowerLens operator one that lenses by -ϕ instead. """
antilensing(L::PowerLens{N,F}) where {N,F} = PowerLens{N,F}(N, (Dict(i=>v*(-1)^i for (i,v)=∂) for ∂=(L.∂xϕⁱ,L.∂xϕⁱ))...)


function *(L::PowerLens{N}, f::Field) where {N}
    f̂ = Ð(f)
    f̃ = 1Ł(f)
    @threadsum for n in 1:N, (a,b) in zip(0:n,n:-1:0)
        @. f̃ += L.∂xϕⁱ[a] * L.∂yϕⁱ[b] * $Ł(∂x^a * ∂y^b * f̂) / factorial(a) / factorial(b)
    end
    f̃
end

function *(f::Field, L::PowerLens{N}) where {N}
    Łf = Ł(f)
    r = 1Ð(f)
    @threadsum for n in 1:N, (a,b) in zip(0:n,n:-1:0)
        @. r += (-1)^n * ∂x^a * ∂y^b * $Ð(@. L.∂xϕⁱ[a] * L.∂yϕⁱ[b] * Łf) / factorial(a) / factorial(b)
    end
    r
end


## PowerLens Jacobian operators

*(fϕ::FΦTuple, J::δfϕₛ_δfϕₜ{1.,0.,<:PowerLens}) = FieldTuple(δf̃_δfᴴ(J.L,fϕ.f1), δf̃_δϕᴴ(J.L,J.fₜ,fϕ.f1) + fϕ.f2)
*(J::δfϕₛ_δfϕₜ{1.,0.,<:PowerLens}, fϕ::FΦTuple) = FieldTuple(δf̃_δf(J.L,fϕ.f1) + δf̃_δϕ(J.L,J.fₜ,fϕ.f2), fϕ.f2)


δf̃_δf(L::PowerLens)           = FuncOp(x->δf̃_δf(L,x),   x->δf̃_δfᴴ(L,x))
δf̃_δϕ(L::PowerLens, f::Field) = FuncOp(x->δf̃_δϕ(L,f,x), x->δf̃_δϕᴴ(L,f,x))

## Jacobian terms

""" δf̃(f,ϕ)/δϕ * v """
function δf̃_δϕ(L::PowerLens{N}, f::F, v::Field) where {N,F<:Field}
    Ðf = Ð(f)
    r = Ł(zero(F))
    ∂ⁿf = Ł(zero(F))
    ∂xv, ∂yv = Ł(∇*v)
    @threadsum for n in 1:N, (a,b) in zip(0:n,n:-1:0)
        @. ∂ⁿf = $Ł(@. ∂x^a * ∂y^b * Ðf) / factorial(a) / factorial(b)
        @. r += (  ((a==0) ? 0 : a * L.∂xϕⁱ[a-1] * L.∂yϕⁱ[b] * ∂xv * ∂ⁿf)
                 + ((b==0) ? 0 : b * L.∂xϕⁱ[a] * L.∂yϕⁱ[b-1] * ∂yv * ∂ⁿf))
    end
    r
end

""" δf̃(f,ϕ)/δf * v """
δf̃_δf(L::PowerLens,v) = L*v


## Jacobian transpose terms

""" (δf̃(f,ϕ)/δf)ᴴ * v """
function δf̃_δfᴴ(L::PowerLens{N}, v::Field) where {N}
    Łv = Ł(v)
    r = 1Ð(v)
    @threadsum for n in 1:N, (a,b) in zip(0:n,n:-1:0)
        @. r += (-1)^n * ∂x^a * ∂y^b * $Ð(@. L.∂xϕⁱ[a] * L.∂yϕⁱ[b] * Łv) / factorial(a) / factorial(b)
    end
    r
end

""" (δf̃(f,ϕ)/δϕ)ᴴ * v """
function δf̃_δϕᴴ(L::PowerLens{N,F}, f::Field, v::Field) where {N,F}
    Łv = Ł(v)
    Ðf = Ð(f)
    r = Ð(zero(F))
    ∂ⁿfᴴ_v = Ł(zero(F))
    @threadsum for n in 1:N, (a,b) in zip(0:n,n:-1:0)
        @. ∂ⁿfᴴ_v = $(Ł(@. ∂x^a * ∂y^b * Ðf)' * Łv) / factorial(a) / factorial(b)
        @. r += -(  ((a==0) ? 0 : (∂x * $Ð(@. a * L.∂xϕⁱ[a-1] * L.∂yϕⁱ[b] * ∂ⁿfᴴ_v)))
                  + ((b==0) ? 0 : (∂y * $Ð(@. b * L.∂xϕⁱ[a] * L.∂yϕⁱ[b-1] * ∂ⁿfᴴ_v))))
    end
    r
end

## Lensing second derivatives

""" Compute wᴴ * δ²f̃(f,ϕ)/δϕ² * v """
function δ²f̃_δϕ²(L::PowerLens{N,F}, f::Field, w::Field, v::Field) where {N,F}
    Łw = Ł(w)
    Ðf = Ð(f)
    ∂xv, ∂yv = Ł(∇*v)
    r = Ð(zero(F))
    ∂ⁿfᴴ_w = Ł(zero(F))
    @threadsum for n in 1:N, (a,b) in zip(0:n,n:-1:0)
        @. ∂ⁿfᴴ_w = $(Ł(@. ∂x^a * ∂y^b * Ðf)' * Łw) / factorial(a) / factorial(b)
        @. r += -(  ((a<2)        ? 0 : (∂x * $Ð(@. ∂xv * a * (a-1) * L.∂xϕⁱ[a-2] * L.∂yϕⁱ[b]   * ∂ⁿfᴴ_w)))
                  + ((a<1 || b<1) ? 0 : (∂x * $Ð(@. ∂yv * a * b     * L.∂xϕⁱ[a-1] * L.∂yϕⁱ[b-1] * ∂ⁿfᴴ_w)))
                  + ((b<2)        ? 0 : (∂y * $Ð(@. ∂yv * b * (b-1) * L.∂yϕⁱ[b-2] * L.∂xϕⁱ[a]   * ∂ⁿfᴴ_w)))
                  + ((a<1 || b<1) ? 0 : (∂y * $Ð(@. ∂xv * a * b     * L.∂yϕⁱ[b-1] * L.∂xϕⁱ[a-1] * ∂ⁿfᴴ_w))))
    end
    r
end


## Hessian terms

δ²f̃_δϕδf(L::PowerLens, f::Field, w::Field, v::Field) = w * δf̃_δϕ(L,v)
δ²f̃_δfδϕ(L::PowerLens, f::Field, w::Field, v::Field) = δf̃_δϕ(L,v) * w
