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


struct PowerLens{D<:Dict} <: ImplicitOp{Bottom}
    order :: Int
    ∇1ϕᵖ  :: D
    ∇2ϕᵖ  :: D
end

PowerLens(order) = x -> PowerLens(x, order)
PowerLens(ϕ::Field, order) = PowerLens(∇*ϕ, order)
function PowerLens(d::FieldVector, order)
    require_unbatched(d[1])
    ∇1ϕ,  ∇2ϕ  = Ł(d)
    ∇1ϕᵖ, ∇2ϕᵖ = (Dict(p => (p==0 ? 1 : ∇iϕ.^p) for p=0:order) for ∇iϕ=(∇1ϕ,∇2ϕ))
    PowerLens(order, ∇1ϕᵖ, ∇2ϕᵖ)
end

""" Create a PowerLens operator that lenses by -ϕ instead. """
function antilensing(L::PowerLens)
    PowerLens(N, (Dict(p=>∇iϕᵖ*(-1)^p for (p,∇iϕᵖ)=coeffs) for coeffs=(L.∇1ϕᵖ,L.∇1ϕᵖ))...)
end

function *(L::PowerLens, f::Field)
    require_unbatched(f)
    Ðf = Ð(f)
    f̃ = copy(Ł(f))
    for n in 1:L.order, (a,b) in zip(0:n,n:-1:0)
        @. f̃ += L.∇1ϕᵖ[a] * L.∇2ϕᵖ[b] * $(Ł(∇[1]^a * ∇[2]^b * Ðf)) / factorial(a) / factorial(b)
    end
    f̃
end

function *(Ladj::Adjoint{<:Any,<:PowerLens}, f::Field)
    require_unbatched(f)
    L = parent(Ladj)
    Łf = Ł(f)
    r = copy(Ð(f))
    for n in 1:L.order, (a,b) in zip(0:n,n:-1:0)
        r .+= (-1)^n .* (∇[1]^a * ∇[2]^b * Ð(@. L.∇1ϕᵖ[a] * L.∇2ϕᵖ[b] * Łf)) ./ factorial(a) ./ factorial(b)
    end
    r
end