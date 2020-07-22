
### functions needed to hook into Zygote forward-mode differentiation

# Zygote forward mode is still quite experimental, much of what is here will
# probably be unnecessary in the future

using Zygote
using Zygote: @adjoint, gradient, Fill, unbroadcast, accum_sum
using Zygote.Forward: @tangent, pushforward
import Zygote.Forward: zerolike

# denotes a function for which gradient is zero 
macro nofgrad(func)
    esc(quote
        @tangent $func(args...) = $func(args...), (_...) -> 0
    end)
end

# denotes a function which is linear in all of its arguments
macro linear(func)
    esc(quote
        @tangent $func(args...) = $func(args...), (ȧrgs...) -> $func(ȧrgs...)
    end)
end


@nofgrad Zygote.cache
@nofgrad Base.nfields
@nofgrad CMBLensing.basetype
@nofgrad one

@linear accum_sum
@linear fft
@linear bfft
@linear promote
@linear adjoint
@linear Diagonal
@linear tr

## generic rules

@tangent function Base.broadcasted(f, args...)
    function _pushforward(_, ȧrgs...)
        broadcast(args..., ȧrgs...) do args_ȧrgs...
            (argsᵢ, ȧrgsᵢ) = args_ȧrgs[1:end÷2], args_ȧrgs[end÷2+1:end]
            pushforward(f, argsᵢ...)(ȧrgsᵢ...)
        end
    end
    broadcast(f, args...), _pushforward
end
@tangent Fill(x, n) = Fill(x,n), (ẋ,_) -> Fill(ẋ,n)
zerolike(A::Array{Any}) = Fill(0,size(A))
@adjoint function norm(x::AbstractVector)
    n = norm(x)
    n, Δ -> (Δ .* x ./ n,)
end
@tangent function norm(x::AbstractVector) 
    n = norm(x)
    n, ẋ -> x'ẋ / n
end 

## CMBLensing-specific rules

@tangent dot(x::Field, y::Field) = dot(x,y), (ẋ,ẏ) -> dot(ẋ,y) + dot(x,ẏ)
@tangent (::Type{B})(f) where {B<:Basis} = B(f), ḟ->B(ḟ)
@tangent (A::LinOp * B::LinOp) = A*B, (Ȧ, Ḃ) -> Ȧ*B + A*Ḃ
@tangent OuterProdOp(x,y)         = OuterProdOp(x,y),  (ẋ,ẏ) -> OuterProdOp(ẋ,y)  + OuterProdOp(x,ẏ)
@tangent (x::Field * y::AdjField) = OuterProdOp(x,y'), (ẋ,ẏ) -> OuterProdOp(ẋ,y') + OuterProdOp(x,ẏ')
@tangent pinv(L::LinOp) = pinv(L), L̇ -> -pinv(L)*L̇*pinv(L)
@tangent (L::ParamDependentOp)(θ) = L(θ), (_,θ̇) -> pushforward(L.recompute_function,θ)(θ̇)
zerolike(L::ParamDependentOp) = nothing
