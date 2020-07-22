
using FFTW
using LinearAlgebra
using Test
using Zygote: @adjoint, gradient, Fill, unbroadcast, accum_sum
using Zygote.Forward: @tangent, pushforward

###

@tangent function Base.broadcasted(f, args...)
    function _pushforward(_, ȧrgs...)
        broadcast((f′,args′...)->f′(args′...), pushforward.(f, args...), ȧrgs...)
    end
    broadcast(f, args...), _pushforward
end
@tangent Fill(x, n) = Fill(x,n), (ẋ,_) -> Fill(ẋ,n)
@tangent accum_sum(x) = accum_sum(x), ẋ->accum_sum(ẋ)
@tangent Zygote.cache(cx) = Zygote.cache(cx), _->nothing
@tangent Base.nfields(x) = nfields(x), ẋ -> nothing
Zygote.Forward.zerolike(A::Array{Any}) = Fill(0,size(A))
@tangent fft(x) = fft(x), ẋ->fft(ẋ)
@tangent bfft(x) = bfft(x), ẋ->bfft(ẋ)
@adjoint function norm(x::AbstractVector)
    n = norm(x)
    n, Δ -> (Δ .* x ./ n,)
end
@tangent function norm(x::AbstractVector) 
    n = norm(x)
    n, ẋ -> x'ẋ / n
end 

N=3
x=rand(N)

using FiniteDiff: finite_difference_hessian, finite_difference_derivative


let x=x
    hcat([pushforward(Aᵢ -> real(gradient(A -> norm(fft(A .* A .* x)), (A=ones(N); A[i]=Aᵢ; A))[1]), 1)(1) for i=1:N]...)
end

let x=x
    finite_difference_hessian(A′ -> norm(fft(A′ .* A′ .* x)), ones(N))
end


##

let x=x
    pushforward(A′ -> real(norm(fft(A′^2 .* x))), 1)(1)
end

let x=x
    finite_difference_derivative(A′ -> real(norm(fft(A′.^2 .* x))), 1.)
end

##

let x=x
    pushforward(A -> gradient(A′ -> norm((A′ .* x)), A)[1], ones(N))(I(N))
end

let x=x
    finite_difference_hessian(A -> gradient(A′ -> norm((A′ .* x)), A)[1], ones(N))
end
