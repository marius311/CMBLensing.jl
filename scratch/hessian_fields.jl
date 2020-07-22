
using CMBLensing
using CMBLensing: OuterProdOp, AdjField, @nofgrad, @linear
using Zygote
using Zygote.Forward: pushforward, @tangent
using LinearAlgebra
using FiniteDiff

f = FlatQUMap(rand(128,128), rand(128,128))
D = Diagonal(f)
L = ParamDependentOp((θ=(A=1,))->θ.A*D)



let f = f, D = Diagonal(f), L = ParamDependentOp((θ=(A=1,))->θ.A*D), λ = L.recompute_function
    pushforward(A -> sum(pinv(L((A=A,))) * f), 1)(1)
    # FiniteDiff.finite_difference_derivative(A -> sum(pinv(L((A=A,))) * f), 1.)
end



let f = f, D = Diagonal(f), L = ParamDependentOp((θ=(A=1,))->θ.A*D)
    (
        pushforward(A -> gradient(A -> dot(f, pinv(L((A=A^2,))) * f), A)[1], 1)(1),
        FiniteDiff.finite_difference_hessian(A -> dot(f, pinv(L((A=A[1]^2,))) * f), [1.])
    )
end



@unpack f,ϕ,ds = load_sim(Nside=32, θpix=3, pol=:P)

pushforward(Aϕ -> gradient(Aϕ -> lnP(0,f,ϕ,(Aϕ=Aϕ,r=0.1),ds), Aϕ)[1], 1)(1)

logdet(ds.Cϕ, (Aϕ=2,))