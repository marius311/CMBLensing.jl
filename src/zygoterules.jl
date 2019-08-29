
@init @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" @eval begin

    using .Zygote: gradient, unbroadcast

    # this one arguably could be in Zygote itself
    # see: https://github.com/FluxML/Zygote.jl/issues/316
    Zygote.@adjoint *(D::Diagonal, v::AbstractVector) = D.diag .* v,
        Δ -> (Diagonal(unbroadcast(D.diag, Δ .* conj.(v))), unbroadcast(v, Δ .* conj.(D.diag)))

end
