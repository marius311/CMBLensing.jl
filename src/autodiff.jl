

# Eventually we want proper Zygote.jl reverse-mode automatic differentiation
# through our posterior. This is close but not fully working yet, so for now we
# define the following simple API and implement it using finite differences. In
# the future, we can then transparently switch to a Zygote-based solution,
# once that is adequately working.


function gradient(f::Function; relϵ=nothing, absϵ=nothing) 
    
    _gradient(x::Number) = gradient_1vecarg(x->f(x[1]))([x])[1]
    _gradient(x::Vector) = gradient_1vecarg(f)(x)
    
    function gradient_1vecarg(f::Function)
        function (x::Vector)
            g = similar(x)
            if relϵ==nothing && absϵ==nothing
                ϵ = sqrt(eps(eltype(x)))
            elseif absϵ!=nothing
                @assert relϵ==nothing "Only one of `relϵ` or `absϵ` must be specifed"
                ϵ = absϵ
            end
            for i in eachindex(x)
                if relϵ!=nothing
                    ϵ = relϵ*x[i]
                end
                x₊,x₋ = copy(x), copy(x)
                x₊[i] += ϵ
                x₋[i] -= ϵ
                g[i] = (f(x₊) - f(x₋))/(2ϵ)
            end
            return g
        end
    end
    
    (args...) -> _gradient(args...)
    
end





# Some adjoint rules for the work-in-progress Zygote-based derivatives

@init @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" @eval begin

    using .Zygote: gradient, unbroadcast

    # this one arguably could be in Zygote itself
    # see: https://github.com/FluxML/Zygote.jl/issues/316
    Zygote.@adjoint *(D::Diagonal, v::AbstractVector) = D.diag .* v,
        Δ -> (Diagonal(unbroadcast(D.diag, Δ .* conj.(v))), unbroadcast(v, Δ .* conj.(D.diag)))

end
