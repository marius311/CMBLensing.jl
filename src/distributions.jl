
function Base.rand(rng::AbstractRNG, dist::MvNormal{<:Any,<:PDiagMat{<:Any,<:Field}}; Nbatch=())
    dist.μ + simulate(rng, Diagonal(dist.Σ.diag); Nbatch)
end
function Distributions.logpdf(dist::MvNormal{<:Any,<:PDiagMat{<:Any,<:Field}}, f::Field)
    z = dist.μ - f
    Σ = Diagonal(dist.Σ)
    -(z' * pinv(Σ) * z + logdet(Σ)) / 2
end
function Distributions.MvNormal(μ::Field, D::Diagonal{T,<:Field{<:Basis,T}}) where {T<:Real}
    Σ = PDiagMat(length(diag(D)), diag(D))
    MvNormal{T,typeof(Σ),typeof(μ)}(μ, Σ)
end
function Distributions.MvNormal(μ::Real, D::DiagOp)
    μ==0 ? MvNormal(zero(diag(D)), D) : error("μ must be a Field or 0")
end

Zygote.@adjoint PDiagMat(dim, dg) = PDiagMat(dim, dg), Δ -> (length(diag(Δ)), diag(Δ))