
using Distributions
using Distributions: PDiagMat

function Base.rand(rng::AbstractRNG, s::MvNormal{<:Any,<:PDiagMat{<:Any,<:Field}})
    s.μ + simulate(rng, Diagonal(s.Σ.diag))
end
function Distributions.logpdf(s::MvNormal{<:Any,<:PDiagMat{<:Any,<:Field}}, f::Field) where {T}
    z = s.μ - f
    -(z' * Diagonal(s.Σ.inv_diag) * z + logdet(Diagonal(s.Σ.diag))) / 2
end
function Distributions.MvNormal(μ::Field, D::DiagOp)
    T = real(eltype(D))
    Σ = PDiagMat{T, typeof(diag(D))}(length(diag(D)), diag(D), diag(pinv(D)))
    MvNormal{T,typeof(Σ),typeof(μ)}(μ, Σ)
end
function Distributions.MvNormal(μ::Real, D::DiagOp)
    MvNormal(μ*one(diag(D)), D)
end
