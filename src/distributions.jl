
function Base.rand(rng::AbstractRNG, dist::MvNormal{<:Any,<:PDiagMat{<:Any,<:Field}}; Nbatch=nothing)
    dist.μ + simulate(rng, Diagonal(dist.Σ.diag); Nbatch)
end
function Distributions.logpdf(dist::MvNormal{<:Any,<:PDiagMat{<:Any,<:Field}}, f::Field)
    z = dist.μ - f
    -(z' * Diagonal(dist.Σ.inv_diag) * z + logdet(Diagonal(dist.Σ.diag))) / 2
end
function Distributions.MvNormal(μ::Field, D::Diagonal{T,<:Field{<:Basis,T}}) where {T<:Union{Real,Complex}}
    Σ = PDiagMat{real(T),typeof(diag(D))}(length(diag(D)), diag(D), diag(pinv(D)))
    MvNormal{real(T),typeof(Σ),typeof(μ)}(μ, Σ)
end
function Distributions.MvNormal(μ::Real, D::DiagOp)
    μ==0 ? MvNormal(zero(diag(D)), D) : error("μ must be a Field or 0")
end
