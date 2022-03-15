
struct PDFieldOpWrapper{L} <: AbstractPDMat{Real}
    op :: L
end
Zygote.ProjectTo(::PDFieldOpWrapper) = identity


function Base.rand(rng::AbstractRNG, dist::MvNormal{<:Any,<:PDFieldOpWrapper}; Nbatch=())
    dist.μ + simulate(rng, dist.Σ.op; Nbatch)
end
function Distributions.logpdf(dist::MvNormal{<:Any,<:PDFieldOpWrapper}, f::Field)
    z = dist.μ - f
    Σ = dist.Σ.op
    -(z' * (pinv(Σ) * z) + logdet(Σ)) / 2
end
function Distributions.MvNormal(μ::Field, Σ::FieldOp{<:Real})
    PDΣ = PDFieldOpWrapper(Σ)
    MvNormal{real(eltype(μ)),typeof(PDΣ),typeof(μ)}(μ, PDΣ)
end
function Distributions.MvNormal(μ::Real, Σ::FieldOp{<:Real})
    μ==0 ? MvNormal(zero(diag(Σ)), Σ) : error("μ must be a Field or 0")
end