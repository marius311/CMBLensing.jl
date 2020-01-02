
using Pkg
pkg"activate ."
@eval Main include("CMBLensing.jl")
using PyPlot

###

struct AnonymousFlowOp{I,t₀,t₁} <: FlowOp{I,t₀,t₁}
    velocity
    velocityᴴ
end

struct AnonymousFlowOpWithAdjoint{I,t₀,t₁} <: FlowOpWithAdjoint{I,t₀,t₁}
    velocity
    velocityᴴ
    negδvelocityᴴ
end

velocity(L::Union{AnonymousFlowOp,AnonymousFlowOpWithAdjoint}, f₀) = L.velocity(L,f₀)
velocityᴴ(L::Union{AnonymousFlowOp,AnonymousFlowOpWithAdjoint}, f₀) = L.velocityᴴ(L,f₀)
negδvelocityᴴ(L::AnonymousFlowOpWithAdjoint, f₀) = L.negδvelocityᴴ(L,f₀)

function FlowOp{I,t₀,t₁}(;velocity=nothing, velocityᴴ=nothing, negδvelocityᴴ=nothing) where {I,t₀,t₁}
    wrap(v) = (_,f₀)->v(f₀)
    if negδvelocityᴴ == nothing
        AnonymousFlowOp{I,t₀,t₁}(wrap(velocity), wrap(velocityᴴ))
    else
        AnonymousFlowOpWithAdjoint{I,t₀,t₁}(wrap(velocity), wrap(velocityᴴ), wrap(negδvelocityᴴ))
    end
end

###

function QuasiLenseFlow(mÐ,N=7)
    function (ϕ,N=N)
        T = eltype(ϕ)
        ∇ϕ,Hϕ = map(Ł, gradhess(ϕ))
        FlowOp{OutOfPlaceRK4Solver{7},0f0,1f0}(
            velocity = function (f₀)
                function v(t,f)
                    p = pinv(Diagonal.(I + T(t)*Hϕ))' * Diagonal.(∇ϕ)
                    # Ł(Diagonal(mÐ) * (p' * (∇ᵢ * (Diagonal(mÐ)*f))))
                    Ł(p' * (∇ᵢ * f))
                end
                v, Ł(f₀)
            end
        )
    end
end

###
@unpack f,f̃,ϕ,ds = load_sim_dataset(
    seed=0, θpix=3, Nside=4, pol=:I, T=Float64, μKarcminT=5, beamFWHM=1, bandpass_mask=LowPass(3000),
);
mÐ = LowPass(3000) * one(f)
###
QL = QuasiLenseFlow(one(Fourier(ϕ)))
ds.Cϕ * gradient(ϕ -> norm(QL(ϕ)*f), ϕ)[1] |> plot
ds.Cϕ * gradient(ϕ -> norm(LenseFlow(ϕ)*f), ϕ)[1] |> plot
##
using Test
@test gradient(f -> (∇*f)' * v, f)[1] ≈ (∇' * v)
@test gradient(f -> sum(Diagonal.(Map.(∇*f))' * Map(v)), f)[1] ≈ (∇' * v)
@test gradient(f -> sum(Diagonal.(Map.(∇*f))' * Fourier(v)), f)[1] ≈ (∇' * v)
##

w = @SVector[f,f]
gradient(ϕ) do ϕ
    g,H = map(Ł,gradhess(ϕ))
    v = Diagonal.(I + 2*H) * Diagonal.(g)
    sum(v' * w)
end

f = Map(f)

gradient(f -> sum(sum(Diagonal.(@SMatrix[f f; f f]) * @SVector[f,f])), f)[1] ≈ 8*f
gradient(f -> sum(sum(@SVector[f,f]      .+ @SVector[f,f])),      f)[1] ≈ 4*one(f)
gradient(f -> sum(sum(@SMatrix[f f; f f] .+ @SMatrix[f f; f f])), f)[1] ≈ 8*one(f)



y,bk = Zygote.pullback(sum, @SVector[f,f])
bk(one(y))[1] * (@SVector[f,f])'

(@SMatrix[f f; f f]') * bk(one(y))[1]



y,bk = Zygote.pullback(norm,Map(f))
(bk(y))

y, bk = Zygote.pullback(gradhess, ϕ)
bk(y)

# g,H = map(Ł,gradhess(ϕ))
# a,b = Diagonal.(H), Diagonal.(g)
# a,b = Map(∇*ϕ)', Map(∇*f)
# a,b = Diagonal.(Map(∇*ϕ))', Fourier(∇*f);
a,b = Diagonal.(Map.(∇*f))', Fourier.(v);
a
b
Zygote.@which a*b
y,bk = Zygote.pullback(*,a,b)
bk(y)[1]
bk(y)[2]
eltype(bk(y)[1])
eltype(bk(y)[2])
