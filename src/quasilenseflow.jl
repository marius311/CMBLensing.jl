
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
                    Ł(Diagonal(mÐ) * (p' * (∇ᵢ * (Diagonal(mÐ) * Ł(f)))))
                end
                v, Ł(f₀)
            end
        )
    end
end

###
@unpack f,f̃,ϕ,ds = load_sim_dataset(
    seed=0, θpix=3, Nside=128, pol=:I, T=Float64, μKarcminT=5, beamFWHM=1, bandpass_mask=LowPass(3000),
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

Lϕ = cache(LenseFlow(ϕ),f);

function ql(ϕ,f)
    ∇ϕ,Hϕ = map(Ł, gradhess(ϕ))
    T = eltype(f)
    OutOfPlaceRK4Solver(f,0,1,7) do t,f
        p = pinv(Diagonal.(I + T(t)*Hϕ))' * Diagonal.(∇ϕ)
        Diagonal(mÐ) * (p' * (∇ᵢ * (Diagonal(mÐ) * f)))
    end
end

function ql(ϕ,f)
    ∇ϕ,Hϕ = map(Ł, gradhess(ϕ))
    p = Diagonal.(Hϕ) * Diagonal.(∇ϕ)
    Diagonal(mÐ) * (p' * Map.(@SVector[f,f]))
end

gradient(ϕ -> sum(Map(ql(ϕ,f))), ϕ)[1]

gradient(ϕ -> f'*ql(ϕ,f), ϕ)[1] - gradient(ϕ -> f'*(LenseFlow(ϕ)*f), ϕ)[1] |> plot

L = Diagonal(f) * OuterProdOp(f,f)



@SMatrix[L L; L L]'
[[L] [L]; [L] [L]]'

struct Elem end
struct AdjointElem end
Base.adjoint(::Elem) = AdjointElem()
@SMatrix[Elem() Elem(); Elem() Elem()]'

@SMatrix[[1,2] [1,2]; [1,2] [1,2]]'










w = @SVector[f,f]
foo(f) = sum(sum(pinv(Diagonal.(Map.(@SMatrix[2f f; f 2f]))) * w))
@time gradient(foo,Map(f))



y1,bk1 = Zygote.pullback(sum, Diagonal.(@SMatrix[ϕ ϕ; ϕ ϕ]))
bk1(y1)

y2,bk2 = Zygote.pullback(pinv, Diagonal.(@SMatrix[ϕ ϕ; ϕ ϕ]))
bk2(bk1(y1)[1])

Δ = bk1(y1)[1]
y

y*Δ
