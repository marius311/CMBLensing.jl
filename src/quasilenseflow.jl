
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

f,g,h = Ł.(@repeated(simulate(ds.Cf),3))
v = @SVector[g,g]
D = Diagonal(f)




@testset "Field inner products" begin
    
    @test ((δ = gradient(f -> sum(Map(f)),     Map(f))[1]); basis(δ)==Map     && δ ≈ one(Map(f)))
    @test ((δ = gradient(f -> sum(Map(f)), Fourier(f))[1]); basis(δ)==Fourier && δ ≈ Fourier(one(Map(f))))
    
    @test gradient(f -> Map(f)' *     Map(g), f)[1] ≈ g
    @test gradient(f -> Map(f)' * Fourier(g), f)[1] ≈ g

    @test gradient(f -> sum(Diagonal(Map(f)) *     Map(g)), f)[1] ≈ g
    @test gradient(f -> sum(Diagonal(Map(f)) * Fourier(g)), f)[1] ≈ g

    @test gradient(f -> sum(Diagonal(Map(∇[1]*f)) *     Map(g)), f)[1] ≈ ∇[1]'*g
    @test gradient(f -> sum(Diagonal(Map(∇[1]*f)) * Fourier(g)), f)[1] ≈ ∇[1]'*g
    
    @test gradient(f -> f'*(D\f), Fourier(f))[1] ≈ D\f + D'\f
    @test gradient(f -> (f'/D)*f, Fourier(f))[1] ≈ D\f + D'\f
    @test gradient(f -> f'*(D\f), Map(f))[1] ≈ D\f + D'\f
    @test gradient(f -> (f'/D)*f, Map(f))[1] ≈ D\f + D'\f
    
    @test gradient(f -> f'*(D*f), Fourier(f))[1] ≈ D*f + D'*f
    @test gradient(f -> (f'*D)*f, Fourier(f))[1] ≈ D*f + D'*f
    @test gradient(f -> f'*(D*f), Map(f))[1] ≈ D*f + D'*f
    @test gradient(f -> (f'*D)*f, Map(f))[1] ≈ D*f + D'*f
    
end

@testset "FieldVector inner products" begin

    @test gradient(f -> Map(∇[1]*f)' *     Map(v[1]) + Map(∇[2]*f)' *     Map(v[2]), f)[1] ≈ ∇' * v
    @test gradient(f -> Map(∇[1]*f)' * Fourier(v[1]) + Map(∇[2]*f)' * Fourier(v[2]), f)[1] ≈ ∇' * v
    @test gradient(f -> sum(Diagonal(Map(∇[1]*f)) * v[1] + Diagonal(Map(∇[2]*f)) * v[2]), f)[1] ≈ ∇' * v

end

@testset "FieldOpVector inner products" begin
    
    @test gradient(f -> @SVector[f,f]' * Map.(@SVector[g,g]), f)[1] ≈ 2g
    @test gradient(f -> @SVector[f,f]' * Fourier.(@SVector[g,g]), f)[1] ≈ 2g
    
    @test gradient(f -> sum(Diagonal.(Map.(∇*f))' * Fourier.(v)), f)[1] ≈ ∇' * v
    @test gradient(f -> sum(Diagonal.(Map.(∇*f))' * Map.(v)), f)[1] ≈ ∇' * v

end
##

@testset "OuterProdOp" begin
    
    @test OuterProdOp(f,g) * h ≈ f*(g'*h)
    @test OuterProdOp(f,g)' * h ≈ g*(f'*h)
    @test diag(OuterProdOp(f,g)) ≈ f .* conj.(g)
    @test diag(OuterProdOp(f,g)') ≈ conj.(f) .* g
    @test diag(OuterProdOp(f,g) + OuterProdOp(f,g)) ≈ 2 .* f .* conj.(g)
    
end



gradient(ϕ) do ϕ
    g,H = map(Ł,gradhess(ϕ))
    v = (Diagonal.(H))' * Diagonal.(g)
    sum(v' * (∇*f))
end

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
