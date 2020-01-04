
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
    seed=0, θpix=3, Nside=128, pol=:I, T=Float64, μKarcminT=1, beamFWHM=3, bandpass_mask=LowPass(3000),
);
###
# mÐ = Diagonal(FlatEBFourier(one(f[:E]), LowPass(1000)*one(f[:B])))
mÐ = LowPass(3000)*one(f)
plot(QuasiLenseFlow(mÐ)(ϕ)*f - f)
###
function getds(mÐℓ,ℓs=50:1000:5092)
    ds′ = deepcopy(ds)
    if spin(f)==S0
        mÐ = BandPassOp(InterpolatedCℓs(ℓs,mÐℓ)) * one(Fourier(f))
    else
        mÐ = Diagonal(FlatEBFourier(one(f[:E]), BandPassOp(InterpolatedCℓs(ℓs,mÐℓ))*one(f[:B])))
    end
    @set! ds′.QL = QuasiLenseFlow(mÐ)
    @set! ds′.D = 1
end
function getds′(mÐℓ; ℓs=50:1000:5092)
    ds′ = deepcopy(ds)
    @set! ds′.D = Diagonal(BandPassOp(InterpolatedCℓs(ℓs,mÐℓ)) * one(Fourier(f)))
end
###
function χ²_end(ds, nsteps=2)
    @unpack tr = MAP_joint(ds(), nsteps=nsteps, progress=false)
    -2 * tr[end][:lnPcur]
end


using Optim
res = optimize(mÐℓ -> χ²_end(getds′([1; mÐℓ], ℓs=[0,2000,3500,5000])), ones(3), method=GradientDescent(), iterations=10)
res.minimizer



##
close(:all)
plot(cov_to_Cℓ(ds.D()))
gcf()
##
@time χ²_end(ds)
@time χ²_end(getds′([1,1,1,1,5,10]))


@time χ²_end(getds([1,1,1,1,1,1]))
@time χ²_end(getds([1,1,1,1,1,1]))


gradient(ϕ->lnP(:mix,f,ϕ,getds([1,1,1,1,1,1])()), ϕ)

Ł(ϕ) .* Ł(f)

QL = QuasiLenseFlow(mÐ) 
gradient(ϕ -> sum(QL(ϕ)*f), ϕ)

f = Ł(f)
gradient(f -> f'*(@. 2f*f + f), f)


gradient(ϕ) do ϕ
    T = eltype(ϕ)
    ∇ϕ,Hϕ = map(Ł, gradhess(ϕ))
    sum(OutOfPlaceRK4Solver(f,0,1,7) do t,f
        p = pinv(Diagonal.(I + T(t)*Hϕ))' * Diagonal.(∇ϕ)
        p' * (∇ᵢ * f)
    end)
end[1][:E] |> plot

gradient(ϕ -> sum(LenseFlow(ϕ)*f), ϕ)[1] |> plot





using Test

sum(sum,f.fs)

Zygote.accum_sum(f, dims=:)
Zygote.accum_sum(f, dims=1)
Zygote.accum_sum(f, dims=2)

@which sum(f[:Q])

Zygote.accum_sum([1,2,3], dims=:)
Zygote.accum_sum([1,2,3], dims=1)
Zygote.accum_sum([1,2,3], dims=2)
Zygote.accum_sum([1,2,3], dims=3)







fwf₀ = argmaxf_lnP(1, ds);
function get_ρℓ_first_step(ds)
    f°, = mix(fwf₀,0ϕ,ds)
    get_ρℓ(gradient(ϕ->lnP(:mix,f°,ϕ,ds), 0ϕ)[1], ϕ)
end
###
close(:all)
semilogx(get_ρℓ_first_step(getds([1,1,1,1,1,1,1,1,1,1,0]))[100:1000])
semilogx(get_ρℓ_first_step(getds([0,0,0,0,0,0,0,0,0,0,0]))[100:1000])
gcf()
##

ϕ = Ł(ϕ)
f = Ł(f)

gradient(ϕ -> sum(ϕ .* f), ϕ)[1].fs |> sum
gradient(ϕ -> sum(ϕ .* f.Q .+ ϕ .* f.U), ϕ)
 
@which broadcast(*, ϕ, f)
