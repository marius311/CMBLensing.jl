module FwdModelCallTests

using CMBLensing
using Random
using Test

const Normal = CMBLensing.Distributions.Normal
const loglikelihood = CMBLensing.Distributions.loglikelihood

module Children
using CMBLensing
const Normal = CMBLensing.Distributions.Normal

@fwdmodel function child(scale; x, offset=0)
    x ~ Normal(2, 1)
    value ← scale * x + offset
end

@fwdmodel function child_with_callback(f; x)
    x ~ Normal(2, 1)
    value ← f(x)
end
end

@fwdmodel function qualified(; x)
    Children.child(2; x, offset=1)
end

@fwdmodel function splatted(; x)
    args = (2,)
    kwargs = (; x, offset=1)
    Children.child(args...; kwargs...)
end

const models = (Children.child,)
@fwdmodel function indexed(; x)
    models[1](2; x, offset=1)
end

const model_lookups = Ref(0)
function choose_model()
    model_lookups[] += 1
    Children.child
end
@fwdmodel function dynamic_model(; x)
    choose_model()(2; x, offset=1)
end

const plain_lookups = Ref(0)
function choose_plain()
    plain_lookups[] += 1
    (x; scale=1) -> scale * x
end
@fwdmodel function dynamic_plain(; x)
    value ← choose_plain()(x; scale=2)
end

@fwdmodel function ordinary_do(; x)
    value ← Base.map(1:3) do i
        i * x
    end
end

@fwdmodel function model_do(; x)
    Children.child_with_callback(; x) do value
        2 * value
    end
end

@fwdmodel function broadcasted(; x)
    value ← x .^ 2 .+ Base.sin.(x)
    total ← Base.sum(value)
end

@fwdmodel function broadcast_macro(; x)
    value ← @. Base.sin(x)
end

@testset "Forward-model call expressions" begin
    expected_loglike = CMBLensing.Distributions.logpdf(Normal(2, 1), 3)
    for model in (qualified, splatted, indexed, dynamic_model)
        result = simulate(MersenneTwister(1), model; x=3)
        @test result.x == 3
        @test result.value == 7
        @test loglikelihood(model; x=3) ≈ expected_loglike
    end
    model_lookups[] = 0
    simulate(MersenneTwister(1), dynamic_model; x=3)
    @test model_lookups[] == 1
    model_lookups[] = 0
    loglikelihood(dynamic_model; x=3)
    @test model_lookups[] == 1

    plain_lookups[] = 0
    @test simulate(MersenneTwister(1), dynamic_plain; x=3).value == 6
    @test plain_lookups[] == 1
    plain_lookups[] = 0
    @test loglikelihood(dynamic_plain; x=3) == 0
    @test plain_lookups[] == 1

    @test simulate(MersenneTwister(1), ordinary_do; x=2).value == [2, 4, 6]
    @test loglikelihood(ordinary_do; x=2) == 0
    result = simulate(MersenneTwister(1), model_do; x=3)
    @test result.x == 3
    @test result.value == 6
    @test loglikelihood(model_do; x=3) ≈ expected_loglike

    x = [0.0, 1.0, 2.0]
    result = simulate(MersenneTwister(1), broadcasted; x)
    @test result.value ≈ x .^ 2 .+ sin.(x)
    @test result.total ≈ sum(result.value)
    @test loglikelihood(broadcasted; x) == 0
    @test simulate(MersenneTwister(1), broadcast_macro; x).value ≈ sin.(x)
    @test loglikelihood(broadcast_macro; x) == 0
end

end
