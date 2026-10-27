module FwdModelCallTests

using CMBLensing
using ForwardDiff
using Random
using Test
using Zygote

const Normal = CMBLensing.Distributions.Normal
const loglikelihood = CMBLensing.Distributions.loglikelihood

macro as_syntax(ex)
    QuoteNode(ex)
end

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

@fwdmodel function ordered_child(f, x; scale)
    value ← f(x) * scale
end

@fwdmodel function double(; x)
    doubled ← 2 * x
end
end

struct ParameterizedChild{T}
    mean::T
end
@fwdmodel function (child::ParameterizedChild)(; x)
    x ~ Normal(child.mean, 1)
end

@fwdmodel function captured_child(container; x)
    container.model(; x)
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

@fwdmodel function argument_scope(; obs)
    first ← Base.identity(begin positional = 3; positional end)
    second ← Base.round(2.345; digits=begin precision = 1; precision end)
    third ← Base.map(begin items = 1:3; items end) do i
        2 * i
    end
    marker ← positional + precision + sum(items)
    obs ~ Normal(marker, 1)
end

const call_events = Symbol[]
positions() = (push!(call_events, :pos); (3,))
keywords() = (push!(call_events, :kw); (; scale=4))
callback(x) = (push!(call_events, :callback); x + 2)
plain_callee() = (push!(call_events, :callee); (f, x; scale) -> (push!(call_events, :body); f(x) * scale))
model_callee() = (push!(call_events, :callee); Children.ordered_child)

@fwdmodel function ordered_plain_do()
    value ← plain_callee()(positions()...; keywords()...) do x
        callback(x)
    end
end

@fwdmodel function ordered_model_do()
    model_callee()(positions()...; keywords()...) do x
        callback(x)
    end
end

const transform = sin
@fwdmodel function do_traversal_order(; x)
    output ← Base.map(begin transform = Children.double; [x] end) do value
        transform(; x=value)
    end
end

@fwdmodel function broadcasted(; x)
    value ← x .^ 2 .+ Base.sin.(x)
    total ← Base.sum(value)
end

@fwdmodel function broadcast_macro(; x)
    value ← @. Base.sin(x)
end

@fwdmodel function quoted_call(; x)
    expression ← :(Base.sin(x))
end

@fwdmodel function interpolated_model(; x)
    expression ← :(1 + $(Children.child(2; x, offset=1)))
end

@fwdmodel function syntax_macro(; x)
    expression ← @as_syntax Base.sin(x)
end

@fwdmodel function quoted_macro_interpolation(; x)
    expression ← :(@inbounds $(Children.child(2; x, offset=1)))
end

@testset "Forward-model call expressions" begin
    expected_loglike = CMBLensing.Distributions.logpdf(Normal(2, 1), 3)
    for model in (qualified, splatted, indexed, dynamic_model)
        result = simulate(MersenneTwister(1), model; x=3)
        @test result.x == 3
        @test result.value == 7
        @test loglikelihood(model; x=3) ≈ expected_loglike
    end
    for model in (qualified, splatted, indexed, model_do)
        likelihood(x) = loglikelihood(model; x)
        @test ForwardDiff.derivative(likelihood, 3.0) ≈ -1
        @test Zygote.gradient(likelihood, 3.0)[1] ≈ -1
    end
    likelihood_with_mean(mean) = loglikelihood(captured_child, (; model=ParameterizedChild(mean)); x=3.0)
    @test ForwardDiff.derivative(likelihood_with_mean, 2.0) ≈ 1
    @test Zygote.gradient(likelihood_with_mean, 2.0)[1] ≈ 1
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

    result = simulate(MersenneTwister(1), argument_scope; obs=10.5)
    @test result.marker == 10
    @test loglikelihood(argument_scope; obs=10.5) ≈ CMBLensing.Distributions.logpdf(Normal(10, 1), 10.5)

    empty!(call_events)
    direct = plain_callee()(positions()...; keywords()...) do x
        callback(x)
    end
    plain_order = copy(call_events)
    @test direct == 20
    for (model, order) in ((ordered_plain_do, plain_order), (ordered_model_do, [:callee, :pos, :kw, :callback]))
        empty!(call_events)
        @test simulate(MersenneTwister(1), model).value == 20
        @test call_events == order
        empty!(call_events)
        @test loglikelihood(model) == 0
        @test call_events == order
    end
    result = simulate(MersenneTwister(1), do_traversal_order; x=3)
    @test result.output == [6]
    @test result.doubled == 6
    @test loglikelihood(do_traversal_order; x=3) == 0

    x = [0.0, 1.0, 2.0]
    result = simulate(MersenneTwister(1), broadcasted; x)
    @test result.value ≈ x .^ 2 .+ sin.(x)
    @test result.total ≈ sum(result.value)
    @test loglikelihood(broadcasted; x) == 0
    @test simulate(MersenneTwister(1), broadcast_macro; x).value ≈ sin.(x)
    @test loglikelihood(broadcast_macro; x) == 0

    @test simulate(MersenneTwister(1), quoted_call; x=2).expression == :(Base.sin(x))
    result = simulate(MersenneTwister(1), interpolated_model; x=3)
    @test result.expression == :(1 + 7)
    @test result.value == 7
    @test loglikelihood(interpolated_model; x=3) ≈ expected_loglike
    @test simulate(MersenneTwister(1), syntax_macro; x=2).expression == :(Base.sin(x))
    result = simulate(MersenneTwister(1), quoted_macro_interpolation; x=3)
    @test result.expression isa Expr && result.expression.head == :macrocall
    @test result.expression.args[end] == 7
    @test result.value == 7
    @test loglikelihood(quoted_macro_interpolation; x=3) ≈ expected_loglike
end

end
