
struct Simulate end
struct Logpdf end

is_simpleppl_model(f) = false

"""
    @fwdmodel function model(...)
        ...
    end

Define a forward model using `~` for stochastic variables and `←` for derived
variables.

When differentiating a likelihood with Zygote, contributions from nested
models called inside a `map` or `foreach` callback may be missing if the
callback result does not contribute to the likelihood. The likelihood value
can be correct while the gradient is `nothing` or incomplete. For a small CPU
collection, an explicit loop is one alternative:

```julia
using CMBLensing, Distributions, Zygote

@fwdmodel function example_child(value)
    value ~ Normal(2, 1)
end

@fwdmodel function example_model(; x)
    for value in (x, 2x)
        example_child(value)
    end
end

Zygote.gradient(x -> loglikelihood(example_model; x), 3.0)[1]  # -9.0
```

Check gradients for your model and backend against ForwardDiff or finite
differences. This example does not establish a GPU workaround.
"""
macro fwdmodel(def)
    sdef = splitdef(def)

    model_type = @capture(sdef[:name], (_::T_)) ? T : :(typeof($(sdef[:name])))
    @capture(sdef[:name], (model_name_::_) | model_name_)
    maybe_local_var = [] # only "maybe" b/c the simple thing here is too conservative c.f. inner functions / let blocks
    @capture(sdef[:name], (var_::_)) && push!(maybe_local_var, var)
       
    # simulate
    rand_vars = []
    body_simulate = postwalk(sdef[:body]) do x
        if @capture(x, ((vars__,) | var_) = rhs_)
            vars != nothing ? append!(maybe_local_var, vars) : push!(maybe_local_var, var)
        elseif @capture(x, var_ ~ dist_)
            push!(rand_vars, var)
            return :(ismissing($var) ? (_vars[$(QuoteNode(var))] = $var = $simulate(rng, $model_name, $dist)) : (_vars[$(QuoteNode(var))] = $var))
        elseif @capture(x, (vars__,) ← rhs_)
            tmpvars = (gensym.(vars)...,)
            return quote
                ($(tmpvars...),) = $rhs
                $(map(vars, tmpvars) do v, t
                    :(_vars[$(QuoteNode(v))] = (Base.@isdefined($v) && !ismissing($v)) ? $v : ($v = $t))
                end...)
            end
        elseif @capture(x, (var_ ← rhs_))
            return :(_vars[$(QuoteNode(var))] = (Base.@isdefined($var) && !ismissing($var) ? $var : ($var = $rhs))) 
        elseif !isexpr(x, :block) && @capture(x, (f_(args__; kwargs__) | f_(args__)))
            kwargs = kwargs == nothing ? () : kwargs
            if (f isa Symbol) && !startswith(string(f),".")
                if !(f in maybe_local_var) && isdefined(__module__, f)
                    if is_simpleppl_model(getfield(__module__, f))
                        return :($f($(Simulate()), _vars, rng, $(args...); $(kwargs...)))
                    end
                else
                    return :($is_simpleppl_model($f) ? $f($(Simulate()), _vars, rng, $(args...); $(kwargs...)) : $x)
                end
            end
        end
        return x
    end
    function arg_default(ex)
        var = isexpr(ex, :kw) ? ex.args[1] : ex
        var in rand_vars ? Expr(:kw, var, missing) : ex
    end
    args_simulate = [[:(::$Simulate),:_vars, :rng]; map(arg_default, sdef[:args])]
    kwargs_simulate = map(arg_default, sdef[:kwargs])
    def_simulate = combinedef(Dict(:name=>sdef[:name], :args=>args_simulate, :kwargs=>kwargs_simulate, :body=>body_simulate, :whereparams=>sdef[:whereparams]))
    
    # logpdf
    body_logpdf = postwalk(sdef[:body]) do x
        if @capture(x, var_ ~ dist_)
            return :(_logpdf[] += $logpdf($dist, $var); $var)
        elseif @capture(x, var_ ← rhs_)
            return :($var = $rhs)
        elseif !isexpr(x, :block) && @capture(x, (f_(args__; kwargs__) | f_(args__)))
            kwargs = kwargs == nothing ? () : kwargs
            if (f isa Symbol) && !startswith(string(f),".")
                if !(f in maybe_local_var) && isdefined(__module__, f)
                    if is_simpleppl_model(getfield(__module__, f))
                        return :($f($(Logpdf()), _logpdf, $(args...); $(kwargs...)))
                    end
                else
                    return :($is_simpleppl_model($f) ? $f($(Logpdf()), _logpdf, $(args...); $(kwargs...)) : $x)
                end
            end
        end
        return x
    end
    args_logpdf = [[:(::$Logpdf),:_logpdf]; sdef[:args]]
    def_logpdf = combinedef(Dict(:name=>sdef[:name], :args=>args_logpdf, :kwargs=>sdef[:kwargs], :body=>body_logpdf, :whereparams=>sdef[:whereparams]))


    esc(quote
        $def_simulate
        $CMBLensing.is_simpleppl_model(::$model_type) where {$(sdef[:whereparams]...)} = true
        $def_logpdf
    end)
end

simulate(rng::AbstractRNG, model, dist::Sampleable) = rand(rng, dist)

simulate(model, args...; kwargs...) = simulate(Random.default_rng(), model, args...; kwargs...)
function simulate(rng::AbstractRNG, model, args...; kwargs...)
    _vars = OrderedDict()
    model(Simulate(), _vars, rng, args...; kwargs...)
    (;_vars...)
end
function Distributions.loglikelihood(model, args...; kwargs...)
    _logpdf = Ref{Real}(0)
    model(Logpdf(), _logpdf, args...; kwargs...)
    _logpdf[]
end
function Distributions.logpdf(model, args...; kwargs...)
    Distributions.loglikelihood(model, args...; kwargs...) + logprior(model, args...; kwargs...)
end
logprior(model, args...; kwargs...) = 0