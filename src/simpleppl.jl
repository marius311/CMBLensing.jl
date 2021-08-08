
###
# TODO: 
# * [ ] API
#   * [ ] handle ds object
# * [ ] normalization terms
# * [ ] mixing
# * [x] _logpdf initial type?


struct SimplePPLModel{S,L}
    simulate! :: S
    logpdf! :: L
end

_simulate!(_vars, rng, m::SimplePPLModel, args...; kwargs...) = m.simulate!(_vars, rng, args...; kwargs...)
_simulate!(_vars, rng, m,                 args...; kwargs...) = m(args...; kwargs...)

_logpdf!(_logpdf, m::SimplePPLModel, args...; kwargs...) = m.logpdf!(_logpdf, args...; kwargs...)
_logpdf!(_logpdf, m,                 args...; kwargs...) = m(args...; kwargs...)


macro fwdmodel(def)
    sdef = splitdef(def)

    # simulate
    body_simulate = postwalk(sdef[:body]) do x
        if @capture(x, var_ ~ dist_)
            quote
                if ismissing($var)
                    _vars[$(QuoteNode(var))] = $var = rand(rng, $dist)
                else
                    _vars[$(QuoteNode(var))] = $var
                end
            end
        elseif @capture(x, (f_(args__; kwargs__) | f_(args__))) && !(isdefined(__module__, f) && (getfield(__module__, f) isa Base.Callable))
            if kwargs == nothing
                kwargs = ()
            end
            :($_simulate!(_vars, rng, $f, $(args...); $(kwargs...)))
        else
            x
        end
    end
    args_simulate = [[:_vars, :rng]; map(x -> Expr(:kw, x, missing), sdef[:args])]
    def_simulate = combinedef(Dict(:args=>args_simulate, :kwargs=>sdef[:kwargs], :body=>body_simulate, :whereparams=>sdef[:whereparams]))
    
    # logpdf
    body_logpdf = postwalk(sdef[:body]) do x
        if @capture(x, var_ ~ dist_)
            quote
                _logpdf[] += logpdf($dist, $var)
                $var
            end
        elseif @capture(x, (f_(args__; kwargs__) | f_(args__))) && !(isdefined(__module__, f) && (getfield(__module__, f) isa Base.Callable))
            if kwargs == nothing
                kwargs = ()
            end
            :($_logpdf!(_logpdf, $f, $(args...); $(kwargs...)))
        else
            x
        end
    end
    args_logpdf = [[:_logpdf]; sdef[:args]]
    def_logpdf = combinedef(Dict(:args=>args_logpdf, :kwargs=>sdef[:kwargs], :body=>body_logpdf, :whereparams=>sdef[:whereparams]))


    esc(quote
        $(sdef[:name]) = $SimplePPLModel(
            $def_simulate,
            $def_logpdf
        )
    end)
end

simulate(m::SimplePPLModel, args...; kwargs...) = simulate(Random.default_rng(), m, args...; kwargs...)
function simulate(rng::AbstractRNG, m::SimplePPLModel, args...; kwargs...)
    _vars = Dict()
    retval = _simulate!(_vars, rng, m, args...; kwargs...)
    retval, (;_vars...)
end
function Distributions.logpdf(m::SimplePPLModel, args...; kwargs...)
    _logpdf = Ref{Real}(0)
    _logpdf!(_logpdf, m, args...; kwargs...)
    _logpdf[]
end
