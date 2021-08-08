
###
# TODO: 
# * [ ] API
#   * [ ] handle ds object
# * [ ] normalization terms
# * [ ] mixing
# * [ ] Union{Missing, T} for type args
# * [x] _logpdf initial type?

struct Simulate end
struct Logpdf end

is_simpleppl_model(f) = false

macro fwdmodel(def)
    sdef = splitdef(def)

    # simulate
    body_simulate = postwalk(sdef[:body]) do x
        if @capture(x, var_ ~ dist_)
            return :(ismissing($var) ? (_vars[$(QuoteNode(var))] = $var = rand(rng, $dist)) : (_vars[$(QuoteNode(var))] = $var))
        elseif @capture(x, (f_(args__; kwargs__) | f_(args__)))
            kwargs = kwargs == nothing ? () : kwargs
            if isdefined(__module__, f)
                if is_simpleppl_model(getfield(__module__, f))
                    return :($f($(Simulate()),_vars, rng, $(args...); $(kwargs...)))
                end
            else
                return :($is_simpleppl_model($f) ? $f($(Simulate()), _vars, rng, $(args...); $(kwargs...)) : $x)
            end
        end
        return x
    end
    args_simulate = [[:(::$Simulate),:_vars, :rng]; map(x -> Expr(:kw, x, missing), sdef[:args])]
    def_simulate = combinedef(Dict(:name=>sdef[:name], :args=>args_simulate, :kwargs=>sdef[:kwargs], :body=>body_simulate, :whereparams=>sdef[:whereparams]))
    
    # logpdf
    body_logpdf = postwalk(sdef[:body]) do x
        if @capture(x, var_ ~ dist_)
            return :(_logpdf[] += logpdf($dist, $var); $var)
        elseif @capture(x, (f_(args__; kwargs__) | f_(args__)))
            kwargs = kwargs == nothing ? () : kwargs
            if isdefined(__module__, f)
                if is_simpleppl_model(getfield(__module__, f))
                    return :($f($(Logpdf()),_logpdf, $(args...); $(kwargs...)))
                end
            else
                return :($is_simpleppl_model($f) ? $f($(Logpdf()), _logpdf, $(args...); $(kwargs...)) : $x)
            end
        end
        return x
    end
    args_logpdf = [[:(::$Logpdf),:_logpdf]; sdef[:args]]
    def_logpdf = combinedef(Dict(:name=>sdef[:name], :args=>args_logpdf, :kwargs=>sdef[:kwargs], :body=>body_logpdf, :whereparams=>sdef[:whereparams]))


    esc(quote
        $def_simulate
        @eval $CMBLensing.is_simpleppl_model(::typeof($(Expr(:$, sdef[:name])))) = true
        $def_logpdf
    end)
end

simulate(model, args...; kwargs...) = simulate(Random.default_rng(), model, args...; kwargs...)
function simulate(rng::AbstractRNG, model, args...; kwargs...)
    _vars = Dict()
    retval = model(Simulate(),_vars, rng, args...; kwargs...)
    retval, (;_vars...)
end
function Distributions.logpdf(model, args...; kwargs...)
    _logpdf = Ref{Real}(0)
    model(Logpdf(),_logpdf, args...; kwargs...)
    _logpdf[]
end
