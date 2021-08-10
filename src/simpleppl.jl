
struct Simulate end
struct Logpdf end

is_simpleppl_model(f) = false

macro fwdmodel(def)
    sdef = splitdef(def)

    model_type = @capture(sdef[:name], (_::T_)) ? T : :(typeof($(sdef[:name])))
    maybe_local_var = [] # only "maybe" b/c the simple thing here is too conservative c.f. inner functions / let blocks
    @capture(sdef[:name], (var_::_)) && push!(maybe_local_var, var)
       
    give_missing_default_value(ex) = isexpr(ex, :kw) ? ex : Expr(:kw, ex, missing)

    # simulate
    body_simulate = postwalk(sdef[:body]) do x
        if @capture(x, ((vars__,) | var_) = rhs_)
            vars != nothing ? append!(maybe_local_var, vars) : push!(maybe_local_var, var)
        elseif @capture(x, var_ ~ dist_)
            return :(ismissing($var) ? (_vars[$(QuoteNode(var))] = $var = rand(rng, $dist)) : (_vars[$(QuoteNode(var))] = $var))
        elseif !isexpr(x, :block) && @capture(x, (f_(args__; kwargs__) | f_(args__)))
            kwargs = kwargs == nothing ? () : kwargs
            if !(f in maybe_local_var) && isdefined(__module__, f)
                if is_simpleppl_model(getfield(__module__, f))
                    return :($f($(Simulate()), _vars, rng, $(args...); $(kwargs...)))
                end
            else
                return :($is_simpleppl_model($f) ? $f($(Simulate()), _vars, rng, $(args...); $(kwargs...)) : $x)
            end
        end
        return x
    end
    args_simulate = [[:(::$Simulate),:_vars, :rng]; map(give_missing_default_value, sdef[:args])]
    kwargs_simulate = map(give_missing_default_value, sdef[:kwargs])
    def_simulate = combinedef(Dict(:name=>sdef[:name], :args=>args_simulate, :kwargs=>kwargs_simulate, :body=>body_simulate, :whereparams=>sdef[:whereparams]))
    
    # logpdf
    body_logpdf = postwalk(sdef[:body]) do x
        if @capture(x, var_ ~ dist_)
            return :(_logpdf[] += logpdf($dist, $var); $var)
        elseif !isexpr(x, :block) && @capture(x, (f_(args__; kwargs__) | f_(args__)))
            kwargs = kwargs == nothing ? () : kwargs
            if !(f in maybe_local_var) && isdefined(__module__, f)
                if is_simpleppl_model(getfield(__module__, f))
                    return :($f($(Logpdf()), _logpdf, $(args...); $(kwargs...)))
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
        @eval $CMBLensing.is_simpleppl_model(::$(Expr(:$, model_type))) = true
        $def_logpdf
    end)
end

simulate(model, args...; kwargs...) = simulate(Random.default_rng(), model, args...; kwargs...)
function simulate(rng::AbstractRNG, model, args...; kwargs...)
    _vars = Dict()
    retval = model(Simulate(), _vars, rng, args...; kwargs...)
    retval, (;_vars...)
end
function Distributions.logpdf(model, args...; kwargs...)
    _logpdf = Ref{Real}(0)
    model(Logpdf(), _logpdf, args...; kwargs...)
    _logpdf[]
end
