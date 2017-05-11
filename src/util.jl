export @repeated, nan2zero

import Base: ==
==(x,y,z,ws...) = x==y && ==(y,z,ws...)

"""
Simply replaces every occurence of ∷ with <:Any
"""
macro ∷(ex)
    visit(ex) = isexpr(ex) ? (map!(visit,ex.args,ex.args); ex) : (ex==:(∷) ? :(<:Any) : ex)
    esc(visit(ex))
end


"""
@typeswap f(a::T1,b::T2) = body

is equivalent to

f(a::T1,b::T2) = body
f(a::T2,b::T1) = body

TODO: phase out use of this entirely, it tends to lead to ambiguities....
"""
macro typeswap(ex)
    if @capture ex ((f_(a_::T1_,b_::T2_) = body_) | (function f_(a_::T1_,b_::T2_) body_ end))
        esc(:($f($a::$T1,$b::$T2)=$body; $f($a::$T2,$b::$T1)=$body))
    elseif @capture ex ((f_(::T1_,::T2_) = body_) | (function f_(::T1_,::T2_) body_ end))
        esc(:($f(::$T1,::$T2)=$body; $f(::$T2,::$T1)=$body))
    else
        error("@typeswap couldn't understand function.")
    end
end

nan2zero{T}(x::T) = isnan(x)?zero(T):x

""" Return a tuple with the expression repeated n times """
macro repeated(ex,n)
    :(tuple($(repeated(esc(ex),n)...)))
end

""" 
Pack some variables in a dictionary 

```
> x = 3
> y = 4
> @dictpack x y z=>5
Dict(:x=>3,:y=>4,:z=>5)
```
"""
macro dictpack(exs...)
    kv(ex::Symbol) = :($(QuoteNode(ex))=>$(esc(ex)))
    kv(ex) = isexpr(ex,:call) && ex.args[1]==:(=>) ? :($(QuoteNode(ex.args[2]))=>$(esc(ex.args[3]))) : error()
    :(Dict($((kv(ex) for ex=exs)...)))
end
