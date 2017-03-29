
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
@swappable f(a,b) = body

is equivalent to 

f(a,b) = body
f(b,a) = body

TODO: phase out use of this entirely, it tends to lead to ambiguities....
"""
macro swappable(ex)
    if @capture ex ((f_(a_,b_) = body_) | (function f_(a_,b_) body_ end))
        esc(:($f($a,$b)=$body; $f($b,$a)=$body))
    else
        error("@swappable couldn't understand function.")
    end
end
