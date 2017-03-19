using MacroTools

#
# we want to be able to write expressions involving Vector/Matrix{<:Field}, and
# have the result be efficiently broacasted, i.e. I write, 
#
# x'*M*y 
#
# where x/y::Vector{<:Field} and M::Matrix{<:Field}, and this essentially
# evaluates,
#
# @. x[1]*M[1,1]*x[1] + x[2]*M[2,1]*x[1] + x[1]*M[1,2]*x[2] + x[2]*M[2,2]*x[2]
# 
# which is one single broacasted loop and hence no temporary memory is allocated.
#
# eventually, we should be able to do this with @generated functions which can
# know both the types of the arguments and the expression being broacasted.
# unfortunately, in 0.6 you can't call broadcast from a @generated function,
# although it seems like at some point in the future that could be fixed. 
# 
# thus, for now, we do this with macro's, and hard code a number of different
# cases. this is kinda ugly and won't work generally since macro's know the
# expression but not the argument types, but worst case scenario is that we just
# don't optimize a particular expression. eventually we should be able to switch
# to a solution based on @generated functions transparently. 


macro ×(ex)
    
    # todo: recurse thru turning ∇ᵀ into ∇'
    
    # check if ex is of the form $(...) and if so allocate a temporary
    # variable to store the result of (...)
    temps = :(begin end) 
    function checktemp!(ex)
        pushtemp!(ex) = (temp=gensym(); push!(temps.args,:($temp = $(esc(ex)))); temp)
        if isexpr(ex) && ex.head==:$
            pushtemp!(ex.args[1])
        elseif isexpr(ex) && ex.head==:call && isexpr(ex.args[1]) && ex.args[1].head==:$
            pushtemp!(Expr(:call,ex.args[1].args[1],ex.args[2:end]...))
        else
            esc(ex)
        end
    end
    
    if @capture(ex, a_' × inv(𝕀 + t_*J_) × c_)
        a,J,c,t = checktemp!.((a,J,c,t))
        quote
            $temps
            @. (($a[1]*(1+$t*$J[2,2])*$c[1] + $a[2]*(1+$t*$J[1,1])*$c[2]
                 - $t*($a[1]*$J[2,1]*$c[2] + $a[2]*$J[1,2]*$c[1])) 
               / ((1+$t*$J[1,1])*(1+$t*$J[2,2]) - $t^2*$J[2,1]*$J[1,2]))
        end
    elseif @capture(ex, a_' × b_ × c_)
        a,b,c = checktemp!.((a,b,c))
        quote
            $temps
            @. ($a[1]*$b[1,1]*$c[1] + $a[1]*$b[1,2]*$c[2] 
              + $a[2]*$b[2,1]*$c[1] + $a[2]*$b[2,2]*$c[2])
        end
    elseif @capture(ex, a_' × b_)
        a,b = checktemp!((a,b))
        :($temps; @. $a[1]*$b[1] + $a[2]*$b[2])
    end

end
