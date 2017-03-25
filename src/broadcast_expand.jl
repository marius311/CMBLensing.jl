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
# which is one single broacasted loop and hence no temporary memory is allocated
# and its fast.
#
# eventually, we should be able to do this with @generated functions which can
# know both the types of the arguments and the expression being broacasted.
# unfortunately, in 0.6 you can't call broadcast from a @generated function,
# although it seems like at some point in the future that could be fixed. 
# 
# thus, for now, we do this with macro's and a custom operator, ⨳, and hard code
# a number of different cases. this is kinda ugly and won't work generally since
# macro's know the expression but not the argument types, but worst case
# scenario is that we just don't optimize a particular expression. the point is
# though eventually we should be able to switch to a solution based on
# @generated functions with no change to the scientific part of the code, just
# the underlying plumbing.


"""
Expands 2D matrix/vector multiplication of Fields into a single broadcasted
expression. E.g., if you write, 

    @⨳ A ⨳ b

this is expands to, 

    @. [A[1,1] * b[1] + A[1,2] * b[2], 
        A[2,1] * b[1] + A[2,2] * b[2]]
        
If b is some sub-expression, it will get spliced into the expanded result and
possibly evaluated multiple times, which might be inefficient depending on the
case. You can force allocation of a temporary variable but putting $(...)
around any subexpression, e.g.

    @⨳ A * \$(1+b)
    
which expands to,

    temp = 1 + b
    @⨳ A ⨳ temp
    
where the second line expands as in the first scenario. This is fewer operations
(although more memory usage).

(TODO: nested temp's don't work yet, e.g. A * (1+\$(2*b)) )

Since macro's dont know about the types of the arguments, this is all done
purely syntatically inferred from ⨳'s and transpose operations. The following
meanings are assumed:

* A ⨳ x          - A::Matrix, x::Vector
* x' ⨳ y         - x/y::Vector
* x' ⨳ (y' ⨳ A)' - x/y::Vector, A::Matrix

In particular, this won't work:

    y = x'
    @⨳ y⨳x
    
because the second expression doesn't know y is a row vector; it instead
interprets this as multiplication between matrix y and vector x according to the
above rules. Once we have the solution based on @generated functions, this case
will work though.  
"""
macro ⨳(ex)
    
    # recurse thru turning ∇ᵀ into ∇' 
    # (we use ∇ᵀ instead of ∇' b/c of a syntax highlighting bug in Juno which
    # hopefully gets fixed soon...)
    convert∇(ex) = isexpr(ex) ? (map!(convert∇,ex.args); ex) : (ex==:(∇ᵀ) ? :(∇') : ex) #'
    ex = convert∇(ex)
    
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
    
    function visit(ex)
        if @capture(ex, a_' ⨳ inv(𝕀 + t_*J_) ⨳ c_)
            a,J,c,t = checktemp!.((a,J,c,t))
            quote
                (($a[1]*(1+$t*$J[2,2])*$c[1] + $a[2]*(1+$t*$J[1,1])*$c[2]
                  - $t*($a[1]*$J[2,1]*$c[2] + $a[2]*$J[1,2]*$c[1])) 
                / ((1+$t*$J[1,1])*(1+$t*$J[2,2]) - $t^2*$J[2,1]*$J[1,2]))
            end
        elseif @capture(ex, b_' ⨳ (a_' ⨳ A_)')
            A,a,b = checktemp!.((A,a,b))
            :($b[1]*($a[1]*$A[1,1]+$a[2]*$A[2,1]) + $b[2]*($a[1]*$A[1,2]+$a[2]*$A[2,2]))
        elseif @capture(ex, a_' ⨳ A_ ⨳ b_)
            A,a,b = checktemp!.((A,a,b))
            :(  $a[1]*$A[1,1]*$b[1] + $a[1]*$A[1,2]*$b[2] 
              + $a[2]*$A[2,1]*$b[1] + $a[2]*$A[2,2]*$b[2])
        elseif @capture(ex, a_' ⨳ b_)
            a,b = checktemp!.((a,b))
            :($a[1]*$b[1] + $a[2]*$b[2])
        elseif @capture(ex, A_ ⨳ b_)
            A,b = checktemp!.((A,b))
            :(@SVector [$A[1,1]*$b[1] + $A[1,2]*$b[2],
                        $A[2,1]*$b[1] + $A[2,2]*$b[2]])
        else
            isexpr(ex) ? (map!(visit,ex.args); ex) : esc(ex)
        end
    end
    
    quote
        $temps
        @. $(visit(ex))
    end
    
end
