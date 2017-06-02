#= 

This file defines "FieldNTuples", arbitrary tuples of fields

To keep things type-stable, until a solution to
https://discourse.julialang.org/t/is-there-a-way-to-forward-getindex-for-tuples-in-a-type-stable-way/2889
comes along, we generate parametric structs for each "N" with a loop and @eval.
i.e., the code below generates more-or-less, 

struct Field2Tuple{F1,F2}
    f1::F1
    f2::F2
end

struct Field3Tuple{F1,F2,F3}
    f1::F1
    f2::F2
    f2::F3
end

etc..., along with all of the necessary functions for each. 

Due to this, the code below can be a bit obtuse; you can also take a look at
docs/field_2tuples.jl which is the version explicitly written out for 2-tuples,
which this file is a generalization of. Additionally, you can uncomment the
"println" line below to just directly print the generate code, which is not too
bad to read either. 

=#


export FieldTuple

*(args::Union{_,Symbol,Int}...) where {_<:Symbol} = Symbol(args...)

Ns = [2,3] # which FieldNTuples to generate

for N in Ns
    
    let 
        local f(i) = :f*i
        local F(i) = :F*i
        local B(i) = :B*i
        local (fs, Fs, Bs) = @. f(1:N), F(1:N), B(1:N)
        local (Fas, Fbs) = @. (F(1:N)*:a), F(1:N)*:b
        local (FNT, BNT)= (:Field*N*:Tuple, :Basis*N*:Tuple)

        (q = (quote
        
            abstract type $(:Basis*N*:Tuple){$((:($(B(i))<:Basis) for i=1:N)...)} <: Basis end
            
            # 
            # I really wish I could define this just as 
            # Field2Tuple{F1<:Field{∷,∷,B1},F2<:Field{∷,∷,B2}} <: Field{Pix,Spin,Basis2Tuple{B1,B2}}
            # but this doesn't exist in Julia (yet?), so instead I use this "hack"
            # see also: https://discourse.julialang.org/t/could-julia-have-implicit-type-parameters/2914/5
            # 
            @∷ struct $FNT{$((:($(F(i))<:Field) for i=1:N)...),$(Bs...)} <: Field{Pix,Spin,$(:Basis*N*:Tuple){$(Bs...)}} 
                $((:($(:f*i)::$(:F*i)) for i=1:N)...)
                # todo:
                $FNT($((:($(:f*i)::$(:F*i)) for i=1:N)...)) where {$(Bs...),$((:($(F(i))<:Field{∷,∷,$(B(i))}) for i=1:N)...)} = new{$(Fs...),$(Bs...)}($(fs...))
                $FNT{$(Fs...),$(Bs...)}($((:($(:f*i)::$(:F*i)) for i=1:N)...)) where {$(Bs...),$((:($(F(i))<:Field{∷,∷,$(B(i))}) for i=1:N)...)} = new{$(Fs...),$(Bs...)}($(fs...))
            end
            
            shortname(::Type{<:$FNT{$(Fs...)}}) where {$(Fs...)} = "{$(join(map(shortname,[$(Fs...)]),","))}"
            
            # Field2Tuple's data
            broadcast_length(::Type{<:$FNT{$(Fs...)}}) where {$(Fs...)} = +($((:(broadcast_length($(F(i)))) for i=1:N)...))
            broadcast_data(::Type{$FNT{$(Fs...)}}, f::$FNT) where {$(Fs...)}  = tuple($((Expr(:(...),:(broadcast_data($(F(i)),f.$(f(i))))) for i=1:N)...))
            # How to broadcast other things as a Field2Tuple
            broadcast_data(::Type{$FNT{$(Fs...)}}, f::Union{Field,LinOp}) where {$(Fs...)} = tuple($((Expr(:(...),:(broadcast_data($(F(i)),f))) for i=1:N)...))
            # needed for ambiguity (todo: get rid of needing this...)
            broadcast_data(::Type{$FNT{$(Fs...)}}, op::FullDiagOp) where {$(Fs...)} = broadcast_data($FNT{$(Fs...)},op.f)


            # the final data type when broadcasting things with Field2Tuple
            containertype(::$FNT{$(Fs...)}) where {$(Fs...)} = $FNT{$((:(containertype($(F(i)))) for i=1:N)...)}
            containertype(::Type{<:$FNT{$(Fs...)}}) where {$(Fs...)} = $FNT{$(Fs...)}
            function promote_containertype(::Type{$FNT{$(Fas...)}}, ::Type{$FNT{$(Fbs...)}}) where {$(Fas...),$(Fbs...)}
                $FNT{$((:(promote_containertype($(F(i)*:a), $(F(i)*:b))) for i=1:N)...)}
            end
            @typeswap function promote_containertype{F<:Field,$(Fs...)}(::Type{F},::Type{$FNT{$(Fs...)}})
                $FNT{$((:(promote_containertype(F,$(F(i)))) for i=1:N)...)}
            end
            @typeswap *(a::Field,b::$FNT) = a.*b
            *(a::$FNT,b::$FNT) = a.*b 

            # Reconstruct FieldNTuple from broadcasted data
            function (::Type{<:$FNT{$(Fs...)}})(args...) where {$(Fs...)}
                lens = broadcast_length.([$(Fs...)])
                starts = [1; cumsum(lens)[1:end-1]+1]
                ends = starts + lens - 1
                $FNT($((:($(F(i))(args[starts[$i]:ends[$i]]...)) for i=1:N)...))
            end

            # promotion / conversion
            function promote_rule(::Type{<:$FNT{$(Fas...)}},::Type{<:$FNT{$(Fbs...)}}) where {$(Fas...),$(Fbs...)} 
                $FNT{$((:(promote_type($(F(i)*:a), $(F(i)*:b))) for i=1:N)...)}
            end
            (::Type{<:$FNT{$(Fs...)}})(f::$FNT) where {$(Fs...)} = $FNT($((:($(F(i))(f.$(f(i)))) for i=1:N)...))
            convert(::Type{<:$FNT{$(Fs...)}},f::$FNT) where {$(Fs...)} = $FNT($((:($(F(i))(f.$(f(i)))) for i=1:N)...))

            # Basis conversions
            (::Type{$BNT{$(Bs...)}})(f::$FNT) where {$(Bs...)} = $FNT($((:($(B(i))(f.$(f(i)))) for i=1:N)...))
            (::Type{B})(f::$FNT) where {B<:Basis} = $FNT($((:(B(f.$(f(i)))) for i=1:N)...))
            (::Type{B})(f::$FNT) where {B<:Basislike} = $FNT($((:(B(f.$(f(i)))) for i=1:N)...)) #needed for ambiguity

            # dot product
            dot(a::$FNT, b::$FNT) = +($((:(dot(a.$(f(i)),b.$(f(i)))) for i=1:N)...))

            # transpose multiplication (todo: optimize...)
            Ac_mul_B(a::$FNT,b::$FNT) = +($((:(Ac_mul_B(a.$(f(i)),b.$(f(i)))) for i=1:N)...))

            # for simulating
            white_noise(::Type{<:$FNT{$(Fs...)}}) where {$(Fs...)} = $FNT($((:(white_noise($(F(i)))) for i=1:N)...))

            zero(::Type{<:$FNT{$(Fs...)}}) where {$(Fs...)} = $FNT($((:(zero($(F(i)))) for i=1:N)...))

            eltype(::Type{<:$FNT{$(Fs...)}}) where {$(Fs...)} = promote_type($((:(eltype($(F(i)))) for i=1:N)...))

            # vector conversion
            getindex(f::$FNT,::Colon) = vcat($((:(f.$(f(i))[:]) for i=1:N)...))
            function fromvec(::Type{F}, vec::AbstractVector) where {$(Fs...),F<:$FNT{$(Fs...)}}
                lens = length.([$(Fs...)])
                starts = [1; cumsum(lens)[1:end-1]+1]
                ends = starts + lens - 1
                F( $((:(fromvec($(F(i)),(@view vec[starts[$i]:ends[$i]]))) for i=1:N)...) )
            end
            length(::Type{<:$FNT{$(Fs...)}}) where {$(Fs...)} = +($((:(length($(F(i)))) for i=1:N)...))
            
            # makes them iterable to allow unpacking, splatting, etc....
            # todo: i think for some reason only the first item is type stable?
            start(f::$FNT) = Val{1}
            next{i}(f::$FNT,::Type{Val{i}}) = getfield(f,i), Val{i+1}
            done{i}(::$FNT,::Type{Val{i}}) = i>$N


        end)) # |> println    # ...uncomment to print generated code
        eval(q)
        
    end
    
end

# containertype(::Type{T}) as opposed to containertype(::T) is only used from
# this file and only because of the "hack" mentioned above, hence we need this
# for other fields. hopefully we eventually can get rid of this
containertype(::Type{F}) where {F<:Field} = F
    
# convenience constructors, FieldTuple(...)
@eval const FieldTupleType = Union{$((:Field*N*:Tuple for N=Ns)...)}
for N in Ns
    @eval FieldTuple($((:f*i for i=1:N)...)) = $(:Field*N*:Tuple)($((:f*i for i=1:N)...))
end

# propagate pixstd (also some minor convenience stuff so it plays nice ODE.jl)
pixstd(f::FieldTupleType) = mean(pixstd.(fieldvalues(f)))
pixstd(arr::AbstractArray{<:Field}) = mean(pixstd.(arr))
pixstd(x,::Int) = pixstd(x)
    
    
# allows some super simple convenience stuff with normal tuples of Fields (but
# not as powerful as FieldTuples)
(::Type{Tuple{F1,F2}})(fs::NTuple{2,Field}) where {F1,F2} = (F1(fs[1]),F2(fs[2]))
(::Type{Tuple{F1,F2,F3}})(fs::NTuple{3,Field}) where {F1,F2,F3} = (F1(fs[1]),F2(fs[2]),F3(fs[3]))
dot(a::NTuple{N,Field},b::NTuple{N,Field}) where N = sum(a[i]⋅b[i] for i=1:N)

# warning: not type-stable and basically can't be without changes to Julia 
getindex(f::FieldTupleType, i::Int) = (f...)[i]
