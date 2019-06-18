# remember: Fields are vectors and technically have linear get/setindex!, but broadcasting is
# overloaded to happen over (atm) 2D matrices...
# 
# 
# * everything is (i,j,spin,tuple) arrays
# 
# * s0 and s2 are (i,j) and (i,j,spin) arrays, tuples are true tuples
# 
# * s0 is (i,j) arrays, s2 and s02 are tuples


# size(ft::FieldTuple) = (sum(map(length,ft.fs)),)
# @propagate_inbounds @inline getindex(ft::FieldTuple, i, j, t) = ft.fs[t][i,j]
# @propagate_inbounds @inline setindex!(ft::FieldTuple, X, i, j, t) = ft.fs[t][i,j] = X

##

struct FieldTuple{FS<:NamedTuple,B,S,P,T} <: Field{B,S,P,T}
    fs::FS
end


# printing
Base.summary(io::IO, ft::FieldTuple{<:NamedTuple{Names},B,S,P,T}) where {Names,B,S,P,T}  = 
    print(io, "$(sum(map(length,values(ft.fs))))-element $(Names) FieldTuple{$(B.name.name),$(S.name.name),$(P.name.name),$T}")
    
function Base.show(io::IO, ::MIME"text/plain", ft::FieldTuple{<:NamedTuple{Names},B,S,P}) where {Names,B,S,P}
    Base.summary(io, ft)
    println(io, ":")
    for (k,f) in pairs(ft.fs)
        print(io, k, " = ")
        summary(IOContext(io, :limit=>true), f)
        Base.show_vector(io, f)
        println(io)
    end
end



## broadcasting 

using Base.Broadcast: Style, broadcasted, Broadcasted, AbstractArrayStyle, DefaultArrayStyle
using Base: @propagate_inbounds

@propagate_inbounds @inline Base.getindex(f::FieldTuple, i) = getindex(f.fs, i)
Base.broadcastable(f::FieldTuple) = f
Base.BroadcastStyle(::Type{<:FieldTuple}) = Style{FieldTuple}()
Base.BroadcastStyle(::Style{FieldTuple}, ::DefaultArrayStyle{0}) = Style{FieldTuple}()
Base.BroadcastStyle(::Style{FieldTuple}, ::Style{Tuple}) = Style{FieldTuple}()
Broadcast.instantiate(bc::Broadcasted{<:Style{FieldTuple}}) = bc

broadcast_data(f::FieldTuple) = values(f.fs)

@inline function Base.copy(bc::Broadcasted{Style{FieldTuple}})
    bc′ = Broadcast.flatten(bc)
    bc″ = Broadcasted{Style{Tuple}}((args...)->broadcast(bc′.f,args...), map(broadcast_data,bc′.args))
    copy(bc″)
end

##


const FlatQUMap{P,T,M}     = FieldTuple{NamedTuple{(:Q,:U),   NTuple{2,FlatMap{P,T,M}}},     QUMap,     S2,  P, T}
FlatQUMap(Qx, Ux; θpix=θpix₀, ∂mode=fourier∂) = FlatQUMap{Flat{size(Qx,2),θpix,∂mode}}(Qx, Ux)
FlatQUMap{P}(Qx::M, Ux::M) where {P,T,M<:AbstractMatrix{T}} = FlatQUMap{P,T,M}((Q=FlatMap{P,T,M}(Qx), U=FlatMap{P,T,M}(Ux)))

##
# 
# f  = FlatMap(rand(128,128))
# ft = FlatQUMap(rand(128,128),rand(128,128))
# 
# show(stdout, MIME("text/plain"), ft)
# println(ft)
# 
# 
# 
# 
# Base.getproperty(f::Field, k) = getproperty(f, Val(k))
# Base.getproperty(f::Field, ::Val{k}) where {k} = getfield(f, k)
# 
# Base.propertynames(f::FlatS2QUMap) = (:Q,:U,:Qx,:Ux)
# Base.getproperty(f::FlatS2QUMap, ::Val{:Qx}) = f.Q.Ix
# Base.getproperty(f::FlatS2QUMap, ::Val{:Ux}) = f.U.Ix
# 
# 
# 
# 
# const FlatQUFourier{P,T,M} = FieldTuple{NamedTuple{(:Q,:U),   NTuple{2,FlatFourier{P,T,M}}}, QUFourier, S2,  P, T}
# const FlatIQUMap{P,T,M}    = FieldTuple{NamedTuple{(:I,:Q,:U),NTuple{3,FlatMap{P,T,M}}},     QUMap      S02, P, T}
# 
