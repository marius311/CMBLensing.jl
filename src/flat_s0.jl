
### FlatMap and FlatFourier types
struct FlatMap{P<:Flat,T<:Real,M<:AbstractRank2or3Array{T}} <: Field{Map,S0,P,T}
    Ix :: M
end
struct FlatFourier{P<:Flat,T<:Real,M<:AbstractRank2or3Array{Complex{T}}} <: Field{Fourier,S0,P,Complex{T}}
    Il :: M
end
const FlatS0{P,T,M} = Union{FlatMap{P,T,M},FlatFourier{P,T,M}}

### convenience constructors
for (F, X, T) in [
    (:FlatMap,     :Ix, :T),
    (:FlatFourier, :Il, :(Complex{T})),
]
    doc = """
        # main constructor:
        $F(
            $X::AbstractArray; $(F==:FlatFourier ? "\n        Nside, # required, size of the map in pixels" : "")
            θpix,  # optional, resolution in arcmin (default: 1)
            ∂mode, # optional, fourier∂ or map∂ (default: fourier∂)
        )
        
        # more low-level:
        $F{P}($X::AbstractArray) # specify pixelization P explicilty
        $F{P,T}($X::AbstractArray) # additionally, convert elements to type $T
        $F{P,T,M<:AbstractArray{$T}}($X::M) # specify everything explicilty
        
    Construct a `$F` object. The top form of the constructor is most convenient
    for interactive work, while the others may be more useful for low-level code.
    """
    @eval begin
        @doc $doc $F
        $F{P}($X::M) where {P,T,M<:AbstractRank2or3Array{$T}} = $F{P,T,M}($X)
        $F{P,T}($X::AbstractRank2or3Array) where {P,T} = $F{P}($T.($X))
    end
    T!=:T && @eval $F{P}($X::M) where {P,T,M<:AbstractRank2or3Array{T}} = $F{P,T}($X)
end
FlatMap(Ix::AbstractRank2or3Array; kwargs...) = 
    FlatMap{Flat(Nside=(size(Ix,1)==size(Ix,2) ? size(Ix,1) : size(Ix)[1:2]), D=size(Ix,3); kwargs...)}(Ix)
FlatFourier(Il::AbstractRank2or3Array; Nside, kwargs...) = 
    FlatFourier{Flat(Nside=Nside, D=size(Il,3); kwargs...)}(Il)


### array interface 
# Base.size and Base.lastindex refer to the splayed-out vector representation of the field:
size(f::FlatS0) = (length(firstfield(f)),)
lastindex(f::FlatS0, i::Int) = lastindex(f.Ix, i)
# content_size and content_ndims refer to the actual array holding the field content:
content_size(::Type{<:FlatMap{    <:Flat{N,<:Any,<:Any,D}}}) where {N,D} = (N .* (1,1)         ..., (D==1 ? () : (D,))...)
content_size(::Type{<:FlatFourier{<:Flat{N,<:Any,<:Any,D}}}) where {N,D} = (N .÷ (2,1) .+ (1,0)..., (D==1 ? () : (D,))...)
content_ndims(::Type{<:FlatS0{<:Flat{<:Any,<:Any,<:Any,D}}}) where {D} = D==1 ? 3 : 2
@propagate_inbounds @inline getindex(f::FlatS0, I...) = getindex(firstfield(f), I...)
@propagate_inbounds @inline setindex!(f::FlatS0, X, I...) = (setindex!(firstfield(f), X, I...); f)
adapt_structure(to, f::F) where {P,F<:FlatS0{P}} = basetype(F){P}(adapt(to,firstfield(f)))
adapt_structure(::Type{T}, f::F) where {T<:Union{Float32,Float64},P,F<:FlatS0{P}} = T(f)
function similar(f::F,::Type{T},dims::Dims) where {P,F<:FlatS0{P},T<:Number}
    @assert size(f)==dims "Tried to make a field similar to $F but dims should have been $(size(f)), not $dims."
    basetype(F){P}(similar(firstfield(f),T))
end
copyto!(dst::Field{B,S0,P}, src::Field{B,S0,P}) where {B,P} = (copyto!(firstfield(dst),firstfield(src)); dst)


### broadcasting
struct FlatS0Style{F,M} <: AbstractArrayStyle{1} end
(::Type{FS})(::Val{1}) where {FS<:FlatS0Style} = FS()
(::Type{FS})(::Val{2}) where {FS<:FlatS0Style} = error("Broadcast expression would create a dense Field operator.")
@generated BroadcastStyle(::Type{F}) where {P,T,M,F<:FlatS0{P,T,M}} = FlatS0Style{basetype(F){P},basetype(M)}()
# both orders needed bc of https://github.com/JuliaLang/julia/pull/35948:
BroadcastStyle(S::FlatS0Style{<:FlatS0{Flat{N,θ,∂m,D}}},  ::FlatS0Style{<:FlatS0{Flat{N,θ,∂m,1}}}) where {N,θ,∂m,D} = S
BroadcastStyle( ::FlatS0Style{<:FlatS0{Flat{N,θ,∂m,1}}}, S::FlatS0Style{<:FlatS0{Flat{N,θ,∂m,D}}}) where {N,θ,∂m,D} = S
BroadcastStyle(S::FieldTupleStyle, ::FlatS0Style) = S
BroadcastStyle(S::FieldOrOpArrayStyle, ::FlatS0Style) = S
BroadcastStyle(S1::FlatS0Style{<:Field{B1}}, S2::FlatS0Style{<:Field{B2}}) where {B1,B2} = 
    invalid_broadcast_error(B1,S1,B2,S2)
instantiate(bc::Broadcasted{<:FlatS0Style}) = bc
similar(::Broadcasted{FS}, ::Type{T}) where {T<:Number,FS<:FlatS0Style} = similar(FS,T)
similar(::Type{FlatS0Style{F,M}}, ::Type{T}) where {F<:FlatS0,M,T<:Number} = 
    F(basetype(M){eltype(F{real(T)})}(undef,content_size(F)))
@inline preprocess(dest::F, bc::Broadcasted) where {F<:FlatS0} = 
    Broadcasted{DefaultArrayStyle{content_ndims(F)}}(bc.f, preprocess_args(dest, bc.args), map(OneTo,content_size(F)))
preprocess(dest::F, arg) where {F<:FlatS0} = broadcastable(F, arg)
broadcastable(::Type{<:FlatS0}, f::FlatS0) = firstfield(f)
broadcastable(::Type{<:FlatS0{<:Flat,T}}, r::Real) where {T} = convert(T,r)
broadcastable(::Any, x) = x
@inline function Broadcast.copyto!(dest::FlatS0, bc::Broadcasted{Nothing})
    bc′ = preprocess(dest, bc)
    @simd for I in eachindex(bc′)
        @inbounds dest[I] = bc′[I]
    end
    return dest
end



### basis conversion
Fourier(f::FlatMap{P}) where {P} = FlatFourier{P}(fieldinfo(f).FFT * f.Ix)
Map(f::FlatFourier{P}) where {P} = FlatMap{P}(fieldinfo(f).FFT \ f.Il)

### inplace conversion
Fourier(f′::FlatFourier, f::FlatMap) =  (mul!(f′.Il, fieldinfo(f).FFT, f.Ix); f′)
Map(f′::FlatMap, f::FlatFourier)     = (ldiv!(f′.Ix, fieldinfo(f).FFT, maybecopy(f.Il)); f′)
# need this for FFTW (but not MKL) see https://github.com/JuliaMath/FFTW.jl/issues/158
@static FFTW.fftw_vendor==:fftw ? maybecopy(x) = copy(x) : maybecopy(x) = x

### properties
getproperty(f::FlatS0, s::Symbol) = getproperty(f,Val(s))
getproperty(f::FlatS0, ::Val{s}) where {s} = getfield(f,s)
getproperty(f::FlatS0, ::Val{:I}) = f
function getindex(f::FlatS0, k::Symbol; full_plane=false)
    maybe_unfold = full_plane ? x->unfold(x,fieldinfo(f).Ny) : identity
    @match k begin
        :I  => f
        :Ix => Map(f).Ix
        :Il => maybe_unfold(Fourier(f).Il)
        _   => throw(ArgumentError("Invalid FlatS0 index: $k"))
    end
end

### dot products
# do in Map space for simplicity, and use sum_kbn to reduce roundoff error
dot(a::FlatS0{<:Flat{N,θ}}, b::FlatS0{<:Flat{N,θ}}) where {N,θ} = batch(sum_kbn(Map(a).Ix .* Map(b).Ix, dims=(1,2)))

### isapprox
≈(a::F, b::F) where {P,T,F<:FlatS0{P,T}} = all(.≈(a[:], b[:], atol=sqrt(eps(T)), rtol=sqrt(eps(T))))

### simulation and power spectra
function white_noise(rng::AbstractRNG, ::Type{F}) where {N,P<:Flat{N},T,M,F<:FlatS0{P,T,M}}
    FlatMap{P}(randn!(rng, basetype(M){T}(undef, content_size(FlatMap{P}))))
end
function Cℓ_to_Cov(::Type{P}, ::Type{T}, ::Type{S0}, Cℓ::InterpolatedCℓs; units=fieldinfo(P).Ωpix) where {P,T}
    Diagonal(FlatFourier{P}(Cℓ_to_2D(P,T,Cℓ)) / units)
end
function Cℓ_to_Cov(::Type{P}, ::Type{T}, ::Type{S0}, (Cℓ, ℓedges, θname)::Tuple; units=fieldinfo(P).Ωpix) where {P,T}
    C₀ = Cℓ_to_Cov(P, T, S0, Cℓ, units=units)
    Cbins = Diagonal.(MidPasses(ℓedges) .* [diag(C₀)])
    BinRescaledOp(C₀,Cbins,θname)
end


function cov_to_Cℓ(L::DiagOp{<:FlatS0{P}}; units=fieldinfo(P).Ωpix) where {P}
    ii = sortperm(fieldinfo(L.diag).kmag[:])
    InterpolatedCℓs(fieldinfo(L.diag).kmag[ii], real.(unfold(L.diag.Il, fieldinfo(L.diag).Ny))[ii] * units, concrete=false)
end

function get_Cℓ(f::FlatS0{P}, f2::FlatS0{P}=f; Δℓ=50, ℓedges=0:Δℓ:16000, Cℓfid=ℓ->1, err_estimate=false) where {P}
    @unpack Nx, Ny ,Δx,kmag = fieldinfo(f)
    α = Nx*Ny/Δx^2

    # faster to excise unused parts:
    kmask = (kmag .> minimum(ℓedges)) .&  (kmag .< maximum(ℓedges)) 
    L = Float64.(kmag[kmask])
    CLobs = real.(dot.(unfold(Float64(f)[:Il],Ny)[kmask], unfold(Float64(f2)[:Il],Ny)[kmask])) ./ α
    w = @. nan2zero((2*Cℓfid(L)^2/(2L+1))^-1)
    
    sum_in_ℓbins(x) = fit(Histogram, L, Weights(x), ℓedges).weights

    local A, Cℓ, ℓ, N, Cℓ²
    Threads.@sync begin
        Threads.@spawn A  = sum_in_ℓbins(w)
        Threads.@spawn Cℓ = sum_in_ℓbins(w .* CLobs)
        Threads.@spawn ℓ  = sum_in_ℓbins(w .* L)
        if err_estimate
            Threads.@spawn N   = sum_in_ℓbins(one.(w)) / 2
            Threads.@spawn Cℓ² = sum_in_ℓbins(w .* CLobs.^2)
        end
    end

    if err_estimate
        σℓ  = sqrt.((Cℓ² ./ A .- Cℓ.^2) ./ N)
        InterpolatedCℓs(ℓ./A,  Cℓ./A .± σℓ)
    else
        InterpolatedCℓs(ℓ./A,  Cℓ./A)
    end
end

"""
    ud_grade(f::Field, θnew, mode=:map, deconv_pixwin=true, anti_aliasing=true)

Up- or down-grades field `f` to new resolution `θnew` (only in integer steps).
Two modes are available specified by the `mode` argument: 

* `:map`     — Up/downgrade by replicating/averaging pixels in map-space
* `:fourier` — Up/downgrade by extending/truncating the Fourier grid

For `:map` mode, two additional options are possible. If `deconv_pixwin` is
true, deconvolves the pixel window function from the downgraded map so the
spectrum of the new and old maps are the same. If `anti_aliasing` is true,
filters out frequencies above Nyquist prior to down-sampling. 

"""
function ud_grade(f::FlatS0{P,T,M}, θnew; mode=:map, deconv_pixwin=(mode==:map), anti_aliasing=(mode==:map)) where {T,M,θ,N,∂mode,P<:Flat{N,θ,∂mode}}
    θnew==θ && return f
    (mode in [:map,:fourier]) || throw(ArgumentError("Available modes: [:map,:fourier]"))

    fac = θnew > θ ? θnew÷θ : θ÷θnew
    (round(Int, fac) ≈ fac) || throw(ArgumentError("Can only ud_grade in integer steps"))
    fac = round(Int, fac)
    Nnew = @. round(Int, N * θ ÷ θnew)
    Pnew = Flat(Nside=Nnew, θpix=θnew, ∂mode=∂mode)
    @unpack Δx,kx,ky,Nx,Ny,nyq = fieldinfo(Pnew,T,M)

    if deconv_pixwin
        PWF = @. T((pixwin(θnew,ky[1:end÷2+1])*pixwin(θnew,kx)')/(pixwin(θ,ky[1:end÷2+1])*pixwin(θ,kx)'))
    else
        PWF = 1
    end

    if θnew>θ
        # downgrade
        if anti_aliasing
            AA = Diagonal(FlatFourier{P}(
                ifelse.((abs.(fieldinfo(P,T,M).ky[1:end÷2+1]) .> nyq) .| (abs.(fieldinfo(P,T,M).kx') .> nyq), 0, 1)
            ))
        else
            AA = 1
        end
        if mode==:map
            fnew = FlatMap{Pnew}(mapslices(mean,reshape((AA*f)[:Ix],(fac,Ny,fac,Nx)),dims=(1,3))[1,:,1,:])
            deconv_pixwin ? FlatFourier{Pnew}(fnew[:Il] ./ PWF) : fnew
        else
            @assert fieldinfo(f).Nside isa Int "Downgrading resolution with `mode=:fourier` only implemented for maps where `Nside isa Int`"
            FlatFourier{Pnew}((AA*f)[:Il][1:(Nnew÷2+1), [1:(isodd(Nnew) ? Nnew÷2+1 : Nnew÷2); (end-Nnew÷2+1):end]])
        end
    else
        # upgrade
        @assert fieldinfo(f).Nside isa Int "Upgrading resolution only implemented for maps where `Nside isa Int`"
        if mode==:map
            fnew = FlatMap{Pnew}(permutedims(hvcat(N,(x->fill(x,(fac,fac))).(f[:Ix])...)))
            deconv_pixwin ? FlatFourier{Pnew}(fnew[:Il] .* PWF' .* PWF[1:Nnew÷2+1]) : fnew
        else
            fnew = FlatFourier{Pnew}(zeros(Nnew÷2+1,Nnew))
            setindex!.(Ref(fnew.Il), f[:Il], 1:(N÷2+1), [findfirst(fieldinfo(fnew).k .≈ fieldinfo(f).k[i]) for i=1:N]')
            deconv_pixwin ? fnew * fac^2 : fnew
        end
    end
end
