# Type defs
# ================================================

struct ProjEquiRect{T} <: CartesianProj

    Ny          :: Int
    Nx          :: Int
    θspan       :: Tuple{Float64,Float64}
    φspan       :: Tuple{Float64,Float64}
    θ           :: Vector{Float64} 
    φ           :: Vector{Float64} 
    θ∂          :: Vector{Float64} 
    φ∂          :: Vector{Float64} 
    Ω           :: Vector{Float64} 
    
    storage

end

struct BlockDiagEquiRect{B<:Basis, P<:ProjEquiRect, T, A<:AbstractArray{T}}  <: ImplicitOp{T}
    blocks :: A
    ## blocks_sqrt :: Ref{A} # lazily computed/saved sqrt of operator
    proj :: P
end

struct AzFourier <: S0Basis end
const  QUAzFourier = Basis2Prod{    𝐐𝐔, AzFourier }
const IQUAzFourier = Basis3Prod{ 𝐈, 𝐐𝐔, AzFourier }

# Type Alias
# ================================================

make_field_aliases(
    "EquiRect",  ProjEquiRect, 
    extra_aliases=OrderedDict(
        "AzFourier"    => AzFourier,
        "QUAzFourier"  => QUAzFourier,
        "IQUAzFourier" => IQUAzFourier,
    ),
)

typealias_def(::Type{<:ProjEquiRect{T}}) where {T} = "ProjEquiRect{$T}"

typealias_def(::Type{F}) where {B,M<:ProjEquiRect,T,A,F<:EquiRectField{B,M,T,A}} = "EquiRect$(typealias(B)){$(typealias(A))}"

# Proj 
# ================================================

function θ_healpix_j_Nside(j_Nside) 
    0 < j_Nside < 1  ? acos(1-abs2(j_Nside)/3)      :
    1 ≤ j_Nside ≤ 3  ? acos(2*(2-j_Nside)/3)        :
    3 < j_Nside < 4  ? acos(-(1-abs2(4-j_Nside)/3)) : 
    error("argument ∉ (0,4)")
end

θ_healpix(Nside) = θ_healpix_j_Nside.((1:4Nside-1)/Nside)

θ_equicosθ(N)    = acos.( ((N-1):-1:-(N-1))/N )

θ_equiθ(N)       = π*(1:N-1)/N

function θ_grid(;θspan::Tuple{T,T}, N::Int, type=:equiθ) where T<:Real
    @assert N > 0
    @assert 0 < θspan[1] < θspan[2] < π

    # θgrid′ is the full grid from 0 to π
    if type==:equiθ
        θgrid′ = θ_equiθ(N)
    elseif type==:equicosθ
        θgrid′ = θ_equicosθ(N)
    elseif type==:healpix
        θgrid′ = θ_healpix(N)
    else
        error("`type` is not valid. Options include `:equiθ`, `:equicosθ` or `:healpix`")
    end 

    # θgrid″ subsets θgrid′ to be within θspan
    # δ½south″ and δ½north″ are the arclength midpoints to the adjacent pixel
    θgrid″   = θgrid′[θspan[1] .≤ θgrid′ .≤ θspan[2]]
    δ½south″ = (circshift(θgrid″,-1)  .- θgrid″) ./ 2
    δ½north″ = (θgrid″ .- circshift(θgrid″,1)) ./ 2   
    
    # now restrict to the interior of the range of θgrid″
    θ       = θgrid″[2:end-1]
    δ½south = δ½south″[2:end-1]
    δ½north = δ½north″[2:end-1]

    # These are the pixel boundaries along polar
    # so length(θ∂) == length(θ)+1
    θ∂ = vcat(θ[1] .- δ½north[1], θ .+ δ½south)

    θ, θ∂
end 

# `φ_grid` Slated for removal or upgraded to include CirculantCov methods 
# that allow φspans of the form `(5.3,1.1)` and `(1.1,5.3)`, the latter 
# denoting the long way around the observational sphere. 
#
# function φ_grid(;φspan::Tuple{T,T}, N::Int) where T<:Real
#     @assert N > 0
#     # TODO: relax this condition ...
#     @assert 0 <= φspan[1] < φspan[2] <= 2π 
#     φ∂    = collect(φspan[1] .+ (φspan[2] - φspan[1])*(0:N)/N)
#     Δφ    = φ∂[2] - φ∂[1]
#     φ     = φ∂[1:end-1] .+ Δφ/2
#     φ, φ∂
# end

@memoize function ProjEquiRect(θ, φ, θ∂, φ∂, ::Type{T}, storage) where {T}
    
    Ny, Nx = length(θ), length(φ)
    θspan = (θ∂[1], θ∂[end])
    φspan = (φ∂[1], φ∂[end])
    Ω  = (φ∂[2] .- φ∂[1]) .* diff(.- cos.(θ∂))

    ProjEquiRect{T}(Ny, Nx, θspan, φspan, θ, φ, θ∂, φ∂, Ω, storage)

end

function ProjEquiRect(; T=Float32, storage=Array, kwargs...)

    arg_error() = error("Constructor takes either (θ, φ, θ∂, φ∂) or (Ny, Nx, θspan, φspan) keyword arguments.")
    
    if all(haskey.(Ref(kwargs), (:θ, :φ, :θ∂, :φ∂)))
        !any(haskey.(Ref(kwargs), (:Ny, :Nx, :θspan, :φspan))) || arg_error()
        @unpack (θ, φ, θ∂, φ∂) = kwargs
    elseif all(haskey.(Ref(kwargs), (:Ny, :Nx, :θspan, :φspan)))
        !all(haskey.(Ref(kwargs), (:θ, :φ, :θ∂, :φ∂))) || arg_error()
        @unpack (Ny, Nx, θspan, φspan) = kwargs
        # the convention for Circulant Cov is that φ ∈ (0,2π] 
        φspan′ = ( @ondemand(CirculantCov.in_0_2π)(φspan[1]), @ondemand(CirculantCov.in_0_2π)(φspan[2]) )
        φ  = @ondemand(CirculantCov.fraccircle)(φspan′[1], φspan′[2], Nx)
        Δφ = @ondemand(CirculantCov.counterclock_Δφ)(φ[1], φ[2])
        φ∂ = vcat(φ, @ondemand(CirculantCov.in_0_2π)(φ[end] + Δφ))
        θ, θ∂ = θ_grid(; θspan, N=Ny, type=:equiθ)
        @show θ
    else
        arg_error()
    end

    ProjEquiRect(θ, φ, θ∂, φ∂, real_type(T), storage)

end



# Field Basis
# ================================================
# CirculantCov: βcovSpin2, βcovSpin0, geoβ,
#multPP̄, multPP, periodize, Jperm # https://github.com/EthanAnderes/CirculantCov.jl

# @init @require CirculantCov="edf8e0bb-e88b-4581-a03e-dda99a63c493" begin
# 
# 
# end

"""
From CirculantCov="edf8e0bb-e88b-4581-a03e-dda99a63c493"...
Jperm(ℓ::Int, n::Int) return the column number in the J matrix U^2
where U is unitary FFT. The J matrix looks like this:

|1   0|
|  / 1|
| / / |
|0 1  |

"""
function Jperm end

function Jperm(ℓ::Int, n::Int)
    @assert 1 <= ℓ <= n
    ℓ==1 ? 1 : n - ℓ + 2
end

# AzFourier <-> Map
function AzFourier(f::EquiRectMap)
    nφ = f.Nx
    EquiRectAzFourier(m_rfft(f.arr, 2) ./ √nφ, f.metadata)
end

function Map(f::EquiRectAzFourier)
    nφ = f.Nx
    EquiRectMap(m_irfft(f.arr, nφ, 2) .* √nφ, f.metadata)
end

# QUAzFourier <-> QUMap
function QUAzFourier(f::EquiRectQUMap)
    nθ, nφ = f.Ny, f.Nx
    Uf = m_fft(f.arr, 2) ./ √nφ
    f▫ = similar(Uf, 2nθ, nφ÷2+1)
    for ℓ = 1:nφ÷2+1
        if (ℓ==1) | ((ℓ==nφ÷2+1) & iseven(nφ))
            f▫[1:nθ, ℓ]     .= Uf[:,ℓ]
            f▫[nθ+1:2nθ, ℓ] .= conj.(Uf[:,ℓ])
        else
            f▫[1:nθ, ℓ]     .= Uf[:,ℓ]
            f▫[nθ+1:2nθ, ℓ] .= conj.(Uf[:,Jperm(ℓ,nφ)])
        end
    end
    EquiRectQUAzFourier(f▫, f.metadata)
end

function QUMap(f::EquiRectQUAzFourier)
    nθₓ2, nφ½₊1 = size(f.arr)
    nθ, nφ = f.Ny, f.Nx
    @assert nφ½₊1 == nφ÷2+1
    @assert 2nθ   == nθₓ2

    pθk = similar(f.arr, nθ, nφ)
    for ℓ = 1:nφ½₊1
        if (ℓ==1) | ((ℓ==nφ½₊1) & iseven(nφ))
            pθk[:,ℓ] .= f.arr[1:nθ,ℓ]
        else
            pθk[:,ℓ]  .= f.arr[1:nθ,ℓ]
            pθk[:,Jperm(ℓ,nφ)] .= conj.(f.arr[nθ+1:2nθ,ℓ])
        end
    end
    EquiRectQUMap(m_ifft(pθk, 2) .* √nφ, f.metadata)
end

Base.getindex(f::EquiRectS0, ::typeof(!)) = AzFourier(f).arr
Base.getindex(f::EquiRectS2, ::typeof(!)) = QUAzFourier(f).arr

Base.getindex(f::EquiRectS0, ::Colon) = Map(f).arr
Base.getindex(f::EquiRectS2, ::Colon) = QUMap(f).arr

function Base.summary(io::IO, f::EquiRectField)
    @unpack Ny,Nx,Nbatch = f
    print(io, "$(length(f))-element $Ny×$Nx$(Nbatch==1 ? "" : "(×$Nbatch)")-pixel ")
    Base.showarg(io, f, true)
end

function Base.summary(io::IO, f::EquiRectAzFourier)
    @unpack Ny,Nx,Nbatch = f
    print(io, "$(length(f))-element $Ny×$(Nx÷2+1)$(Nbatch==1 ? "" : "(×$Nbatch)")-pixel ")
    Base.showarg(io, f, true)
end

function Base.summary(io::IO, f::EquiRectQUAzFourier)
    @unpack Ny,Nx,Nbatch = f
    print(io, "$(length(f))-element $(2Ny)×$(Nx÷2+1)$(Nbatch==1 ? "" : "(×$Nbatch)")-pixel ")
    Base.showarg(io, f, true)
end

# block-diagonal operator
# ================================================

# ## Constructors

function BlockDiagEquiRect{B}(block_matrix::A, proj::P) where {B<:Basis, P<:ProjEquiRect, T, A<:AbstractArray{T}}
    BlockDiagEquiRect{B,P,T,A}(block_matrix, proj)
end

# Allows construction by a vector of blocks
function BlockDiagEquiRect{B}(vector_of_blocks::Vector{A}, proj::P) where {B<:Basis, P<:ProjEquiRect, T, A<:AbstractMatrix{T}}
    block_matrix = Array{T}(undef, size(vector_of_blocks[1])..., length(vector_of_blocks))
    for b in eachindex(vector_of_blocks)
        block_matrix[:,:,b] .= vector_of_blocks[b]
    end
    BlockDiagEquiRect{B}(block_matrix, proj)
end

# ## Linear Algebra basics 

*(M::BlockDiagEquiRect{B}, f::EquiRectField) where {B<:Basis} = M * B(f)

function *(M::BlockDiagEquiRect{AzFourier}, f::EquiRectAzFourier)
    promote_metadata_strict(M.proj, f.proj) # ensure same projection
    EquiRectAzFourier(@tullio(Bf[p,iₘ] := M.blocks[p,q,iₘ] * f.arr[q,iₘ]), f.metadata)
end

function *(M::BlockDiagEquiRect{QUAzFourier}, f::EquiRectQUAzFourier)
    promote_metadata_strict(M.proj, f.proj) # ensure same projection
    EquiRectQUAzFourier(@tullio(Bf[p,iₘ] := M.blocks[p,q,iₘ] * f.arr[q,iₘ]), f.metadata)
end

# TODO: Figure out how reduce duplication so I can define methods like this ...

# function *(M::BlockDiagEquiRect{T}, f::EquiRect{T}) where {T<:Az}
# ...

# ## mapblocks for fun.(Mblocks,eachcol(f))

function mapblocks(fun::Function, M::BlockDiagEquiRect{B}, f::EquiRectField) where {B<:Basis} 
    mapblocks(fun, M, B(f))
end

function mapblocks(fun::Function, M::BlockDiagEquiRect{QUAzFourier}, f::EquiRectQUAzFourier)
    promote_metadata_strict(M.proj, f.proj) # ensure same projection
    Mfarr = similar(f.arr)
    for i ∈ axes(M.blocks,3)
        Mfarr[:,i] = fun(M.blocks[:,:,i], f.arr[:,i])
    end
    EquiRectQUAzFourier(Mfarr, f.metadata)
end 

function mapblocks(fun::Function, M::BlockDiagEquiRect{AzFourier}, f::EquiRectAzFourier)
    promote_metadata_strict(M.proj, f.proj) # ensure same projection
    Mfarr = similar(f.arr)
    for i ∈ axes(M.blocks,3)
        Mfarr[:,i] = fun(M.blocks[:,:,i], f.arr[:,i])
    end
    EquiRectAzFourier(Mfarr, f.metadata)
end 

# ## mapblocks for fun.(Mblocks...)

function mapblocks(fun::Function, Ms::BlockDiagEquiRect{B}...) where {B<:Basis}
    map(M->promote_metadata_strict(M.proj, Ms[1].proj), Ms) 
    BlockDiagEquiRect{B}(
        map(i->fun(getindex.(getproperty.(Ms,:blocks),:,:,i)...), axes(Ms[1].blocks,3)),
        Ms[1].proj,
    )
end 

# ## Other methods 

function adapt_structure(storage, L::BlockDiagEquiRect{B}) where {B}
    BlockDiagEquiRect{B}(adapt(storage, L.blocks), adapt(storage, L.proj))
end

# function adapt_structure(storage, L::BlockDiagEquiRect{B}) where {B}
#     BlockDiagEquiRect{B}(adapt(storage, L.blocks), adapt(storage, L.blocks_sqrt), adapt(storage, L.proj))
# end



# ## make BlockDiagEquiRect an iterable over the last index
# ... so that 
#     `M½ = BlockDiagEquiRect{AzFourier}(sqrt.(Hermitian.(M)), M.proj)`
# works
# Base.parent(M::BlockDiagEquiRect) = M.blocks # for convienience
# Base.length(M::BlockDiagEquiRect) = size(parent(M),3)
# Base.eltype(::Type{BlockDiagEquiRect{B,P,T}}) where {B,P,T} = T 
# Base.firstindex(M::BlockDiagEquiRect) = 1
# Base.lastindex(M::BlockDiagEquiRect) = length(M)
# Base.iterate(M::BlockDiagEquiRect) = (Σ=parent(M) ; isempty(Σ) ? nothing : (Σ[:,:,1],1))
# Base.iterate(M::BlockDiagEquiRect, st) = st+1 > length(M) ? nothing : (parent(M)[:,:,st+1],  st+1)

# function Base.getindex(M::BlockDiagEquiRect, i::Int) 
#     1 <= i <= length(M) || throw(BoundsError(M, i))
#     return parent(M)[:,:,i]
# end

# function Base.setindex!(M::BlockDiagEquiRect, m::Matrix, i::Int)
#     1 <= i <= length(M) || throw(BoundsError(M, i))  
#     setindex!(parent(M)[:,:,i], m)
# end





# size(L::BlockDiagEquiRect) = (fill(L.proj.Nx * L.proj.Ny, 2)...,)

# function sqrt(L::BlockDiagEquiRect{B}) where {B}
#     if !isassigned(L.blocks_sqrt)
#         L.blocks_sqrt[] = mapslices(sqrt, L.blocks, dims=(1,2))
#     end
#     BlockDiagEquiRect{B}(L.blocks_sqrt[], L.proj)
# end

# function simulate(rng::AbstractRNG, L::BlockDiagEquiRect{AzFourier,ProjEquiRect{T}}) where {T}
#     @unpack Ny, Nx, θspan = L.proj
#     z = EquiRectMap(randn(rng, T, Ny, Nx) .* sqrt.(sin.(range(θspan..., length=Ny))), L.proj)
#     sqrt(L) * z
# end

# function simulate(rng::AbstractRNG, L::BlockDiagEquiRect{AzFourier,ProjEquiRect{T}}) where {T}
#     @unpack Ny, Nx, θspan = L.proj
#     z = EquiRectMap(randn(rng, T, Ny, Nx) .* sqrt.(sin.(range(θspan..., length=Ny))), L.proj)
#     sqrt(L) * z
# end

# covariance operators
# ================================================

# can't depend on Legendre.jl since its not in the general registry
Cℓ_to_Cov(::Val, ::ProjEquiRect{T}, args...; kwargs...) where {T} = 
    error("You must run `using Legendre` for this method to be available.")

@init @require Legendre="7642852e-7f09-11e9-134e-0940411082b6" begin

    function Cℓ_to_Cov(::Val{:I}, proj::ProjEquiRect{T}, Cℓ::InterpolatedCℓs; units=1, ℓmax=500) where {T}
        @unpack Ny, Nx, θspan, φspan = proj
        φspan_ratio = round(Int, 2π / abs(-(φspan...)))
        Cℓ = T.(nan2zero.(Cℓ[0:ℓmax]))
        Nm = Nx÷2+1
        θs = T.(range(reverse(θspan)..., length=Ny))
        λ = T.(Legendre.λlm(0:ℓmax, 0:φspan_ratio*(Nm-1), cos.(θs))[:,:,1:φspan_ratio:end])
        @tullio blocks[p,q,iₘ] := λ[p,ℓ,iₘ] * λ[q,ℓ,iₘ] * Cℓ[ℓ] * (iₘ==1 ? 2 : 4)
        BlockDiagEquiRect{AzFourier}(blocks, proj)
    end

    function Cℓ_to_Cov(::Val{:P}, proj::ProjEquiRect{T}, Cℓ::InterpolatedCℓs; units=1, ℓmax=500) where {T}
        error("Not implemented")
        # TODO: implement building S2 covariance
    end

end


# promotion
# ================================================

promote_basis_generic_rule(::Map, ::AzFourier) = Map()

promote_basis_generic_rule(::QUMap, ::QUAzFourier) = QUMap()

# used in broadcasting to decide the resulting metadata when
# broadcasting over two fields
function promote_metadata_strict(metadata₁::ProjEquiRect{T₁}, metadata₂::ProjEquiRect{T₂}) where {T₁,T₂}

    if (
        metadata₁.Ny    === metadata₂.Ny    &&
        metadata₁.Nx    === metadata₂.Nx    &&
        metadata₁.θspan === metadata₂.θspan &&   
        metadata₁.φspan === metadata₂.φspan   
    )
        
        # always returning the "wider" metadata even if T₁==T₂ helps
        # inference and is optimized away anyway
        promote_type(T₁,T₂) == T₁ ? metadata₁ : metadata₂
        
    else
        error("""Can't broadcast two fields with the following differing metadata:
        1: $(select(fields(metadata₁),(:Ny,:Nx,:θspan,:φspan)))
        2: $(select(fields(metadata₂),(:Ny,:Nx,:θspan,:φspan)))
        """)
    end

end


# used in non-broadcasted algebra to decide the resulting metadata
# when performing some operation across two fields. this is free to do
# more generic promotion than promote_metadata_strict (although this
# is currently not used, but in the future could include promoting
# resolution, etc...). the result should be a common metadata which we
# can convert both fields to then do a succesful broadcast
promote_metadata_generic(metadata₁::ProjEquiRect, metadata₂::ProjEquiRect) = 
    promote_metadata_strict(metadata₁, metadata₂)


### preprocessing
# defines how ImplicitFields and BatchedReals behave when broadcasted
# with ProjEquiRect fields. these can return arrays, but can also
# return `Broadcasted` objects which are spliced into the final
# broadcast, thus avoiding allocating any temporary arrays.

function preprocess((_,proj)::Tuple{<:Any,<:ProjEquiRect}, r::Real)
    r isa BatchedReal ? adapt(proj.storage, reshape(r.vals, 1, 1, 1, :)) : r
end
# need custom adjoint here bc Δ can come back batched from the
# backward pass even though r was not batched on the forward pass
@adjoint function preprocess(m::Tuple{<:Any,<:ProjEquiRect}, r::Real)
    preprocess(m, r), Δ -> (nothing, Δ isa AbstractArray ? batch(real.(Δ[:])) : Δ)
end



### adapting

# dont adapt the fields in proj, instead re-call into the memoized
# ProjLambert so we always get back the singleton ProjEquiRect object
# for the given set of parameters (helps reduce memory usage and
# speed-up subsequent broadcasts which would otherwise not hit the
# "===" branch of the "promote_*" methods)
function adapt_structure(storage, proj::ProjEquiRect{T}) where {T}
    # TODO: make sure these are consistent with any arguments that
    # were added to the memoized constructor
    @unpack Ny, Nx, θspan, φspan = proj
    T′ = eltype(storage)
    ProjEquiRect(;Ny, Nx, T=(T′==Any ? T : real(T′)), θspan, φspan, storage)
end
adapt_structure(::Nothing, proj::ProjEquiRect{T}) where {T} = proj


### etc...
# TODO: see proj_lambert.jl and adapt the things there for EquiRect
# maps, or even better, figure out what can be factored out into
# generic code that works for both Lambert and EquiRect
