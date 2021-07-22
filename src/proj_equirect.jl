# Type defs
# ================================================

struct ProjEquiRect{T} <: CartesianProj

    Ny    :: Int
    Nx    :: Int
    θspan :: Tuple{Float64,Float64}
    φspan :: Tuple{Float64,Float64}

    storage

end

struct BlockDiagEquiRect{B<:Basis, P<:ProjEquiRect, T, A<:AbstractArray{T}} <: ImplicitOp{T}
    blocks :: A
    blocks_sqrt :: Ref{A} # lazily computed/saved sqrt of operator
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
        # This doesn't work ???
        # "EquiRectS0"   => Union{EquiRectMap, EquiRectAzFourier},
        # "EquiRectS2"   => Union{EquiRectQUMap, EquiRectQUAzFourier},
    ),
)

# This also gives an error ... so something below automatically defines
# EquiRectS0 and EquiRectS2 someplace
#
# const EquiRectS0 = Union{EquiRectMap, EquiRectAzFourier}
# const EquiRectS2 = Union{EquiRectQUMap, EquiRectQUAzFourier}

typealias_def(::Type{<:ProjEquiRect{T}}) where {T} = "ProjEquiRect{$T}"

typealias_def(::Type{F}) where {B,M<:ProjEquiRect,T,A,F<:EquiRectField{B,M,T,A}} = "EquiRect$(typealias(B)){$(typealias(A))}"

# Proj 
# ================================================


function ProjEquiRect(;Ny, Nx, θspan, φspan, T=Float32, storage=Array)
    ProjEquiRect(Ny, Nx, θspan, φspan, real_type(T), storage)
end

@memoize function ProjEquiRect(Ny, Nx, θspan, φspan, ::Type{T}, storage) where {T}
    
    # make span always be (low, high)
    θspan = (Float64.(sort(collect(θspan)))...,)
    φspan = (Float64.(sort(collect(φspan)))...,)

    φspan_ratio = 2π / abs(-(φspan...))
    if !(φspan_ratio ≈ round(Int, φspan_ratio))
        error("φspan=$φspan must span integer multiple of 2π")
    end

    ProjEquiRect{T}(Ny, Nx, θspan, φspan, storage)

end

function Base.summary(io::IO, f::EquiRectField)
    @unpack Ny,Nx,Nbatch = f
    print(io, "$(length(f))-element $Ny×$Nx$(Nbatch==1 ? "" : "(×$Nbatch)")-pixel ")
    Base.showarg(io, f, true)
end


# Field Basis
# ================================================
# NOTE: I still don't fully understand what AzFourer, Map ... etc is. 
# EquiRectAzFourier, EquiRectQUAzFourier ... seem to be aliases for a base field type.
# Are then AzFourier, Map just methods for making the conversion? Why not use the 
# field types themselfs for the conversion?

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


# block-diagonal operator
# ================================================

function BlockDiagEquiRect{B}(block_matrix::A, proj::P) where {B<:Basis, P<:ProjEquiRect, T, A<:AbstractArray{T}}
    BlockDiagEquiRect{B,P,T,A}(block_matrix, Ref{A}(), proj)
end

size(L::BlockDiagEquiRect) = (fill(L.proj.Nx * L.proj.Ny, 2)...,)

function sqrt(L::BlockDiagEquiRect{B}) where {B}
    if !isassigned(L.blocks_sqrt)
        L.blocks_sqrt[] = mapslices(sqrt, L.blocks, dims=(1,2))
    end
    BlockDiagEquiRect{B}(L.blocks_sqrt[], L.proj)
end

*(L::BlockDiagEquiRect{B}, f::EquiRectField) where {B<:Basis} = L * B(f)

function *(B::BlockDiagEquiRect{AzFourier}, f::EquiRectAzFourier)
    promote_metadata_strict(B.proj, f.proj) # ensure same projection
    EquiRectAzFourier(@tullio(Bf[p,iₘ] := B.blocks[p,q,iₘ] * f.arr[q,iₘ]), f.metadata)
end

function *(B::BlockDiagEquiRect{QUAzFourier}, f::EquiRectQUAzFourier)
    # TODO: implement S2 multiplication
    error("not implemented")
end

function adapt_structure(storage, L::BlockDiagEquiRect{B}) where {B}
    BlockDiagEquiRect{B}(adapt(storage, L.blocks), adapt(storage, L.blocks_sqrt), adapt(storage, L.proj))
end

function simulate(rng::AbstractRNG, L::BlockDiagEquiRect{AzFourier,ProjEquiRect{T}}) where {T}
    @unpack Ny, Nx, θspan = L.proj
    z = EquiRectMap(randn(rng, T, Ny, Nx) .* sqrt.(sin.(range(θspan..., length=Ny))), L.proj)
    sqrt(L) * z
end


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
