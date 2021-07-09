
struct ProjEquiRect{T} <: CartesianProj

    Ny    :: Int
    Nx    :: Int
    θspan :: Tuple{Float64,Float64}
    ϕspan :: Tuple{Float64,Float64}

    storage

end


# some extra Bases only relevant for EquiRect
struct AzFourier <: S0Basis end
const  QUAzFourier = Basis2Prod{    𝐐𝐔, AzFourier }
const IQUAzFourier = Basis3Prod{ 𝐈, 𝐐𝐔, AzFourier }

# make EquiRectMap, EquiRectFourier, etc... type aliases
make_field_aliases("EquiRect",  ProjEquiRect, extra_aliases=OrderedDict(
    "AzFourier"    => AzFourier,
    "QUAzFourier"  => QUAzFourier,
    "IQUAzFourier" => IQUAzFourier,
))


# for printing
typealias_def(::Type{<:ProjEquiRect{T}}) where {T} = "ProjEquiRect{$T}"


function ProjEquiRect(;Ny, Nx, θspan, ϕspan, T=Float32, storage=Array)
    ProjEquiRect(Ny, Nx, θspan, ϕspan, real_type(T), storage)
end

@memoize function ProjEquiRect(Ny, Nx, θspan, ϕspan, ::Type{T}, storage) where {T}
    
    # TODO: precompute block diagonal transform matrices here and
    # store them in the constructed object. note that this function is
    # memoized so its only actually called once, and its arguments
    # should be everything that uniquely defined a ProjEquiRect
    
    # span is always (low, high)
    θspan = (Float64.(sort(collect(θspan)))...,)
    ϕspan = (Float64.(sort(collect(ϕspan)))...,)

    ϕspan_ratio = 2π / abs(-(ϕspan...))
    if !(ϕspan_ratio ≈ round(Int, ϕspan_ratio))
        error("ϕspan=$ϕspan must span integer multiple of 2π")
    end

    ProjEquiRect{T}(Ny, Nx, θspan, ϕspan, storage)

end

typealias_def(::Type{F}) where {B,M<:ProjEquiRect,T,A,F<:EquiRectField{B,M,T,A}} = "EquiRect$(typealias(B)){$(typealias(A))}"
function Base.summary(io::IO, f::EquiRectField)
    @unpack Ny,Nx,Nbatch = f
    print(io, "$(length(f))-element $Ny×$Nx$(Nbatch==1 ? "" : "(×$Nbatch)")-pixel ")
    Base.showarg(io, f, true)
end


### basis conversion

AzFourier(f::EquiRectMap) = EquiRectAzFourier(m_rfft(f.arr, (2,)), f.metadata)
Map(f::EquiRectAzFourier) = EquiRectMap(m_irfft(f.arr, f.Nx, (2,)), f.metadata)

QUAzFourier(f::EquiRectQUMap) = EquiRectQUAzFourier(m_rfft(f.arr, (2,)), f.metadata)
QUMap(f::EquiRectQUAzFourier) = EquiRectQUMap(m_irfft(f.arr, f.Nx, (2,)), f.metadata)

IQUAzFourier(f::EquiRectIQUMap) = EquiRectIQUAzFourier(m_rfft(f.arr, (2,)), f.metadata)
IQUMap(f::EquiRectIQUAzFourier) = EquiRectIQUMap(m_irfft(f.arr, f.Nx, (2,)), f.metadata)


# TODO: remaining conversion rules


### block-diagonal operator

struct BlockDiagEquiRect{B<:Basis, P<:ProjEquiRect, T, A<:AbstractArray{T}} <: ImplicitOp{T}
    blocks :: A
    blocks_sqrt :: Ref{A} # lazily computed/saved sqrt of operator
    proj :: P
end
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



### covariance operators

# can't depend on Legendre.jl since its not in the general registry
Cℓ_to_Cov(::Val, ::ProjEquiRect{T}, args...; kwargs...) where {T} = 
    error("You must run `using Legendre` for this method to be available.")

@init @require Legendre="7642852e-7f09-11e9-134e-0940411082b6" begin

    function Cℓ_to_Cov(::Val{:I}, proj::ProjEquiRect{T}, Cℓ::InterpolatedCℓs; units=1, ℓmax=500) where {T}
        @unpack Ny, Nx, θspan, ϕspan = proj
        ϕspan_ratio = round(Int, 2π / abs(-(ϕspan...)))
        Cℓ = T.(nan2zero.(Cℓ[0:ℓmax]))
        Nm = Nx÷2+1
        θs = T.(range(reverse(θspan)..., length=Ny))
        λ = T.(Legendre.λlm(0:ℓmax, 0:ϕspan_ratio*(Nm-1), cos.(θs))[:,:,1:ϕspan_ratio:end])
        @tullio blocks[p,q,iₘ] := λ[p,ℓ,iₘ] * λ[q,ℓ,iₘ] * Cℓ[ℓ] * (iₘ==1 ? 2 : 4)
        BlockDiagEquiRect{AzFourier}(blocks, proj)
    end

    function Cℓ_to_Cov(::Val{:P}, proj::ProjEquiRect{T}, Cℓ::InterpolatedCℓs; units=1, ℓmax=500) where {T}
        error("Not implemented")
        # TODO: implement building S2 covariance
    end

end


### promotion

# used in broadcasting to decide the resulting metadata when
# broadcasting over two fields
function promote_metadata_strict(metadata₁::ProjEquiRect{T₁}, metadata₂::ProjEquiRect{T₂}) where {T₁,T₂}

    if (
        metadata₁.Ny    === metadata₂.Ny    &&
        metadata₁.Nx    === metadata₂.Nx    &&
        metadata₁.θspan === metadata₂.θspan &&   
        metadata₁.ϕspan === metadata₂.ϕspan   
    )
        
        # always returning the "wider" metadata even if T₁==T₂ helps
        # inference and is optimized away anyway
        promote_type(T₁,T₂) == T₁ ? metadata₁ : metadata₂
        
    else
        error("""Can't broadcast two fields with the following differing metadata:
        1: $(select(fields(metadata₁),(:Ny,:Nx,:θspan,:ϕspan)))
        2: $(select(fields(metadata₂),(:Ny,:Nx,:θspan,:ϕspan)))
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
    @unpack Ny, Nx, θspan, ϕspan = proj
    T′ = eltype(storage)
    ProjEquiRect(;Ny, Nx, T=(T′==Any ? T : real(T′)), θspan, ϕspan, storage)
end
adapt_structure(::Nothing, proj::ProjEquiRect{T}) where {T} = proj


### etc...
# TODO: see proj_lambert.jl and adapt the things there for EquiRect
# maps, or even better, figure out what can be factored out into
# generic code that works for both Lambert and EquiRect
