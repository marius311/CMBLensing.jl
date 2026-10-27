
struct ProjEquiRect{T,V,M} <: CartesianProj

    Ny          :: Int
    Nx          :: Int
    θspan       :: Tuple{Float64,Float64}
    φspan       :: Tuple{Float64,Float64}
    θ           :: V
    φ           :: V
    θedges      :: V
    φedges      :: V
    Ω           :: V
    ℓx          :: M
    
    storage

end

struct BlockDiagEquiRect{B<:Basis, T<:Real, P<:ProjEquiRect{T}, A<:AbstractArray}  <: ImplicitOp{T}

    blocks :: A
    blocks_sqrt :: Ref{A}
    blocks_pinv :: Ref{A}
    logabsdet :: Ref{Tuple{T,Complex{T}}}
    proj :: P

end

function BlockDiagEquiRect{B}(
    blocks :: A, 
    blocks_sqrt :: Ref{A}, 
    blocks_pinv :: Ref{A}, 
    logabsdet :: Ref{Tuple{T,Complex{T}}}, 
    proj :: P
) where {B<:Basis, T<:Real, P<:ProjEquiRect{T}, A<:AbstractArray}
    BlockDiagEquiRect{B,T,P,A}(blocks, blocks_sqrt, blocks_pinv, logabsdet, proj)
end

struct AzFourier <: S0Basis end
const  QUAzFourier = Basis2Prod{    𝐐𝐔, AzFourier }
const IQUAzFourier = Basis3Prod{ 𝐈, 𝐐𝐔, AzFourier }
const AzBasis = Union{AzFourier, QUAzFourier, IQUAzFourier}


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

# allow proj.T
getproperty(proj::ProjEquiRect, k::Symbol) = getproperty(proj, Val(k))
getproperty(proj::ProjEquiRect{T}, ::Val{:T}) where {T} = T
getproperty(proj::ProjEquiRect, ::Val{k}) where {k} = getfield(proj, k)



# Proj 
# ================================================

@memoize function ProjEquiRect(θ, φ, θedges, φedges, θspan, φspan, ::Type{T}, storage) where {T}
    Ny, Nx = length(θ), length(φ)
    Ω  = rem2pi(φedges[2] .- φedges[1], RoundDown) .* diff(.- cos.(θedges))
    Δx = sin.(θ) .* abs(-(φspan...)) / Nx
    Δℓx = @. 2π/(Nx*Δx)
    ℓx = ifftshift(-Nx÷2:(Nx-1)÷2)' .* Δℓx
    (θ, φ, θedges, φedges, Ω, ℓx) = adapt.(storage, (T.(θ), T.(φ), T.(θedges), T.(φedges), T.(Ω), T.(ℓx)))
    V = typeof(φ)
    M = typeof(ℓx)
    ProjEquiRect{T,V,M}(Ny, Nx, θspan, φspan, θ, φ, θedges, φedges, Ω, ℓx, storage)
end

"""
    ProjEquiRect(; Ny::Int, Nx::Int, θspan::Tuple, φspan::Tuple,         T=Float32, storage=Array)
    ProjEquiRect(; θ::Vector, φ::Vector, θedges::Vector, φedges::Vector, T=Float32, storage=Array)

Construct an EquiRect projection object. The projection can either be
specified by:

* The number of pixels `Ny` and `Nx` (corresponding to the `θ` and `φ`
  angular directions, respectively) and the span in radians of the
  field in these directions, `θspan` and `φspan`. The order in which
  the span tuples are given is irrelevant, either order will refer to
  the same field. Note, the spans correspond to the field size between
  outer pixel edges, not from pixel centers. If one wishes to call
  `Cℓ_to_Cov` with this projection, `φspan` must be an integer
  multiple of 2π, but other functionality will be available if this is
  not the case. 
* A manual list of pixels centers and pixel edges, `θ`, `φ`, `θedges`,
  `φedges`.

Healpix conversions currently use coordinate helpers with a half-pixel
offset on regular EquiRect grids, and longitude handling has known seam
limitations. For manually specified grids, coordinate helpers use dimensions
and spans rather than the supplied centers, and compatibility checks do not
compare individual centers or edges. Area and azimuthal Fourier calculations
assume uniformly spaced longitudes. Storage adaptation preserves the stored
coordinates and derived arrays, converting their element type when requested.

"""
function ProjEquiRect(; T=Float32, storage=Array, kwargs...)

    arg_error() = error("Constructor takes either (θ, φ, θedges, φedges) or (Ny, Nx, θspan, φspan) keyword arguments.")
    
    if all(haskey.(Ref(kwargs), (:θ, :φ, :θedges, :φedges)))
        !any(haskey.(Ref(kwargs), (:Ny, :Nx, :θspan, :φspan))) || arg_error()
        @unpack (θ, φ, θedges, φedges) = kwargs
        θspan = (θedges[1], θedges[end])
        φspan = (φedges[1], φedges[end])
    elseif all(haskey.(Ref(kwargs), (:Ny, :Nx, :θspan, :φspan)))
        !all(haskey.(Ref(kwargs), (:θ, :φ, :θedges, :φedges))) || arg_error()
        @unpack (Nx, Ny, θspan, φspan) = kwargs
        θspan = (sort(collect(θspan))...,)
        φspan = (sort(collect(φspan))...,)
        φedges = rem2pi.(range(φspan..., length=Nx+1),           RoundDown)
        φ      = rem2pi.(range(φspan..., length=2Nx+1)[2:2:end], RoundDown)
        θedges = range(θspan..., length=Ny+1)
        θ      = range(θspan..., length=2Ny+1)[2:2:end]
    else
        arg_error()
    end

    ProjEquiRect(θ, φ, θedges, φedges, θspan, φspan, real_type(T), storage)

end



# Field Basis
# ================================================
"""
Jperm(ℓ::Int, n::Int) return the column number in the J matrix U^2
where U is unitary FFT. The J matrix looks like this:

|1   0|
|  / 1|
| / / |
|0 1  |

"""
function Jperm(ℓ::Int, n::Int)
    @assert 1 <= ℓ <= n
    ℓ==1 ? 1 : n - ℓ + 2
end

# AzFourier <-> Map
function AzFourier(f::EquiRectMap)
    nφ, T = f.Nx, real(f.T)
    EquiRectAzFourier(m_rfft(f.arr, 2) ./ T(√nφ), f.proj)
end

function Map(f::EquiRectAzFourier)
    nφ, T = f.Nx, real(f.T)
    EquiRectMap(m_irfft(f.arr, nφ, 2) .* T(√nφ), f.proj)
end

# QUAzFourier <-> QUMap
function QUAzFourier(f::EquiRectQUMap)
    nθ, nφ, T = f.Ny, f.Nx, real(f.T)
    P_map = complex.(f.Qx, f.Ux)
    P_azfft = m_fft(P_map, 2) ./ T(√nφ)
    P_azfft_perm = similar(P_azfft, 2nθ, nφ÷2+1)
    P_azfft_perm[1:nθ,:] .= P_azfft[:,1:nφ÷2+1]
    # Zero frequency is followed by the negative frequencies. For even
    # widths the Nyquist frequency occurs in both halves; for odd widths
    # the highest positive and negative frequencies are distinct.
    negative_indices = [1; nφ:-1:(nφ - nφ÷2 + 1)]
    P_azfft_perm[nθ+1:end,:] .= conj.(P_azfft[:, negative_indices])
    EquiRectQUAzFourier(P_azfft_perm, f.proj)
end

function QUMap(f::EquiRectQUAzFourier)
    nθ, nφ, T = f.Ny, f.Nx, real(f.T)
    P_azfft_perm = f.arr
    P_azfft = similar(P_azfft_perm, nθ, nφ)
    P_azfft[:, 1:nφ÷2+1] .= P_azfft_perm[1:nθ,:]
    negative_indices = [1; nφ:-1:(nφ - nφ÷2 + 1)]
    P_azfft[:, negative_indices] .= conj(P_azfft_perm[nθ+1:end,:])
    P_map = m_ifft(P_azfft, 2) .* T(√nφ)
    EquiRectQUMap(cat(real(P_map), imag(P_map), dims=3), f.proj)
end


function Base.getindex(f::EquiRectS0, k::Symbol)
    @match k begin
        :Ix => Map(f).arr
        :Il => AzFourier(f).arr
        _ => error("Invalid EquiRectS0 index $k")
    end
end
function Base.getindex(f::EquiRectS2, k::Symbol)
    @match k begin
        :Qx => QUMap(f).Qx
        :Ux => QUMap(f).Ux
        :Px => (qu=QUMap(f); complex.(qu.Qx,qu.Ux))
        :Pl => QUAzFourier(f).arr
        _ => error("Invalid EquiRectS2 index $k")
    end
end

function Base.summary(io::IO, f::EquiRectField)
    @unpack Ny,Nx,Nbatch = f
    print(io, "$(length(f))-element $Ny×$Nx$(Nbatch==1 ? "" : "(×$Nbatch)")-pixel ")
    Base.showarg(io, f, true)
end

# block-diagonal operator
# ================================================

# ## Constructors

function BlockDiagEquiRect{B}(block_matrix::A, proj::P) where {B<:AzBasis, T<:Real, P<:ProjEquiRect{T}, A<:AbstractArray}
    real(eltype(A)) == T || error("Mismatched eltype between $P and $A")
    BlockDiagEquiRect{B,T,P,A}(block_matrix, Ref{A}(), Ref{A}(), Ref{Tuple{T,Complex{T}}}((0,0)), proj)
end

# The following allows construction by a vector of blocks

function BlockDiagEquiRect{B}(vector_of_blocks::Vector{A}, proj::P) where {B<:AzBasis, T<:Real, P<:ProjEquiRect{T}, A<:AbstractMatrix}
    block_matrix = similar(vector_of_blocks[1], size(vector_of_blocks[1])..., length(vector_of_blocks))
    for b in eachindex(vector_of_blocks)
        block_matrix[:,:,b] .= vector_of_blocks[b]
    end
    BlockDiagEquiRect{B}(block_matrix, proj)
end

# ## Linear Algebra: tullio accelerated (operator, field)

# M * f

(*)(M::BlockDiagEquiRect{B}, f::EquiRectField) where {B<:Basis} = M * B(f)

@uses_tullio function (*)(M::BlockDiagEquiRect{B}, f::F) where {B<:AzBasis, F<:EquiRectField{B}}
    promote_metadata_strict(M.proj, f.proj) # ensure same projection
    F(@tullio(Bf[p,iₘ] := M.blocks[p,q,iₘ] * f.arr[q,iₘ]), f.proj)
end

(*)(M::Adjoint{T,<:BlockDiagEquiRect{B}}, f::EquiRectField) where {T, B<:Basis} = M * B(f)

@uses_tullio function (*)(M::Adjoint{T,<:BlockDiagEquiRect{B}}, f::F) where {T, B<:AzBasis, F<:EquiRectField{B}}
    promote_metadata_strict(M.parent.proj, f.proj) # ensure same projection
    F(@tullio(Bf[p,iₘ] := conj(M.parent.blocks[q,p,iₘ]) * f.arr[q,iₘ]), f.proj)
end

@uses_tullio function rrule(::typeof(*), M::BlockDiagEquiRect{B}, f::EquiRectField{B′}) where {B<:Basis, B′<:Basis}
    function times_pullback(Δ)
        BΔ, Bf = B(Δ), B(f)
        Zygote.ChainRules.NoTangent(), @thunk(BlockDiagEquiRect{B}(@tullio(M̄[p,q,iₘ] := Bf.arr[p,iₘ] * conj(BΔ.arr[q,iₘ])), M.proj)'), B′(M' * BΔ)
    end
    M * f, times_pullback
end


# ## Linear Algebra: tullio accelerated (operator, operator)

# M₁ * M₂
@uses_tullio function (*)(M₁::BlockDiagEquiRect{B}, M₂::BlockDiagEquiRect{B}) where {B<:AzBasis}
    promote_metadata_strict(M₁.proj, M₂.proj) # ensure same projection
    BlockDiagEquiRect{B}(@tullio(M₃[p,q,iₘ] := M₁.blocks[p,j,iₘ] * M₂.blocks[j,q,iₘ]), M₁.proj)
end

# M₁' * M₂
@uses_tullio function (*)(M₁::Adjoint{T,<:BlockDiagEquiRect{B}}, M₂::BlockDiagEquiRect{B}) where {T, B<:AzBasis}
    promote_metadata_strict(M₁.parent.proj, M₂.proj) # ensure same projection
    BlockDiagEquiRect{B}(@tullio(M₃[p,q,iₘ] := conj(M₁.parent.blocks[j,p,iₘ]) * M₂.blocks[j,q,iₘ]), M₁.parent.proj)
end

# M₁ * M₂'
@uses_tullio function (*)(M₁::BlockDiagEquiRect{B}, M₂::Adjoint{T,<:BlockDiagEquiRect{B}}) where {T, B<:AzBasis}
    promote_metadata_strict(M₁.proj, M₂.parent.proj) # ensure same projection
    BlockDiagEquiRect{B}(@tullio(M₃[p,q,iₘ] := M₁.blocks[p,j,iₘ] * conj(M₂.parent.blocks[q,j,iₘ])), M₁.proj)
end

# M₁ + M₂, M₁ - M₂, M₁ \ M₂, M₁ / M₂ ... also with mixed adjoints
# QUESTION: some of these may be sped up with @tullio

for op in (:+, :-, :/, :\)

    @eval function Base.$op(M₁::BlockDiagEquiRect{B}, M₂::BlockDiagEquiRect{B}) where {B<:AzBasis}
        promote_metadata_strict(M₁.proj, M₂.proj) # ensure same projection
        BlockDiagEquiRect{B}(
            map( $op, eachslice(M₁.blocks;dims=3), eachslice(M₂.blocks;dims=3) ),
            M₁.proj,
        )
    end

    @eval function Base.$op(M₁::Adjoint{T,<:BlockDiagEquiRect{B}}, M₂::BlockDiagEquiRect{B}) where {T, B<:AzBasis}
        promote_metadata_strict(M₁.parent.proj, M₂.proj) # ensure same projection
        BlockDiagEquiRect{B}(
            map( (m1,m2)->$op(m1',m2), eachslice(M₁.parent.blocks;dims=3), eachslice(M₂.blocks;dims=3) ),
            M₁.proj
        )
    end

    @eval function Base.$op(M₁::BlockDiagEquiRect{B}, M₂::Adjoint{T,<:BlockDiagEquiRect{B}}) where {T, B<:AzBasis}
        promote_metadata_strict(M₁.proj, M₂.parent.proj) # ensure same projection
        BlockDiagEquiRect{B}(
            map( (m1,m2)->$op(m1,m2'), eachslice(M₁.blocks;dims=3), eachslice(M₂.parent.blocks;dims=3) ),
            M₁.proj,
        )
    end

end

for op in (:*, :/)
    @eval Base.$op(a::Scalar, M::BlockDiagEquiRect{B}) where {B} = BlockDiagEquiRect{B}(broadcast($op, a, M.blocks), M.proj)
    @eval Base.$op(M::BlockDiagEquiRect{B}, a::Scalar) where {B} = BlockDiagEquiRect{B}(broadcast($op, a, M.blocks), M.proj)
    @eval Base.$op(a::Scalar, M::Adjoint{T,<:BlockDiagEquiRect{B}}) where {B,T} = (conj(a) * M.parent)'
    @eval Base.$op(M::Adjoint{T,<:BlockDiagEquiRect{B}}, a::Scalar) where {B,T} = (conj(a) * M.parent)'
end



# ## Linear Algebra: with arguments (operator, )

function LinearAlgebra.sqrt(M::BlockDiagEquiRect{B}) where {B<:AzBasis}
    if !isassigned(M.blocks_sqrt)
        M.blocks_sqrt[] = blocks_sqrt = similar(M.blocks)
        for i = 1:size(M.blocks,3)
            # use SVD since it works on both CPU/GPU
            U, S, V = svd(M.blocks[:,:,i])
            blocks_sqrt[:,:,i] .= U * Diagonal(real.(sqrt.(S))) * V'
        end
    end
    BlockDiagEquiRect{B}(M.blocks_sqrt[], M.proj)
end

function LinearAlgebra.pinv(M::BlockDiagEquiRect{B}) where {B<:AzBasis}
    if !isassigned(M.blocks_pinv)
        M.blocks_pinv[] = blocks_pinv = similar(M.blocks)
        for i = 1:size(M.blocks,3)
            blocks_pinv[:,:,i] .= pinv(M.blocks[:,:,i])
        end
    end
    BlockDiagEquiRect{B}(M.blocks_pinv[], M.proj)
end

# logdet and logabsdet

function LinearAlgebra.logdet(M::BlockDiagEquiRect{B}) where {B<:AzBasis} 
    l, s = logabsdet(M)
    l + log(s)
end

function LinearAlgebra.logabsdet(M::BlockDiagEquiRect{B}) where {B<:AzBasis} 
    if M.logabsdet[] == (0,0)
        M.logabsdet[] = mapreduce(logabsdet, ((l1,s1),(l2,s2))->(l1+l2,s1*s2), eachslice(M.blocks, dims=3))
    end
    M.logabsdet[]
end

@adjoint function LinearAlgebra.logabsdet(M::BlockDiagEquiRect)
    logabsdet(M), Δ -> (Δ[1] * pinv(M)',)
end

# dot products

LinearAlgebra.dot(a::EquiRectField, b::EquiRectField) = dot(Ł(a).arr, Ł(b).arr)

# needed by AD
@uses_tullio function LinearAlgebra.dot(M₁::Adjoint{T,<:BlockDiagEquiRect{B}}, M₂::BlockDiagEquiRect{B}) where {T, B<:AzBasis}
    (@tullio a[] := conj(M₁.parent.blocks[q,p,iₘ]) * M₂.blocks[p,q,iₘ])[]
end



# mapblocks 
# =====================================

function mapblocks(fun::Function, M::BlockDiagEquiRect{B}, f::EquiRectField) where {B<:AzBasis} 
    mapblocks(fun, M, B(f))
end

function mapblocks(fun::Function, M::BlockDiagEquiRect{B}, f::F) where {B<:AzBasis, F<:EquiRectField{B}}
    promote_metadata_strict(M.proj, f.proj) # ensure same projection
    Mfarr = similar(f.arr)
    y_    = eachcol(Mfarr)
    x_    = eachcol(f.arr)
    Mb_   = eachslice(M.blocks; dims = 3) 
    for (y, x, Mb) in zip(y_, x_, Mb_)
        y .= fun(Mb, x)
    end
    F(Mfarr, f.proj)
end 

function mapblocks(fun::Function, Ms::BlockDiagEquiRect{B}...) where {B<:AzBasis}
    map(M->promote_metadata_strict(M.proj, Ms[1].proj), Ms) 
    BlockDiagEquiRect{B}(
        map(
            i->fun(getindex.(getproperty.(Ms,:blocks),:,:,i)...), # This looks miserable:(
            axes(Ms[1].blocks,3),
        ),
        Ms[1].proj,
    )
end 

# Other methods
# ========================================= 

# ## simulation

function simulate(rng::AbstractRNG, M::BlockDiagEquiRect{AzFourier,T}) where {T}
    sqrt(M) * EquiRectMap(randn!(rng, similar(M.blocks, T, M.proj.Ny, M.proj.Nx)), M.proj)
end

function simulate(rng::AbstractRNG, M::BlockDiagEquiRect{QUAzFourier,T}) where {T}
    sqrt(M) * EquiRectQUMap(randn!(rng, similar(M.blocks, T, M.proj.Ny, M.proj.Nx, 2)), M.proj)
end

# adapt_structure

function adapt_structure(storage, L::BlockDiagEquiRect{B}) where {B}
    blocks = adapt(storage, L.blocks)
    BlockDiagEquiRect{B}(
        blocks,
        isassigned(L.blocks_sqrt) ? Ref(adapt(storage, L.blocks_sqrt[])) : Ref{typeof(blocks)}(),
        isassigned(L.blocks_pinv) ? Ref(adapt(storage, L.blocks_pinv[])) : Ref{typeof(blocks)}(),
        L.logabsdet,
        adapt(storage, L.proj)
    )
end

function Base.size(L::BlockDiagEquiRect{<:AzBasis})  
    n,m,p = size(L.blocks)
    @assert n==m
    sz = n*p
    return (sz, sz)
end

# covariance and beam operators
# ================================================

function Cℓ_to_Cov(::Val, proj::ProjEquiRect, args...; kwargs...)
    error("Run `using CirculantCov` to use this function.")
end

@init @require CirculantCov="edf8e0bb-e88b-4581-a03e-dda99a63c493" begin

    function Cℓ_to_Cov(::Val{:I}, proj::ProjEquiRect{T}, CI::Cℓs; units=1, ℓmax=10_000, progress=true) where {T}
        
        @unpack θ, φ, Ω = proj
        @cpu! θ φ Ω
        nθ, nφ  = length(θ), length(φ)
        ℓ       = 0:ℓmax

        CIℓ = nan2zero.(CI(ℓ))

        @assert real(T) == T
        blocks = zeros(T, nθ, nθ, nφ÷2+1)
        # TODO: do we want ngrid as an optional argmuent to Cℓ_to_Cov?
        Γ_I  = CirculantCov.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, CIℓ; ngrid=50_000)
        # using full resolution ComplexF64 for internal construction
        ptmW    = FFTW.plan_fft(Vector{ComplexF64}(undef, nφ)) 

        pbar = Progress(nθ, progress ? 1 : Inf, "Cℓ_to_Cov: ")
        for k = 1:nθ
            for j = 1:nθ
                Iγⱼₖℓ⃗ = CirculantCov.γθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ_I, ptmW)
                for ℓ = 1:nφ÷2+1
                    blocks[j,k,ℓ] = real(Iγⱼₖℓ⃗[ℓ])
                end
            end
            next!(pbar)
        end

        return BlockDiagEquiRect{AzFourier}(blocks, proj)
        
    end

    function Cℓ_to_Cov(::Val{:P}, proj::ProjEquiRect{T}, CEE::Cℓs, CBB::Cℓs; units=1, ℓmax=10_000, progress=true) where {T}
        
        @unpack θ, φ, Ω = proj
        @cpu! θ φ Ω
        nθ, nφ  = length(θ), length(φ)
        ℓ       = 0:ℓmax

        CBBℓ = nan2zero.(CBB(ℓ))
        CEEℓ = nan2zero.(CEE(ℓ))

        @assert real(T) == T
        blocks = zeros(Complex{T},2nθ,2nθ,nφ÷2+1)
        # TODO: do we want ngrid as an optional argmuent to Cℓ_to_Cov?
        ΓC_EB = CirculantCov.ΓCθ₁θ₂φ₁φ⃗_CMBpol(ℓ, CEEℓ, CBBℓ; ngrid=50_000)    
        # using full resolution ComplexF64 for internal construction
        ptmW = FFTW.plan_fft(Vector{ComplexF64}(undef, nφ)) 
        
        pbar = Progress(nθ, progress ? 1 : Inf, "Cℓ_to_Cov: ")
        for k = 1:nθ
            for j = 1:nθ
                EBγⱼₖℓ⃗, EBξⱼₖℓ⃗ = CirculantCov.γθ₁θ₂ℓ⃗_ξθ₁θ₂ℓ⃗(θ[j], θ[k], φ, ΓC_EB..., ptmW)
                for ℓ = 1:nφ÷2+1
                    Jℓ = Jperm(ℓ, nφ)
                    blocks[j,    k,    ℓ] = EBγⱼₖℓ⃗[ℓ]
                    blocks[j,    k+nθ, ℓ] = EBξⱼₖℓ⃗[ℓ]
                    blocks[j+nθ, k,    ℓ] = conj(EBξⱼₖℓ⃗[Jℓ])
                    blocks[j+nθ, k+nθ, ℓ] = conj(EBγⱼₖℓ⃗[Jℓ])
                end
            end
            next!(pbar)
        end

        return BlockDiagEquiRect{QUAzFourier}(blocks, proj)

    end

end

@uses_tullio function Cℓ_to_Beam(::Val{:I}, proj::ProjEquiRect{T}, CI::Cℓs; units=1, ℓmax=10_000, progress=true) where {T}

    @unpack Ω = proj
    @cpu! Ω
    Ω′ = T.(Ω)

    Cov = Cℓ_to_Cov(:I, proj, CI; units, ℓmax, progress)
    @tullio Cov.blocks[j,k,iₘ] *= Ω′[k]

    return Cov
end

@uses_tullio function Cℓ_to_Beam(::Val{:P}, proj::ProjEquiRect{T}, CI::Cℓs; units=1, ℓmax=10_000, progress=true) where {T}

    @unpack θ, Ω = proj
    @cpu! Ω
    Ω′ = T.(Ω)

    Cov   = Cℓ_to_Cov(:I, proj, CI; units, ℓmax, progress)
    dcatΩ = Diagonal(vcat(Ω′, Ω′))
    zB    = zeros(T, length(θ), length(θ))

    Beam = BlockDiagEquiRect{QUAzFourier}(
        map(B->[B zB;zB B]*dcatΩ, eachslice(Cov.blocks; dims=3)),
        proj,
    )

    return Beam
end

Cℓ_to_Beam(pol::Symbol, args...; kwargs...) = Cℓ_to_Beam(Val(pol), args...; kwargs...)


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

# Preserve manual coordinates and derived arrays instead of rebuilding a
# regular grid from the spans. Adapt determines the resulting precision.
function adapt_structure(storage, proj::ProjEquiRect)
    θ, φ, θedges, φedges, Ω, ℓx = map(x -> adapt(storage, x),
        (proj.θ, proj.φ, proj.θedges, proj.φedges, proj.Ω, proj.ℓx))
    T = real(eltype(θ))
    ProjEquiRect{T,typeof(φ),typeof(ℓx)}(proj.Ny, proj.Nx,
        proj.θspan, proj.φspan, θ, φ, θedges, φedges, Ω, ℓx, storage)
end
adapt_structure(::Nothing, proj::ProjEquiRect) = proj


hash(proj::ProjEquiRect, h::UInt64) = foldr(hash, (ProjEquiRect, proj.Ny, proj.Nx, proj.θspan, proj.φspan, proj.storage), init=h)
