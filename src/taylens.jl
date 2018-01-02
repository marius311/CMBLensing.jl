
export Taylens

struct Taylens{N,T<:Real} <: LenseOp
    # pixel remapping
    i::Matrix{Int}
    j::Matrix{Int}

    # residual displacement
    rx::Matrix{T}
    ry::Matrix{T}

    # precomputed quantities
    kα::Dict{Any,Matrix{Complex{T}}}
    xα::Dict{Any,Matrix{T}}

end



function Taylens{N}(ϕ::FlatS0{T,P}) where {N,T,P}

    g = FFTgrid(T,P)
    Nside = g.nside
    
    # total displacement
    d = ∇*ϕ
    dx, dy = d[1][:Tx], d[2][:Tx]

    # nearest pixel displacement
    indexwrap(ind::Int64, uplim)  = mod(ind - 1, uplim) + 1
    di, dj = (round(Int,d/g.Δx) for d=(dx,dy))
    i = indexwrap.(di .+ (1:Nside)', Nside)
    j = indexwrap.(dj .+ (1:Nside) , Nside)

    # residual displacement
    rx, ry = ((d - i.*g.Δx) for (d,i)=[(dx,di),(dy,dj)])

    # precomputation
    kα = Dict{Any,Matrix{Complex{T}}}()
    xα = Dict{Any,Matrix{T}}()
    for n in 1:N, α₁ in 0:n
        kα[n,α₁] = im ^ n .* g.k' .^ α₁ .* g.k[1:round(Int,Nside/2+1)] .^ (n - α₁)
        xα[n,α₁] = rx .^ α₁ .* ry .^ (n - α₁) ./ factorial(α₁) ./ factorial(n - α₁)
    end

    Taylens{N,T}(i,j,rx,ry,kα,xα)
end

# our implementation of Taylens
function *(L::Taylens{N}, f::FlatS0Map{T,P}) where {N,T,P}

    intlense(fx) = broadcast_getindex(fx, L.j, L.i)
    fl = f[:Tl]

    # lens to the nearest whole pixel
    Lfx = intlense(f.Tx)

    # add in Taylor series correction
    for n in 1:N, α₁ in 0:n
        Lfx .+= L.xα[n,α₁] .* intlense(ℱ{P} \ (L.kα[n,α₁] .* fl))
    end

    FlatS0Map{T,P}(Lfx)
end
*(L::Taylens, f::F) where {T,P,F<:FlatS2QUMap{T,P}} = F((L*FlatS0Map{T,P}(f.Qx))[:Tx], (L*FlatS0Map{T,P}(f.Ux))[:Tx])
*(L::Taylens, f::FlatField) = L*Ł(f)
