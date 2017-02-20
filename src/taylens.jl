using BayesLensSPTpol: indexwrap

export FlatS0TaylensOp

immutable FlatS0TaylensOp{T<:Real,P<:Flat} <: LinearOp{P,S0,Map}
    # pixel remapping
    i::Matrix{Int}
    j::Matrix{Int}

    # residual displacement
    rx::Matrix{T}
    ry::Matrix{T}

    # precomputed quantities
    kα::Dict{Any,Matrix{Complex{T}}}
    xα::Dict{Any,Matrix{T}}

    order::Int
    taylens::Bool
end

function FlatS0TaylensOp{T,P}(ϕ::FlatS0{T,P}; order=4, taylens=true)

    g = FFTgrid(T,P)
    Nside = g.nside
    
    # total displacement
    d = ∇*ϕ
    dx, dy = d[1][:Tx], d[2][:Tx]

    # nearest pixel displacement
    if taylens
        di, dj = (round(Int,d/g.Δx) for d=(dx,dy))
        i = indexwrap.(di .+ (1:Nside)', Nside)
        j = indexwrap.(dj .+ (1:Nside) , Nside)
    else
        di = dj = i = j = zeros(Int,Nside,Nside)
    end

    # residual displacement
    rx, ry = ((d - i.*g.Δx) for (d,i)=[(dx,di),(dy,dj)])

    # precomputation
    kα = Dict{Any,Matrix{Complex{T}}}()
    xα = Dict{Any,Matrix{T}}()
    for n in 1:order, α₁ in 0:n
        kα[n,α₁] = im ^ n .* g.k' .^ α₁ .* g.k[1:round(Int,Nside/2+1)] .^ (n - α₁)
        xα[n,α₁] = rx .^ α₁ .* ry .^ (n - α₁) ./ factorial(α₁) ./ factorial(n - α₁)
    end

    FlatS0TaylensOp{T,P}(i,j,rx,ry,kα,xα,order,taylens)
end

# our implementation of Taylens
function *{T,P}(lens::FlatS0TaylensOp, f::FlatS0Map{T,P})

    intlense(fx) = lens.taylens ? broadcast_getindex(fx, lens.j, lens.i) : fx
    fl = f[:Tl]

    # lens to the nearest whole pixel
    Lfx = intlense(f.Tx)

    # add in Taylor series correction
    for n in 1:lens.order, α₁ in 0:n
        Lfx .+= lens.xα[n,α₁] .* intlense(ℱ{P} \ (lens.kα[n,α₁] .* fl))
    end

    FlatS0Map{T,P}(Lfx,meta(f)...)
end
