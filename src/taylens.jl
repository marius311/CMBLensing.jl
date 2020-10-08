

struct Taylens{N,T<:Real} <: ImplicitOp{Basis,Spin,Pix}
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



function Taylens(ϕ::FlatS0{P,T}, N) where {P,T}

    @unpack kx,ky,Nx,Ny,Δx = fieldinfo(ϕ)
    
    # total displacement
    d = ∇*ϕ
    dx, dy = d[1][:Ix], d[2][:Ix]

    # nearest pixel displacement
    indexwrap(ind::Int64, uplim)  = mod(ind - 1, uplim) + 1
    di, dj = (round.(Int,d/Δx) for d=(dx,dy))
    i = indexwrap.(di .+ (1:Nx)', Nx)
    j = indexwrap.(dj .+ (1:Ny) , Ny)

    # residual displacement
    rx, ry = ((d - i.*Δx) for (d,i)=[(dx,di),(dy,dj)])

    # precomputation
    kα = Dict{Any,Matrix{Complex{T}}}()
    xα = Dict{Any,Matrix{T}}()
    for n in 1:N, α₁ in 0:n
        kα[n,α₁] = im ^ n .* kx' .^ α₁ .* ky[1:Ny÷2+1] .^ (n - α₁)
        xα[n,α₁] = rx .^ α₁ .* ry .^ (n - α₁) ./ factorial(α₁) ./ factorial(n - α₁)
    end

    Taylens{N,T}(i,j,rx,ry,kα,xα)
end

# our implementation of Taylens
function *(L::Taylens{N}, f::FlatS0{P,T}) where {N,P,T}

    intlense(fx) = getindex.(Ref(fx), L.j, L.i)
    fl = f[:Il]

    # lens to the nearest whole pixel
    Lfx = intlense(f[:Ix])

    # add in Taylor series correction
    for n in 1:N, α₁ in 0:n
        Lfx .+= L.xα[n,α₁] .* intlense(fieldinfo(f).FFT \ (L.kα[n,α₁] .* fl))
    end

    FlatMap{P}(Lfx)
end

function *(L::Taylens, f::FieldTuple)
    Łf = Ł(f)
    F = typeof(Łf)
    F(map(f->L*f, Łf.fs))
end
