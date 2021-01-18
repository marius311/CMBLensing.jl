

struct Taylens{L<:PowerLens} <: ImplicitOp{Bottom}

    # nearest pixel remapping
    i :: Matrix{Int}
    j :: Matrix{Int}

    # residual lensing (via PowerLens)
    residual :: L

end

Taylens(order) = x -> Taylens(x, order)
Taylens(ϕ::FlatS0, order) = Taylens(∇*ϕ, order)
function Taylens(d::FieldVector, order)

    require_unbatched(d[1])
    @unpack ℓx,ℓy,Nx,Ny,Δx,metadata = d[1]
    
    # total displacement
    dx, dy = d[1][:Ix], d[2][:Ix]

    # nearest pixel displacement
    indexwrap(i, N) = mod(i - 1, N) + 1
    di = round.(Int,dy/Δx)
    dj = round.(Int,dx/Δx)
    i = indexwrap.(di .+ (1:Ny) , Ny)
    j = indexwrap.(dj .+ (1:Nx)', Nx)

    # residual displacement
    r = @SVector[
        FlatMap(dx .- dj .* Δx, metadata),
        FlatMap(dy .- di .* Δx, metadata),
    ]
    residual = PowerLens(r, order)

    Taylens(i, j, residual)

end

function (*)(Lϕ::Taylens, f::FlatField)
    require_unbatched(f)
    function nearest_pixel_remapping(Łf)
        Łf.arr .= getindex.(Ref(copy(Łf.arr)), Lϕ.i, Lϕ.j, (reshape(1:Npol, 1, 1, :) for Npol in size(Łf.arr)[3:end])...)
        Łf
    end
    @unpack ∇1ϕᵖ, ∇2ϕᵖ, order = Lϕ.residual
    Ðf = Ð(f)
    # nearest pixel remapping
    f̃ = nearest_pixel_remapping(Ł(f))
    # add in residual Taylor correction
    for n in 1:order, (a,b) in zip(0:n,n:-1:0)
        @. f̃ += ∇1ϕᵖ[a] * ∇2ϕᵖ[b] * $(nearest_pixel_remapping(Ł(∇[1]^a * ∇[2]^b * Ðf))) / factorial(a) / factorial(b)
    end
    f̃
end