push!(LOAD_PATH, pwd()*"/src")
using Base.Test
using CMBFields
using CMBFields: LazyBinaryOp
using StaticArrays

nside = 4
P = Flat{1,nside}
T = Float64

d = rand(nside,nside)
f = Fourier(FlatS0Map{T,P}(d))

fvec = @SVector [f, f]
opvec = @SVector [∂x, ∂y]
fmat = @SMatrix [f f; f f]
opmat = @SMatrix [∂x ∂y; ∂x ∂y]

@testset "vector algebra" begin

    # vec inner
    @test (fvec' ⨳ fvec).Tl == @. 2*f.Tl^2
    @test (opvec' ⨳ opvec) isa LazyBinaryOp

    # vec outer
    @test all(map(g->(g.Tl == f.Tl.^2),(fvec * fvec')))
    @test (opvec ⨳ opvec') isa StaticMatrix{<:LazyBinaryOp}
    
    # vec inner op on fields
    @test (opvec' ⨳ fvec) isa Field

    # op broadcasting
    @test (opvec*f) isa StaticVector
    
    # matrix*vector
    @test (fmat ⨳ fvec) isa StaticVector{<:Field}
    @test (opmat ⨳ fvec) isa StaticVector{<:Field}
    @test (opmat ⨳ opvec) isa StaticVector{<:LazyBinaryOp}

    # vector*matrix
    @test (fvec' ⨳ fmat) isa RowVector{<:Field}
    @test (opvec' ⨳ opmat) isa RowVector{<:LazyBinaryOp}
    
    # matrix*matrix
    @test (fmat ⨳ fmat) isa StaticMatrix{<:Field}
    @test (opmat ⨳ opmat) isa StaticMatrix{<:LazyBinaryOp}
    
end
