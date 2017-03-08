push!(LOAD_PATH, pwd()*"/src")
using Base.Test
using CMBFields


nside = 64
P = Flat{1,nside}
T = Float64

d = rand(nside,nside)
f = FlatS0Map{T,P}(d)

@testset "vector algebra" begin

    # vec inner
    @test ([f,f]' * [f,f]).Tx == 2.*d.^2
    @test_noerror [∂x,∂x]' * [∂x,∂x]

    # vec outer
    @test all(map(f->f.Tx == d.^2,([f,f] * [f,f]')))
    @test_noerror [∂x,∂x] * [∂x,∂x]'
    
    # vec inner op on fields
    @test_noerror [∂x,∂y]' * [f,f]

    # op broadcasting
    @test_noerror ([∂x,∂y]*f) :: Vector
    
    # matrix*vector
    @test_noerror ([f f; f f] * [f, f]) :: Matrix
    @test_noerror ([∂x ∂x; ∂x ∂x] * [f, f]) :: Matrix
    @test_noerror ([∂x ∂x; ∂x ∂x] * [∂x, ∂x]) :: Matrix

    # vector*matrix
    @test_noerror ([f f] * [f f; f f]) :: Matrix
    @test_noerror ([∂x ∂x] * [∂x ∂x; ∂x ∂x]) :: Matrix
    
    # matrix*matrix
    @test_noerror ([f f; f f] * [f f; f f]) :: Matrix
end
