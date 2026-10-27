@testset "CUDA zero-dimensional reshape" begin
    a = CuArray([1.0])
    scalar = reshape(a, ())
    @test scalar isa CuArray
    @test size(scalar) == ()
    @test Array(scalar)[] == 1.0
    @test reshape(scalar, ()) === scalar

    fill!(scalar, 2.0)
    @test Array(a) == [2.0]
    @test_throws DimensionMismatch reshape(CuArray([1.0, 2.0]), ())
end
