@testset "EquiRect block application returns the computed field type" begin
    Ny, Nx, Nk = 4, 8, 5
    proj = ProjEquiRect(; Ny, Nx, T=Float32, θspan=(0.45,2.35), φspan=(0.0,2π), storage=Array)
    input = reshape(Float32[sin(i/7)+0.2cos(i/3) for i in 1:Ny*Nx], Ny, Nx)
    direction = reshape(Float32[cos(i/9)-0.15sin(i/5) for i in 1:Ny*Nx], Ny, Nx)
    f0 = AzFourier(EquiRectMap(input, proj))
    d = AzFourier(EquiRectMap(direction, proj))

    blocks(T, n) = [complex(T((i == j ? 1.1 : 0) + 0.006*(i-2j+k)),
                            T(k == 1 || k == Nk ? 0 : 0.004*(i+2j+k)))
                    for i in 1:n, j in 1:n, k in 1:Nk]
    dense_apply(A, x; adj=false) = hcat(((adj ? A[:,:,k]' : A[:,:,k])*x[:,k] for k in 1:Nk)...)
    function view_field(B, x, p)
        n, nk = size(x)
        parent = zeros(eltype(x), 2n, nk)
        arr = view(parent, 1:2:2n, :)
        arr .= x
        CMBLensing.BaseField{B}(arr, p)
    end

    # Each view skips rows of a padded parent, so it cannot be treated as a
    # dense matrix. Polarized fields here exercise the block algebra directly.
    for (B,n) in ((AzFourier,Ny), (QUAzFourier,2Ny), (IQUAzFourier,3Ny))
        data = B == AzFourier ? f0.arr :
            reshape(ComplexF32[complex(sin(i/9),cos(i/11)) for i in 1:n*Nk], n, Nk)
        f = view_field(B, data, proj)
        A = blocks(Float32, n)
        op = BlockDiagEquiRect{B}(A, proj)
        @test f.arr isa SubArray
        @test stride(f.arr,1) == 2
        for (y, expected) in ((op*f, dense_apply(A,data)), (op'*f, dense_apply(A,data; adj=true)))
            @test y isa CMBLensing.BaseField{B}
            @test y.proj === proj
            @test isapprox(y.arr, expected; rtol=1f-5, atol=1f-5)
        end
    end

    # The field and operator share the same projection, but Tullio promotes
    # the actual array result to ComplexF64.
    proj64 = ProjEquiRect(; Ny, Nx, T=Float64, θspan=(0.45,2.35), φspan=(0.0,2π), storage=Array)
    f32 = CMBLensing.BaseField{AzFourier}(ComplexF32.(f0.arr), proj64)
    A64 = blocks(Float64, Ny)
    op64 = BlockDiagEquiRect{AzFourier}(A64, proj64)
    for (y, expected) in ((op64*f32, dense_apply(A64,f32.arr)),
                          (op64'*f32, dense_apply(A64,f32.arr; adj=true)))
        @test y.proj === proj64
        @test eltype(y.arr) === ComplexF64
        @test isapprox(y.arr, expected; rtol=1e-12, atol=1e-12)
    end

    f = view_field(AzFourier, f0.arr, proj)
    A = blocks(Float32, Ny)
    op = BlockDiagEquiRect{AzFourier}(A, proj)
    objective(x) = dot(op*x, op*x)
    grad = only(Zygote.gradient(objective, f))
    h = 0.01f0
    finite = (objective(f+h*d)-objective(f-h*d))/(2h)
    @test isapprox(dot(grad,d), finite; rtol=1f-3, atol=1f-4)
    @test isapprox(grad.arr, 2 .* dense_apply(A,dense_apply(A,f.arr); adj=true); rtol=1f-4, atol=1f-5)
end
