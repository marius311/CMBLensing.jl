using CMBLensing
using CMBLensing: @SMatrix
using LinearAlgebra
using Test

@testset "Nonsymmetric 2×2 diagonal-operator matrices" begin
    for T in (Float32, Float64), dims in ((4, 4), (4, 4, 2))
        # Distinct off-diagonal entries and positive eigenvalues at every
        # pixel. The third dimension exercises the batched-field layout.
        x = reshape(T.(1:prod(dims)), dims) ./ T(prod(dims))
        a = @. T(2) + T(0.1) * x
        b = @. T(3) + T(0.2) * x
        c = @. T(1) + T(0.05) * x
        d = @. T(4) + T(0.1) * x
        field(arr) = FlatMap(arr; θpix=3)
        A = @SMatrix [Diagonal(field(a)) Diagonal(field(b));
                      Diagonal(field(c)) Diagonal(field(d))]

        determinant = det(A).diag.arr
        root = sqrt(A)
        inverse = pinv(A)
        inverse_inplace = similar.(A)
        CMBLensing.pinv!(inverse_inplace, A)

        for k in eachindex(a)
            dense = T[a[k] b[k]; c[k] d[k]]
            root_ref, inverse_ref = sqrt(dense), inv(dense)
            @test determinant[k] ≈ det(dense) rtol=10eps(T)
            for i in 1:2, j in 1:2
                @test root[i,j].diag.arr[k] ≈ root_ref[i,j] rtol=100eps(T)
                @test inverse[i,j].diag.arr[k] ≈ inverse_ref[i,j] rtol=100eps(T)
                @test inverse_inplace[i,j].diag.arr[k] ≈ inverse_ref[i,j] rtol=100eps(T)
            end
        end
    end
end
