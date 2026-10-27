using CMBLensing
using CMBLensing: Ł!
using FFTW
using Test

@testset "in-place inverse FFT preserves array-backed views" begin
    for T in (Float32, Float64), N in (7, 8, 9), nbat in (1, 2)
        # MKL cannot run this package's batched CPU FFT path.
        nbat > 1 && FFTW.fftw_provider == "mkl" && continue
        x = T[sin(T(i)) + cos(T(j)/3) for i in 1:N, j in 1:N]
        input = nbat == 1 ? x : cat(x, T(0.4) .* x; dims=3)
        f = FlatMap(input)
        F = Fourier(f)
        inds = ntuple(_ -> Colon(), ndims(F.arr))
        backing = copy(F.arr)
        Fview = CMBLensing.BaseField{Fourier}(view(backing, inds...), F.metadata)
        before = copy(backing)
        dst = similar(f)
        Ł!(dst, Fview)
        @test backing == before
        @test isapprox(dst.arr, f.arr; rtol=100eps(T), atol=100eps(T))
        # Repeating the conversion should give the same answer.
        Ł!(dst, Fview)
        @test backing == before
        @test isapprox(dst.arr, f.arr; rtol=100eps(T), atol=100eps(T))
    end
end
