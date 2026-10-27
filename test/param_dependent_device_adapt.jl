using CMBLensing, Adapt, LinearAlgebra, Test

# The recomputation function intentionally returns a CPU operator even after
# the ParamDependentOp's cached operator has been adapted to another backend.
const param_dependent_cpu_operator = Ref{Any}()
param_dependent_cpu_recompute(; amplitude=1, _...) = amplitude * param_dependent_cpu_operator[]

@testset "ParamDependentOp recomputation device transfer" begin
    for complex_data in (false, true), source_type in (Float32, Float64)
        p32 = ProjLambert(; Ny=8, Nx=8, θpix=4, T=Float32)
        p64 = ProjLambert(; Ny=8, Nx=8, θpix=4, T=Float64)
        values64 = reshape(Float64.(1:40), 5, 8)
        if complex_data
            values64 = complex.(values64, reverse(values64; dims=2))
        end
        values32 = complex_data ? ComplexF32.(values64) : Float32.(values64)
        cpu32 = Diagonal(FlatFourier(values32, p32))
        cpu64 = Diagonal(FlatFourier(values64, p64))
        source, other = source_type === Float32 ? (cpu32, cpu64) : (cpu64, cpu32)
        source_values = source_type === Float32 ? values32 : values64
        source_proj = source_type === Float32 ? p32 : p64
        param_dependent_cpu_operator[] = source

        # Same-backend evaluation already preserves the recomputed operator's
        # precision, even if the cached operator has a different precision.
        same_backend = ParamDependentOp(other, param_dependent_cpu_recompute, [:amplitude])
        same = same_backend((; amplitude=source_type(1.25)))
        @test eltype(diag(same).arr) == eltype(source_values)
        @test eltype(diag(same).proj.ℓx) == source_type
        @test diag(same).arr ≈ source_type(1.25) .* source_values

        if storage != Array
            # Exercise matching and mixed precision across CPU -> GPU. A
            # concrete 2-D array type cannot adapt the 1-D projection vectors.
            for cached in (source, other)
                gpu = ParamDependentOp(adapt(storage, cached), param_dependent_cpu_recompute, [:amplitude])
                result = gpu((; amplitude=source_type(1.25)))
                f = diag(result)
                @test f.arr isa CuArray
                @test eltype(f.arr) == eltype(source_values)
                @test Array(f.arr) ≈ source_type(1.25) .* source_values
                @test (f.proj.Ny, f.proj.Nx, f.proj.θpix, f.proj.rotator) ==
                    (source_proj.Ny, source_proj.Nx, source_proj.θpix, source_proj.rotator)
                for name in (:ℓx, :ℓy, :λ_rfft, :ℓmag, :sin2ϕ, :cos2ϕ)
                    actual, expected = getproperty(f.proj, name), getproperty(source_proj, name)
                    @test actual isa CuArray
                    @test eltype(actual) == eltype(expected)
                    @test Array(actual) ≈ expected
                end
            end
        end
    end
end
