using CMBLensing, CUDA, Random, Test

@testset "CUDA scalar normal RNG" begin
    ext = @static if isdefined(Base, :get_extension)
        Base.get_extension(CMBLensing, :CMBLensingCUDAExt)
    else
        CMBLensing.CMBLensingCUDAExt
    end
    @test ext !== nothing

    @static if isdefined(CUDA, :cuRAND)
        R = CUDA.cuRAND.LibraryRNG
        # CUDA 6 owns scalar methods, and the extension must not overwrite or
        # shadow them.
        for T in (Float32, Float64)
            @test which(randn, (R, Type{T})).module === CUDA.cuRAND
        end
        @test which(randn, (R, Union{Type{Float32},Type{Float64}})).module === CUDA.cuRAND
    else
        # Preserve the original method for CUDA 3–5, including its signature.
        signature = Tuple{typeof(randn), CUDA.CURAND.RNG, Random.BitFloatType}
        @test any(m -> m.module === ext && m.sig == signature, methods(randn))
    end
end
