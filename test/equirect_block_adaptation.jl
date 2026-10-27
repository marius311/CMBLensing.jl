using CMBLensing, Adapt, LinearAlgebra, Test

struct BlockPrecisionAdaptor{T} end
Adapt.adapt_storage(::BlockPrecisionAdaptor{T}, x::Array{S}) where {T,S<:AbstractFloat} = T.(x)
Adapt.adapt_storage(::BlockPrecisionAdaptor{T}, x::Array{Complex{S}}) where {T,S<:AbstractFloat} = Complex{T}.(x)

@testset "EquiRect block operator precision adaptation" begin
    projection(T) = ProjEquiRect(; θ=[0.4, 0.95, 1.7, 2.4],
        φ=[0.15, 0.7, 1.4, 2.2], θedges=[0.2, 0.65, 1.3, 2.1, 2.8],
        φedges=[0.0, 0.4, 1.0, 1.8, 2.6], T)

    for T in (Float32, Float64), complex_blocks in (false, true)
        S = T === Float32 ? Float64 : Float32
        E = complex_blocks ? Complex{T} : T
        F = complex_blocks ? Complex{S} : S
        blocks = repeat(reshape(Matrix(Diagonal(E[1, 1e-8, 2, 3])), 4, 4, 1), 1, 1, 3)
        original = BlockDiagEquiRect{AzFourier}(blocks, projection(T))
        sqrt(original); pinv(original); logabsdet(original)

        same = adapt(BlockPrecisionAdaptor{T}(), original)
        @test same.blocks == original.blocks
        @test same.blocks_sqrt[] == original.blocks_sqrt[]
        @test same.blocks_pinv[] == original.blocks_pinv[]
        @test same.logabsdet === original.logabsdet

        converted = adapt(BlockPrecisionAdaptor{S}(), original)
        reference = BlockDiagEquiRect{AzFourier}(F.(blocks), projection(S))
        @test !isassigned(converted.blocks_sqrt)
        @test !isassigned(converted.blocks_pinv)
        @test converted.logabsdet !== original.logabsdet
        @test converted.proj.θ == S.(original.proj.θ)
        @test pinv(converted).blocks == pinv(reference).blocks
        @test maximum(abs.(pinv(converted).blocks .- F.(original.blocks_pinv[]))) > 1e7
        @test sqrt(converted).blocks ≈ sqrt(reference).blocks
        @test logabsdet(converted) == logabsdet(reference)
    end

    # Rounding this matrix to Float32 changes it from nonsingular to singular.
    blocks = repeat(reshape([1.0 1.0 0 0; 1.0 1.0+1e-8 0 0;
                            0 0 2.0 0; 0 0 0 3.0], 4, 4, 1), 1, 1, 3)
    original = BlockDiagEquiRect{AzFourier}(blocks, projection(Float64))
    @test isfinite(logabsdet(original)[1])
    converted = adapt(BlockPrecisionAdaptor{Float32}(), original)
    @test logabsdet(converted)[1] == -Inf32
end
