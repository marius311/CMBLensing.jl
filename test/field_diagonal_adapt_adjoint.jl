using CMBLensing, Adapt, LinearAlgebra, Test, Zygote

@testset "field diagonal adaptation pullback" begin
    for T in (Float32, Float64), complex_data in (false, true), isview in (false, true)
        other = T === Float32 ? Float64 : Float32
        values = reshape(T.(1:40) ./ T(10), 5, 8)
        values = complex_data ? complex.(values, reverse(values; dims=2)) : values
        wider_values = complex_data ? Complex{other}.(values) : other.(values)
        projection = ProjLambert(; Ny=8, Nx=8, θpix=4, T)
        other_projection = ProjLambert(; Ny=8, Nx=8, θpix=4, T=other)
        cpu = FlatFourier(values, projection)
        cases = storage == Array ? ((cpu, Array),) :
            ((cpu, Array), (cpu, storage), (adapt(storage, cpu), Array))
        for (field, target) in cases
            input_array = isview ? view(field.arr, :, :) : field.arr
            input = Diagonal(CMBLensing.BaseField{Fourier}(input_array, field.metadata))
            output, back = Zygote.pullback(x -> adapt(target, x), input)
            cotangent = only(back(output))
            @test Array(diag(cotangent).arr) ≈ values
            @test CMBLensing._fft_arr_type(diag(cotangent).arr) ===
                CMBLensing._fft_arr_type(input_array)
            @test Array(diag(input).arr) ≈ values
            @test Array(diag(output).arr) ≈ values

            # Preserve the seed's precision while returning it to the input's
            # device, consistent with the generic array adaptation rule.
            seed = adapt(target, Diagonal(FlatFourier(wider_values, other_projection)))
            returned_seed = only(back(seed))
            @test eltype(diag(returned_seed).arr) === eltype(wider_values)
            @test Array(diag(returned_seed).arr) ≈ wider_values
            @test CMBLensing._fft_arr_type(diag(returned_seed).arr) ===
                CMBLensing._fft_arr_type(input_array)
            @test CMBLensing._fft_arr_type(diag(returned_seed).metadata.ℓx) ===
                CMBLensing._fft_arr_type(input_array)
            for zero in (nothing, CMBLensing.ChainRulesCore.ZeroTangent())
                result = back(zero)
                @test isnothing(result) || (length(result) == 1 &&
                    (isnothing(result[1]) || result[1] isa CMBLensing.ChainRulesCore.AbstractZero))
            end

            objective(a) = sum(real.(diag(adapt(target, a*input)).arr))
            a = T(1.25)
            h = T === Float32 ? T(1e-3) : T(1e-5)
            tol = T === Float32 ? 2e-2 : 3e-7
            reverse_grad = gradient(objective, a)[1]
            finite = (objective(a+h) - objective(a-h)) / (2h)
            @test isapprox(reverse_grad, sum(real.(values)); atol=tol, rtol=tol)
            @test isapprox(reverse_grad, finite; atol=tol, rtol=tol)
        end
    end

    # A FieldTuple can hold components with different element types and bases;
    # it must continue using its existing adaptation rule.
    tuplefield = FieldTuple(; map=FlatMap(ones(Float32,4,4)),
        fourier=Fourier(FlatMap(ones(Float64,4,4))))
    tuplediag = Diagonal(tuplefield)
    _, tuple_back = Zygote.pullback(x -> adapt(Array, x), tuplediag)
    returned_tuple = only(tuple_back(tuplediag))
    @test diag(returned_tuple).map.arr == tuplefield.map.arr
    @test diag(returned_tuple).fourier.arr == tuplefield.fourier.arr

    # The generic raw-array adaptation rule remains in use.
    raw = reshape(collect(1.0:12.0), 3, 4)
    @test gradient(a -> sum(adapt(Array, a*raw)), 1.5)[1] ≈ sum(raw)
    if storage != Array
        gpu_raw = adapt(storage, raw)
        @test gradient(a -> sum(adapt(storage, a*raw)), 1.5)[1] ≈ sum(raw)
        @test gradient(a -> sum(adapt(Array, a*gpu_raw)), 1.5)[1] ≈ sum(raw)
    end
end
