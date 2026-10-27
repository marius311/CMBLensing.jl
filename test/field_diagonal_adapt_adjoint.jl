using CMBLensing, Adapt, LinearAlgebra, Test, Zygote

field_diagonal_test_backend(arr) = CMBLensing.basetype(typeof(parent(arr)))

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
            @test field_diagonal_test_backend(diag(cotangent).arr) ===
                field_diagonal_test_backend(input_array)
            @test Array(diag(input).arr) ≈ values
            @test Array(diag(output).arr) ≈ values

            # Preserve the seed's precision while returning it to the input's
            # device, consistent with the generic array adaptation rule.
            seed = adapt(target, Diagonal(FlatFourier(wider_values, other_projection)))
            returned_seed = only(back(seed))
            @test eltype(diag(returned_seed).arr) === eltype(wider_values)
            @test Array(diag(returned_seed).arr) ≈ wider_values
            @test field_diagonal_test_backend(diag(returned_seed).arr) ===
                field_diagonal_test_backend(input_array)
            @test field_diagonal_test_backend(diag(returned_seed).metadata.ℓx) ===
                field_diagonal_test_backend(input_array)
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

    # Backing arrays can contain more than one wrapper.  In particular, a
    # noncontiguous GPU view may sit under a PermutedDimsArray.
    template = FlatMap(reshape(Float32.(1:16)./10,4,4))
    cpu_buffer = vcat(vec(template.arr),vec(template.arr))
    cpu_nested = reshape(view(reshape(view(cpu_buffer,1:16),2,8),:,1:8),4,4)
    @test !(parent(cpu_nested) isa DenseArray)
    cpu_field = CMBLensing.BaseField{Map}(cpu_nested,template.metadata)
    if storage == Array
        nested_cases = ((cpu_field,Array,Array),)
    else
        gpu_template = adapt(storage,template)
        gpu_flat = vec(gpu_template.arr)
        gpu_buffer = reshape(vcat(gpu_flat,gpu_flat,gpu_flat,gpu_flat),8,8)
        gpu_nested = PermutedDimsArray(view(gpu_buffer,1:2:8,1:2:8),(2,1))
        @test !(parent(gpu_nested) isa DenseArray)
        gpu_field = CMBLensing.BaseField{Map}(gpu_nested,gpu_template.metadata)
        nested_cases = ((cpu_field,storage,Array),(gpu_field,Array,storage))
    end
    for (field,target,backend) in nested_cases
        input = Diagonal(field)
        original = Array(field.arr)
        _,back = Zygote.pullback(x->adapt(target,x),input)
        seed = adapt(target,Diagonal(FlatMap(reshape(Float64.(16:-1:1)./20,4,4))))
        seed_values = Array(diag(seed).arr)
        returned = diag(only(back(seed)))
        @test field_diagonal_test_backend(returned.arr) === backend
        @test field_diagonal_test_backend(returned.metadata.ℓx) === backend
        @test eltype(returned.arr) === Float64
        @test Array(returned.arr) ≈ seed_values
        @test Array(field.arr) == original
        @test Array(diag(seed).arr) == seed_values
    end

    # An unrecognized wrapper retains the prior one-parent fallback.
    unknown = transpose(cpu_nested)
    unknown_input = Diagonal(CMBLensing.BaseField{Map}(unknown,template.metadata))
    for target in (storage == Array ? (Array,) : (Array,storage))
        _,back = Zygote.pullback(x->adapt(target,x),unknown_input)
        seed = adapt(target,Diagonal(FlatMap(ones(Float64,4,4))))
        expected = diag(adapt(CMBLensing.basetype(typeof(parent(unknown))),seed))
        returned = diag(only(back(seed)))
        @test field_diagonal_test_backend(returned.arr) === field_diagonal_test_backend(expected.arr)
        @test Array(returned.arr) == Array(expected.arr)
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
