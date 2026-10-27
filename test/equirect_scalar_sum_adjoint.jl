@testset "EquiRect scalar Fourier sum gradients" begin
    for T in (Float32, Float64), nx in (7, 8)
        ny = 3
        proj = ProjEquiRect(; Ny=ny, Nx=nx, T, θspan=(0.4,2.3), φspan=(0,2π), storage=Array)
        values = reshape(T[sin(i/7) + cos(i/11) for i in 1:ny*nx], ny, nx)
        direction = reshape(T[cos(i/5) - sin(i/13) for i in 1:ny*nx], ny, nx)
        f = adapt(storage, AzFourier(EquiRectMap(values, proj)))
        d = adapt(storage, AzFourier(EquiRectMap(direction, proj)))
        h = T === Float32 ? T(0.01) : T(1e-5)
        tol = T === Float32 ? 2e-3 : 2e-7
        weight = complex(T(0.7), T(1.3))
        # Include unconstrained complex endpoint entries in the raw direction.
        arr_direction = adapt(storage, reshape(Complex{T}[
            complex(sin(i/9), cos(i/8)) for i in 1:length(f.arr)], size(f.arr)))

        objectives = (x -> real(sum(x)), x -> imag(sum(x)),
                      x -> real(weight * sum(x)))
        for objective in objectives
            grad = only(Zygote.gradient(objective, f))
            finite = (objective(f + h*d) - objective(f - h*d)) / (2h)
            @test eltype(grad.arr) === eltype(f.arr)
            @test isapprox(dot(grad, d), finite; rtol=tol, atol=tol)

            # Raw coefficient perturbations also exercise the constructor.
            field = a -> CMBLensing.BaseField{AzFourier}(a, f.metadata)
            raw_objective = a -> objective(field(a))
            raw_gradient = only(Zygote.gradient(raw_objective, f.arr))
            raw_finite = (raw_objective(f.arr + h*arr_direction) -
                          raw_objective(f.arr - h*arr_direction)) / (2h)
            raw_actual = real(dot(vec(Array(raw_gradient)), vec(Array(arr_direction))))
            @test isapprox(raw_actual, raw_finite; rtol=tol, atol=tol)
        end

        # The FieldTuple rule delegates to each component's sum pullback.
        ft = FieldTuple(; field=f, scalar=T(2))
        for objective in objectives
            ft_gradient = only(Zygote.gradient(objective, ft))
            tuple_objective = a -> objective(FieldTuple(; field=f+a*d, scalar=T(2)+a*T(0.3)))
            finite = (tuple_objective(h) - tuple_objective(-h)) / (2h)
            @test isapprox(dot(ft_gradient.field, d) + T(0.3)*ft_gradient.scalar,
                           finite; rtol=tol, atol=tol)
        end

        _, back = Zygote.pullback(sum, f)
        seed = complex(T(0.3), T(0.7))
        first = only(back(seed))
        @test Array(only(back(2seed)).arr) ≈ 2 .* Array(first.arr)
        thunk = CMBLensing.ChainRulesCore.@thunk seed
        @test Array(only(back(thunk)).arr) ≈ Array(first.arr)
        for zero_seed in (nothing, CMBLensing.ChainRulesCore.ZeroTangent())
            result = back(zero_seed)
            @test result === nothing || (length(result) == 1 &&
                (result[1] === nothing || result[1] isa CMBLensing.ChainRulesCore.AbstractZero))
        end
    end

    # Wider metadata must not change the stored cotangent precision.
    proj = ProjEquiRect(; Ny=3, Nx=8, T=Float64, θspan=(0.4,2.3), φspan=(0,2π))
    f = CMBLensing.BaseField{AzFourier}(zeros(ComplexF32, 3, 5), proj)
    for objective in (x -> real(sum(x)), x -> imag(sum(x)))
        @test eltype(only(Zygote.gradient(objective, f)).arr) === ComplexF32
    end
end
