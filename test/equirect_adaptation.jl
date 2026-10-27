using CMBLensing, Adapt, LinearAlgebra, Serialization, Test, Zygote

struct EquiRectFloat32Adaptor end
Adapt.adapt_storage(::EquiRectFloat32Adaptor, x::Array{T}) where {T<:AbstractFloat} = Float32.(x)

@testset "EquiRect projection adaptation" begin
    components = (:θ, :φ, :θedges, :φedges, :Ω, :ℓx)
    manual(T) = ProjEquiRect(; θ=[0.4, 0.95, 1.7, 2.4],
        φ=[0.15, 0.7, 1.4, 2.2], θedges=[0.2, 0.65, 1.3, 2.1, 2.8],
        φedges=[0.0, 0.4, 1.0, 1.8, 2.6], T)
    regular(T) = ProjEquiRect(; Ny=4, Nx=8, θspan=(0.2, 2.8),
        φspan=(π/8, 2π+π/8), T)

    for make_projection in (manual, regular), T in (Float32, Float64)
        proj = make_projection(T)
        @test adapt(Array, proj) === proj
        @test adapt(nothing, proj) === proj

        other = T === Float32 ? Float64 : Float32
        for (target, S) in ((Array{other}, other), (EquiRectFloat32Adaptor(), Float32))
            converted = adapt(target, proj)
            @test converted.T === S
            @test converted.θspan === proj.θspan && converted.φspan === proj.φspan
            @test all(k -> eltype(getproperty(converted, k)) === S, components)
            @test all(k -> getproperty(converted, k) == S.(getproperty(proj, k)), components)
        end

        # Adaptation of a field and its reverse derivative must keep the same
        # stored geometry as direct projection adaptation.
        values = reshape(T.(1:proj.Ny*proj.Nx) ./ T(10), proj.Ny, proj.Nx)
        field = EquiRectMap(values, proj)
        converted = adapt(Array{other}, field)
        @test converted.arr == other.(values)
        @test all(k -> getproperty(converted.metadata, k) == other.(getproperty(proj, k)), components)
        grad = only(gradient(x -> sum(abs2, adapt(Array{other}, x).arr), field))
        @test Array(grad.arr) ≈ 2values rtol=3e-7
        @test all(k -> getproperty(grad.metadata, k) == eltype(grad.arr).(getproperty(converted.metadata, k)), components)

        io = IOBuffer()
        serialize(io, converted.metadata)
        seekstart(io)
        restored = deserialize(io)
        @test restored.θspan === proj.θspan && restored.φspan === proj.φspan
        @test all(k -> getproperty(restored, k) == getproperty(converted.metadata, k), components)

        if storage != Array
            device = adapt(storage, field)
            @test all(k -> getproperty(device.metadata, k) isa storage, components)
            @test Array(device.arr) == values
            @test all(k -> Array(getproperty(device.metadata, k)) == getproperty(proj, k), components)
            roundtrip = adapt(Array, device)
            @test all(k -> getproperty(roundtrip.metadata, k) == getproperty(proj, k), components)
            grad_device = only(gradient(x -> sum(abs2, adapt(storage, x).arr), field))
            @test Array(grad_device.arr) ≈ 2values rtol=3e-7
            @test all(k -> Array(getproperty(grad_device.metadata, k)) == getproperty(proj, k), components)
        end
    end
end
