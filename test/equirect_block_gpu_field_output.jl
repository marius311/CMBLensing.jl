# Device-only controls for a strided field view. The public CPU regression is
# in equirect_block_field_output.jl and always runs, including without CUDA.
@testset "CUDA EquiRect block application with field views" begin
    modern = isdefined(CUDA, :CUDABackend) && isdefined(CUDA, :AnyCuArray) && CMBLensing.versionof(CMBLensing.Tullio) >= v"0.3.7"
    if modern
        @test isdefined(CMBLensing, :CMBLensingCUDAEquiRectExt)
        if isdefined(CMBLensing, :CMBLensingCUDAEquiRectExt)
            ext = getfield(CMBLensing, :CMBLensingCUDAEquiRectExt)
            Ny, Nx, Nk = 4, 8, 5
            p = ProjEquiRect(; Ny, Nx, T=Float32, θspan=(0.45,2.35), φspan=(0.0,2π), storage=Array)
            pg = adapt(CUDA.CuArray, p)
            input = reshape(Float32[sin(i/7)+0.2cos(i/3) for i in 1:Ny*Nx], Ny, Nx)
            direction = reshape(Float32[cos(i/9)-0.15sin(i/5) for i in 1:Ny*Nx], Ny, Nx)
            f0 = AzFourier(EquiRectMap(input, p))
            d0 = AzFourier(EquiRectMap(direction, p))
            blocks(T, n) = [complex(T((i == j ? 1.1 : 0) + 0.006*(i-2j+k)),
                                    T(k == 1 || k == Nk ? 0 : 0.004*(i+2j+k)))
                            for i in 1:n, j in 1:n, k in 1:Nk]
            dense_apply(A, x; adj=false) = hcat(((adj ? A[:,:,k]' : A[:,:,k])*x[:,k] for k in 1:Nk)...)
            function device_view_field(B, data, p)
                n, nk = size(data)
                parent = CUDA.zeros(eltype(data), 2n, nk)
                arr = view(parent, 1:2:2n, :)
                arr .= CUDA.CuArray(data)
                CMBLensing.BaseField{B}(arr, p)
            end

            for (B,n) in ((AzFourier,Ny), (QUAzFourier,2Ny), (IQUAzFourier,3Ny))
                data = B == AzFourier ? f0.arr :
                    reshape(ComplexF32[complex(sin(i/9),cos(i/11)) for i in 1:n*Nk], n, Nk)
                f = device_view_field(B,data,pg)
                A = blocks(Float32,n)
                op = BlockDiagEquiRect{B}(CUDA.CuArray(A),pg)
                @test f.arr isa SubArray
                @test f.arr isa CUDA.AnyCuArray
                @test which(*, Tuple{typeof(op),typeof(f)}).module === ext
                @test which(*, Tuple{typeof(op'),typeof(f)}).module === ext
                for (y,expected) in ((op*f,dense_apply(A,data)), (op'*f,dense_apply(A,data; adj=true)))
                    @test y isa CMBLensing.BaseField{B}
                    @test y.proj === pg
                    @test isapprox(Array(y.arr),expected;rtol=1f-4,atol=1f-5)
                end
            end

            p64 = ProjEquiRect(; Ny, Nx, T=Float64, θspan=(0.45,2.35), φspan=(0.0,2π), storage=Array)
            p64g = adapt(CUDA.CuArray,p64)
            data32 = ComplexF32.(f0.arr)
            f32 = device_view_field(AzFourier,data32,p64g)
            A64 = blocks(Float64,Ny)
            op64 = BlockDiagEquiRect{AzFourier}(CUDA.CuArray(A64),p64g)
            for (y,expected) in ((op64*f32,dense_apply(A64,data32)), (op64'*f32,dense_apply(A64,data32; adj=true)))
                @test y.proj === p64g
                @test eltype(y.arr) === ComplexF64
                @test isapprox(Array(y.arr),expected;rtol=1e-10,atol=1e-11)
            end

            f = device_view_field(AzFourier,f0.arr,pg)
            d = CMBLensing.BaseField{AzFourier}(CUDA.CuArray(d0.arr),pg)
            A = blocks(Float32,Ny)
            op = BlockDiagEquiRect{AzFourier}(CUDA.CuArray(A),pg)
            objective(x) = dot(op*x,op*x)
            g = only(Zygote.gradient(objective,f))
            h = 0.01f0
            finite = (objective(f+h*d)-objective(f-h*d))/(2h)
            @test isapprox(dot(g,d),finite;rtol=1f-3,atol=1f-4)
            @test isapprox(Array(g.arr),2 .* dense_apply(A,dense_apply(A,f0.arr); adj=true);rtol=1f-4,atol=1f-5)
        end
    else
        @info "Skipping CUDA EquiRect field-view tests on this legacy dependency stack"
    end
end
