@testset "EquiRect block contractions" begin
    T = Float32
    Ny, Nx, Nk = 4, 8, 5
    proj = ProjEquiRect(; Ny, Nx, T, θspan=(0.45,2.35), φspan=(0.0,2π), storage=Array)
    input = reshape(T[sin(i/7)+0.2cos(i/3) for i in 1:Ny*Nx], Ny, Nx)
    direction = reshape(T[cos(i/9)-0.15sin(i/5) for i in 1:Ny*Nx], Ny, Nx)
    f0 = AzFourier(EquiRectMap(input, proj))
    d0 = AzFourier(EquiRectMap(direction, proj))
    A = Array{Complex{T}}(undef, Ny, Ny, Nk)
    B = similar(A)
    for k in 1:Nk, j in 1:Ny, i in 1:Ny
        ai = (k == 1 || k == Nk) ? 0 : 0.021*(i+2j+k)
        bi = (k == 1 || k == Nk) ? 0 : 0.014*(2i-j+k)
        A[i,j,k] = Complex{T}((i == j ? 1.25 : 0)+0.04*(i-2j)+0.013k, ai)
        B[i,j,k] = Complex{T}((i == j ? 0.75 : 0)+0.03*(2i-j)+0.009k, bi)
    end
    op0 = BlockDiagEquiRect{AzFourier}(A, proj)
    other0 = BlockDiagEquiRect{AzFourier}(B, proj)
    dense_apply(blocks, arr; adj=false) = hcat(((adj ? blocks[:,:,k]' : blocks[:,:,k])*arr[:,k] for k in axes(arr,2))...)
    expected_forward = dense_apply(A, f0.arr)
    expected_adjoint = dense_apply(A, f0.arr; adj=true)
    expected_gradient = 2 .* dense_apply(A, expected_forward; adj=true)
    dense_product(left, right) = cat((left[:,:,k]*right[:,:,k] for k in axes(left,3))...; dims=3)
    adjblocks(blocks) = permutedims(conj.(blocks), (2,1,3))

    # CPU is always exercised. Modern CUDA stacks must load the extension;
    # only known legacy versions may skip the device checks.
    targets = Tuple{String,Function}[("CPU", identity)]
    ext = nothing
    if storage != Array
        ext = if isdefined(CMBLensing, :CMBLensingCUDAEquiRectExt)
            getfield(CMBLensing, :CMBLensingCUDAEquiRectExt)
        else
            nothing
        end
        modern = isdefined(CUDA, :CUDABackend) && isdefined(CUDA, :AnyCuArray) && CMBLensing.versionof(CMBLensing.Tullio) >= v"0.3.7"
        if modern
            @test ext !== nothing
            push!(targets, ("CUDA", maybegpu))
        else
            @info "Skipping optional CUDA EquiRect contractions on this legacy dependency stack"
        end
    end

    for (label, place) in targets
        @testset "$label" begin
            f, d, op, other = place.((f0, d0, op0, other0))
            @test isapprox(Array((op*f).arr), expected_forward; rtol=1e-4, atol=1e-5)
            @test isapprox(Array((op'*f).arr), expected_adjoint; rtol=1e-4, atol=1e-5)
            @test isapprox(Array((op*other).blocks), dense_product(A,B); rtol=1e-4, atol=1e-5)
            @test isapprox(Array((op'*other).blocks), dense_product(adjblocks(A),B); rtol=1e-4, atol=1e-5)
            @test isapprox(Array((op*other').blocks), dense_product(A,adjblocks(B)); rtol=1e-4, atol=1e-5)
            realop = place(BlockDiagEquiRect{AzFourier}(real.(A), proj))
            @test isapprox(Array((realop*f).arr), dense_apply(real.(A), f0.arr); rtol=1e-4, atol=1e-5)
            dense_dot = sum(dot(adjoint(A[:,:,k]), B[:,:,k]) for k in 1:Nk)
            @test isapprox(dot(op', other), dense_dot; rtol=1e-4, atol=1e-5)

            objective(x) = dot(op*x, op*x)
            grad = only(Zygote.gradient(objective, f))
            @test isapprox(Array(grad.arr), expected_gradient; rtol=1e-4, atol=1e-5)
            h = 0.01f0
            finite = (objective(f+h*d)-objective(f-h*d))/(2h)
            @test isapprox(finite, dot(grad,d); rtol=1e-3, atol=1e-4)

            # Exercise the public operator argument's pullback as well as the
            # field derivative above. Dense outer products are independent of
            # the Tullio implementation.
            y, pullback = CMBLensing.rrule(*, op, f)
            _, op̄, f̄ = pullback(y)
            op̄ = CMBLensing.unthunk(op̄)
            expected_op̄ = cat((expected_forward[:,k] * f0.arr[:,k]' for k in 1:Nk)...; dims=3)
            actual_op̄ = permutedims(conj.(Array(op̄.parent.blocks)), (2,1,3))
            @test isapprox(actual_op̄, expected_op̄; rtol=1e-4, atol=1e-5)
            @test isapprox(Array(f̄.arr), expected_gradient ./ 2; rtol=1e-4, atol=1e-5)

            if label == "CUDA"
                viewop = BlockDiagEquiRect{AzFourier}(view(op.blocks,:,:,:), op.proj)
                for (fun, args) in (
                    (*, (op, f)), (*, (op', f)), (*, (op, other)),
                    (*, (op', other)), (*, (op, other')),
                    (dot, (op', other)), (CMBLensing.rrule, (*, op, f)),
                    (*, (viewop, f)),
                )
                    @test which(fun, Tuple{map(typeof, args)...}).module === ext
                end
                # Mixed host/device inputs are not supported, but dispatch
                # should still be deterministic and should not be mistaken
                # for a second all-device method.
                @test which(*, Tuple{typeof(op),typeof(f0)}).module === ext
                @test which(*, Tuple{typeof(op0),typeof(f)}).module === CMBLensing
                @test isapprox(Array((viewop*f).arr), expected_forward; rtol=1e-4, atol=1e-5)
            end
        end
    end

    # Polarized blocks use a different basis and twice as many rows, but the
    # same contraction should agree with independent dense matrix products.
    fpol0 = QUAzFourier(EquiRectQUMap(input, direction, proj))
    n = size(fpol0.arr,1)
    P = Array{Complex{T}}(undef, n, n, Nk)
    for k in 1:Nk, j in 1:n, i in 1:n
        P[i,j,k] = Complex{T}((i == j ? 1 : 0)+0.01*(i-2j), 0.005*(i+j+k))
    end
    pop0 = BlockDiagEquiRect{QUAzFourier}(P, proj)
    pol_forward = dense_apply(P, fpol0.arr)
    pol_adjoint = dense_apply(P, fpol0.arr; adj=true)
    for (label, place) in targets
        @testset "$label polarized" begin
            pop, fpol = place.((pop0, fpol0))
            @test isapprox(Array((pop*fpol).arr), pol_forward; rtol=1e-4, atol=1e-5)
            @test isapprox(Array((pop'*fpol).arr), pol_adjoint; rtol=1e-4, atol=1e-5)
            if label == "CUDA"
                @test which(*, Tuple{typeof(pop),typeof(fpol)}).module === ext
                @test which(*, Tuple{typeof(pop'),typeof(fpol)}).module === ext
            end
        end
    end
end
