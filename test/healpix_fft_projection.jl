@testset "Healpix FFT projections" begin
    ny, nx = 8, 16
    hpx_proj = ProjHealpix(2)
    cart_projs = (
        ProjLambert(; Ny=ny, Nx=nx, θpix=600, T=Float64, rotator=(0, 30, 0)),
        ProjEquiRect(; Ny=ny, Nx=nx, θspan=(0, π), φspan=(0, 2π), T=Float64),
    )
    grid_i = reshape(sin.(collect(1.0:ny*nx) ./ 9), ny, nx)
    grid_q = reshape(cos.(collect(1.0:ny*nx) ./ 13), ny, nx)
    grid_u = reshape(sin.(collect(1.0:ny*nx) ./ 7), ny, nx)
    hpx_i = sin.(collect(1.0:48.0) ./ 5)
    hpx_q = cos.(collect(1.0:48.0) ./ 6)
    hpx_u = sin.(collect(1.0:48.0) ./ 8)
    weights = cos.(collect(1.0:ny*nx) ./ 5)

    for cart_proj in cart_projs
        @testset "$(typeof(cart_proj))" begin
            projector = CMBLensing.Projector(cart_proj => hpx_proj; method=:fft)
            patch = projector.hpx_idxs_in_patch
            @test !isempty(patch)

            # Compare the complete scalar projector with the direct nonuniform
            # DFT rather than another approximate FFT implementation.
            grid_nodes = reduce(hcat, [[(i-ny÷2-1)/ny, (j-nx÷2-1)/nx] for i=1:ny, j=1:nx])
            hpx_nodes = reduce(hcat, [[(projector.is[k]-ny÷2-1)/ny, (projector.js[k]-nx÷2-1)/nx] for k in patch])
            grid_dft = NFFT.NDFTPlan(grid_nodes, (ny, nx))
            hpx_dft = NFFT.NDFTPlan(hpx_nodes, (ny, nx))
            direct_cart = real.(grid_dft * (adjoint(hpx_dft) * complex.(hpx_i[patch]))) ./ length(patch)
            direct_hpx = zeros(48)
            direct_hpx[patch] = real.(hpx_dft * (adjoint(grid_dft) * complex.(grid_i[:]))) ./ (ny*nx)

            to_cart(v) = project(projector, CMBLensing.HealpixMap(v, hpx_proj) => cart_proj).arr
            to_hpx(v) = project(projector, CMBLensing.BaseMap(v, cart_proj) => hpx_proj).arr
            @test to_cart(hpx_i)[:] ≈ direct_cart rtol=1e-5 atol=1e-6
            @test to_hpx(grid_i) ≈ direct_hpx rtol=1e-5 atol=1e-6

            # The adjoint of this linear projection has a direct DFT oracle.
            loss(v) = sum(weights .* to_cart(v)[:])
            expected_grad = zeros(48)
            expected_grad[patch] = real.(hpx_dft * (adjoint(grid_dft) * complex.(weights))) ./ length(patch)
            @test Zygote.gradient(loss, hpx_i)[1] ≈ expected_grad rtol=1e-5 atol=1e-6

            # Check the scalar-to-polarization composition and rotation for
            # QU and IQU in both directions, including EquiRect output types.
            ψcart = projector.ψpol_θϕs
            qcart, ucart = to_cart(hpx_q), to_cart(hpx_u)
            qcart_expected = @. qcart * cos(2ψcart) - ucart * sin(2ψcart)
            ucart_expected = @. ucart * cos(2ψcart) + qcart * sin(2ψcart)
            hpx_qu = CMBLensing.HealpixQUMap(hpx_q, hpx_u)
            hpx_iqu = CMBLensing.HealpixIQUMap(hpx_i, hpx_q, hpx_u)
            cart_qu = project(projector, hpx_qu => cart_proj)
            cart_iqu = project(projector, hpx_iqu => cart_proj)
            @test cart_qu.proj == cart_proj
            @test cart_qu.arr[:, :, 1] ≈ qcart_expected
            @test cart_qu.arr[:, :, 2] ≈ ucart_expected
            @test cart_iqu.proj == cart_proj
            @test cart_iqu.arr[:, :, 1] ≈ to_cart(hpx_i)
            @test cart_iqu.arr[:, :, 2] ≈ qcart_expected
            @test cart_iqu.arr[:, :, 3] ≈ ucart_expected

            ψhpx = projector.ψpol_ijs
            qhpx, uhpx = to_hpx(grid_q), to_hpx(grid_u)
            qhpx_expected = @. qhpx * cos(2ψhpx) + uhpx * sin(2ψhpx)
            uhpx_expected = @. uhpx * cos(2ψhpx) - qhpx * sin(2ψhpx)
            cart_qu_input = CMBLensing.BaseQUMap(cat(grid_q, grid_u; dims=3), cart_proj)
            cart_iqu_input = CMBLensing.BaseIQUMap(cat(grid_i, grid_q, grid_u; dims=3), cart_proj)
            hpx_qu_output = project(projector, cart_qu_input => hpx_proj)
            hpx_iqu_output = project(projector, cart_iqu_input => hpx_proj)
            @test hpx_qu_output.arr[:, 1] ≈ qhpx_expected
            @test hpx_qu_output.arr[:, 2] ≈ uhpx_expected
            @test hpx_iqu_output.arr[:, 1] ≈ to_hpx(grid_i)
            @test hpx_iqu_output.arr[:, 2] ≈ qhpx_expected
            @test hpx_iqu_output.arr[:, 3] ≈ uhpx_expected

            direction = cos.(collect(1.0:48.0) ./ 4)
            weights_qu = sin.(collect(1.0:(ny*nx*2)) ./ 7)
            weights_iqu = sin.(collect(1.0:(ny*nx*3)) ./ 7)
            qu_loss(t) = sum(weights_qu .* project(projector,
                CMBLensing.HealpixQUMap(hpx_q .+ t .* direction, hpx_u) => cart_proj).arr[:])
            iqu_loss(t) = sum(weights_iqu .* project(projector,
                CMBLensing.HealpixIQUMap(hpx_i, hpx_q .+ t .* direction, hpx_u) => cart_proj).arr[:])
            for polarized_loss in (qu_loss, iqu_loss)
                expected = polarized_loss(1.0) - polarized_loss(0.0)
                @test Zygote.gradient(polarized_loss, 0.3)[1] ≈ expected rtol=1e-8 atol=1e-8
            end
        end
    end
end
