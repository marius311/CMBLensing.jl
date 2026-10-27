@testset "Optional NFFT projector" begin
    cart_proj = ProjLambert(; Ny=8, Nx=16, θpix=600, T=Float64, rotator=(0, 30, 0))
    hpx_proj = ProjHealpix(2)
    projector = CMBLensing.Projector(cart_proj => hpx_proj; method=:fft)
    cart = CMBLensing.FlatMap(ones(8, 16), cart_proj)
    hpx = CMBLensing.HealpixMap(ones(48), hpx_proj)
    cart_to_hpx = project(projector, cart => hpx_proj).arr
    hpx_to_cart = project(projector, hpx => cart_proj).arr
    @test !isempty(projector.hpx_idxs_in_patch)
    @test all(isfinite, hpx_to_cart)
    patch = projector.hpx_idxs_in_patch
    patch_values = cart_to_hpx[patch]
    matches_constant = all(isapprox.(patch_values, 1.0; atol=1e-5))
    if !matches_constant
        ny, nx = cart_proj.Ny, cart_proj.Nx
        grid_nodes = reduce(hcat, [[(i-ny÷2-1)/ny, (j-nx÷2-1)/nx] for i=1:ny, j=1:nx])
        patch_nodes = reduce(hcat, [[(projector.is[k]-ny÷2-1)/ny, (projector.js[k]-nx÷2-1)/nx] for k in patch])
        grid_dft = NFFT.NDFTPlan(grid_nodes, (ny, nx))
        patch_dft = NFFT.NDFTPlan(patch_nodes, (ny, nx))
        direct = real.(patch_dft * (adjoint(grid_dft) * complex.(cart.arr[:]))) ./ (ny*nx)
        @info "NFFT constant-map diagnostic" patch=repr(patch) patch_values=repr(patch_values) direct=repr(direct) max_error=maximum(abs.(patch_values .- direct))
    end
    @test matches_constant
end
