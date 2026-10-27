@testset "Healpix-to-Lambert FFT array adjoint gradient" begin
    # The existing reverse FFT operation is a scaled array adjoint; it is not
    # a sky interpolation. Build an exact NDFT reference at independently
    # calculated Lambert coordinates, and test its reverse-mode derivative.
    ny, nx, nside = 8, 8, 4
    for T in (Float32, Float64)
        p = ProjLambert(; Ny=ny, Nx=nx, θpix=600, rotator=(0,0,0), T)
        hp = ProjHealpix(nside)
        projector = CMBLensing.Projector(p => hp; method=:fft)
        npix = 12*nside^2
        θϕ = CMBLensing.pix2angRing.(Ref(CMBLensing.Resolution(nside)), 1:npix)
        θ, ϕ = first.(θϕ), last.(θϕ)
        r = 2 .* cos.(θ ./ 2)
        i = @. -r*cos(ϕ)/p.Δx + ny÷2 + 0.5
        j = @. -r*sin(ϕ)/p.Δx + nx÷2 + 0.5
        patch = projector.hpx_idxs_in_patch
        @test patch == findall((1 .<= i) .& (i .<= ny) .& (1 .<= j) .& (j .<= nx))
        grid_nodes = reduce(hcat, [[(a-ny÷2-1)/ny, (b-nx÷2-1)/nx] for a=1:ny, b=1:nx])
        hp_nodes = reduce(hcat, [[(i[k]-ny÷2-1)/ny, (j[k]-nx÷2-1)/nx] for k in patch])
        grid_dft = NFFT.NDFTPlan(grid_nodes, (ny, nx))
        hp_dft = NFFT.NDFTPlan(hp_nodes, (ny, nx))

        values = T.(sin.(collect(1.0:npix) ./ 7))
        direction = T.(cos.(collect(1.0:npix) ./ 11))
        weights = T.(cos.(collect(1.0:ny*nx) ./ 5))
        project_map(v) = project(projector, CMBLensing.HealpixMap(v, hp) => p).arr[:]
        exact_map = real.(grid_dft * (adjoint(hp_dft) * complex.(Float64.(values[patch])))) ./ length(patch)
        tol = T === Float32 ? 5e-5 : 1e-6
        @test project_map(values) ≈ exact_map rtol=tol atol=tol

        loss(v) = sum(weights .* project_map(v))
        reverse = Zygote.gradient(loss, values)[1]
        exact_grad = zeros(Float64, npix)
        exact_grad[patch] = real.(hp_dft * (adjoint(grid_dft) * complex.(Float64.(weights)))) ./ length(patch)
        @test reverse ≈ exact_grad rtol=tol atol=tol
        step = T === Float32 ? T(1e-2) : T(1e-5)
        finite = (loss(values .+ step .* direction) - loss(values .- step .* direction))/(2step)
        @test dot(reverse, direction) ≈ finite rtol=2e-3 atol=2e-4

        # This is the scaled Euclidean-array transpose of the forward
        # Cartesian-to-Healpix FFT operation, not a constant-preserving inverse.
        wmap = CMBLensing.FlatMap(reshape(weights, ny, nx), p)
        forward = project(projector, wmap => hp).arr
        @test dot(weights, project_map(values)) ≈ (ny*nx/length(patch))*dot(values, forward) rtol=tol atol=tol

        qu = T.(cos.(collect(1.0:npix) ./ 9))
        quweights = T.(sin.(collect(1.0:2*ny*nx) ./ 13))
        function qu_loss(t)
            out = project(projector, CMBLensing.HealpixQUMap(values .+ t .* direction, qu) => p)
            sum(quweights .* out.arr[:])
        end
        qu_ad = Zygote.gradient(qu_loss, T(0.3))[1]
        qu_fd = (qu_loss(T(0.3)+step)-qu_loss(T(0.3)-step))/(2step)
        @test qu_ad ≈ qu_fd rtol=2e-3 atol=2e-4
    end
end
