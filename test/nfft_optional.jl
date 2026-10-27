# From the repository root, exercise an isolated environment and either
# supported NFFT series (use 0.13.3 for Julia 1.8):
#
#   nfft_env=$(mktemp -d)
#   julia --project="$nfft_env" -e 'using Pkg; Pkg.develop(path=pwd())'
#   julia --project="$nfft_env" test/nfft_optional.jl absent
#   NFFT_VERSION=0.14.3 julia --project="$nfft_env" -e 'using Pkg; Pkg.add(PackageSpec(name="NFFT", version=VersionNumber(ENV["NFFT_VERSION"])))'
#   julia --project="$nfft_env" test/nfft_optional.jl before
#   julia --project="$nfft_env" test/nfft_optional.jl after
using Test
using Pkg

mode = only(ARGS)
if mode == "before"
    @eval using NFFT
    @eval using CMBLensing
elseif mode == "after" || mode == "absent"
    @eval using CMBLensing
    @test !isdefined(CMBLensing, :plan_nfft)
    if mode == "after"
        @eval using NFFT
    end
else
    error("Expected absent, before, or after")
end

@testset "Optional NFFT ($mode)" begin
    hpx_proj = ProjHealpix(2)
    if mode == "absent"
        @test !any(info -> info.name == "NFFT", values(Pkg.dependencies()))
        cart_proj = ProjEquiRect(; Ny=8, Nx=16, θspan=(0, π), φspan=(0, 2π), T=Float64)
        cart = CMBLensing.EquiRectMap(ones(8, 16), cart_proj)
        projector = CMBLensing.Projector(cart_proj => hpx_proj; method=:bilinear)
        result = project(projector, cart => hpx_proj)
        @test all(isapprox.(result.arr[projector.hpx_idxs_in_patch], 1.0))
        err = try
            CMBLensing.Projector(cart_proj => hpx_proj; method=:fft)
            nothing
        catch exception
            exception
        end
        @test err isa ErrorException
        @test occursin("Load the `NFFT` package", sprint(showerror, err))
    else
        @test isdefined(CMBLensing, :plan_nfft)
        cart_proj = ProjLambert(; Ny=8, Nx=16, θpix=600, T=Float64, rotator=(0, 30, 0))
        projector = CMBLensing.Projector(cart_proj => hpx_proj; method=:fft)
        cart = CMBLensing.FlatMap(ones(8, 16), cart_proj)
        hpx = CMBLensing.HealpixMap(ones(48), hpx_proj)
        cart_to_hpx = project(projector, cart => hpx_proj).arr
        hpx_to_cart = project(projector, hpx => cart_proj).arr
        @test !isempty(projector.hpx_idxs_in_patch)
        @test all(isfinite, hpx_to_cart)
        @test all(isapprox.(cart_to_hpx[projector.hpx_idxs_in_patch], 1.0; atol=1e-5))

        # NFFT and its adjoint must both retain the dual-number dispatch.
        nodes = [-0.31 0.01 0.26; 0.2 -0.3 0.14]
        plan = NFFT.plan_nfft(Array, nodes, (4, 4))
        a = reshape(complex.(collect(1.0:16.0), collect(16.0:-1.0:1.0)), 4, 4)
        b = (0.2 + 0.4im) .* reverse(a, dims=1)
        asreal(z) = vcat(real.(vec(z)), imag.(vec(z)))
        coeffs = [0.4, -0.7]
        forward(x) = asreal(plan * (x[1] .* a .+ x[2] .* b))
        @test CMBLensing.ForwardDiff.jacobian(forward, coeffs) ≈ hcat(asreal(plan*a), asreal(plan*b))
        c, d = plan*a, plan*b
        backward(x) = asreal(adjoint(plan) * (x[1] .* c .+ x[2] .* d))
        @test CMBLensing.ForwardDiff.jacobian(backward, coeffs) ≈ hcat(asreal(adjoint(plan)*c), asreal(adjoint(plan)*d))
    end
end
