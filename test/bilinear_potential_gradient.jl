using CMBLensing
using LinearAlgebra
using Random
using Test
using Zygote
using CMBLensing: BatchedReal

@testset "BilinearLens potential pullback" begin
    # Points away from grid boundaries have a genuine derivative. Use random
    # samples to test the actual interpolation, rather than a spectral model.
    for T in (Float32, Float64), (Ny,Nx) in ((6,8), (7,9), (7,8)), pol in (1,2), B in (Map,Fourier)
        @testset "$T, $(Ny)×$(Nx), polarization $pol, $B" begin
            rng = MersenneTwister(50123 + Ny + Nx + pol)
            proj = ProjLambert(;Ny,Nx,θpix=60*180/π,T)
            scalar() = FlatMap(randn(rng,T,Ny,Nx), proj)
            polarized() = FlatQUMap(scalar().Ix, scalar().Ix, proj)
            ϕ, η = B(T(0.35)*scalar()), B(T(0.05)*scalar())
            f = B(pol == 1 ? scalar() : polarized())
            g = B(pol == 1 ? scalar() : polarized())
            δ = B(pol == 1 ? scalar() : polarized())
            dx,dy = getindex.((∇*ϕ)./proj.Δx, :Ix)
            gap = minimum(minimum(abs.(d .- round.(d))) for d in (dx,dy))
            h = T == Float32 ? T(1e-3) : T(1e-5)
            direction = getindex.((∇*η)./proj.Δx, :Ix)
            @test gap > h * maximum(maximum(abs.(d)) for d in direction)
            objective = a -> dot(g, BilinearLens(ϕ+a*η) * (f+a*δ))
            gradϕ, gradf = gradient((p,x) -> dot(g, BilinearLens(p)*x), ϕ, f)
            reverse = dot(gradϕ,η) + dot(gradf,δ)
            finite = (objective(h)-objective(-h))/(2h)
            @test isapprox(reverse, finite; atol=T == Float32 ? 4e-3 : 2e-8, rtol=T == Float32 ? 4e-3 : 2e-8)
            @test gradf ≈ B(BilinearLens(ϕ)'*g)
        end
    end

    # A locally affine map supplies an independent analytic check of axis,
    # scale and sign. The one nonzero component of g is strictly inside the
    # image and displaced away from both boundaries.
    Ny,Nx=8,10
    proj=ProjLambert(;Ny,Nx,θpix=60*180/π,T=Float64)
    ii,jj=3,4
    field(fn)=FlatMap(Float64[fn(i,j) for i=1:Ny,j=1:Nx],proj)
    ϕ=field((i,j)->0.3sin(2π*(j-1)/Nx+0.19) + 0.4sin(2π*(i-1)/Ny+0.23))
    η=field((i,j)->0.07cos(2π*(j-1)/Nx+0.2) + 0.09sin(2π*(i-1)/Ny+0.3))
    f=field((i,j)->2i+3j)
    g=field((i,j)->Float64(i==ii && j==jj))
    dx,dy=getindex.((∇*ϕ)./proj.Δx, :Ix)
    @test 0 < abs(dx[ii,jj]) < 0.5
    @test 0 < abs(dy[ii,jj]) < 0.5
    expected = 3*(-0.07*2π/Nx*sin(2π*(jj-1)/Nx+0.2)) +
               2*( 0.09*2π/Ny*cos(2π*(ii-1)/Ny+0.3))
    objective = a -> dot(g, BilinearLens(ϕ+a*η)*f)
    @test isapprox(gradient(objective,0.0)[1], expected; atol=2e-12, rtol=2e-12)
    @test isapprox((objective(1e-5)-objective(-1e-5))/(2e-5), expected; atol=1e-9, rtol=1e-9)

    # At zero deflection the map has a corner. Select the positive-index
    # cell's derivative; a central difference need not agree at the corner.
    ϕ0=zero(ϕ)
    η0=field((i,j)->sin(2π*(j-jj)/Nx) + sin(2π*(i-ii)/Ny))
    nonlinear=field((i,j)->i^2+3j^2)
    corner = a -> dot(g,BilinearLens(ϕ0+a*η0)*nonlinear)
    reverse=gradient(corner,0.0)[1]
    right=(corner(1e-7)-corner(0.0))/1e-7
    left=(corner(0.0)-corner(-1e-7))/1e-7
    @test isapprox(reverse, right; atol=1e-5, rtol=1e-5)
    @test abs(right-left) > 0.1
end

@testset "BilinearLens Fourier potential at the Nyquist row" begin
    # Test both real-map-derived and arbitrary stored complex directions.
    # The latter distinguish the inverse-FFT adjoints used by FFTW and MKL.
    for T in (Float32, Float64), (Ny, Nx) in ((6,8), (7,8), (7,9))
        @testset "$T, $(Ny)×$(Nx)" begin
            rng = MersenneTwister(50124 + Ny + Nx)
            proj = ProjLambert(; Ny, Nx, θpix=60*180/π, T)
            scalar() = FlatMap(randn(rng,T,Ny,Nx), proj)
            ϕ, η = Fourier(T(0.35)*scalar()), Fourier(T(0.05)*scalar())
            f, g = scalar(), scalar()
            h = T == Float32 ? T(1e-3) : T(1e-5)
            dx, dy = getindex.((∇*ϕ)./proj.Δx, :Ix)
            gap = minimum(minimum(abs.(d .- round.(d))) for d in (dx,dy))
            objective = p -> dot(g, BilinearLens(p)*f)
            ϕ̄ = gradient(objective, ϕ)[1]
            mapϕ̄ = gradient(objective, Map(ϕ))[1]
            @test isapprox(dot(ϕ̄,η), dot(mapϕ̄,Map(η)); atol=T == Float32 ? 5e-4 : 2e-11, rtol=T == Float32 ? 5e-4 : 2e-11)

            directions = [η]
            # The first and last y-frequency rows, positive and negative x
            # frequencies, and fully self-conjugate corners, including imaginary
            # directions, all have distinct inverse-real-FFT behavior.
            rows = unique((1, Ny÷2+1))
            cols = unique((1, 2, Nx÷2+1, Nx÷2+2, Nx))
            for i in rows, j in cols, value in (one(Complex{T}), im*one(Complex{T}))
                a = zeros(Complex{T}, Ny÷2+1, Nx)
                a[i,j] = value
                push!(directions, FlatFourier(a, proj))
            end
            for direction in directions
                dη = getindex.((∇*direction)./proj.Δx, :Ix)
                @test gap > h * maximum(maximum(abs.(d)) for d in dη)
                finite = (objective(ϕ+h*direction)-objective(ϕ-h*direction))/(2h)
                @test isapprox(dot(ϕ̄,direction), finite; atol=T == Float32 ? 4e-3 : 2e-8, rtol=T == Float32 ? 4e-3 : 2e-8)
            end
        end
    end
end

bilinear_test_scalar(x) = x isa BatchedReal ? sum(unbatch(x)) : x
@testset "BilinearLens shared potential, batches and tuples" begin
    for mode in (:batch, :tuple, :named_tuple, :tuple_batch, :named_tuple_batch, :broadcast_cotangent)
        @testset "$mode" begin
            rng=MersenneTwister(51262)
            Ny,Nx=7,8
            proj=ProjLambert(;Ny,Nx,θpix=60*180/π,T=Float64)
            scalar()=FlatMap(randn(rng,Ny,Nx),proj)
            polar()=FlatIQUMap(scalar().Ix,scalar().Ix,scalar().Ix,proj)
            batched=mode in (:batch,:tuple_batch,:named_tuple_batch)
            maybe_batch(fn) = batched ? batch(fn(),fn()) : fn()
            f1,g1,δ1 = maybe_batch(scalar),maybe_batch(scalar),maybe_batch(scalar)
            if mode in (:tuple,:named_tuple,:tuple_batch,:named_tuple_batch)
                f2,g2,δ2=maybe_batch(polar),maybe_batch(polar),maybe_batch(polar)
                named=mode in (:named_tuple,:named_tuple_batch)
                f=named ? FieldTuple(;scalar=f1,polar=f2) : FieldTuple(f1,f2)
                g=named ? FieldTuple(;scalar=g1,polar=g2) : FieldTuple(g1,g2)
                δ=named ? FieldTuple(;scalar=δ1,polar=δ2) : FieldTuple(δ1,δ2)
            elseif mode == :broadcast_cotangent
                f,δ=f1,δ1
                g=batch(scalar(),scalar())
            else
                f,g,δ=f1,g1,δ1
            end
            ϕ,η=0.3scalar(),0.03scalar()
            dx,dy=getindex.((∇*ϕ)./proj.Δx,:Ix)
            dη=getindex.((∇*η)./proj.Δx,:Ix)
            gap=minimum(minimum(abs.(d .- round.(d))) for d in (dx,dy))
            h=1e-5
            @test gap > h*maximum(maximum(abs.(d)) for d in dη)
            objective = (p,x) -> bilinear_test_scalar(dot(g,BilinearLens(p)*x))
            ϕbar,fbar=gradient(objective,ϕ,f)
            finiteϕ=(objective(ϕ+h*η,f)-objective(ϕ-h*η,f))/(2h)
            finitef=(objective(ϕ,f+h*δ)-objective(ϕ,f-h*δ))/(2h)
            projectedϕ=dot(ϕbar,η)
            projectedf=bilinear_test_scalar(dot(fbar,δ))
            αbar=gradient(a -> objective(ϕ+a*η,f),0.0)[1]
            @test batch_length(ϕbar)==1
            @test batch_length(fbar)==batch_length(f)
            @test isapprox(projectedϕ,finiteϕ;atol=2e-8,rtol=2e-8)
            @test isapprox(αbar,finiteϕ;atol=2e-8,rtol=2e-8)
            @test isapprox(projectedf,finitef;atol=2e-8,rtol=2e-8)
        end
    end
end

@testset "BilinearLens Fourier source directions" begin
    # With L fixed, its action on the source is linear. Compare its pullback
    # to that exact action for unrestricted stored Fourier directions.
    for T in (Float32,Float64), (Ny,Nx) in ((6,8),(7,9)), spin in (:scalar,:QU,:EB,:IQU,:IEB), shared in (false,true)
        @testset "$T, $(Ny)×$(Nx), $spin, shared=$shared" begin
            rng = MersenneTwister(81623+Ny+Nx)
            proj = ProjLambert(;Ny,Nx,θpix=60*180/π,T)
            scalar() = FlatMap(randn(rng,T,Ny,Nx),proj)
            qu() = FlatQUMap(scalar().Ix,scalar().Ix,proj)
            iqu() = FlatIQUMap(scalar().Ix,scalar().Ix,scalar().Ix,proj)
            mapfield() = spin == :scalar ? scalar() : spin in (:QU,:EB) ? qu() : iqu()
            tofourier(x) = spin == :EB ? EBFourier(x) : spin == :IEB ? IEBFourier(x) : Fourier(x)
            L = BilinearLens(T(0.3)*scalar())
            f = tofourier(mapfield())
            g = shared ? batch(mapfield(),mapfield()) : mapfield()
            f_before, g_before = copy(f.arr), copy(g.arr)
            objective = x -> bilinear_test_scalar(dot(g,L*x))
            f̄ = gradient(objective,f)[1]
            @test batch_length(f̄)==1
            @test f.arr == f_before
            @test g.arr == g_before
            directions = [typeof(f)(randn(rng,eltype(f),size(f.arr)...),proj)]
            if spin == :scalar
                rows = unique((1,Ny÷2+1))
                cols = unique((1,2,Nx÷2+1,Nx))
                for i in rows, j in cols, value in (one(Complex{T}),im*one(Complex{T}))
                    a = zeros(Complex{T},size(f.arr))
                    a[i,j] = value
                    push!(directions,typeof(f)(a,proj))
                end
            end
            for direction in directions
                exact = bilinear_test_scalar(dot(g,L*direction))
                @test isapprox(dot(f̄,direction),exact;atol=T==Float32 ? 3e-5 : 3e-12,rtol=T==Float32 ? 3e-5 : 3e-12)
            end
        end
    end
end

@testset "BilinearLens zero-potential source identity" begin
    rng=MersenneTwister(82311)
    Ny,Nx=6,8
    proj=ProjLambert(;Ny,Nx,θpix=60*180/π,T=Float64)
    scalar()=FlatMap(randn(rng,Ny,Nx),proj)
    qu()=FlatQUMap(scalar().Ix,scalar().Ix,proj)
    iqu()=FlatIQUMap(scalar().Ix,scalar().Ix,scalar().Ix,proj)
    L=BilinearLens(zero(scalar()))
    for template in (Fourier(scalar()), QUFourier(qu()), EBFourier(qu()), IQUFourier(iqu()), IEBFourier(iqu()))
        # Fully arbitrary coefficients intentionally need not be Hermitian.
        f=typeof(template)(randn(rng,eltype(template),size(template.arr)...),proj)
        seed=typeof(template)(randn(rng,eltype(template),size(template.arr)...),proj)
        f_before,seed_before=copy(f.arr),copy(seed.arr)
        result,back=Zygote.pullback(x->L*x,f)
        f̄,=back(seed)
        @test result === f
        @test f̄.arr == seed_before
        @test f.arr == f_before
        @test seed.arr == seed_before
    end
end

@testset "BilinearLens Fourier source with mixed storage precision" begin
    rng=MersenneTwister(81970)
    Ny,Nx=7,8
    proj=ProjLambert(;Ny,Nx,θpix=60*180/π,T=Float64)
    wide()=FlatMap(randn(rng,Ny,Nx),proj)
    narrow()=FlatMap(randn(rng,Float32,Ny,Nx),proj)
    L=BilinearLens(0.3wide())
    f=Fourier(narrow())
    g=batch(wide(),wide())
    δ=typeof(f)(randn(rng,ComplexF32,size(f.arr)...),proj)
    f_before,g_before=copy(f.arr),copy(g.arr)
    objective=x->bilinear_test_scalar(dot(g,L*x))
    f̄=gradient(objective,f)[1]
    @test eltype(f̄) === ComplexF32
    @test f.arr == f_before
    @test g.arr == g_before
    @test isapprox(dot(f̄,δ),bilinear_test_scalar(dot(g,L*δ));atol=1e-4,rtol=1e-4)
end

@testset "BilinearLens mixed storage precision" begin
    rng=MersenneTwister(31556)
    Ny,Nx=7,8
    proj=ProjLambert(;Ny,Nx,θpix=60*180/π,T=Float64)
    wide()=FlatMap(randn(rng,Ny,Nx),proj)
    narrow()=FlatMap(randn(rng,Float32,Ny,Nx),proj)
    ϕ,η,g,f,δ=0.3wide(),0.03wide(),wide(),narrow(),narrow()
    h=1e-3
    dx,dy=getindex.((∇*ϕ)./proj.Δx,:Ix)
    dη=getindex.((∇*η)./proj.Δx,:Ix)
    gap=minimum(minimum(abs.(d .- round.(d))) for d in (dx,dy))
    @test gap > h*maximum(maximum(abs.(d)) for d in dη)
    objective=(p,x)->dot(g,BilinearLens(p)*x)
    ϕbar,fbar=gradient(objective,ϕ,f)
    fdϕ=(objective(ϕ+h*η,f)-objective(ϕ-h*η,f))/(2h)
    fdf=(objective(ϕ,f+Float32(h)*δ)-objective(ϕ,f-Float32(h)*δ))/(2h)
    projectedϕ, projectedf=dot(ϕbar,η),dot(fbar,δ)
    @test eltype(fbar) === Float32
    @test isapprox(projectedϕ,fdϕ;atol=3e-3,rtol=3e-3)
    @test isapprox(projectedf,fdf;atol=3e-3,rtol=3e-3)
end
