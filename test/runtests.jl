
using CUDA
if CUDA.functional()
    CUDA.allowscalar(false)
    maybegpu(x) = adapt(CuArray,x)
    storage = CuArray
    @info "Running tests on $(repr("text/plain", CUDA.device()))"
else
    maybegpu(x) = x
    storage = Array
    @info "Running tests on CPU"
end

##

using CMBLensing
using CMBLensing: @SMatrix, @SVector, AbstractCℓs, basis, Basis,
    LinearInterpolation, Measurement, RK4Solver, ±, typealias, BatchedReal

##

using Adapt
using FileIO
using FFTW
using FiniteDifferences
using LinearAlgebra
using Random
using Requires
using Serialization
using SparseArrays
using Test
using Zygote

##

try
    push!(LOAD_PATH, "@v#.#") # assumes you have CirculantCov in your global environment
    using CirculantCov
catch
    @warn """CirculantCov.jl not found, not testing EquiRect fields.
    Run `pkg> add https://github.com/EthanAnderes/CirculantCov.jl` to add this package.
    """
end

##

macro test_real_gradient(f, x, tol=:(rtol=1e-3))
    esc(:(@test real(gradient($f,$x)[1]) ≈ central_fdm(5,1)($f,$x) $tol))
end

Nsides     = [(8,8), (4,8), (8,4)]
Nsides_big = [(128,128), (64,128), (128,64)]

Random.seed!(1)

has_batched_fft = (FFTW.fftw_provider != "mkl") || (storage != Array)

##

@testset "CMBLensing" begin

##

# basic printing sanity checks, which were super annoying to get right...
# see also: https://discourse.julialang.org/t/dispatching-on-the-result-of-unwrap-unionall-seems-weird/25677

@testset "Printing" begin

    # concrete types:
    for f in [maybegpu(FlatMap(rand(4,4))), maybegpu(FlatQUMap(rand(4,4),rand(4,4)))]
        @test occursin("pixel",sprint(show, MIME("text/plain"), f))
    end
    
end

##

@testset "Flat" begin 

    Ny,Nx = Nside = first(Nsides)

    @testset "Constructors" begin

        @testset "batch=$D" for D in [(), (3,)]

            Ix = maybegpu(rand(Ny,Nx,D...))
            Il = maybegpu(rand(Ny÷2+1,Nx,D...))

            @testset "$(basis(F))" for (F,ks,args,kwargs) in [
                (FlatMap,        (:Ix,),        (Ix,),      ()),
                (FlatFourier,    (:Il,),        (Il,),      (;Ny,)),
                (FlatQUMap,      (:Qx,:Qx),     (Ix,Ix),    ()),
                (FlatQUFourier,  (:Ql,:Ql),     (Il,Il),    (;Ny,)),
                (FlatEBMap,      (:Ex,:Bx),     (Ix,Ix),    ()),
                (FlatEBFourier,  (:El,:Bl),     (Il,Il),    (;Ny,)),
                (FlatIQUMap,     (:Ix,:Qx,:Qx), (Ix,Ix,Ix), ()),
                (FlatIQUFourier, (:Il,:Ql,:Ql), (Il,Il,Il), (;Ny,)),
                (FlatIEBMap,     (:Ix,:Ex,:Bx), (Ix,Ix,Ix), ()),
                (FlatIEBFourier, (:Il,:El,:Bl), (Il,Il,Il), (;Ny,)),
            ]
                local f
                @test (f = F(args...; kwargs...)) isa F
                @test @inferred(F(getproperty.(Ref(f),ks)..., f.metadata)) == f
                @test (io=IOBuffer(); serialize(io,f); seekstart(io); deserialize(io) == f)
                @test (save(".test_field.jld2", "f", cpu(f)); load(".test_field.jld2", "f") == cpu(f))
                @test_throws ErrorException F(args..., ProjLambert(Nx=Nx+1, Ny=Ny+1))

            end
        
        end

        rm(".test_field.jld2", force=true)
    
    end

    @testset "Basis conversions" begin
        @testset "$(typealias(Bin)) → $(typealias(Bout))" for (f,Bin,Bout) in [
            (f,Bin,Bout)
            for (f,Bs) in [
                (FlatMap(rand(Nside...)), (Map,Fourier)),
                (FlatQUMap(rand(Nside...),rand(Nside...)), (QUMap,QUFourier,EBMap,EBFourier))
            ]
            for Bin in Bs
            for Bout in Bs
        ]

            @test basis(@inferred(Bout(Bin(f)))) == Bout
            @test Bin(Bout(Bin(f))) ≈ f

        end
    end

end

##

@testset "Algebra" begin
    
    @testset "Nside = $Nside" for Nside in Nsides

        fs = ((B0,f0),(B2,f2),(Bt,ft)) = [
            (Fourier,    maybegpu(FlatMap(rand(Nside...)))), 
            (EBFourier,  maybegpu(FlatQUMap(rand(Nside...),rand(Nside...)))), 
            (Fourier,    maybegpu(FieldTuple(FlatMap(rand(Nside...)),FlatMap(rand(Nside...)))))
            # inference currently broken for this case:
            # (IEBFourier, maybegpu(FlatIQUMap(rand(Nside...),rand(Nside...),rand(Nside...)))), 
        ]
        # MKL doesnt seem to support batched FFTs, not that theyre really useful on CPU
        has_batched_fft && append!(fs, [
            (Fourier,    maybegpu(FlatMap(rand(Nside...,2)))),
            (EBFourier,  maybegpu(FlatQUMap(rand(Nside...,2),rand(Nside...,2)))),
        ])
            
        @testset "f :: $(typeof(f)) " for (B,f) in fs
            
            local Ðf, Ðv, g, H
            
            @test similar(f) isa typeof(f)
            @test zero(f) isa typeof(f)
            @test similar(f,Float32) isa Field
            @test eltype(similar(f,Float32)) == Float32
                        
            # used in lots of tests
            @test f ≈ f
            @test !(f ≈ 2f)

            # broadcasting
            @test (@inferred f .+ f) isa typeof(f)
            @test (@inferred f .+ Float32.(f)) isa typeof(f)
            
            # promotion
            @test (@inferred f + B(f)) isa typeof(f)
            @test (@inferred f + B(Float32.(f))) isa typeof(f)
            
            # Diagonal broadcasting
            @test (@inferred Diagonal(f) .* Diagonal(f) .* Diagonal(f)) isa typeof(Diagonal(f))
            
            # inverses
            @test (@inferred pinv(Diagonal(f))) isa Diagonal{<:Any,<:typeof(f)}
            @test_throws Exception inv(Diagonal(0*f))
            
            # trace
            @test all(tr(Diagonal(f)' * Diagonal(f)) ≈ f'f)
            @test all(tr(Diagonal(f) * Diagonal(f)') ≈ f'f)
            @test_broken all(tr(f*f') ≈ f'f) # broken by (intentionally) removing OuterProdOp

            # Field dot products
            D = Diagonal(f)
            @test (@inferred f' * f) isa Real
            @test (@inferred f' * B(f)) isa Real
            @test (@inferred f' * D * f) isa Real
            @test sum(f, dims=:) ≈ sum(f[:])

            # Explicit vs. lazy DiagOp algebra
            @test (Diagonal(Ð(f)) + Diagonal(Ð(f))) isa DiagOp{<:Field{basis(Ð(f))}}
            @test (Diagonal(Ł(f)) + Diagonal(Ð(f))) isa LazyBinaryOp
            @test (Diagonal(Ł(f)) + Diagonal(Ł(f))) isa DiagOp{<:Field{basis(Ł(f))}}

            if !(f isa FieldTuple)

                # gradients
                @test (Ðf = @inferred ∇[1]*f) isa Field
                @test all(∇[1]'*f ≈ -∇[1]*f)
                @test all(-∇[1]'*f ≈ ∇[1]*f)
                @test (@inferred mul!(Ðf,∇[1],Ð(f))) isa Field
                @test (Ðv = @inferred ∇*f) isa FieldVector
                @test (@inferred mul!(Ðv,∇,Ð(f))) isa FieldVector
                @test ((g,H) = map(Ł, (@inferred gradhess(f)))) isa NamedTuple{<:Any, <:Tuple{FieldVector, FieldMatrix}}
                
                # FieldVector dot product
                @test (@inferred Diagonal.(g)' * g) isa typeof(g[1])
                @test (@inferred mul!(similar(g[1]), Diagonal.(g)', g)) isa typeof(g[1])
                
                # FieldMatrix-FieldVector product
                @test (@inferred Diagonal.(H) * g) isa FieldVector
                @test (@inferred Diagonal.(H) * Diagonal.(g)) isa FieldOrOpVector
                @test (@inferred mul!(Diagonal.(similar.(g)), Diagonal.(H), Diagonal.(g))) isa FieldOrOpVector
                
            end

        end
        
        # # tuple adjoints
        # f0b = identity.(batch(f0, batch_length(f)))
        # v = similar.(@SVector[f0b, f0b])
        # @test (@inferred mul!(f0b, spin_adjoint(f), f)) isa Field{<:Any,S0}
        # @test (@inferred mul!(v, spin_adjoint(f), @SVector[f,f])) isa FieldVector{<:Field{<:Any,S0}}

        # mixed-spin
        @test (@inferred f0 .* f2) isa typeof(f2)
        
        # matrix type promotion
        @test (@inferred FlatMap(rand(Float64,2,2)) .+ FlatMap(view(rand(Float32,2,2),:,:))) isa FlatMap{<:Any,Float64,Matrix{Float64}}
        
    end

end

##

@testset "Log/Trace" begin

    @testset "logdet(Diagonal(::Map))" begin
        @test logdet(Diagonal(FlatMap([1 -2; 3 -4])))                                           ≈  log(24)
        @test logdet(Diagonal(FlatQUMap([1 -2; 3 -4], [1 -2; 3 -4])))                           ≈ 2log(24)
        @test logdet(Diagonal(FlatIQUMap([1 -2; 3 -4], [1 -2; 3 -4], [1 -2; 3 -4])))            ≈ 3log(24)
        @test logdet(Diagonal(FieldTuple(FlatMap([1 -2; 3 -4]), FlatMap([1 -2; 3 -4]))))        ≈ 2log(24)
        @test all(logdet(Diagonal(FlatMap(cat([1 -2; 3 -4],[1 -2; 3 -4],dims=3))))::BatchedReal ≈  log(24))
    end

    @testset "logdet(Diagonal(::Fourier)) Nside=$Nside" for Nside in Nsides_big
        x = rand(Nside...)
        @test logdet(Diagonal(Fourier(FlatMap(x))))                                                   ≈ real( logdet(Diagonal(fft(x)[:])))
        @test logdet(Diagonal(QUFourier(FlatQUMap(x,x))))                                             ≈ real(2logdet(Diagonal(fft(x)[:])))
        @test logdet(Diagonal(IQUFourier(FlatIQUMap(x,x,x))))                                         ≈ real(3logdet(Diagonal(fft(x)[:])))
        @test logdet(Diagonal(FieldTuple(Fourier(FlatMap(x)), Fourier(FlatMap(x)))))                  ≈ real(2logdet(Diagonal(fft(x)[:])))
        has_batched_fft && @test all(logdet(Diagonal(Fourier(FlatMap(cat(x,x,dims=3)))))::BatchedReal ≈ real( logdet(Diagonal(fft(x)[:]))))
    end

    @testset "tr(Diagonal(::Map))" begin
        @test tr(Diagonal(FlatMap([1 -2; 3 -4])))                                           ≈  -2
        @test tr(Diagonal(FlatQUMap([1 -2; 3 -4], [1 -2; 3 -4])))                           ≈  -4
        @test tr(Diagonal(FlatIQUMap([1 -2; 3 -4], [1 -2; 3 -4], [1 -2; 3 -4])))            ≈  -6
        @test tr(Diagonal(FieldTuple(FlatMap([1 -2; 3 -4]), FlatMap([1 -2; 3 -4]))))        ≈  -4
        @test all(tr(Diagonal(FlatMap(cat([1 -2; 3 -4],[1 -2; 3 -4],dims=3))))::BatchedReal ≈  -2)
    end

    @testset "tr(Diagonal(::Fourier)) Nside=$Nside" for Nside in Nsides_big
        x = rand(Nside...)
        @test tr(Diagonal(Fourier(FlatMap(x))))                                                   ≈  tr(Diagonal(fft(x)[:]))
        @test tr(Diagonal(QUFourier(FlatQUMap(x,x))))                                             ≈ 2tr(Diagonal(fft(x)[:]))
        @test tr(Diagonal(IQUFourier(FlatIQUMap(x,x,x))))                                         ≈ 3tr(Diagonal(fft(x)[:]))
        @test tr(Diagonal(FieldTuple(Fourier(FlatMap(x)), Fourier(FlatMap(x)))))                  ≈ 2tr(Diagonal(fft(x)[:]))
        has_batched_fft && @test all(tr(Diagonal(Fourier(FlatMap(cat(x,x,dims=3)))))::BatchedReal ≈ real(tr(Diagonal(fft(x)[:]))))
    end

end

##

@testset "FlatS2" begin
    @testset "Nside = $Nside" for Nside in Nsides
        C = maybegpu(Diagonal(EBFourier(FlatEBMap(rand(Nside...), rand(Nside...)))))
        f = maybegpu(FlatQUMap(rand(Nside...), rand(Nside...)))
        @test C*f ≈ FlatQUFourier(C[:QQ]*f[:Q]+C[:QU]*f[:U], C[:UU]*f[:U]+C[:UQ]*f[:Q])
    end
end

##

@testset "BatchedReal" begin

    @testset "Nside = $Nside" for Nside in Nsides

        r  = 1.
        rb = batch([1.,2])
        
        @testset "f :: $(typeof(f))" for (f,fb) in [
            (maybegpu(FlatMap(rand(Nside...))),                  maybegpu(FlatMap(rand(Nside...,2)))),
            (maybegpu(FlatQUMap(rand(Nside...),rand(Nside...))), maybegpu(FlatQUMap(rand(Nside...,2),rand(Nside...,2))))
        ]

            @test @inferred(r * f)  == f
            @test @inferred(r * fb) == fb
            @test unbatch(@inferred(rb * f)) == [f, 2f]
            @test unbatch(@inferred(rb * fb)) == [batch_index(fb,1), 2batch_index(fb,2)]

        end

    end
    
end

## 

@testset "Misc" begin
    
    @testset "Nside = $Nside" for Nside in Nsides

        f = maybegpu(FlatMap(rand(Nside...)))
        
        @test                  (MidPass(100,200) .* Diagonal(Fourier(f))) isa Diagonal
        @test_throws Exception  MidPass(100,200) .* Diagonal(        f)

    end
    
end

##

@testset "Cℓs" begin
    
    @test InterpolatedCℓs(1:100, rand(100))      isa AbstractCℓs{Float64}
    @test InterpolatedCℓs(1:100, rand(100) .± 1) isa AbstractCℓs{Measurement{Float64}}
    @test (InterpolatedCℓs(1:100, 1:100) * ℓ²)[50] == 50^3
    
end

##

@testset "ParamDependentOp" begin
    
    D = Diagonal(maybegpu(FlatMap(rand(4,4))))
    
    @test ParamDependentOp((;x=1, y=1)->x*y*D)() ≈ D
    @test ParamDependentOp((;x=1, y=1)->x*y*D)(z=2) ≈ D
    @test ParamDependentOp((;x=1, y=1)->x*y*D)(x=2) ≈ 2D
    @test ParamDependentOp((;x=1, y=1)->x*y*D)((x=2,y=2)) ≈ 4D # tuple calling form
    @test_throws MethodError ParamDependentOp((;x=1, y=1)->x*y*D)(2) # only Tuple unnamed arg is OK

end

##

@testset "Chains" begin

    chains = CMBLensing.wrap_chains([
        [Dict(:i=>1, :b=>2), Dict(:i=>2       ), Dict(:i=>3, :b=>2)],
        [Dict(:i=>1, :b=>3), Dict(:i=>2, :b=>3), Dict(:i=>3, :b=>3)],
    ])
    
    # basic
    @test chains[1, 1, :i] == 1
    @test chains[:, 1, :i] == [1,1]
    @test chains[:, :, :i] == [[1, 2, 3], [1, 2, 3]]
    
    # slices
    @test chains[1, 1:2, :i] == [1, 2]
    @test chains[:, 1:2, :i] == [[1,2], [1,2]]
    
    # implied : in first dims
    @test chains[:i] == [[1,2,3],[1,2,3]]
    @test chains[1,:i] == [1,2,3]
    
    # missing
    @test all(chains[1,:b] .=== [2, missing, 2])

end;


##

@testset "Zygote" begin

    @testset "Nside = $Nside" for Nside in Nsides

        @testset "$FMap" for (FMap, FFourier, Npol) in [
            (FlatMap,   FlatFourier,   1),
            (FlatQUMap, FlatQUFourier, 2)
        ]
            
            Ny,Nx = Nside
            Ixs = collect(maybegpu(rand(Nside...)) for i=1:Npol)
            Ils = rfft.(Ixs)
            f,g,h = @repeated(maybegpu(FMap(Ixs...)),3)
            v = @SVector[f,f]
            D = Diagonal(f)
            A = 2

            @testset "Fields" begin
                
                @testset "sum" begin
                    @test ((δ = gradient(f -> sum(Map(f)),     Map(f))[1]); basis(δ)==basis(Map(f))     && δ ≈ one(Map(f)))
                    @test ((δ = gradient(f -> sum(Map(f)), Fourier(f))[1]); basis(δ)==basis(Fourier(f)) && δ ≈ Fourier(one(Map(f))))
                end
                
                @testset "B1=$B1, B2=$B2, B3=$B3" for B1=[Map,Fourier], B2=[Map,Fourier], B3=[Map,Fourier]
                    
                    @test gradient(f -> dot(B1(f),B2(f)), B3(f))[1] ≈ 2f
                    @test gradient(f -> norm(B1(f)), B3(f))[1] ≈ f/norm(f)
                    @test gradient(f -> B1(f)' * B2(g), B3(f))[1] ≈ g
                    @test gradient(f -> sum(Diagonal(Map(f)) * B2(g)), B3(f))[1] ≈ g
                    @test gradient(f -> sum(Diagonal(Map(∇[1]*f)) * B2(g)), B3(f))[1] ≈ ∇[1]'*g
                    @test gradient(f -> B1(f)'*(D\B2(f)), B3(f))[1] ≈ D\f + D'\f
                    @test gradient(f -> (B1(f)'/D)*B2(f), B3(f))[1] ≈ D\f + D'\f
                    @test gradient(f -> B1(f)'*(D*B2(f)), B3(f))[1] ≈ D*f + D'*f
                    @test gradient(f -> (B1(f)'*D)*B2(f), B3(f))[1] ≈ D*f + D'*f
                    if B2==Map
                        @test gradient(f -> B1(f)'*Diagonal(B2(f))*f, B3(f))[1] ≈ @. 3*$B2(f)^2
                    else
                        @test_broken gradient(f -> B1(f)'*Diagonal(B2(f))*f, B3(f))[1] ≈ @. 3*$B2(f)^2
                    end
                    
                end

                @testset "Broadcasting" begin
                    @test gradient(f -> sum(@. f*f + 2*f + 1), f)[1] ≈ 2*f+2
                    @test gradient(f -> sum(@. f^2 + 2*f + 1), f)[1] ≈ 2*f+2
                end
                
            end

            @testset "FieldVectors" begin
            
                @test gradient(f -> Map(∇[1]*f)' *     Map(v[1]) + Map(∇[2]*f)' *     Map(v[2]), f)[1] ≈ ∇' * v
                @test gradient(f -> Map(∇[1]*f)' * Fourier(v[1]) + Map(∇[2]*f)' * Fourier(v[2]), f)[1] ≈ ∇' * v
                @test gradient(f -> sum(Diagonal(Map(∇[1]*f)) * v[1] + Diagonal(Map(∇[2]*f)) * v[2]), f)[1] ≈ ∇' * v
            
                @test gradient(f -> @SVector[f,f]' * Map.(@SVector[g,g]), f)[1] ≈ 2g
                @test gradient(f -> @SVector[f,f]' * Fourier.(@SVector[g,g]), f)[1] ≈ 2g
                
                @test gradient(f -> sum(sum(@SVector[f,f])),                            f)[1] ≈ 2*one(f)
                @test gradient(f -> sum(sum(@SVector[f,f]      .+ @SVector[f,f])),      f)[1] ≈ 4*one(f)
                @test gradient(f -> sum(sum(@SMatrix[f f; f f] .+ @SMatrix[f f; f f])), f)[1] ≈ 8*one(f)
                
                # these were broken by (intentionally) removing OuterProdOp .they
                # seem like a fairly unusual, but keeping them here as broken for now... 
                @test_broken gradient(f -> sum(Diagonal.(Map.(∇*f))' * Fourier.(v)), f)[1] ≈ ∇' * v
                @test_broken gradient(f -> sum(Diagonal.(Map.(∇*f))' * Map.(v)), f)[1] ≈ ∇' * v
                @test_broken gradient(f -> sum(sum(Diagonal.(@SMatrix[f f; f f]) * @SVector[f,f])), f)[1] ≈ 8*f

            end
            
            if f isa FlatS0
                
                @testset "logdet" begin
                    @test gradient(x->logdet(x*Diagonal(Map(f))),     1)[1] ≈ size(Map(f))[1]
                    @test gradient(x->logdet(x*Diagonal(Fourier(f))), 1)[1] ≈ size(Map(f))[1]
                    L = ParamDependentOp((;x=1)->x*Diagonal(Fourier(f)))
                    @test gradient(x->logdet(L(x=x)), 1)[1] ≈ size(Map(f))[1]
                end
                
                @test gradient(x -> norm(x*Fourier(f)), 1)[1] ≈ norm(f)
                @test gradient(x -> norm(x*Map(f)), 1)[1] ≈ norm(f)

                L₀ = Diagonal(Map(f))
                @test gradient(x -> norm((x*L₀)*f), 1)[1] ≈ norm(L₀*f)
                L₀ = Diagonal(Fourier(f))
                @test gradient(x -> norm((x*L₀)*f), 1)[1] ≈ norm(L₀*f)

                @test gradient(x -> norm((x*Diagonal(Map(f)))*f), 1)[1] ≈ norm(Diagonal(Map(f))*f)
                @test gradient(x -> norm((x*Diagonal(Fourier(f)))*f), 1)[1] ≈ norm(Diagonal(Fourier(f))*f)

            end

            @testset "Array Bailout" begin

                @testset "Fourier" begin

                    @test_real_gradient(A -> logdet(Diagonal(FFourier((A.*Ils)...; Ny))), A)
                    @test_real_gradient(A -> logdet(Diagonal(A*FFourier(Ils...; Ny))), A)
                    @test_real_gradient(A -> logdet(A*Diagonal(FFourier(Ils...; Ny))), A)
                    
                    @test_real_gradient(A -> f' * (Diagonal(FFourier((A.*Ils)...; Ny))) * f, A)
                    @test_real_gradient(A -> f' * (Diagonal(A*FFourier(Ils...; Ny))) * f, A)
                    @test_real_gradient(A -> f' * (A*Diagonal(FFourier(Ils...; Ny))) * f, A)
                    
                    @test_real_gradient(A -> norm(FFourier((A.*Ils)...; Ny)), A)
                    @test_real_gradient(A -> norm(A*FFourier(Ils...; Ny)), A)

                end

                @testset "Map" begin
                
                    @test_real_gradient(A -> logdet(Diagonal(FMap((A.*Ixs)...))), A)
                    @test_real_gradient(A -> logdet(Diagonal(A*FMap(Ixs...))), A)
                    @test_real_gradient(A -> logdet(A*Diagonal(FMap(Ixs...))), A)
                    
                    @test_real_gradient(A -> f' * (Diagonal(FMap((A.*Ixs)...))) * f, A)
                    @test_real_gradient(A -> f' * (Diagonal(A*FMap(Ixs...))) * f, A)
                    @test_real_gradient(A -> f' * (A*Diagonal(FMap(Ixs...))) * f, A)
                    
                    @test_real_gradient(A -> norm(FMap((A.*Ixs)...)), A)
                    @test_real_gradient(A -> norm(A*FMap(Ixs...)), A)

                end
                
            end
        
        end
        
    end

    @testset "LinearInterpolation" begin
        @test gradient(x->LinearInterpolation([1,2,3],[1,2,3])(x), 2)[1] == 1
        @test gradient(x->LinearInterpolation([1,2,3],[1,x,3])(2), 2)[1] == 1
        @test gradient(x->LinearInterpolation([1,x,3],[1,2,3])(2), 2)[1] == -1
    end
        
end

##

@testset "Lensing" begin
    
    local f, ϕ, Lϕ
    Cℓ = camb().unlensed_total

    @testset "$L" for (L,atol) in [(BilinearLens,300), (LenseFlow,0.2)]

        @testset "Nside = ($Ny,$Nx)" for (Ny,Nx) in Nsides_big

            @testset "T :: $T" for T in (Float32, Float64)
                
                proj = ProjLambert(;Ny,Nx,T,storage)

                Cϕ = maybegpu(Cℓ_to_Cov(:I, proj, Cℓ.ϕϕ))
                @test (ϕ = @inferred simulate(Cϕ)) isa FlatS0
                
                ## S0
                Cf = maybegpu(Cℓ_to_Cov(:I, proj, Cℓ.TT))
                @test (f = @inferred simulate(Cf)) isa FlatS0
                @test (Lϕ = cache(LenseFlow(ϕ),f)) isa CachedLenseFlow
                @test (@inferred Lϕ*f) isa FlatS0
                # adjoints
                f,g = simulate(Cf),simulate(Cf)
                @test f' * (Lϕ * g) ≈ (f' * Lϕ) * g
                # gradients
                δf, δϕ = simulate(Cf), simulate(Cϕ)
                @test_real_gradient(α -> norm(L(ϕ+α*δϕ)*(f+α*δf)), 0, atol=atol)

                # S2 lensing
                Cf = maybegpu(Cℓ_to_Cov(:P, proj, Cℓ.EE, Cℓ.BB))
                @test (f = @inferred simulate(Cf)) isa FlatS2
                @test (Lϕ = cache(LenseFlow(ϕ),f)) isa CachedLenseFlow
                @test (@inferred Lϕ*f) isa FlatS2
                # adjoints
                f,g = simulate(Cf),simulate(Cf)
                @test f' * (Lϕ * g) ≈ (f' * Lϕ) * g
                # gradients
                δf, δϕ = simulate(Cf), simulate(Cϕ)
                @test_real_gradient(α -> norm(L(ϕ+α*δϕ)*(f+α*δf)), 0, atol=atol)
                
            end

        end

    end

end

##

@testset "Posterior" begin
    
    Cℓ = camb()
    L = LenseFlow{RK4Solver{7}}
    T = Float64

    @testset "Nside = $Nside" for Nside in Nsides_big

        @testset "pol = $pol" for pol in (:I, :P)
            
            @unpack f,f̃,ϕ,ds,ds₀ = load_sim(
                Cℓ       = Cℓ,
                θpix     = 3,
                Nside    = Nside,
                T        = T,
                beamFWHM = 3,
                pol      = pol,
                storage  = storage,
                pixel_mask_kwargs = (edge_padding_deg=1,)
            )
            @unpack Cf, Cϕ = ds₀
            f°, ϕ° = mix(ds; f, ϕ)

            @test logpdf(ds; f, ϕ) ≈ logpdf(Mixed(ds); f°, ϕ°) rtol=1e-4

            δf,δϕ = simulate(Cf), simulate(Cϕ)

            @test_real_gradient(α -> logpdf(      ds;  f  = f  + α * δf, ϕ  = ϕ  + α * δϕ), 0, atol=2)
            @test_real_gradient(α -> logpdf(Mixed(ds); f° = f° + α * δf, ϕ° = ϕ° + α * δϕ), 0, atol=2)
            
        end
        
    end

end

##

@require CirculantCov="edf8e0bb-e88b-4581-a03e-dda99a63c493" begin

@testset "EquiRect" begin

    θspan  = (π/2 .- deg2rad.((-40,-70)))
    φspan  = deg2rad.((-60, 60))
    φspan′ = deg2rad.((-50, 50))
    Cℓ     = camb()
    rtol   = 1e-5

    @testset "T = $T" for T in (Float32, Float64)

        @testset "Nside = $Nside" for Nside in [(32,64)]

            Ny, Nx = Nside

            # non-periodic
            proj′ = ProjEquiRect(;Ny, Nx, T, θspan, φspan=φspan′)
            @test (proj′.Ny == length(proj′.θ) == Ny) && (proj′.Nx == length(proj′.φ) == Nx)

            # Make a linear list `θedges[1], θ[1], θedges[2], θ[2], ..., θedges[n], θ[n], θedges[n+1]`
            # and test that it is strictly increasing. 
            ∂θedges = vcat(vcat(proj′.θedges[1:end-1]', proj′.θ')[:],  proj′.θedges[end])
            @test all(diff(∂θedges) .> 0)

            proj′Δφpix   = rem2pi(proj′.φ[2]-proj′.φ[1],   RoundDown)
            proj′Δφspan  = rem2pi(proj′.φ[end]-proj′.φ[1], RoundDown) + proj′Δφpix
            inputΔφspan = φspan′ |> x->rem2pi(x[end]-x[1], RoundDown)
            @test proj′Δφspan ≈ inputΔφspan

            # with integer fraction of 2π (and Float64)
            proj = ProjEquiRect(;Ny, Nx, T, θspan, φspan)
            P = typeof(proj)
            f0   = EquiRectMap(randn(T, Nside...), proj)
            f2   = EquiRectQUMap(randn(T, Nside...), randn(T, Nside...), proj)

            @test f0 isa EquiRectMap
            @test f2 isa EquiRectQUMap

            # transform
            @test Map(AzFourier(f0)) ≈ f0
            @test QUMap(QUAzFourier(f2)) ≈ f2
            @test real(eltype(Map(AzFourier(f0)))) == T
            @test real(eltype(QUMap(QUAzFourier(f2)))) == T

            # transform (testing equality independent of dot)
            @test AzFourier(f0)[:Ix]   ≈ f0[:Ix]
            @test QUAzFourier(f2)[:Px] ≈ f2[:Px]
            @test Map(f0)[:Il]   ≈ f0[:Il]
            @test QUMap(f2)[:Pl] ≈ f2[:Pl]

            # dot product independent of basis
            @test dot(f0,f0) ≈ dot(AzFourier(f0), AzFourier(f0))
            @test dot(f2,f2) ≈ dot(QUAzFourier(f2), QUAzFourier(f2))

            # creating block-diagonal covariance operators
            Cf0 = Cℓ_to_Cov(:I, f0.proj, Cℓ.total.TT)
            Cf2 = Cℓ_to_Cov(:P, f2.proj, Cℓ.total.EE, Cℓ.total.BB)
            Bf0 = CMBLensing.Cℓ_to_Beam(:I, f0.proj, Cℓ.total.TT)
            Bf2 = CMBLensing.Cℓ_to_Beam(:P, f2.proj, Cℓ.total.TT)
            @test Cf0 isa BlockDiagEquiRect{AzFourier,T}
            @test Cf2 isa BlockDiagEquiRect{QUAzFourier,T}
            @test Bf0 isa BlockDiagEquiRect{AzFourier,T}
            @test Bf2 isa BlockDiagEquiRect{QUAzFourier,T}

            # sqrt
            @test (sqrt(Cf0) * sqrt(Cf0) * f0) ≈ (Cf0 * f0) rtol=rtol
            @test (sqrt(Cf2) * sqrt(Cf2) * f2) ≈ (Cf2 * f2) rtol=rtol

            # simulation
            @test real(eltype(simulate(Cf0) :: EquiRectS0)) == T
            @test real(eltype(simulate(Cf2) :: EquiRectS2)) == T
            
            # pinv
            @test (pinv(Cf0) * Cf0 * f0) ≈ f0 rtol=rtol 
            @test (pinv(Cf2) * Cf2 * f2) ≈ f2 rtol=rtol 
        
            @test (Cf0 \ Cf0 * f0) ≈ f0 rtol=rtol
            @test (Cf2 \ Cf2 * f2) ≈ f2 rtol=rtol

            @test (Cf0 / Cf0 * f0) ≈ f0 rtol=rtol
            @test (Cf2 / Cf2 * f2) ≈ f2 rtol=rtol

            # some operator algebra on ops
            @test (Cf0 + Cf0) * f0 ≈ Cf0 * (2 * f0) ≈ (2 * Cf0) * f0
            @test (Cf2 + Cf2) * f2 ≈ Cf2 * (2 * f2) ≈ (2 * Cf2) * f2

            # logdet
            @test logdet(Cf0) ≈ logabsdet(Cf0)[1]
            @test logdet(Cf2) ≈ logabsdet(Cf2)[1]

            # adjoint
            g0 = simulate(Cf0)
            g2 = simulate(Cf2)
            @test f0' * (Cf0 * g0) ≈ (f0' * Cf0) * g0 rtol=rtol
            @test f2' * (Cf2 * g2) ≈ (f2' * Cf2) * g2 rtol=rtol
            
            # Test for correct Fourier symmetry in monopole and nyquist f2 
            let f2kk = f2[:Pl], f2xx = f2[:Px]
                v = f2kk[1:end÷2,1]
                w = f2kk[end÷2+1:end,1]
                @test v ≈ conj.(w)
                if iseven(size(f2xx,2))
                    v′ = f2kk[1:end÷2,end]
                    w′ = f2kk[end÷2+1:end,end]
                    @test v′ ≈ conj.(w′)
                end
            end

            # gradients w.r.t. fields
            @test gradient(f -> dot(f,f), f0)[1] ≈ 2*f0
            @test gradient(f -> dot(f,f), f2)[1] ≈ 2*f2
            @test gradient(f0 -> f0' * (pinv(Cf0) * f0), f0)[1] ≈ (2 * pinv(Cf0) * f0)
            @test gradient(f2 -> f2' * (pinv(Cf2) * f2), f2)[1] ≈ (2 * pinv(Cf2) * f2)

            # gradients through entries of BlockDiagEquiRect
            @test gradient(α -> logabsdet(α * Cf0)[1], 1)[1] ≈ size(Cf0,1)  rtol=1e-2
            @test gradient(α -> logabsdet(α * Cf2)[1], 1)[1] ≈ size(Cf2,1)  rtol=1e-2
            @test_broken gradient(α -> f0' * (pinv(α * Cf0) * f0), 1)[1] isa Real 
            @test_broken gradient(α -> f2' * (pinv(α * Cf2) * f2), 1)[1] isa Real
            
        end

    end

end

end


##

end
