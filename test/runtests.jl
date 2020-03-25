using CMBLensing
using CMBLensing: 
    @SMatrix, @SVector, AbstractCℓs, basis, Basis, BasisTuple,
    LinearInterpolation, Measurement, RK4Solver, seed!, ±

##

using Test
using SparseArrays
using LinearAlgebra
using Zygote

##

@testset "CMBLensing" begin

##

# basic printing sanity checks, which were super annoying to get right...
# see also: https://discourse.julialang.org/t/dispatching-on-the-result-of-unwrap-unionall-seems-weird/25677

@testset "Printing" begin

    # concrete types:
    for f in [FlatMap(rand(4,4)), FlatQUMap(rand(4,4),rand(4,4))]
        @test occursin("pixels",sprint(show, MIME("text/plain"), f))
        @test occursin("pixels",sprint(show, MIME("text/plain"), [f,f]))
    end
    
    for m in ((), (MIME("text/plain"),))
        # unionall types: (the presence of "where" indicates printing correctly
        # forwarded to the default behavior)
        @test occursin("where",sprint(show, m..., FieldTuple{<:Any,<:NamedTuple{(:Q,:U)}}))
        @test occursin("where",sprint(show, m..., FlatMap{<:Any,<:Any,<:Matrix{Real}}))
        @test occursin("where",sprint(show, m..., FlatQUMap))
    end

end
##

@testset "FieldTuples" begin 

    f = FlatMap(rand(4,4))

    @testset "Constructors" begin

        # Enumerate the combinations of
        # * 1) no basis or names specified, 2) only basis specified, 3) only names specified, 4) basis and names specified
        # * 1) args/kwargs form 2) single Tuple/NamedTuple argument form

        for F in [
            FieldTuple,
            FieldTuple{QUMap},
            FieldTuple{QUMap,<:NamedTuple{(:Q,:U)}},
            FieldTuple{<:Basis,<:NamedTuple{(:Q,:U)}}
            ]

            @testset "F :: $F" begin
                @test (@inferred F(;Q=f, U=f)) isa F
                @test (@inferred F((Q=f, U=f))) isa F
                @test (@inferred F(f,f)) isa F
                @test (@inferred F((f,f))) isa F
            end
            
        end

    end

    @testset "Basis conversions" begin

        # basis conversions
        for f_basistuple in [FieldTuple(f, f), FieldTuple(A=f, B=f)] # named and unnamed
            @test basis(@inferred    Fourier(f_basistuple)) <: BasisTuple{Tuple{Fourier,Fourier}}
            @test basis(@inferred        Map(f_basistuple)) <: BasisTuple{Tuple{Map,Map}}
            @test basis(@inferred DerivBasis(f_basistuple)) <: BasisTuple{Tuple{Fourier,Fourier}}
            @test basis(@inferred BasisTuple{Tuple{Fourier,Fourier}}(f_basistuple)) <: BasisTuple{Tuple{Fourier,Fourier}}
        end

        f_concretebasis = FieldTuple{QUMap, <:NamedTuple{(:Q,:U)}}(f,f)
        @test basis(@inferred    Fourier(f_concretebasis)) <: QUFourier
        @test basis(@inferred        Map(f_concretebasis)) <: QUMap
        @test basis(@inferred DerivBasis(f_concretebasis)) <: QUFourier
        
    end
            

end

##

@testset "Flat Constructors" begin
    
    N = 2
    θpix = 3
    kwargs = (Nside=N, θpix=θpix)
    P = Flat(;kwargs...)
    Ix = rand(N,N)
    Il = rand(N÷2+1,N) + im*rand(N÷2+1,N)
    
    for (F,args) in [
            (FlatMap,        (Ix,)),
            (FlatFourier,    (Il,)),
            (FlatQUMap,      (Ix,Ix)),
            (FlatQUFourier,  (Il,Il)),
            (FlatEBMap,      (Ix,Ix)),
            (FlatEBFourier,  (Il,Il)),
            (FlatIQUMap,     (Ix,Ix,Ix)),
            (FlatIQUFourier, (Il,Il,Il)),
            (FlatIEBMap,     (Ix,Ix,Ix)),
            (FlatIEBFourier, (Il,Il,Il))
        ]
        @testset "f::$F" begin
            @test F(args...; kwargs...) isa F{P}
            @test (@inferred F{P}(args...)) isa F{P}
            @test (@inferred broadcast(real, (F{P}(args...)))) isa F{P}
            if eltype(args[1]) <: Complex
                @test (@inferred F{P}(map(real,args)...)) isa F{P}
            end
            @test real(eltype(@inferred F{P,Float32}(args...))) == Float32
        end
    end

end


##

@testset "Algebra" begin
    
    fs = ((B0,f0),(B2,f2),(Bt,ft)) = [
        (Fourier,   FlatMap(rand(4,4))), 
        (EBFourier, FlatQUMap(rand(4,4),rand(4,4))), # named FieldTuple
        (Fourier,   FieldTuple(FlatMap(rand(4,4)),FlatMap(rand(4,4)))), # unnamed FieldTuple
        (BasisTuple{Tuple{Fourier,EBFourier}}, FlatIQUMap(rand(4,4),rand(4,4),rand(4,4))), # named nested FieldTuple,
        (BasisTuple{Tuple{Fourier,EBFourier}}, FieldTuple(FlatMap(rand(4,4)),FlatQUMap(rand(4,4),rand(4,4)))) # unnamed nested FieldTuple
    ]
    
    for (B,f) in fs
        
        @testset "f::$(typeof(f))" begin
            
            local Ðf, Ðv, g, H
            
            @test similar(f) isa typeof(f)
            @test zero(f) isa typeof(f)
            @test similar(f,Float32) isa Field
            @test eltype(similar(f,Float32)) == Float32
                        
            # broadcasting
            @test (@inferred f .+ f) isa typeof(f)
            @test (@inferred f .+ Float32.(f)) isa typeof(f)
            
            # promotion
            @test (@inferred f + B(f)) isa typeof(f)
            @test (@inferred f + B(Float32.(f))) isa typeof(f)
            
            # gradients
            @test (Ðf = @inferred ∇[1]*f) isa Field
            @test (∇[1]'*f ≈ -∇[1]*f)
            @test (-∇[1]'*f ≈ ∇[1]*f)
            @test (@inferred mul!(Ðf,∇[1],Ð(f))) isa Field
            @test (Ðv = @inferred ∇*f) isa FieldVector
            @test (@inferred mul!(Ðv,∇,Ð(f))) isa FieldVector
            @test ((g,H) = map(Ł, (@inferred gradhess(f)))) isa NamedTuple{<:Any, <:Tuple{FieldVector, FieldMatrix}}
            
            # Diagonal broadcasting
            @test (@inferred Diagonal(f) .* Diagonal(f) .* Diagonal(f)) isa typeof(Diagonal(f))
            
            # inverses
            @test (@inferred pinv(Diagonal(f))) isa Diagonal{<:Any,<:typeof(f)}
            @test_throws SingularException inv(Diagonal(0*f))
            
            # Field dot products
            D = Diagonal(f)
            @test (@inferred f' * f) isa Real
            @test (@inferred f' * B(f)) isa Real
            @test (@inferred f' * D * f) isa Real
            @test sum(f, dims=:) ≈ sum(f[:])
            @test_throws Any sum(f, dims=1)
            @test sum(f, dims=2) == f

            
            if f isa FlatS0
                # FieldVector dot product
                @test (@inferred Diagonal.(g)' * g) isa typeof(g[1])
                @test (@inferred mul!(similar(g[1]), Diagonal.(g)', g)) isa typeof(g[1])
                
                # FieldMatrix-FieldVector product
                @test (@inferred Diagonal.(H) * g) isa FieldVector
                @test (@inferred Diagonal.(H) * Diagonal.(g)) isa FieldOrOpVector
                @test (@inferred mul!(Diagonal.(similar.(g)), Diagonal.(H), Diagonal.(g))) isa FieldOrOpVector
            
            end
            
            # Explicit vs. lazy DiagOp algebra
            @test (Diagonal(Ð(f)) + Diagonal(Ð(f))) isa DiagOp{<:Field{basis(Ð(f))}}
            @test (Diagonal(Ł(f)) + Diagonal(Ð(f))) isa LazyBinaryOp
            @test (Diagonal(Ł(f)) + Diagonal(Ł(f))) isa DiagOp{<:Field{basis(Ł(f))}}
        
            # tuple adjoints
            v = similar.(@SVector[f0,f0])
            @test (@inferred mul!(f0, tuple_adjoint(f), f)) isa Field{<:Any,S0}
            @test (@inferred mul!(v, tuple_adjoint(f), @SVector[f,f])) isa FieldVector{<:Field{<:Any,S0}}

        end
        
    end
    
    # mixed-spin
    @test (@inferred f0 .* f2) isa typeof(f2)
    @test (@inferred f0 .* ft) isa typeof(ft)
    
    # matrix type promotion
    @test_broken (@inferred FlatMap(rand(Float32,2,2)) + FlatMap(spzeros(Float64,2,2))) isa FlatMap{<:Any,Float64,<:Matrix}
    
end

##

@testset "FlatS02" begin
    
    ΣTT, ΣTE, ΣEE, ΣBB = [Diagonal(Fourier(FlatMap(rand(3,3)))) for i=1:4]
    L = FlatIEBCov(@SMatrix([ΣTT ΣTE; ΣTE ΣEE]), ΣBB)
    f = IEBFourier(FlatIEBMap(rand(3,3),rand(3,3),rand(3,3)))

    @test (sqrt(L) * @inferred(@inferred(sqrt(L)) * f)) ≈ (L * f)
    @test (L * @inferred(@inferred(pinv(L)) * f)) ≈ f
    @test @inferred(L * L) isa FlatIEBCov
    @test @inferred(L + L) isa FlatIEBCov
    @test L * Diagonal(f) isa FlatIEBCov
    @test Diagonal(f) * L isa FlatIEBCov
    @test_broken @inferred L * Diagonal(f)
    @test @inferred(diag(L)) isa FlatIEBFourier
    @test @inferred(L + I) isa FlatIEBCov
    @test @inferred(2 * L) isa FlatIEBCov
    @test @inferred(similar(L)) isa FlatIEBCov
    @test (L .= 2L) isa FlatIEBCov

end

##

@testset "Gradients" begin
    
    @test (@inferred ∇[1] * FlatMap(rand(3,3), ∂mode=fourier∂)) isa FlatFourier
    @test (@inferred ∇[1] * FlatQUMap(rand(3,3), rand(3,3), ∂mode=fourier∂)) isa FlatQUFourier
    @test (@inferred ∇[1] * FlatIQUMap(rand(3,3), rand(3,3), rand(3,3), ∂mode=fourier∂)) isa FlatIQUFourier
    
    @test (@inferred ∇[1] * Fourier(FlatMap(rand(3,3), ∂mode=map∂))) isa FlatMap
    @test (@inferred ∇[1] * QUFourier(FlatQUMap(rand(3,3), rand(3,3), ∂mode=map∂))) isa FlatQUMap
    @test (@inferred ∇[1] * BasisTuple{Tuple{Fourier,QUFourier}}(FlatIQUMap(rand(3,3), rand(3,3), rand(3,3), ∂mode=map∂))) isa FlatIQUMap
    
end

##

@testset "Misc" begin
    
    f = FlatMap(rand(4,4))
    
    @test                  @inferred(MidPass(100,200) .* Diagonal(Fourier(f))) isa Diagonal
    @test_throws Exception           MidPass(100,200) .* Diagonal(        f)
    
end

##

@testset "Cℓs" begin
    
    @test InterpolatedCℓs(1:100, rand(100))      isa AbstractCℓs{Float64}
    @test InterpolatedCℓs(1:100, rand(100) .± 1) isa AbstractCℓs{Measurement{Float64}}
    @test (InterpolatedCℓs(1:100, 1:100) * ℓ²)[50] == 50^3
    
end

##

@testset "ParamDependentOp" begin
    
    D = Diagonal(FlatMap(rand(4,4)))
    mem = similar(D)
    
    @test_throws ArgumentError ParamDependentOp((;x=1, y=1)->x*y*D)(mem) # passing memory to non-inplace op
    @test_throws ArgumentError ParamDependentOp((mem;x=1, y=1)->mem.=x*y*D,similar(D))(1) # passing wrong-type memory
    
    @test ParamDependentOp((;x=1, y=1)->x*y*D)() ≈ D
    @test ParamDependentOp((;x=1, y=1)->x*y*D)(z=2) ≈ D
    @test ParamDependentOp((;x=1, y=1)->x*y*D)(x=2) ≈ 2D
    @test ParamDependentOp((;x=1, y=1)->x*y*D)((x=2,y=2)) ≈ 4D # tuple calling form
    @test ParamDependentOp((mem;x=1, y=1)->mem.=x*y*D,similar(D))(D) ≈ D # inplace 
end

##

@testset "Zygote" begin

    for (f,g,h) in [
        @repeated(FlatMap(rand(2,2)),3), 
        @repeated(FlatQUMap(rand(2,2),rand(2,2)),3)
    ]
    
        @testset "$(typeof(f))" begin
        
            v = @SVector[f,f]
            D = Diagonal(f)

            @testset "Fields" begin
                
                @testset "sum" begin
                    @test ((δ = gradient(f -> sum(Map(f)),     Map(f))[1]); basis(δ)==basis(Map(f))     && δ ≈ one(Map(f)))
                    @test ((δ = gradient(f -> sum(Map(f)), Fourier(f))[1]); basis(δ)==basis(Fourier(f)) && δ ≈ Fourier(one(Map(f))))
                end
                
                for B1=[Map,Fourier], B2=[Map,Fourier], B3=[Map,Fourier]
                    @testset "B1=$B1, B2=$B2, B3=$B3" begin
                        @test gradient(f -> dot(B1(f),B2(f)), B3(f))[1] ≈ 2f
                        @test gradient(f -> norm(B1(f)), B3(f))[1] ≈ f/norm(f)
                        @test gradient(f -> B1(f)' * B2(g), B3(f))[1] ≈ g
                        @test gradient(f -> sum(Diagonal(Map(f)) * B2(g)), B3(f))[1] ≈ g
                        @test gradient(f -> sum(Diagonal(Map(∇[1]*f)) * B2(g)), B3(f))[1] ≈ ∇[1]'*g
                        @test gradient(f -> B1(f)'*(D\B2(f)), B3(f))[1] ≈ D\f + D'\f
                        @test gradient(f -> (B1(f)'/D)*B2(f), B3(f))[1] ≈ D\f + D'\f
                        @test gradient(f -> B1(f)'*(D*B2(f)), B3(f))[1] ≈ D*f + D'*f
                        @test gradient(f -> (B1(f)'*D)*B2(f), B3(f))[1] ≈ D*f + D'*f
                        @test gradient(f -> B1(f)'*Diagonal(B2(f))*f, B3(f))[1] ≈ @. 3*$B2(f)^2
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
                
                @test gradient(f -> sum(Diagonal.(Map.(∇*f))' * Fourier.(v)), f)[1] ≈ ∇' * v
                @test gradient(f -> sum(Diagonal.(Map.(∇*f))' * Map.(v)), f)[1] ≈ ∇' * v
                
                @test gradient(f -> sum(sum(@SVector[f,f])),                            f)[1] ≈ 2*one(f)
                @test gradient(f -> sum(sum(@SVector[f,f]      .+ @SVector[f,f])),      f)[1] ≈ 4*one(f)
                @test gradient(f -> sum(sum(@SMatrix[f f; f f] .+ @SMatrix[f f; f f])), f)[1] ≈ 8*one(f)
                
                @test gradient(f -> sum(sum(Diagonal.(@SMatrix[f f; f f]) * @SVector[f,f])), f)[1] ≈ 8*f
                @test_broken gradient(f -> sum(sum(@SMatrix[f f] * @SMatrix[f f; f f])), f)[1] ≈ 8*f

            end
            
            @testset "OuterProdOp" begin
                
                @test OuterProdOp(f,g) * h ≈ f*(g'*h)
                @test OuterProdOp(f,g)' * h ≈ g*(f'*h)
                @test diag(OuterProdOp(f,g)) ≈ f .* conj.(g)
                @test diag(OuterProdOp(f,g)') ≈ conj.(f) .* g
                @test diag(OuterProdOp(f,g) + OuterProdOp(f,g)) ≈ 2 .* f .* conj.(g)
                
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
    
    local f,ϕ
    
    Cℓ = camb().unlensed_total
    nside = 128
    
    for T in (Float32, Float64)
        
        @testset "T :: $T" begin
                
            ε = sqrt(eps(T))
            Cϕ = Cℓ_to_Cov(Flat(Nside=nside), T, S0, Cℓ.ϕϕ)
            @test (ϕ = @inferred simulate(Cϕ)) isa FlatS0
            Lϕ = LenseFlow(ϕ)
            
            ## S0
            Cf = Cℓ_to_Cov(Flat(Nside=nside), T, S0, Cℓ.TT)
            @test (f = @inferred simulate(Cf)) isa FlatS0
            @test (@inferred Lϕ*f) isa FlatS0
            # adjoints
            f,g = simulate(Cf),simulate(Cf)
            @test f' * (Lϕ * g) ≈ (f' * Lϕ) * g
            # gradients
            δf, δϕ = simulate(Cf), simulate(Cϕ)
            @test FieldTuple(gradient((f′,ϕ) -> f'*(LenseFlow(ϕ)*f′), f, ϕ))' * FieldTuple(δf,δϕ) ≈ 
                (f'*((LenseFlow(ϕ+ε*δϕ)*(f+ε*δf))-(LenseFlow(ϕ-ε*δϕ)*(f-ε*δf)))/(2ε)) rtol=1e-2

            # S2 lensing
            Cf = Cℓ_to_Cov(Flat(Nside=nside), T, S2, Cℓ.EE, Cℓ.BB)
            @test (f = @inferred simulate(Cf)) isa FlatS2
            @test (@inferred Lϕ*f) isa FlatS2
            # adjoints
            f,g = simulate(Cf),simulate(Cf)
            @test f' * (Lϕ * g) ≈ (f' * Lϕ) * g
            # gradients
            δf, δϕ = simulate(Cf), simulate(Cϕ)
            @test FieldTuple(gradient((f′,ϕ) -> f'*(LenseFlow(ϕ)*f′), f, ϕ))' * FieldTuple(δf,δϕ) ≈ 
                (f'*((LenseFlow(ϕ+ε*δϕ)*(f+ε*δf))-(LenseFlow(ϕ-ε*δϕ)*(f-ε*δf)))/(2ε)) rtol=1e-2
        
        end
        
    end
    
end

##

@testset "Posterior" begin
    
    Cℓ = camb()
    L = LenseFlow{RK4Solver{7}}
    T = Float64
    
    for pol in (:I,:P)
        
        @testset "pol = $pol" begin
            
            @unpack f,f̃,ϕ,ds,ds₀ = load_sim_dataset(
                seed  = 0,
                Cℓ    = Cℓ,
                θpix  = 3,
                Nside = 128,
                T     = T,
                beamFWHM = 3,
                pol   = pol,
                L     = L,
                pixel_mask_kwargs = (edge_padding_deg=2,)
                );
            @unpack Cf,Cϕ,D = ds₀
            f° = L(ϕ)*D*f

            @test lnP(0,f,ϕ,ds) ≈ lnP(1,    f̃,  ϕ ,ds) rtol=1e-4
            @test lnP(0,f,ϕ,ds) ≈ lnP(:mix, f°, ϕ, ds) rtol=1e-4

            ε = sqrt(eps(T))
            seed!(0)
            δf,δϕ = simulate(Cf),simulate(Cϕ)
            
            @test FieldTuple(gradient((f,ϕ)->lnP(0,f,ϕ,ds),f,ϕ))'*FieldTuple(δf,δϕ) ≈ 
                (lnP(0,f+ε*δf,ϕ+ε*δϕ,ds)-lnP(0,f-ε*δf,ϕ-ε*δϕ,ds))/(2ε)  rtol=1e-3
            @test FieldTuple(gradient((f̃,ϕ)->lnP(1,f̃,ϕ,ds),f̃,ϕ))'*FieldTuple(δf,δϕ) ≈ 
                (lnP(1,f̃+ε*δf,ϕ+ε*δϕ,ds)-lnP(1,f̃-ε*δf,ϕ-ε*δϕ,ds))/(2ε)  rtol=1e-2
            @test FieldTuple(gradient((f°,ϕ)->lnP(:mix,f°,ϕ,ds),f°,ϕ))'*FieldTuple(δf,δϕ) ≈ 
                (lnP(:mix,f°+ε*δf,ϕ+ε*δϕ,ds)-lnP(:mix,f°-ε*δf,ϕ-ε*δϕ,ds))/(2ε)  rtol=1e-2
            
        end
        
    end
    
end

##

end
