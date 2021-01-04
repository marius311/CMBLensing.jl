
if get(ENV, "CMBLENSING_TEST_CUDA", false) == "1"
    using Adapt, CUDA
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
using CMBLensing: 
    @SMatrix, @SVector, AbstractCℓs, basis, Basis, BasisTuple,
    LinearInterpolation, Measurement, RK4Solver, seed!, ±

##

using Test
using SparseArrays
using LinearAlgebra
using Zygote
using AbstractFFTs

##

Nsides  = [(8,  (8,8)),     ((8,8),    (8,8)),     ((4,8),    (4,8))]
Nsides′ = [(128,(128,128)), ((128,128),(128,128)), ((196,128),(196,128))]

##

@testset "CMBLensing" begin

##

# basic printing sanity checks, which were super annoying to get right...
# see also: https://discourse.julialang.org/t/dispatching-on-the-result-of-unwrap-unionall-seems-weird/25677

@testset "Printing" begin

    # concrete types:
    for f in [maybegpu(FlatMap(rand(4,4))), maybegpu(FlatQUMap(rand(4,4),rand(4,4)))]
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

@testset "Flat" begin 

    @testset "Constructors" begin

        @testset "batch=$D" for D in [(), (3,)]

            N = 8
            Ix = maybegpu(rand(N,N,D...))
            Il = maybegpu(rand(N÷2+1,N,D...))

            @testset "$(basis(F))" for (F,ks,args,kwargs) in [
                (FlatMap,        (:Ix,),        (Ix,),      ()),
                (FlatFourier,    (:Il,),        (Il,),      (Ny=N,)),
                (FlatQUMap,      (:Qx,:Qx),     (Ix,Ix),    ()),
                (FlatQUFourier,  (:Ql,:Ql),     (Il,Il),    (Ny=N,)),
                (FlatEBMap,      (:Ex,:Bx),     (Ix,Ix),    ()),
                (FlatEBFourier,  (:El,:Bl),     (Il,Il),    (Ny=N,)),
                (FlatIQUMap,     (:Ix,:Qx,:Qx), (Ix,Ix,Ix), ()),
                (FlatIQUFourier, (:Il,:Ql,:Ql), (Il,Il,Il), (Ny=N,)),
                (FlatIEBMap,     (:Ix,:Ex,:Bx), (Ix,Ix,Ix), ()),
                (FlatIEBFourier, (:Il,:El,:Bl), (Il,Il,Il), (Ny=N,)),
            ]
                local f
                @test (f = F(args...; kwargs...)) isa F
                @test @inferred(F(getproperty.(Ref(f),ks)..., f.metadata)) == f
            end
        
        end
    
    end

    @testset "Basis conversions" begin
        @testset "$(typealias(Bin)) → $(typealias(Bout))" for (f,Bin,Bout) in [
            (f,Bin,Bout)
            for (f,Bs) in [
                (FlatMap(rand(N,N)),             (Map,Fourier)),
                (FlatQUMap(rand(N,N),rand(N,N)), (QUMap,QUFourier,EBMap,EBFourier))
            ]
            for Bin in Bs
            for Bout in Bs
        ]

            @test basis(@inferred(Bout(Bin(f)))) == Bout
            # @test Bin(Bout(Bin(f))) == f

        end
    end

end

##

@testset "Algebra" begin
    
    @testset "Nside = $Nside" for (Nside,Nside2D) in Nsides

        fs = ((B0,f0),(B2,f2),(Bt,ft)) = [
            (Fourier,    maybegpu(FlatMap(rand(Nside2D...)))), 
            (EBFourier,  maybegpu(FlatQUMap(rand(Nside2D...),rand(Nside2D...)))), # named FieldTuple
            (Fourier,    maybegpu(FieldTuple(FlatMap(rand(Nside2D...)),FlatMap(rand(Nside2D...))))), # unnamed FieldTuple
            (IEBFourier, maybegpu(FlatIQUMap(rand(Nside2D...),rand(Nside2D...),rand(Nside2D...)))), # named nested FieldTuple
            (IEBFourier, maybegpu(FieldTuple(FlatMap(rand(Nside2D...)),FlatQUMap(rand(Nside2D...),rand(Nside2D...))))), # unnamed nested FieldTuple
            (Fourier,    maybegpu(FlatMap(rand(Nside2D...,2)))), # batched S0 
            (EBFourier,  maybegpu(FlatQUMap(rand(Nside2D...,2),rand(Nside2D...,2)))), # batched S2
        ]
            
        @testset "f :: $(typeof(f))" for (B,f) in fs
            
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
            if f isa FlatField && batchsize(f)>1 # batched fields not inferred
                @test (f' * f) isa Real
                @test (f' * B(f)) isa Real
                @test (f' * D * f) isa Real
            else
                @test (@inferred f' * f) isa Real
                @test (@inferred f' * B(f)) isa Real
                @test (@inferred f' * D * f) isa Real
            end
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
            f0b = identity.(batch(f0,batchsize(f)))
            v = similar.(@SVector[f0b,f0b])
            @test (@inferred mul!(f0b, tuple_adjoint(f), f)) isa Field{<:Any,S0}
            @test (@inferred mul!(v, tuple_adjoint(f), @SVector[f,f])) isa FieldVector{<:Field{<:Any,S0}}

        end
            
        # mixed-spin
        @test (@inferred f0 .* f2) isa typeof(f2)
        @test (@inferred f0 .* ft) isa typeof(ft)
        
        # matrix type promotion
        @test_broken (@inferred FlatMap(rand(Float32,2,2)) + FlatMap(spzeros(Float64,2,2))) isa FlatMap{<:Any,Float64,<:Matrix}
        
    end

end

##

@testset "FlatS2" begin
    @testset "Nside = $Nside" for (Nside,Nside2D) in Nsides
        C = maybegpu(Diagonal(EBFourier(FlatEBMap(rand(Nside2D...), rand(Nside2D...)))))
        f = maybegpu(FlatQUMap(rand(Nside2D...), rand(Nside2D...)))
        @test C*f ≈ FlatQUFourier(C[:QQ]*f[:Q]+C[:QU]*f[:U], C[:UU]*f[:U]+C[:UQ]*f[:Q])
    end
end

##

@testset "FlatS02" begin
    
    @testset "Nside = $Nside" for (Nside,Nside2D) in Nsides

        ΣTT, ΣTE, ΣEE, ΣBB = [Diagonal(Fourier(maybegpu(FlatMap(rand(Nside2D...))))) for i=1:4]
        L = FlatIEBCov(@SMatrix([ΣTT ΣTE; ΣTE ΣEE]), ΣBB)
        f = maybegpu(IEBFourier(FlatIEBMap(rand(Nside2D...),rand(Nside2D...),rand(Nside2D...))))

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

end

##

@testset "BatchedReal" begin

    @testset "Nside = $Nside" for (Nside,Nside2D) in Nsides

        Nside2D = Nside .* (1,1)

        r  = 1.
        rb = batch([1.,2])
        
        @testset "f :: $(typeof(f))" for (f,fb) in [
            (maybegpu(FlatMap(rand(Nside2D...))),                    maybegpu(FlatMap(rand(Nside2D...,2)))),
            (maybegpu(FlatQUMap(rand(Nside2D...),rand(Nside2D...))), maybegpu(FlatQUMap(rand(Nside2D...,2),rand(Nside2D...,2))))
        ]

            @test @inferred(r * f)  == f
            @test @inferred(r * fb) == fb
            @test unbatch(@inferred(rb * f)) == [f, 2f]
            @test unbatch(@inferred(rb * fb)) == [batchindex(fb,1), 2batchindex(fb,2)]

        end

    end
    
end

## 

@testset "Gradients" begin

    @testset "Nside = $Nside" for (Nside,Nside2D) in Nsides
    
        @test (@inferred ∇[1] * maybegpu(FlatMap(rand(Nside2D...), ∂mode=fourier∂))) isa FlatFourier
        @test (@inferred ∇[1] * maybegpu(FlatQUMap(rand(Nside2D...), rand(Nside2D...), ∂mode=fourier∂))) isa FlatQUFourier
        @test (@inferred ∇[1] * maybegpu(FlatIQUMap(rand(Nside2D...), rand(Nside2D...), rand(Nside2D...), ∂mode=fourier∂))) isa FlatIQUFourier
        
        @test (@inferred ∇[1] * Fourier(FlatMap(rand(Nside2D...), ∂mode=map∂))) isa FlatMap
        @test (@inferred ∇[1] * QUFourier(FlatQUMap(rand(Nside2D...), rand(Nside2D...), ∂mode=map∂))) isa FlatQUMap
        @test (@inferred ∇[1] * BasisTuple{Tuple{Fourier,QUFourier}}(FlatIQUMap(rand(Nside2D...), rand(Nside2D...), rand(Nside2D...), ∂mode=map∂))) isa FlatIQUMap

    end
    
end

##

@testset "Misc" begin
    
    @testset "Nside = $Nside" for (Nside,Nside2D) in Nsides

        f = maybegpu(FlatMap(rand(Nside2D...)))
        
        @test                  @inferred(MidPass(100,200) .* Diagonal(Fourier(f))) isa Diagonal
        @test_throws Exception           MidPass(100,200) .* Diagonal(        f)

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

    @testset "Nside = $Nside" for (Nside,Nside2D) in Nsides

        @testset "$(typeof(f))" for (f,g,h) in [
            @repeated(maybegpu(FlatMap(rand(Nside2D...))),3), 
            @repeated(maybegpu(FlatQUMap(rand(Nside2D...),rand(Nside2D...))),3)
        ]
        
            v = @SVector[f,f]
            D = Diagonal(f)

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
                
                @test gradient(f -> sum(Diagonal.(Map.(∇*f))' * Fourier.(v)), f)[1] ≈ ∇' * v
                @test gradient(f -> sum(Diagonal.(Map.(∇*f))' * Map.(v)), f)[1] ≈ ∇' * v
                
                @test gradient(f -> sum(sum(@SVector[f,f])),                            f)[1] ≈ 2*one(f)
                @test gradient(f -> sum(sum(@SVector[f,f]      .+ @SVector[f,f])),      f)[1] ≈ 4*one(f)
                @test gradient(f -> sum(sum(@SMatrix[f f; f f] .+ @SMatrix[f f; f f])), f)[1] ≈ 8*one(f)
                
                @test gradient(f -> sum(sum(Diagonal.(@SMatrix[f f; f f]) * @SVector[f,f])), f)[1] ≈ 8*f

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
    seed!(0)

    @testset "Nside = $Nside" for (Nside,) in Nsides′

        @testset "T :: $T" for T in (Float32, Float64)
            
            ε = sqrt(eps(T))
            Cϕ = maybegpu(Cℓ_to_Cov(Flat(Nside=Nside), T, S0, Cℓ.ϕϕ))
            @test (ϕ = @inferred simulate(Cϕ)) isa FlatS0
            Lϕ = LenseFlow(ϕ)
            
            ## S0
            Cf = maybegpu(Cℓ_to_Cov(Flat(Nside=Nside), T, S0, Cℓ.TT))
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
            Cf = maybegpu(Cℓ_to_Cov(Flat(Nside=Nside), T, S2, Cℓ.EE, Cℓ.BB))
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

    @testset "Nside = $Nside" for (Nside,) in Nsides′

        @testset "pol = $pol" for pol in (:I,:P)
            
            @unpack f,f̃,ϕ,ds,ds₀ = load_sim(
                seed     = 0,
                Cℓ       = Cℓ,
                θpix     = 3,
                Nside    = Nside,
                T        = T,
                beamFWHM = 3,
                pol      = pol,
                L        = L,
                storage  = storage,
                pixel_mask_kwargs = (edge_padding_deg=1,)
            )
            @unpack Cf,Cϕ,D = ds₀
            f° = L(ϕ)*D*f

            @test lnP(0,f,ϕ,ds) ≈ lnP(1,    f̃,  ϕ ,ds) rtol=1e-4
            @test lnP(0,f,ϕ,ds) ≈ lnP(:mix, f°, ϕ, ds) rtol=1e-4

            ε = sqrt(eps(T))
            seed!(1)
            δf,δϕ = simulate(Cf),simulate(Cϕ)
            
            @test FieldTuple(gradient((f,ϕ)->lnP(0,f,ϕ,ds),f,ϕ))'*FieldTuple(δf,δϕ) ≈ 
                (lnP(0,f+ε*δf,ϕ+ε*δϕ,ds)-lnP(0,f-ε*δf,ϕ-ε*δϕ,ds))/(2ε)  rtol=3e-2
            @test FieldTuple(gradient((f̃,ϕ)->lnP(1,f̃,ϕ,ds),f̃,ϕ))'*FieldTuple(δf,δϕ) ≈ 
                (lnP(1,f̃+ε*δf,ϕ+ε*δϕ,ds)-lnP(1,f̃-ε*δf,ϕ-ε*δϕ,ds))/(2ε)  rtol=3e-2
            @test FieldTuple(gradient((f°,ϕ)->lnP(:mix,f°,ϕ,ds),f°,ϕ))'*FieldTuple(δf,δϕ) ≈ 
                (lnP(:mix,f°+ε*δf,ϕ+ε*δϕ,ds)-lnP(:mix,f°-ε*δf,ϕ-ε*δϕ,ds))/(2ε)  rtol=3e-2
            
        end
        
    end

end

##

end
