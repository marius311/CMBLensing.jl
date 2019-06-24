using CMBLensing
using CMBLensing: basis, BasisTuple

##

using Test
using SparseArrays
using LinearAlgebra

##

@testset "CMBLensing" begin

##

@testset "Algebra" begin
    
    f0,f2 = [FlatMap(rand(4,4)), FlatQUMap(rand(4,4),rand(4,4))]
    
    for f in [f0,f2]
        
        @testset "f::$(typeof(f))" begin
            
            local Ðf, Ðv, g, H
            
            @test (@inferred f + f) isa typeof(f)
            
            # gradients
            @test (Ðf = @inferred ∇₀*f) isa Field
            @test (@inferred mul!(Ðf,∇₀,Ð(f))) isa Field
            @test (Ðv = @inferred ∇*f) isa FieldVector
            @test (@inferred mul!(Ðv,∇,Ð(f))) isa FieldVector
            @test ((g,H) = Ł.(@inferred gradhess(f))) isa Tuple{FieldVector, FieldMatrix}
            
            # Diagonal broadcasting
            @test (@inferred Diagonal(f) .* Diagonal(f) .* Diagonal(f)) isa typeof(Diagonal(f))
            
            # inverses
            @test (@inferred pinv(Diagonal(f))) isa Diagonal{<:Any,<:typeof(f)}
            @test_throws SingularException inv(Diagonal(0*f))
            
            # Field dot products
            @test (@inferred f' * f) isa eltype(f)
            # @test (@inferred @SVector[f,f]' * @SVector[f,f]) isa eltype(f)
            
            if f isa FlatS0
                # FieldVector dot product
                @test (@inferred Diagonal.(g)' * g) isa typeof(g[1])
                @test (@inferred mul!(similar(g[1]), Diagonal.(g)', g)) isa typeof(g[1])
                
                # FieldMatrix-FieldVector product
                @test (@inferred Diagonal.(H) * g) isa FieldVector
                @test (@inferred Diagonal.(H) * Diagonal.(g)) isa FieldOrOpVector
                @test (@inferred mul!(Diagonal.(similar.(g)), Diagonal.(H), Diagonal.(g))) isa FieldOrOpVector
            end
            
        end
        
        # eltype promotion
        @test (@inferred broadcast(+, FlatMap(rand(Float32,2,2)), FlatMap(rand(Float64,2,2)))) isa FlatMap{<:Any,Float64}
        # matrix type promotion
        @test (@inferred FlatMap(rand(Float32,2,2)) + FlatMap(spzeros(Float64,2,2))) isa FlatMap{<:Any,Float64,<:Matrix}

    end
    
end

##

@testset "FieldTuple basis conversions" begin 

    f = FlatMap(rand(4,4))
    f_basistuple = FieldTuple(A=f, B=f)

    @test basis(@inferred    Fourier(f_basistuple)) <: BasisTuple{Tuple{Fourier,Fourier}}
    @test basis(@inferred        Map(f_basistuple)) <: BasisTuple{Tuple{Map,Map}}
    @test basis(@inferred DerivBasis(f_basistuple)) <: BasisTuple{Tuple{Fourier,Fourier}}

    @test_broken BasisTuple{Tuple{Fourier,Fourier}}(f_basistuple)

    f_concretebasis = FlatQUMap(rand(4,4), rand(4,4))

    @test basis(@inferred    Fourier(f_concretebasis)) <: BasisTuple{Tuple{Fourier,Fourier}}
    @test basis(@inferred        Map(f_concretebasis)) <: BasisTuple{Tuple{Map,Map}}
    @test basis(@inferred DerivBasis(f_concretebasis)) <: QUFourier

end

##

using Zygote

@testset "Autodiff" begin

    f = FlatMap(rand(2,2))

    check_grad(f) = Zygote.gradient(x -> (y=(x .* f); y⋅y), 1)[1]

    @test (@inferred check_grad(f.Ix)) ≈ 2*norm(f,2)^2
    @test (@inferred check_grad(f)) ≈ 2*norm(f,2)^2
    
end

##

@testset "Lensing" begin
    
    @test begin 
        
        Cℓ = camb().unlensed_total
        Cf = Cℓ_to_cov(Flat(Nside=128), Float64, S0, Cℓ.TT)
        Cϕ = Cℓ_to_cov(Flat(Nside=128), Float64, S0, Cℓ.ϕϕ)
        f = simulate(Cf)
        ϕ = simulate(Cϕ)
        
        (@inferred LenseFlow(ϕ)*f) isa Field
    
    end
    
end


##

end
