using CMBLensing
using CMBLensing: basis, BasisTuple, @SVector

##

using Test
using SparseArrays
using LinearAlgebra

##

@testset "CMBLensing" begin

##

@testset "Printing" begin

    # basic printing sanity checks, which were super annoying to get right...
    # see: https://discourse.julialang.org/t/dispatching-on-the-result-of-unwrap-unionall-seems-weird/25677
    for m in ((), (MIME("text/plain"),))
        # concrete types:
        for f in [FlatMap(rand(4,4)), FlatQUMap(rand(4,4),rand(4,4))]
            @test 5 < length(sprint(show, m..., f))                                 < 1000
            @test 5 < length(sprint(show, m..., [f,f]))                             < 1000
        end
        # unionall types: (the #s indicates printing correctly dropped to the default behavior)
        @test occursin("#s",sprint(show, m..., FieldTuple{<:Any,<:NamedTuple{(:Q,:U)}}))
        @test occursin("#s",sprint(show, m..., FlatMap{<:Any,<:Any,<:Matrix{Real}}))
        @test occursin("#s",sprint(show, m..., FΦTuple))
        # this is the case that we need the @safe_get for in field_tuples.jl:
        @test_broken occursin("#s",sprint(show, m..., FlatQUMap))
    end

end
##

@testset "Algebra" begin
    
    f0,f2 = [FlatMap(rand(4,4)), FlatQUMap(rand(4,4),rand(4,4))]
    
    for f in [f0,f2]
        
        @testset "f::$(typeof(f))" begin
            
            local Ðf, Ðv, g, H
            
            @test (@inferred f + f) isa typeof(f)
            
            # gradients
            @test (Ðf = @inferred ∇[1]*f) isa Field
            @test (∇[1]'*f ≈ -∇[1]*f)
            @test (-∇[1]'*f ≈ ∇[1]*f)
            @test (@inferred mul!(Ðf,∇[1],Ð(f))) isa Field
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

        # tuple adjoints
        v = similar.(@SVector[f0,f0])
        @test (@inferred mul!(v, tuple_adjoint(f), @SVector[f,f])) isa FieldVector{<:Field{<:Any,S0}}

    end
    
end

##

@testset "FieldTuple constructors & conversions" begin 

    f = FlatMap(rand(4,4))

    # constructors
    @test FieldTuple(Q=f, U=f) isa FieldTuple{<:BasisTuple, <:NamedTuple{(:Q,:U)}}
    @test FieldTuple((Q=f, U=f)) isa FieldTuple{<:BasisTuple, <:NamedTuple{(:Q,:U)}}
    @test FieldTuple{QUMap}((Q=f, U=f)) isa FieldTuple{<:QUMap, <:NamedTuple{(:Q,:U)}}
    @test FieldTuple{<:Any, <:NamedTuple{(:Q,:U)}}(f,f) isa FieldTuple

    # basis conversions
    f_basistuple = FieldTuple(A=f, B=f)
    f_concretebasis = FlatQUMap(rand(4,4), rand(4,4))
    @test basis(@inferred    Fourier(f_basistuple)) <: BasisTuple{Tuple{Fourier,Fourier}}
    @test basis(@inferred        Map(f_basistuple)) <: BasisTuple{Tuple{Map,Map}}
    @test basis(@inferred DerivBasis(f_basistuple)) <: BasisTuple{Tuple{Fourier,Fourier}}
    @test_broken BasisTuple{Tuple{Fourier,Fourier}}(f_basistuple)
    @test basis(@inferred    Fourier(f_concretebasis)) <: BasisTuple{Tuple{Fourier,Fourier}}
    @test basis(@inferred        Map(f_concretebasis)) <: BasisTuple{Tuple{Map,Map}}
    @test basis(@inferred DerivBasis(f_concretebasis)) <: QUFourier

end

##

using Zygote


# make sure we can take type-stable gradients of scalar functions of our Fields
# (like the posterior)
@testset "Autodiff" begin

    f = FlatMap(rand(2,2))

    check_grad(f) = Zygote.gradient(x -> (y=(x .* f); y⋅y), 1)[1]

    @test (@inferred check_grad(f.Ix)) ≈ 2*norm(f,2)^2
    @test (@inferred check_grad(f)) ≈ 2*norm(f,2)^2
    
end

##

@testset "Lensing" begin
    
    local f,ϕ
    
    Cℓ = camb().unlensed_total
    
    for T in (Float32, Float64)
        
        ϵ = sqrt(eps(T))
        Cϕ = Cℓ_to_cov(Flat(Nside=128), Float64, S0, Cℓ.ϕϕ)
        @test (ϕ = @inferred simulate(Cϕ)) isa FlatS0
        Lϕ = LenseFlow(ϕ)
        
        ## S0
        Cf = Cℓ_to_cov(Flat(Nside=128), Float64, S0, Cℓ.TT)
        @test (f = @inferred simulate(Cf)) isa FlatS0
        @test (@inferred Lϕ*f) isa FlatS0
        # adjoints
        f,g = simulate(Cf),simulate(Cf)
        @test f' * (Lϕ * g) ≈ (f' * Lϕ) * g
        # gradients
        δf, δϕ = simulate(Cf), simulate(Cϕ)
        @test (FΦTuple(δf,δϕ)'*(δf̃ϕ_δfϕ(Lϕ,Lϕ*f,f)'*FΦTuple(f,ϕ))) ≈ (f'*((LenseFlow(ϕ+ϵ*δϕ)*(f+ϵ*δf))-(LenseFlow(ϕ-ϵ*δϕ)*(f-ϵ*δf)))/(2ϵ)) rtol=1e-3

        # S2 lensing
        Cf = Cℓ_to_cov(Flat(Nside=128), Float64, S2, Cℓ.EE, Cℓ.BB)
        @test (f = @inferred simulate(Cf)) isa FlatS2
        @test (@inferred Lϕ*f) isa FlatS2
        #adjoints
        f,g = simulate(Cf),simulate(Cf)
        @test f' * (Lϕ * g) ≈ (f' * Lϕ) * g
    end
    
end

##

end
