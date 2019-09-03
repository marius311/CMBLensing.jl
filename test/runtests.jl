using CMBLensing
using CMBLensing: basis, BasisTuple, @SVector, RK4Solver

##

using Test
using SparseArrays
using LinearAlgebra
using Zygote: gradient

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
        @test occursin("where",sprint(show, m..., FΦTuple))
        @test occursin("where",sprint(show, m..., FlatQUMap))
    end

end
##

@testset "FieldTuples" begin 

    f = FlatMap(rand(4,4))

    # constructors
    @test FieldTuple(Q=f, U=f) isa FieldTuple{<:BasisTuple, <:NamedTuple{(:Q,:U)}}
    @test FieldTuple((Q=f, U=f)) isa FieldTuple{<:BasisTuple, <:NamedTuple{(:Q,:U)}}
    @test FieldTuple{QUMap}((Q=f, U=f)) isa FieldTuple{<:QUMap, <:NamedTuple{(:Q,:U)}}
    @test FieldTuple{<:Any, <:NamedTuple{(:Q,:U)}}(f,f) isa FieldTuple

    # basis conversions
    for f_basistuple in [FieldTuple(f, f), FieldTuple(A=f, B=f)] # named and unnamed
        @test basis(@inferred    Fourier(f_basistuple)) <: BasisTuple{Tuple{Fourier,Fourier}}
        @test basis(@inferred        Map(f_basistuple)) <: BasisTuple{Tuple{Map,Map}}
        @test basis(@inferred DerivBasis(f_basistuple)) <: BasisTuple{Tuple{Fourier,Fourier}}
        @test basis(@inferred BasisTuple{Tuple{Fourier,Fourier}}(f_basistuple)) <: BasisTuple{Tuple{Fourier,Fourier}}
    end
    f_concretebasis = FlatQUMap(rand(4,4), rand(4,4))
    @test basis(@inferred    Fourier(f_concretebasis)) <: BasisTuple{Tuple{Fourier,Fourier}}
    @test basis(@inferred        Map(f_concretebasis)) <: BasisTuple{Tuple{Map,Map}}
    @test basis(@inferred DerivBasis(f_concretebasis)) <: QUFourier

end

##

@testset "Flat Constructors" begin
    
    N = 2
    θpix = 3
    kwargs = (Nside=N, θpix=θpix)
    P = Flat(;kwargs...)
    Ix = rand(N,N)
    Il = complex(rand(N÷2+1,N))
    
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

# make sure we can take type-stable gradients of scalar functions of our Fields
# (like the posterior)
@testset "Autodiff" begin

    f = FlatMap(rand(2,2))
    D = Diagonal(f)

    # the very basics
    grad1() = gradient(function (θ)
        g = θ * f
        dot(g,g)
    end, 1)[1]
    @test @inferred(grad1()) ≈ 2*norm(f,2)^2

    grad2() = gradient(function (θ)
        g = (θ * D * f)
        dot(g,g)
    end, 1)[1]
    @test                  grad2()  ≈ 2*norm(f.^2,2)^2
    @test_broken @inferred(grad2()) ≈ 2*norm(f.^2,2)^2 # would be nice to get this inferred
    
    # derivatives through ParamDependentOps 
    Dr = ParamDependentOp((;r=1)-> r * D)
    grad3() = gradient(function (r)
        g = (Dr(r=r) * f)
        dot(g,g)
    end,1)[1]
    @test                  grad3()  ≈ 2*norm(f.^2,2)^2
    @test_broken @inferred(grad3()) ≈ 2*norm(f.^2,2)^2 # would be nice to get this inferred
    
    @test @inferred(gradient(r->logdet(Dr(r=r)), 3)[1]) ≈ 4/3
    
    
end

##

@testset "Lensing" begin
    
    local f,ϕ
    
    Cℓ = camb().unlensed_total
    nside = 128
    
    for T in (Float32, Float64)
        
        @testset "T :: $T" begin
                
            ϵ = sqrt(eps(T))
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
            @test (FΦTuple(δf,δϕ)'*(δf̃ϕ_δfϕ(Lϕ,Lϕ*f,f)'*FΦTuple(f,ϕ))) ≈ (f'*((LenseFlow(ϕ+ϵ*δϕ)*(f+ϵ*δf))-(LenseFlow(ϕ-ϵ*δϕ)*(f-ϵ*δf)))/(2ϵ)) rtol=1e-2

            # S2 lensing
            Cf = Cℓ_to_Cov(Flat(Nside=nside), T, S2, Cℓ.EE, Cℓ.BB)
            @test (f = @inferred simulate(Cf)) isa FlatS2
            @test (@inferred Lϕ*f) isa FlatS2
            # adjoints
            f,g = simulate(Cf),simulate(Cf)
            @test f' * (Lϕ * g) ≈ (f' * Lϕ) * g
            # gradients
            δf, δϕ = simulate(Cf), simulate(Cϕ)
            @test (FΦTuple(δf,δϕ)'*(δf̃ϕ_δfϕ(Lϕ,Lϕ*f,f)'*FΦTuple(f,ϕ))) ≈ (f'*((LenseFlow(ϕ+ϵ*δϕ)*(f+ϵ*δf))-(LenseFlow(ϕ-ϵ*δϕ)*(f-ϵ*δf)))/(2ϵ)) rtol=1e-2
        
        end
        
    end
    
end

##

@testset "Posterior" begin
    
    Cℓ = camb()
    L = LenseFlow{RK4Solver{7}}
    T = Float64
    
    for use in (:I,:P)
        
        @testset "use = $use" begin
            
            @unpack f,f̃,ϕ,ds,ds₀ = load_sim_dataset(
                seed  = 0,
                Cℓ    = Cℓ,
                θpix  = 3,
                Nside = 128,
                T     = T,
                beamFWHM = 3,
                use   = use,
                L     = L,
                pixel_mask_kwargs = (edge_padding_deg=2,)
                );
            @unpack Cf,Cϕ,D = ds₀
            f° = L(ϕ)*D*f

            @test lnP(0,f,ϕ,ds) ≈ lnP(1,    f̃,  ϕ ,ds) rtol=1e-4
            @test lnP(0,f,ϕ,ds) ≈ lnP(:mix, f°, ϕ, ds) rtol=1e-4

            ε = sqrt(eps(T))
            δf,δϕ = simulate(Cf),simulate(Cϕ)

            @test δlnP_δfϕₜ(0,f,ϕ,ds)'*FΦTuple(δf,δϕ)     ≈ (lnP(0,f+ε*δf,ϕ+ε*δϕ,ds)-lnP(0,f-ε*δf,ϕ-ε*δϕ,ds))/(2ε)          rtol=1e-2
            @test δlnP_δfϕₜ(1,f̃,ϕ,ds)'*FΦTuple(δf,δϕ)     ≈ (lnP(1,f̃+ε*δf,ϕ+ε*δϕ,ds)-lnP(1,f̃-ε*δf,ϕ-ε*δϕ,ds))/(2ε)          rtol=1e-1
            @test δlnP_δfϕₜ(:mix,f°,ϕ,ds)'*FΦTuple(δf,δϕ) ≈ (lnP(:mix,f°+ε*δf,ϕ+ε*δϕ,ds)-lnP(:mix,f°-ε*δf,ϕ-ε*δϕ,ds))/(2ε)  rtol=1e-2

        end
        
    end
    
end

##

end
