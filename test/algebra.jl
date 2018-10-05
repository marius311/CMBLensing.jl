include("general.jl")

using CMBLensing: LP, SymmetricFuncOp, FlatS02, FlatTEBCov, LazyBinaryOp

##
N = 4

f0   = FlatS0Map(rand(N,N))
f2   = FlatS2QUMap(rand(N,N),rand(N,N))
f02  = FieldTuple(f0,f2)
f220 = FieldTuple(f2,f2,f0)
fn   = FieldTuple(f02,f02)
LTEB = FlatTEBCov{Float64,Flat{1,N}}(rand(N÷2+1,N),zeros(N÷2+1,N),rand(N÷2+1,N),rand(N÷2+1,N))
##

@testset "Basic Algebra" begin
    
    for f in (f0,)
        
        F = typeof(f)
        L = FullDiagOp(f)
        
        @testset "f::$(shortname(typeof(f)))" begin
        
            # basic conversion / construction
            @test_noerr @inferred F(f)
            @test F(f)==f
            
            # field algebra
            @test_noerr @inferred f+1
            @test_noerr @inferred 2*f
            @test_noerr @inferred f+f
            @test_noerr @inferred f*f
            
            # field basis conversion
            @test_noerr @inferred f+Ð(f)
            @test_noerr @inferred f+Ł(f)
            
            # type-stable operators on fields
            Ls = Any[FullDiagOp(f), FullDiagOp(Ð(f)), ∇₁, LP(500)]
            isa(f,FlatS02) && push!(Ls, LTEB)
            for L in Ls
                @testset "L::$(shortname(typeof(L)))" begin
                    @test_noerr @inferred L*f
                    @test_noerr @inferred L\f
                    if L!=∇₁
                        @test_noerr @inferred f'*L
                        @test_noerr @inferred L'*f
                        @test_noerr @inferred L'\f
                    end
                end
            end
            
            # not-type-stable operators on fields
            for L in (SymmetricFuncOp(op=(x->2x), op⁻¹=(x->x/2)),)
                @test_noerr L*f
                @test_noerr L\f
                @test_noerr f*L
                @test_noerr L'*f
                @test_noerr L'\f
            end
            
            # stuff with ∇
            @test eltype(∇) == eltype(∇')
            @test_noerr @inferred ∇ * f
            @test_noerr @inferred ∇' * (∇*f)
            @test_noerr @inferred gradhess(f)
            
            v = @SVector[f,f]
            M = @SMatrix[f f; f f]
            @test_noerr @inferred ∇' * M
            @test_noerr @inferred v' * M
            @test_noerr @inferred M * v
            
            
            # FullDiagOp explicit and lazy operations
            @test_noerr @inferred(FullDiagOp(Ł(f)) + FullDiagOp(Ł(f)))::FullDiagOp
            @test_noerr @inferred(2*FullDiagOp(Ł(f)))::FullDiagOp
            @test_noerr @inferred(FullDiagOp(Ł(f)) + FullDiagOp(Ð(f)))::LazyBinaryOp

            @test_noerr @inferred(f⋅f)::Real
            @test_noerr @inferred(Ac_mul_B(f,f))::FlatS0Map
            
            # broadcasting
            @test_noerr (@. f = 2f + 3f)
        end
        
    end
    
    # @testset "LTEB" begin
    #     @test_noerr @inferred(LTEB * LTEB)::FlatTEBCov
    #     @test_noerr @inferred(LTEB + 1I)::FlatTEBCov
    #     @test_noerr @inferred(2 * LTEB)::FlatTEBCov
    #     @test_noerr @inferred simulate(LTEB)
    #     @test_noerr @inferred LTEB \ f02
    # end
    
    # @testset "S0/S2" begin
    #     @test_noerr f0*f2
    #     @test_noerr f0*f02
    # end
end
