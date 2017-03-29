push!(LOAD_PATH, pwd()*"/src")
using Base.Test
using CMBFields
using MacroTools

# checks on error thrown and is type-stable
macro mytest(ex) :(@test (@inferred ($(esc(ex))); true)) end
macro mytest2(ex) :(@test ($(esc(ex)); true)) end
macro mytest(F,ex) :(@test (((@inferred ($(esc(ex))))::$(esc(F))); true)) end
    
    
f0 = FlatS0Map(rand(4,4))
f2 = FlatS2QUMap(rand(4,4),rand(4,4))
f02 = FieldTuple(f0,f2)
f220 = FieldTuple(f2,f2,f0)
fn = FieldTuple(f02,f02)

@testset "Basic Algebra" begin
    for f in (f0,f2,f02,f220,fn)
        F = typeof(f)
        @testset "$(shortname(typeof(f)))" begin
            @mytest F f+1
            @mytest F 2*f
            @mytest F f+f
            @mytest F f*f
            @mytest F FullDiagOp(f)*f
            @mytest F FullDiagOp(f)*Ð(f)
            @mytest F FullDiagOp(f)*Ł(f)
            
            @mytest f+Ð(f)
            @mytest f+Ł(f)
            @mytest ∂x*f
            @mytest ∂x*FullDiagOp(f)*f
            @mytest ∂x.*FullDiagOp(Ð(f)).*Ð(f)
            @mytest simulate(FullDiagOp(f))
            
            @mytest f⋅f
            @mytest f'*f
            
            # @mytest2 (@. f = 2f + 3f)
            # @mytest2 (@. f = 2*∂x*f + 3*∂y*f)
        end
    end
    @testset "S0/S2" begin
        @mytest f0*f2
        @mytest f0*f02
    end
end
