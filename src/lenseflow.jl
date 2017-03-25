using ODE

export LenseFlowOp, LenseBasis, δlenseflow

abstract type ODESolver end
abstract type ode45{reltol,abstol,maxsteps,debug} <: ODESolver  end
abstract type ode4{nsteps} <: ODESolver  end


struct LenseFlowOp{I<:ODESolver,F<:Field} <: LenseOp
    ϕ::F
    ∇ϕ::SVector{2,F}
    Jϕ::SMatrix{2,2,F,4}
end

function LenseFlowOp{I<:ODESolver}(ϕ::Field{<:Pix,<:S0,<:Basis}, ::Type{I}=ode45{1e-3,1e-3,100,false})
    ∇ϕ = ∇*ϕ
    ϕ = Map(ϕ)
    LenseFlowOp{I,typeof(ϕ)}(ϕ, ∇ϕ, ∇⨳(∇ϕ'))
end

function kwargs(::Type{ode45{ϵr,ϵa,N,dbg}}) where {ϵr,ϵa,N,dbg}
    Dict(:reltol=>ϵr, :abstol=>ϵa, :minstep=>1/N, :points=>(dbg ? :all : :specified))
end
dbg(::Type{ode45{ϵr,ϵa,N,d}}) where {ϵr,ϵa,N,d} = d

# the LenseFlow algorithm 
velocity(L::LenseFlowOp, f::Field, t::Real) = @⨳ L.∇ϕ' ⨳ inv(𝕀 + t*L.Jϕ) ⨳ $Ł(∇*f)

function lenseflow(L::LenseFlowOp{I}, f::F, ts) where {I,F<:Field}
    ys = ODE.ode45((t,y)->F(velocity(L,y[~f],t))[:], f[:], ts; kwargs(I)...)
    if dbg(I)
        info("lenseflow: ode45 took $(length(ys[2])) steps")
        ys
    else
        ys[2][end][~f]::F # <-- ODE.jl not type stable
    end
end

function lenseflow(L::LenseFlowOp{ode4{N}}, f::F, ts) where {N,F<:Field}
    ODE.ode4((t,y)->F(velocity(L,y[~f],t))[:], f[:], linspace(ts...,N))[2][end][~f]::F
end


*(L::LenseFlowOp, f::Field) = lenseflow(L,Ð(f),[0.,1])
\(L::LenseFlowOp, f::Field) = lenseflow(L,Ð(f),[1.,0])

# transpose lenseflow

*(J::δf̃_δfϕᵀ{<:LenseFlowOp}, δPδf̃::Field) = δf̃_δfϕᵀ(J.L,Ł(J.f),Ł(δPδf̃))

""" Compute [(δf̃(f)/δf)ᵀ * δP/δf̃, (δf̃(f)/δϕ)ᵀ * δP/δf̃] """
function δf̃_δfϕᵀ(L::LenseFlowOp{I,F}, f::F1, δPδf̃::F2, δLδϕ::F3=zero(F)) where {I,F,F1<:Field,F2<:Field,F3<:Field}
    
    # first get lensed field at t=1
    f̃ = F1(L*f)
    
    # now run negative transpose perturbed lense flow backwards
    Fs = Tuple{F1,F2,F3}
    ys = ODE.ode45(
        (t,y)->((Fs(δvelocityᵀ(L,y[Fs]...,t)))[:]), 
        [f̃,δPδf̃,δLδϕ][:], [1.,0]; 
        kwargs(I)...)
        
    if dbg(I)
        info("δf̃_δfϕᵀ: ode45 took $(length(ys[2])) steps")
        ys
    else
        ys[2][end][Fs][2:3] :: Tuple{F2,F3} # <-- tuple indexing with UnitRange not type stable (yet?)
    end
end

# function dLdf_dfdf̃ϕ{reltol,abstol,maxsteps,F}(L::LenseFlowOp{ode45{reltol,abstol,maxsteps},F}, f::Field, dLdf::Field, δPδϕ::F=zero(F); debug=false)
#     
#     # now run negative transpose perturbed lense flow forwards
#     ys = ODE.ode45(
#         (t,y)->δvelocityᵀ(L,y[~(f,dLdf,δPδϕ)]...,t)[:], 
#         [f,dLdf,δPδϕ][:], [0.,1]; 
#         reltol=reltol, abstol=abstol, points=:all, minstep=1/maxsteps)
#         
#     if debug
#         info("dLdf_dfdf̃ϕ: ode45 took $(length(ys)) steps")
#         ys
#     else:
#         ys[2][end][~(f,dLdf,δPδϕ)][2:3]
#     end
#     
# end

function δvelocityᵀ(L::LenseFlowOp, f::Field, δPδf̃::Field, δPδϕ::Field, t::Real)
    
    ŁδPδf̃       = Ł(δPδf̃)
    iM          = Ł(inv(𝕀 + t*L.Jϕ))
    ∇f          = Ł(∇*f)
    iM_δPδf̃ᵀ_∇f = Ł(iM ⨳ (ŁδPδf̃'*∇f))
    iM_∇ϕ       = Ł(iM ⨳ L.∇ϕ)
    
    f′    = @⨳ L.∇ϕ' ⨳ iM ⨳ ∇f
    δPδf̃′ = @⨳ ∇ᵀ ⨳ $Ð(ŁδPδf̃*iM_∇ϕ)
    δPδϕ′ = @⨳ ∇ᵀ ⨳ $Ð(iM_δPδf̃ᵀ_∇f) + t*(∇ᵀ ⨳ ((∇ᵀ ⨳ $Ð(iM_∇ϕ ⨳ iM_δPδf̃ᵀ_∇f'))'))
    
    (f′, δPδf̃′, δPδϕ′)

end
