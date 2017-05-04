
export LenseFlow

abstract type ODESolver end

struct LenseFlow{I<:ODESolver,t1,t2,F<:Field} <: LenseOp
    ϕ::F
    ∇ϕ::SVector{2,F}
    Hϕ::SMatrix{2,2,F,4}
end


@∷ function LenseFlow(ϕ::Field{∷,<:S0}, ::Type{I}=ode4{10}, t1=0., t2=1.) where {I<:ODESolver}
    ∇ϕ = ∇*ϕ
    ϕ = Map(ϕ)
    LenseFlow{I,t1,t2,typeof(ϕ)}(ϕ, ∇ϕ, ∇⨳(∇ϕ'))
end
LenseFlow{I<:ODESolver}(ϕ) = LenseFlow(ϕ,I)

# the ODE solvers
abstract type ode45{reltol,abstol,maxsteps,debug} <: ODESolver  end
abstract type ode4{nsteps} <: ODESolver  end
kwargs{ϵr,ϵa,N,dbg}(::Type{ode45{ϵr,ϵa,N,dbg}}) = Dict(:norm=>pixstd, :reltol=>ϵr, :abstol=>ϵa, :minstep=>1/N, :points=>((dbg[1] || dbg[2]) ? :all : :specified))
kwargs(::Type{<:ode4}) = Dict()
run_ode(::Type{<:ode45}) = ODE.ode45
run_ode(::Type{<:ode4}) = ODE.ode4
dbg(::Type{ode45{ϵr,ϵa,N,d}}) where {ϵr,ϵa,N,d} = d
dbg(::Type{<:ode4}) = (false,false)
tts(::Type{ode4{N}},ts) where {N} = linspace(ts...,N)
tts(::Type{<:ode45},ts) = ts


""" ODE velocity for LenseFlow """
velocity(L::LenseFlow, f::Field, t::Real) = @⨳ L.∇ϕ' ⨳ inv(𝕀 + t*L.Hϕ) ⨳ $Ł(∇*f)

function lenseflow(L::LenseFlow{I}, f::F, ts) where {I,F<:Field}
    ys = run_ode(I)((t,y)->F(velocity(L,y,t)), f, tts(I,ts); kwargs(I)...)
    dbg(I)[1] && info("lenseflow: ode45 took $(length(ys[2])) steps")
    dbg(I)[2] ? ys : ys[2][end]::F # <-- ODE.jl not type stable
end


@∷ _getindex(L::LenseFlow{I,∷,∷,F}, ::→{t1,t2}) where {I,t1,t2,F} = LenseFlow{I,t1,t2,F}(L.ϕ,L.∇ϕ,L.Hϕ)
@∷ *(L::LenseFlow{∷,t1,t2}, f::Field) where {t1,t2} = lenseflow(L,Ð(f),Float32[t1,t2])
@∷ \(L::LenseFlow{∷,t1,t2}, f::Field) where {t1,t2} = lenseflow(L,Ð(f),Float32[t2,t1])



## LenseFlow Jacobian operators

*(J::δfϕₛ_δfϕₜ{s,t,<:LenseFlow}, fϕ::FΦTuple) where {s,t} = δfϕₛ_δfϕₜ(J.L,Ł(J.fₜ),Ł(fϕ)...,s,t)
*(fϕ::FΦTuple, J::δfϕₛ_δfϕₜ{s,t,<:LenseFlow}) where {s,t} = δfϕₛ_δfϕₜᴴ(J.L,Ł(J.fₛ),Ł(fϕ)...,s,t)


## Jacobian

""" (δfϕₛ(fₜ,ϕ)/δfϕₜ) * (δf,δϕ) """
function δfϕₛ_δfϕₜ(L::LenseFlow{I}, fₜ::Ff, δf::Fδf, δϕ::Fδϕ, s::Real, t::Real) where {I,Ff<:Field,Fδf<:Field,Fδϕ<:Field}
    Fy = Field2Tuple{Ff,Fδf}
    ∇δϕ,Hδϕ = Ł.(gradhess(δϕ))
    ys = run_ode(I)(
        (t,y)->Fy(δvelocity(L,y...,δϕ,t,∇δϕ,Hδϕ)),
        FieldTuple(fₜ,δf), tts(I,Float32[t,s]);
        kwargs(I)...)
    dbg(I)[1] && info("δfϕₛ_δfϕₜ: ode45 took $(length(ys[2])) steps")
    dbg(I)[2] ? ys : FieldTuple(ys[2][end][2]::Fδf,δϕ)
end

""" ODE velocity for the Jacobian flow """
function δvelocity(L::LenseFlow, f::Field, δf::Field, δϕ::Field, t::Real, ∇δϕ, Hδϕ)

    @unpack ∇ϕ,Hϕ = L
    M⁻¹ = Ł(inv(𝕀 + t*Hϕ))
    ∇f  = Ł(∇*f)
    ∇δf = Ł(∇*δf)

    f′  = @⨳ ∇ϕ' ⨳ M⁻¹ ⨳ ∇f
    δf′ = (∇ϕ' ⨳ M⁻¹ ⨳ ∇δf) + (∇δϕ' ⨳ M⁻¹ ⨳ ∇f) - t*(∇ϕ' ⨳ M⁻¹ ⨳ Hδϕ ⨳ M⁻¹ ⨳ ∇f)

    FieldTuple(f′, δf′)

end


## transpose Jacobian

""" Compute (δfϕₛ(fₛ,ϕ)/δfϕₜ)' * (δf,δϕ) """
function δfϕₛ_δfϕₜᴴ(L::LenseFlow{I}, fₛ::Ff, δf::Fδf, δϕ::Fδϕ, s::Real, t::Real) where {I,Ff<:Field,Fδf<:Field,Fδϕ<:Field}
    # this specifies the basis in which we do the ODE, which is taken to be the
    # basis in which the fields come into this function
    Fy = Field3Tuple{Ff,Fδf,Fδϕ}
    # now run negative transpose perturbed lense flow backwards
    ys = run_ode(I)(
        (t,y)->Fy(negδvelocityᵀ(L,y...,t)),
        FieldTuple(fₛ,δf,δϕ), tts(I,Float32[s,t]);
        kwargs(I)...)
    dbg(I)[1] && info("δfϕₛ_δfϕₜᴴ: ode45 took $(length(ys[2])) steps")
    dbg(I)[2] ? ys : FieldTuple(ys[2][end][2:3]...) :: Field2Tuple{Fδf,Fδϕ}
end


""" ODE velocity for the negative transpose Jacobian flow """
function negδvelocityᵀ(L::LenseFlow, f::Field, δf::Field, δϕ::Field, t::Real)

    Łδf        = Ł(δf)
    M⁻¹        = Ł(inv(𝕀 + t*L.Hϕ))
    ∇f         = Ł(∇*f)
    M⁻¹_δfᵀ_∇f = Ł(M⁻¹ ⨳ (Łδf'*∇f))
    M⁻¹_∇ϕ     = Ł(M⁻¹ ⨳ L.∇ϕ)
    
    f′  = @⨳ L.∇ϕ' ⨳ M⁻¹ ⨳ ∇f
    δf′ = @⨳ ∇ᵀ ⨳ $Ð(Łδf*M⁻¹_∇ϕ)
    δϕ′ = @⨳ ∇ᵀ ⨳ $Ð(M⁻¹_δfᵀ_∇f) + t*(∇ᵀ ⨳ ((∇ᵀ ⨳ $Ð(M⁻¹_∇ϕ ⨳ M⁻¹_δfᵀ_∇f'))'))
    
    FieldTuple(f′, δf′, δϕ′)

end
