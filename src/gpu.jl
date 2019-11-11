using .CuArrays

preprocess(dest::F, bc::Broadcasted) where {F<:FlatS0{<:Any,<:Any,<:CuArray}} = Broadcasted{Nothing}(bc.f, preprocess_args(dest, bc.args), map(OneTo,size_2d(F)))
preprocess(dest::F, arg) where {F<:FlatS0{<:Any,<:Any,<:CuArray}} = broadcastable(F, arg)
function copyto!(dest::F, bc::Broadcasted{Nothing}) where {F<:FlatS0{<:Any,<:Any,<:CuArray}}
    bc′ = preprocess(dest, bc)
    copyto!(firstfield(dest), bc′)
    return dest
end

function pinv(D::Diagonal{T,<:FlatS0{<:Any,<:Any,<:CuArray}}) where {T}
    Diagonal(@. ifelse(isfinite(inv(D.diag)), inv(D.diag), $zero(T)))
end

copy(f::F) where {F<:FlatS0} = F(copy(firstfield(f)))
