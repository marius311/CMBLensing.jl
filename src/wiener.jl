export 𝕎

# Generic Wiener filter
struct WienerFilter{tol,TS<:LinOp,TN<:LinOp} <: LinOp{Pix,Spin,Basis}
    S::TS
    N::TN
end
const 𝕎 = WienerFilter

@∷ 𝕎(S::LinDiagOp{∷,∷,B},N::LinDiagOp{∷,∷,B}) where {B} = @. nan2zero(S*(S+N)^-1)

𝕎(S::TS,N::TN,tol=1e-3) where {TS,TN} = 𝕎{tol,TS,TN}(S,N)

# otherwise, we solve using conjugate gradient
function *(w::𝕎{tol}, d::Field) where {tol}
    swf, hist = cg(FuncOp(d->(w.S\d+w.N\d))[~d], (w.N\d)[:], tol=tol, log=true)
    hist.isconverged ? swf[~d] : error("Conjugate gradient solution of Wiener filter did not converge.")
end
