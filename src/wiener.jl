# Generic Wiener filter
struct WienerFilter{tol,TS<:LinOp,TN<:LinOp} <: LinOp{Pix,Spin,Basis}
    S::TS
    N::TN
end
const 𝕎 = WienerFilter
𝕎{TS,TN}(S::TS,N::TN,tol=1e-3) = 𝕎{tol,TS,TN}(S,N)
function *{tol}(w::𝕎{tol}, d::Field)
    A = w.S^-1+w.N^-1
    if isa(A,LinDiagOp)  
        # if S & N are diagonal in the same basis they can be added/inverted directly
        A^-1 * w.N^-1 * d
    else
        # otherwise solve using conjugate gradient
        swf, hist = cg(A[~d], (w.N^-1*d)[:], tol=tol, log=true)
        hist.isconverged ? swf[~d] : error("Conjugate gradient solution of Wiener filter did not converge.")
    end
end
