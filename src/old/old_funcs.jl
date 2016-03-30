################################################
#   
# Code for computing the variance and normalizing constants for the quadratic delenser
#
################################################

"""
Aell_fun computes the normalizing constant when the weights factor
"""
function Aell_fun(amat, bmat, cmat, cPhatPhat, cPhatP, cPP)
    τsq     = sum( (k1.^2) .* cPP .* dk ./ ((2π)^d) )
    kdic    = [1 => k1.*k1, 2 => k1.*k2, 3 => k2.*k2]
    tmpAx   = Array(Complex{Float64}, size(amat))
    preApqk = squash!(abs2(cPhatP) ./ cmat ./ cPhatPhat)
    tmpBx   = ifft2(squash!(1 ./ amat),deltk)
    rtn     = zeros(Complex{Float64}, size(amat))
    for cntr = 1:3
        tmpAx[:,:]  = ifft2(kdic[cntr] .* preApqk, deltk)
        rtn[:,:]   += (cntr == 2?2:1) .* kdic[cntr] .* fft2(tmpAx .* conj(tmpBx), deltx)
    end
    rtn[:,:] .*= exp(- abs2(magk) * τsq)
    rtn[:,:] ./= bmat 
    rtn[:,:] ./= (2π)^(d/2)
    return  squash!(1 ./ real(rtn)) 
end



"""
Aell_optim(ℓ_ind_1, ℓ_ind_2; mode = :B) computes the optimal weighting normalizing 
factor. This call signatures uses a slow loop. Mostly used to check correctness of faster
methods.
"""
function Aell_optim(ℓ_ind_1, ℓ_ind_2; mode = :B)  # only works for optimal choice of w
    τsq     = sum( (k1.^2) .* cPP .* dk ./ ((2π)^d) )
    ℓ1, ℓ2  = k1[ℓ_ind_1, ℓ_ind_2],  k2[ℓ_ind_1, ℓ_ind_2]
    cos2kl  = cos(φ2_l - 2*angle(ℓ1+im*ℓ2)).^2
    sin2kl  = sin(φ2_l - 2*angle(ℓ1+im*ℓ2)).^2
    trm1    = abs2(ℓ1*(k1+ℓ1) + ℓ2*(k2+ℓ2))
    trm1  .*= exp(-(ℓ1^2 + ℓ2^2) * τsq)
    if mode == :B
        trm1  ./= cos2kl .* cBBobs + sin2kl .* cEEobs
    elseif mode == :E
        trm1  ./= sin2kl .* cBBobs + cos2kl .* cEEobs
    else
        error("argument mode needs to either be :B or :E")
    end
    trm2    = abs2(cPhatP)./cPhatPhat
    trm1[(magk .> lmax) | isnan(trm1) | (abs(trm1) .== Inf)] = 0.0
    trm2[(magk .> lmax) | isnan(trm2) | (abs(trm2) .== Inf)] = 0.0
    invAell = (1/(2π)^(d/2)) * real(fft2( ifft2(trm2) .* conj(ifft2(trm1)) )[ℓ_ind_1, ℓ_ind_2])
    return 1 / invAell
end


"""
Aell_optim(; mode = :B) computes the optimal weighting normalizing 
factor by taking advantage of a rotational symmetry.
"""
function Aell_optim(cEEobs, cBBobs, cPhatPhat, cPhatP, cPP; mode = :B) # this interpolates Aell when l = (l1,0) along the axis 
    cos2kl  =  cos(φ2_l).^2
    sin2kl  =  sin(φ2_l).^2
    if mode == :B
        amat = (cos2kl .* cBBobs + sin2kl .* cEEobs) ./ (magk .< lmax)
        # amat = abs2(cos(φ2_l) .* conj(Bobs) + sin(φ2_l) .* conj(Eobs)) ./ (magk .< lmax)
    elseif mode == :E
        amat = (sin2kl .* cBBobs + cos2kl .* cEEobs) ./ (magk .< lmax)
        # amat = abs2(-sin(φ2_l) .* conj(Bobs) + cos(φ2_l) .* conj(Eobs)) ./ (magk .< lmax)
    else
        error("argument mode needs to either be :B or :E")
    end
    cmat    =  ones(size(cBBobs)) ./ (magk .< lmax)
    bmat    =  ones(size(cBBobs))
    Aell    =  Aell_fun(amat, bmat, cmat, cPhatPhat, cPhatP, cPP)
    ellvec  =  fftshift(k2[:,1]) # the fftshift is to make the grid montonic.
    Aellvec =  fftshift(Aell[:,1]) 
    return linear_interp1(ellvec, Aellvec, magk)
end



"""
N0_fun computes the spectral density of the QE delenser when the weights factor.
"""
function N0_fun(amat, bmat, cmat, cEEobs, cBBobs, cPhatPhat, cPhatP, cPP; mode = :B)
    τsq       = sum( (k1.^2) .* cPP .* dk ./ ((2π)^d) )
    kdic      = [1 => k1 .* k1, 2 => k1 .* k2, 3 => k2 .* k2]
    preApqk   = squash!(abs2(cPhatP) ./ abs2(cmat) ./ cPhatPhat)
    Bx_term1  = ifft2(squash!((cEEobs + cBBobs) ./ abs2(amat)), deltk) 
    if mode == :B
        Bx_term23 = ifft2(squash!(exp(im * 2 * φ2_l) .* (cBBobs - cEEobs) ./ abs2(amat)), deltk)
    elseif mode == :E
        Bx_term23 = ifft2(squash!(exp(im * 2 * φ2_l) .* (-cBBobs + cEEobs) ./ abs2(amat)), deltk) 
    end
    tmpAx     = Array(Complex{Float64}, size(amat))
    tmpk      = Array(Complex{Float64}, size(amat))
    rtn       = zeros(Complex{Float64}, size(amat))
    for cntr = 1:3
        rtn[:,:]    += (cntr == 2?2:1) .* kdic[cntr] .* fft2(tmpAx .* conj(Bx_term1), deltx)
        tmpAx[:,:]   = ifft2(kdic[cntr] .* preApqk, deltk)
        tmpk[:,:]    = fft2(tmpAx .* conj(Bx_term23), deltx)
        rtn[:,:]    += (cntr == 2?2:1) .* kdic[cntr] .* cos(2 * φ2_l) .* real(tmpk)
        rtn[:,:]    += (cntr == 2?2:1) .* kdic[cntr] .* sin(2 * φ2_l) .* imag(tmpk)
    end
    rtn[:,:] .*= abs2(Aell_fun(amat, bmat, cmat, cPhatPhat, cPhatP, cPP))
    rtn[:,:] .*= exp(- abs2(magk) * τsq)
    rtn[:,:] ./= abs2(bmat) 
    rtn[:,:] .*= 0.5 / (2π)^(d/2)
    return  real(rtn) 
end


"""
N0_fun computes the "observed" spectral density of the QE delenser when the weights factor.
"""
function N0_obs_fun(amat, bmat, cmat, Eobs, Bobs, hatϕ, cPhatPhat, cPhatP, cPP; mode = :B)
    τsq       = sum( (k1.^2) .* cPP .* dk ./ ((2π)^d) )
    kdic      = [1 => k1 .* k1, 2 => k1 .* k2, 3 => k2 .* k2]

    tmpk      = squash!(abs2(cPhatP) ./ cPhatPhat ./ abs2(cmat))
    # tmpk      = squash!(abs2(hatϕ*deltk) .* abs2(cPhatP) ./ abs2(cPhatPhat) ./ abs2(cmat))
    Axdic = [1 => ifft2(kdic[1] .* tmpk, deltk), 
             2 => ifft2(kdic[2] .* tmpk, deltk), 
             3 => ifft2(kdic[3] .* tmpk, deltk)]
    
    if mode == :B
        cosBdk   = cos(φ2_l) .* conj(Bobs)
        cosBdk  += sin(φ2_l) .* conj(Eobs)
        cosBdk .*= deltk
        scEBdk   = sin(φ2_l) .* conj(Bobs)
        scEBdk  -= cos(φ2_l) .* conj(Eobs)
        scEBdk .*= deltk
    elseif mode == :E
        cosBdk   = cos(φ2_l) .* conj(Eobs)
        cosBdk  -= sin(φ2_l) .* conj(Bobs)
        cosBdk .*= deltk
        scEBdk   = sin(φ2_l) .* conj(Eobs)
        scEBdk  += cos(φ2_l) .* conj(Bobs)
        scEBdk .*= deltk
    end
    Bx1   = [:inside => ifft2(squash!( cosBdk .* conj(cosBdk) ./ abs2(amat)), deltk),
             :outsde => cos(φ2_l) .* cos(φ2_l) ]
    Bx2   = [:inside => ifft2(squash!( scEBdk .* conj(cosBdk) ./ abs2(amat)), deltk),
             :outsde => sin(φ2_l) .* cos(φ2_l) ]  
    Bx3   = [:inside => ifft2(squash!( cosBdk .* conj(scEBdk) ./ abs2(amat)), deltk),
             :outsde => sin(φ2_l) .* cos(φ2_l) ]  
    Bx4   = [:inside => ifft2(squash!( scEBdk .* conj(scEBdk) ./ abs2(amat)), deltk),
             :outsde => sin(φ2_l) .* sin(φ2_l) ]  

    rtn = zeros(Complex{Float64}, size(amat))
    for ct = 1:3
        rtn[:,:] += (ct==2?2:1).*kdic[ct].*Bx1[:outsde] .* fft2(Axdic[ct].*conj(Bx1[:inside]), deltx)
        rtn[:,:] += (ct==2?2:1).*kdic[ct].*Bx2[:outsde] .* fft2(Axdic[ct].*conj(Bx2[:inside]), deltx)
        rtn[:,:] += (ct==2?2:1).*kdic[ct].*Bx3[:outsde] .* fft2(Axdic[ct].*conj(Bx3[:inside]), deltx)
        rtn[:,:] += (ct==2?2:1).*kdic[ct].*Bx4[:outsde] .* fft2(Axdic[ct].*conj(Bx4[:inside]), deltx)
    end
    rtn[:,:] .*= abs2(Aell_fun(amat, bmat, cmat, cPhatPhat, cPhatP, cPP))
    rtn[:,:] .*= exp(- abs2(magk) * τsq)
    rtn[:,:] ./= abs2(bmat) 
    rtn[:,:] ./= (2π)^(d/2)
    return  real(rtn) 
end




"""
hatEB computes the quadratic delenser
"""
function hatEB(Eobs, Bobs, hatϕ, amat, bmat, cmat, cPhatPhat, cPhatP, cPP; mode = :B)
    τsq      = sum( (k1.^2) .* cPP .* dk ./ ((2π)^d) )
    kdic     = [1 => k1, 2 => k2]
    Axdic    = [1 => ifft2(squash!(im .* k1 .* hatϕ .* cPhatP ./ cPhatPhat ./ cmat), deltk),
                2 => ifft2(squash!(im .* k2 .* hatϕ .* cPhatP ./ cPhatPhat ./ cmat), deltk)]
    if mode == :B
        Bx_term1 = ifft2(squash!(Eobs .* sin(φ2_l) ./ amat), deltk) 
        Bx_term2 = ifft2(squash!(Eobs .* cos(φ2_l) ./ amat), deltk) 
        Bx_term3 = ifft2(squash!(Bobs .* cos(φ2_l) ./ amat), deltk) 
        Bx_term4 = ifft2(squash!(Bobs .* sin(φ2_l) ./ amat), deltk) 
        mult1    = - im .* cos(φ2_l) ./ bmat
        mult2    =   im .* sin(φ2_l) ./ bmat
        mult3    = - im .* cos(φ2_l) ./ bmat
        mult4    = - im .* sin(φ2_l) ./ bmat
    elseif mode == :E
        Bx_term1 = ifft2(squash!(Eobs .* cos(φ2_l) ./ amat), deltk) 
        Bx_term2 = ifft2(squash!(Eobs .* sin(φ2_l) ./ amat), deltk) 
        Bx_term3 = ifft2(squash!(Bobs .* sin(φ2_l) ./ amat), deltk) 
        Bx_term4 = ifft2(squash!(Bobs .* cos(φ2_l) ./ amat), deltk) 
        mult1    = - im .* cos(φ2_l)
        mult2    = - im .* sin(φ2_l)
        mult3    =   im .* cos(φ2_l)
        mult4    = - im .* sin(φ2_l)
    end
    rtn = zeros(Complex{Float64}, size(amat))
    for q = 1:2
        rtn[:,:]  += kdic[q] .* mult1 .* fft2(Axdic[q] .* conj(Bx_term1), deltx)
        rtn[:,:]  += kdic[q] .* mult2 .* fft2(Axdic[q] .* conj(Bx_term2), deltx)
        rtn[:,:]  += kdic[q] .* mult3 .* fft2(Axdic[q] .* conj(Bx_term3), deltx)
        rtn[:,:]  += kdic[q] .* mult4 .* fft2(Axdic[q] .* conj(Bx_term4), deltx)
    end
    rtn[:,:] .*= Aell_fun(amat, bmat, cmat, cPhatPhat, cPhatP, cPP)
    rtn[:,:] .*= exp(- abs2(magk) * τsq / 2)
    rtn[:,:] ./= bmat 
    return  rtn 
end




####################################
#
# Compute some lensing spectra
#
#######################################




"""
Spectra for First order B lensing contribution from primordial E and B.
"""
function frst_ordr_spec_decomp(cEE, cBB, cPP; mode = :B, src = :B)
    # This generates the spectra for (mode)^{len, src}, i.e. the power of 'src' in first order lensed 'mode'
    kdic    = [1 => k1, 2 => k2]
    esign   = (mode == src) ? 1.0 : -1.0 
    cXX     = (src == :B)   ? cBB : cEE   
    tmpA1x  = Array(Float64, size(cXX))
    tmpA2x  = Array(Float64, size(cXX))
    tmpA3x  = Array(Float64, size(cXX))
    tmpBx   = Array(Float64, size(cXX))
    rtn     = zeros(Complex{Float64}, size(cXX))
    for cntrq = 1:2, cntrp = 1:2
        tmpA1x[:,:]  = ifft2r(kdic[cntrq] .* kdic[cntrp] .* cXX, deltk)
        tmpA2x[:,:]  = ifft2r(kdic[cntrq] .* kdic[cntrp] .* cXX .* cos(2 * φ2_l), deltk)
        tmpA3x[:,:]  = ifft2r(kdic[cntrq] .* kdic[cntrp] .* cXX .* sin(2 * φ2_l), deltk)
        tmpBx[:,:]   = ifft2r(kdic[cntrq] .* kdic[cntrp] .* cPP, deltk)
        rtn[:,:]    += fft2(tmpA1x .* tmpBx, deltx)
        rtn[:,:]    += esign .* fft2(tmpA2x .* tmpBx, deltx) .* cos(2 * φ2_l)
        rtn[:,:]    += esign .* fft2(tmpA3x .* tmpBx, deltx) .* sin(2 * φ2_l)
    end
    rtn[:,:] .*= 0.5 / (2π)^(d/2)
    return  real(rtn) 
end


function clenElenB(cEE, cBB, cPP)
    kdic  = [1 => k1 .* k1, 2 => k1 .* k2, 3 => k2 .* k1, 4 => k2 .* k2]
    Axdic = [1 => ifft2(kdic[1] .* cPP, deltk), 
             2 => ifft2(kdic[2] .* cPP, deltk), 
             3 => ifft2(kdic[3] .* cPP, deltk), 
             4 => ifft2(kdic[4] .* cPP, deltk)]
    Bk1   = [:inside => sin(φ2_l) .* cos(φ2_l) .* (cEE - cBB),
             :outsde => cos(φ2_l).^2 - sin(φ2_l).^2 ]
    Bk2   = [:inside => (sin(φ2_l).^2 - cos(φ2_l).^2) .* (cEE - cBB),
             :outsde => sin(φ2_l) .* cos(φ2_l) ]
    tmpBx1  = Array(Complex{Float64}, size(cEE))
    tmpBx2  = Array(Complex{Float64}, size(cEE))
    rtn = zeros(Complex{Float64}, size(cEE))
    for ct = 1:4
        tmpBx1[:,:]  = ifft2(kdic[ct] .* Bk1[:inside], deltk)
        tmpBx2[:,:]  = ifft2(kdic[ct] .* Bk2[:inside], deltk)
        rtn[:,:] += Bk1[:outsde] .* fft2(Axdic[ct] .* conj(tmpBx1), deltx)
        rtn[:,:] += Bk2[:outsde] .* fft2(Axdic[ct] .* conj(tmpBx2), deltx)
    end         
    rtn[:,:] ./= (2π)^(d/2)
    return  real(rtn) 
end







####################################
#
# Computations related to Blake's methodology
#
#######################################



"""
Spectrum for the error when using Blake's method to estimate first order lense
"""
function blakes_error_spec(cEEobs, cEE, cPhatPhat, cPhatP, cPP)
    kd    = [1 => k1, 2 => k2]
    tmpAa1x  = Array(Float64, size(cEE))
    tmpAa2x  = Array(Float64, size(cEE))
    tmpAa3x  = Array(Float64, size(cEE))
    tmpAb1x  = Array(Float64, size(cEE))
    tmpAb2x  = Array(Float64, size(cEE))
    tmpAb3x  = Array(Float64, size(cEE))
    tmpBax   = Array(Float64, size(cEE))
    tmpBbx   = Array(Float64, size(cEE))
    rtn      = zeros(Complex{Float64}, size(cEE))
    for cq = 1:2, cp = 1:2
        tmpAa1x[:,:]  = ifft2r(kd[cq] .* kd[cp] .* cEE, deltk)
        tmpAa2x[:,:]  = ifft2r(kd[cq] .* kd[cp] .* cos(2 * φ2_l).* cEE , deltk)
        tmpAa3x[:,:]  = ifft2r(kd[cq] .* kd[cp] .* sin(2 * φ2_l).* cEE , deltk)

        tmpAb1x[:,:]  = ifft2r(squash!(kd[cq] .* kd[cp] .* abs2(cEE) ./ cEEobs), deltk)
        tmpAb2x[:,:]  = ifft2r(squash!(kd[cq] .* kd[cp] .* cos(2 * φ2_l) .* abs2(cEE) ./ cEEobs), deltk)
        tmpAb3x[:,:]  = ifft2r(squash!(kd[cq] .* kd[cp] .* sin(2 * φ2_l) .* abs2(cEE) ./ cEEobs), deltk)

        tmpBax[:,:]  = ifft2r(kd[cq] .* kd[cp] .* cPP, deltk)

        tmpBbx[:,:]  = - ifft2r(squash!(kd[cq] .* kd[cp] .* abs2(cPhatP) ./ cPhatPhat), deltk)

        rtn[:,:]    += fft2(tmpAa1x .* tmpBax, deltx)
        rtn[:,:]    += fft2(tmpAb1x .* tmpBbx, deltx)
        rtn[:,:]    -= fft2(tmpAa2x .* tmpBax, deltx) .* cos(2 * φ2_l)
        rtn[:,:]    -= fft2(tmpAb2x .* tmpBbx, deltx) .* cos(2 * φ2_l)
        rtn[:,:]    -= fft2(tmpAa3x .* tmpBax, deltx) .* sin(2 * φ2_l)
        rtn[:,:]    -= fft2(tmpAb3x .* tmpBbx, deltx) .* sin(2 * φ2_l)
    end
    rtn[:,:] .*= 0.5 / (2π)^(d/2)
    return  real(rtn) 
end




"""
frst_ordr_lnBfromE computes Blakes first order lensing of B from E
"""
function frst_ordr_lnBfromE(Efield, ϕfield)
    kdic    = [1 => k1, 2 => k2]
    multc   =   cos(φ2_l) 
    mults   = - sin(φ2_l)     
    rtn     = zeros(Complex{Float64}, size(Efield))
    for q = 1:2
        Ax  = ifft2(im .* kdic[q] .* ϕfield, deltk)
        Bxs = ifft2(im .* kdic[q] .* sin(φ2_l) .* Efield, deltk) |> conj
        Bxc = ifft2(im .* kdic[q] .* cos(φ2_l) .* Efield, deltk) |> conj
        rtn[:,:] += multc .* fft2(Ax .* Bxs, deltx)
        rtn[:,:] += mults .* fft2(Ax .* Bxc, deltx)
    end
    return  rtn 
end
function frst_ordr_lnBfromE(Eobs, ϕobs, cEEobs, cEE, cPhatPhat, cPhatP)
    kdic    = [1 => k1, 2 => k2]
    multc   =   cos(φ2_l) 
    mults   = - sin(φ2_l) 
    ϕfield  = squash!(ϕobs .* cPhatP ./ cPhatPhat)
    Efield  = squash!(Eobs .* cEE ./ cEEobs)
    rtn     = zeros(Complex{Float64}, size(Eobs))
    for q = 1:2
        Ax  = ifft2(im .* kdic[q] .* ϕfield, deltk)
        Bxs = ifft2(im .* kdic[q] .* sin(φ2_l) .* Efield, deltk) |> conj
        Bxc = ifft2(im .* kdic[q] .* cos(φ2_l) .* Efield, deltk) |> conj
        rtn[:,:] += multc .* fft2(Ax .* Bxs, deltx)
        rtn[:,:] += mults .* fft2(Ax .* Bxc, deltx)
    end
    return  rtn 
end








################################################
#   
# The type QUandFriends is a container for the fields and derivatives
#
################################################

immutable QUandFriends
    Qx   ::Array{Float64,2}
    Ux   ::Array{Float64,2}
    ∂1Qx ::Array{Float64,2}
    ∂2Qx ::Array{Float64,2}
    ∂1Ux ::Array{Float64,2}
    ∂2Ux ::Array{Float64,2}
    ∂11Qx::Array{Float64,2}
    ∂12Qx::Array{Float64,2}
    ∂22Qx::Array{Float64,2}
    ∂11Ux::Array{Float64,2}
    ∂12Ux::Array{Float64,2}
    ∂22Ux::Array{Float64,2}
end

immutable TandFriends
    Tx   ::Array{Float64,2}
    ∂1Tx ::Array{Float64,2}
    ∂2Tx ::Array{Float64,2}
    ∂11Tx::Array{Float64,2}
    ∂12Tx::Array{Float64,2}
    ∂22Tx::Array{Float64,2}
end



""" here is the basic constructor """
function QUandFriends(Qx::Array{Float64,2}, Ux::Array{Float64,2})
    Q, U = fft2(Qx, deltx), fft2(Ux, deltx)
    return QUandFriends(
        Qx, 
        Ux,
        ifft2r(im .* k1 .* Q),
        ifft2r(im .* k2 .* Q),
        ifft2r(im .* k1 .* U),
        ifft2r(im .* k2 .* U),
        ifft2r(im .* k1 .* k1 .* Q),
        ifft2r(im .* k1 .* k2 .* Q),
        ifft2r(im .* k2 .* k2 .* Q),
        ifft2r(im .* k1 .* k1 .* U),
        ifft2r(im .* k1 .* k2 .* U),
        ifft2r(im .* k2 .* k2 .* U)
        )
end

function TandFriends(Tx::Array{Float64,2})
    T = fft2(Tx, deltx)
    return TandFriends(
        Tx, 
        ifft2r(im .* k1 .* T),
        ifft2r(im .* k2 .* T),
        ifft2r(im .* k1 .* k1 .* T),
        ifft2r(im .* k1 .* k2 .* T),
        ifft2r(im .* k2 .* k2 .* T)
        )
end


"""  here is the basic constructor """
function QUandFriends(Q::Array{Complex{Float64},2}, U::Array{Complex{Float64},2})
    return QUandFriends(
        ifft2r(Q), 
        ifft2r(U),
        ifft2r(im .* k1 .* Q),
        ifft2r(im .* k2 .* Q),
        ifft2r(im .* k1 .* U),
        ifft2r(im .* k2 .* U),
        ifft2r(im .* k1 .* k1 .* Q),
        ifft2r(im .* k1 .* k2 .* Q),
        ifft2r(im .* k2 .* k2 .* Q),
        ifft2r(im .* k1 .* k1 .* U),
        ifft2r(im .* k1 .* k2 .* U),
        ifft2r(im .* k2 .* k2 .* U)
        )
end

function TandFriends(T::Array{Complex{Float64},2})
    return TandFriends(
        ifft2r(T), 
        ifft2r(im .* k1 .* T),
        ifft2r(im .* k2 .* T),
        ifft2r(im .* k1 .* k1 .* T),
        ifft2r(im .* k1 .* k2 .* T),
        ifft2r(im .* k2 .* k2 .* T),
        )
end




""" another constructor """
function QUandFriends(row, col)
    return QUandFriends(
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col)
        )
end


function TandFriends(row, col)
    return TandFriends(
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col),
        Array(Float64,row, col)
        )
end




################################################
#   
# Lensing and delensing functions
#
################################################

function easylense(QUdata::QUandFriends, ϕ, ψ) 
    # first decompose the lensing
    row, col = size(ϕ)
    displx   = ifft2r(im .* k1 .* ϕ) + ifft2r(im .* k2 .* ψ)
    disply   = ifft2r(im .* k2 .* ϕ) - ifft2r(im .* k1 .* ψ)
    rdisplx  = Array(Float64, row, col)
    rdisply  = Array(Float64, row, col)
    indcol   = Array(Int64, row, col)
    indrow   = Array(Int64, row, col)
    decomplense!(indcol, indrow, rdisplx, rdisply, x, y, displx, disply, deltx)
    
    # now do integer lensing on the taylor terms
    lQUdata = gridlense(QUdata, indcol, indrow)

    # now do sub-grid taylor corrections
    Qϕψx    = Array(Float64, row, col)
    Uϕψx    = Array(Float64, row, col)
    ∂1Qϕψx  = Array(Float64, row, col)
    ∂1Uϕψx  = Array(Float64, row, col)
    ∂2Qϕψx  = Array(Float64, row, col)
    ∂2Uϕψx  = Array(Float64, row, col)
    for j = 1:col
        for i = 1:row
            Qϕψx[i,j]    = lQUdata.Qx[i,j]  
            Qϕψx[i,j]   += (lQUdata.∂1Qx[i,j]  * rdisplx[i,j]) + (lQUdata.∂2Qx[i,j]  * rdisply[i,j])
            Qϕψx[i,j]   += 0.5 * (rdisplx[i,j] * lQUdata.∂11Qx[i,j] * rdisplx[i,j]) 
            Qϕψx[i,j]   +=       (rdisplx[i,j] * lQUdata.∂12Qx[i,j] * rdisply[i,j]) 
            Qϕψx[i,j]   += 0.5 * (rdisply[i,j] * lQUdata.∂22Qx[i,j] * rdisply[i,j]) 

            Uϕψx[i,j]    = lQUdata.Ux[i,j]   
            Uϕψx[i,j]   += (lQUdata.∂1Ux[i,j]  * rdisplx[i,j]) + (lQUdata.∂2Ux[i,j]  * rdisply[i,j])
            Uϕψx[i,j]   += 0.5 * (rdisplx[i,j] * lQUdata.∂11Ux[i,j] * rdisplx[i,j]) 
            Uϕψx[i,j]   +=       (rdisplx[i,j] * lQUdata.∂12Ux[i,j] * rdisply[i,j]) 
            Uϕψx[i,j]   += 0.5 * (rdisply[i,j] * lQUdata.∂22Ux[i,j] * rdisply[i,j])   

            # should we also go to a deeper Taylor term for these?
            ∂1Qϕψx[i,j]  = lQUdata.∂1Qx[i,j] + (lQUdata.∂11Qx[i,j] * rdisplx[i,j]) + (lQUdata.∂12Qx[i,j] * rdisply[i,j])
            ∂1Uϕψx[i,j]  = lQUdata.∂1Ux[i,j] + (lQUdata.∂11Ux[i,j] * rdisplx[i,j]) + (lQUdata.∂12Ux[i,j] * rdisply[i,j])
            ∂2Qϕψx[i,j]  = lQUdata.∂2Qx[i,j] + (lQUdata.∂12Qx[i,j] * rdisplx[i,j]) + (lQUdata.∂22Qx[i,j] * rdisply[i,j])
            ∂2Uϕψx[i,j]  = lQUdata.∂2Ux[i,j] + (lQUdata.∂12Ux[i,j] * rdisplx[i,j]) + (lQUdata.∂22Ux[i,j] * rdisply[i,j])
        end
    end
    
    return Qϕψx, Uϕψx, ∂1Qϕψx, ∂1Uϕψx, ∂2Qϕψx, ∂2Uϕψx
end


function easylense(Tdata::TandFriends, ϕ, ψ) 
    # first decompose the lensing
    row, col = size(ϕ)
    displx   = ifft2r(im .* k1 .* ϕ) + ifft2r(im .* k2 .* ψ)
    disply   = ifft2r(im .* k2 .* ϕ) - ifft2r(im .* k1 .* ψ)
    rdisplx  = Array(Float64, row, col)
    rdisply  = Array(Float64, row, col)
    indcol   = Array(Int64, row, col)
    indrow   = Array(Int64, row, col)
    decomplense!(indcol, indrow, rdisplx, rdisply, x, y, displx, disply, deltx)
    
    # now do integer lensing on the taylor terms
    lTdata = gridlense(Tdata, indcol, indrow)

    # now do sub-grid taylor corrections
    Tϕψx    = Array(Float64, row, col)
    ∂1Tϕψx  = Array(Float64, row, col)
    ∂2Tϕψx  = Array(Float64, row, col)
    for j = 1:col
        for i = 1:row
            Tϕψx[i,j]    = lTdata.Tx[i,j]  
            Tϕψx[i,j]   += (lTdata.∂1Tx[i,j]  * rdisplx[i,j]) + (lTdata.∂2Tx[i,j]  * rdisply[i,j])
            Tϕψx[i,j]   += 0.5 * (rdisplx[i,j] * lTdata.∂11Tx[i,j] * rdisplx[i,j]) 
            Tϕψx[i,j]   +=       (rdisplx[i,j] * lTdata.∂12Tx[i,j] * rdisply[i,j]) 
            Tϕψx[i,j]   += 0.5 * (rdisply[i,j] * lTdata.∂22Tx[i,j] * rdisply[i,j]) 
            # should we also go to a deeper Taylor term for these?
            ∂1Tϕψx[i,j]  = lTdata.∂1Tx[i,j] + (lTdata.∂11Tx[i,j] * rdisplx[i,j]) + (lTdata.∂12Tx[i,j] * rdisply[i,j])
            ∂2Tϕψx[i,j]  = lTdata.∂2Tx[i,j] + (lTdata.∂12Tx[i,j] * rdisplx[i,j]) + (lTdata.∂22Tx[i,j] * rdisply[i,j])
        end
    end
    
    return Tϕψx, ∂1Tϕψx, ∂2Tϕψx
end


function gridlense(QUdata::QUandFriends, indcol, indrow) 
    row, col = size(QUdata.Qx)
    lQUdata  = QUandFriends(row, col)  # allocate a integer delese
    for j = 1:col
        for i = 1:row
            lQUdata.Qx   [i,j] = QUdata.Qx   [indrow[i,j], indcol[i,j]]
            lQUdata.Ux   [i,j] = QUdata.Ux   [indrow[i,j], indcol[i,j]]
            lQUdata.∂1Qx [i,j] = QUdata.∂1Qx [indrow[i,j], indcol[i,j]]
            lQUdata.∂2Qx [i,j] = QUdata.∂2Qx [indrow[i,j], indcol[i,j]]
            lQUdata.∂1Ux [i,j] = QUdata.∂1Ux [indrow[i,j], indcol[i,j]]
            lQUdata.∂2Ux [i,j] = QUdata.∂2Ux [indrow[i,j], indcol[i,j]]
            lQUdata.∂11Qx[i,j] = QUdata.∂11Qx[indrow[i,j], indcol[i,j]]
            lQUdata.∂12Qx[i,j] = QUdata.∂12Qx[indrow[i,j], indcol[i,j]]
            lQUdata.∂22Qx[i,j] = QUdata.∂22Qx[indrow[i,j], indcol[i,j]]
            lQUdata.∂11Ux[i,j] = QUdata.∂11Ux[indrow[i,j], indcol[i,j]]
            lQUdata.∂12Ux[i,j] = QUdata.∂12Ux[indrow[i,j], indcol[i,j]]
            lQUdata.∂22Ux[i,j] = QUdata.∂22Ux[indrow[i,j], indcol[i,j]]
        end
    end
    # Note that the entries of lQUdata are now not related by derivatives of each-other.
    # Instead, they correspond to integerLensing of the corresponding derivatives. 
    return lQUdata
end


function gridlense(Tdata::TandFriends, indcol, indrow) 
    row, col = size(Tdata.Tx)
    lTdata  = TandFriends(row, col)  # allocate a integer delese
    for j = 1:col
        for i = 1:row
            lTdata.Tx   [i,j] = Tdata.Tx   [indrow[i,j], indcol[i,j]]
            lTdata.∂1Tx [i,j] = Tdata.∂1Tx [indrow[i,j], indcol[i,j]]
            lTdata.∂2Tx [i,j] = Tdata.∂2Tx [indrow[i,j], indcol[i,j]]
            lTdata.∂11Tx[i,j] = Tdata.∂11Tx[indrow[i,j], indcol[i,j]]
            lTdata.∂12Tx[i,j] = Tdata.∂12Tx[indrow[i,j], indcol[i,j]]
            lTdata.∂22Tx[i,j] = Tdata.∂22Tx[indrow[i,j], indcol[i,j]]
        end
    end
    # Note that the entries of lTdata are now not related by derivatives of each-other.
    # Instead, they correspond to integerLensing of the corresponding derivatives. 
    return lTdata
end




function gridlense!(lF1ₓ, F1ₓ, indcol, indrow)
    # mutates: lF1ₓ
    row, col = size(F1ₓ)
    for j = 1:col
        for i = 1:row
            lF1ₓ[i,j]  = F1ₓ[indrow[i,j], indcol[i,j]]
        end
    end
    return nothing
end

function gridlense!(lF1ₓ, lF2ₓ, F1ₓ, F2ₓ, indcol, indrow)
    # mutates: lF1ₓ, lF2ₓ
    row, col = size(F1ₓ)
    for j = 1:col
        for i = 1:row
            lF1ₓ[i,j]  = F1ₓ[indrow[i,j], indcol[i,j]]
            lF2ₓ[i,j]  = F2ₓ[indrow[i,j], indcol[i,j]]
        end
    end
    return nothing
end

indexwrap(ind::Int64, uplim)  = mod(ind - 1, uplim) + 1

function decomplense!(indcol, indrow, rdisplx, rdisply, x, y, displx, disply, deltx)
    # the arguments (indcol, indrow, rdisplx, rdisply) are mutated
    row, col = size(x) 
    period = maximum(x)
    for j = 1:col, i = 1:row
        @inbounds indcol[i,j]  = indexwrap(j + round(Int64, displx[i,j]/deltx), col)
        @inbounds indrow[i,j]  = indexwrap(i + round(Int64, disply[i,j]/deltx), row)
        rdisplx[i,j] = displx[i,j] - deltx * round(Int64, displx[i,j]/deltx)
        rdisply[i,j] = disply[i,j] - deltx * round(Int64, disply[i,j]/deltx)
    end
    return nothing
end


#=   # Here are some tests for speed of gridlense! and delense!
indcol, indrow   = Array(Int, size(x)), Array(Int, size(x))
rdisplx, rdisply = Array(Float64, size(x)), Array(Float64, size(x))
gc()
@time  decomplense!(indcol, indrow, rdisplx, rdisply, x, y, displx, disply, deltx);

glenQx = Array(Float64, size(x))
gc()
@time gridlense!(glenQx, Qx, indcol, indrow);

using ProfileView
@profileview gridlense!(glenQx, Qx, indcol, indrow);

g = @code_typed gridlense!(glenQx, Qx, indcol, indrow);
g[1].args[2][2]

g = @code_typed decomplense!(indcol, indrow, rdisplx, rdisply, x, y, displx, disply, deltx)
g[1].args[2][2]
=#




################################################
#   
# Gradient update functions
#
################################################

# I'm turning off the ψ updates.
function gradupdate!(ϕcurr, ψcurr, QUdata::QUandFriends, maxitr = 20, sg1 = 1e-8, sg2 = 1e-8)
    for cntr = 1:maxitr
        ϕgrad, ψgrad = ϕψgrad_master(ϕcurr, ψcurr, QUdata, Mq, Mu, Mqu)
        # ϕcurr[:] = ϕcurr + ϕgrad .* sg1 .* (cPh + √(magk).*cPh) .* (magk .< maskupP) 
        # ψcurr[:] = ψcurr + ψgrad .* sg2 .* (cPs + √(magk).*cPs) .* (magk .< maskupP)
        ϕcurr[:] = ϕcurr + ϕgrad .* sg1 .* (cPh) .* (magk .< maskupP ) 
        ψcurr[:] = ψcurr + ψgrad .* sg2 .* (cPs) .* (magk .< maskupP ) 
    end
end


function ϕψgrad_master(ϕ, ψ, QUdata::QUandFriends, Mq, Mu, Mqu)
    # -- antilense QUdata 
    Qϕψx, Uϕψx, ∂1Qϕψx, ∂1Uϕψx, ∂2Qϕψx, ∂2Uϕψx = easylense(QUdata, -ϕ, -ψ)
    Qϕψ     = fft2(Qϕψx)
    Uϕψ     = fft2(Uϕψx)

    # --- compute loglike
    Eϕψ = - Qϕψ .* cos(φ2_l) - Uϕψ .* sin(φ2_l) 
    Bϕψ =   Qϕψ .* sin(φ2_l) - Uϕψ .* cos(φ2_l) 
    loglike   = - 0.5 * sum(squash!( abs2(Eϕψ .* (magk .<= maskupC)) ./ cEE )) * dk
    loglike  += - 0.5 * sum(squash!( abs2(Bϕψ .* (magk .<= maskupC)) ./ cBB )) * dk
    loglike  += - 0.5 * sum(squash!( abs2(ϕ.* (magk .<= maskupP))./cPh + abs2(ψ .* (magk .<= maskupP))./cPs )) * dk
    @show loglike

    # this calls ϕψgrad with different weights and inputs
    ϕgradQ, ψgradQ   = ϕψgrad(∂1Qϕψx, ∂2Qϕψx, Qϕψ, Mq)
    ϕgradU, ψgradU   = ϕψgrad(∂1Uϕψx, ∂2Uϕψx, Uϕψ, Mu)
    ϕgradQU, ψgradQU = ϕψgrad(∂1Qϕψx, ∂2Qϕψx, Qϕψ, ∂1Uϕψx, ∂2Uϕψx, Uϕψ, Mqu)
    return  ϕgradQ + ϕgradU + ϕgradQU - squash!(ϕ ./ cPh),   ψgradQ + ψgradU + ψgradQU - squash!(ψ ./ cPs)
end


function ϕψgrad(∂₁Xϕψₓ, ∂₂Xϕψₓ, Xϕψ, ∂₁Yϕψₓ, ∂₂Yϕψₓ, Yϕψ, M)
    X₁YM = fft2(∂₁Xϕψₓ .* ifft2(Yϕψ .* M))
    X₂YM = fft2(∂₂Xϕψₓ .* ifft2(Yϕψ .* M))
    Y₁XM = fft2(∂₁Yϕψₓ .* ifft2(Xϕψ .* M))
    Y₂XM = fft2(∂₂Yϕψₓ .* ifft2(Xϕψ .* M))
    ϕgrad = similar(X₁YM)
    ψgrad = similar(X₁YM)
    for i2 = 1:n
        for i1 = 1:n
        ϕgrad[i1, i2] = im * ( k1[i1, i2] * X₁YM[i1, i2] 
                             + k2[i1, i2] * X₂YM[i1, i2]
                             + k1[i1, i2] * Y₁XM[i1, i2]
                             + k2[i1, i2] * Y₂XM[i1, i2] )
        ψgrad[i1, i2] = - im * ( k1[i1, i2] * X₂YM[i1, i2] 
                             - k2[i1, i2] * X₁YM[i1, i2]
                             + k1[i1, i2] * Y₂XM[i1, i2]
                             - k2[i1, i2] * Y₁XM[i1, i2] )
        end
    end
    return ϕgrad, ψgrad
end


function ϕψgrad(∂₁Xϕψₓ, ∂₂Xϕψₓ, Xϕψ, M)
    X₁XM = fft2(∂₁Xϕψₓ .* ifft2(Xϕψ .* M))
    X₂XM = fft2(∂₂Xϕψₓ .* ifft2(Xϕψ .* M))
    ϕgrad = similar(X₁XM)
    ψgrad = similar(X₁XM)
    for i2 = 1:n
        for i1 = 1:n
        ϕgrad[i1, i2] = 2 * im * ( k1[i1, i2] * X₁XM[i1, i2] 
                                 + k2[i1, i2] * X₂XM[i1, i2])
        ψgrad[i1, i2] = - 2 * im * ( k1[i1, i2] * X₂XM[i1, i2] 
                                 - k2[i1, i2] * X₁XM[i1, i2])
        end
    end
    return ϕgrad, ψgrad
end


################################################
#   
# Misc
#
################################################



function squash!{T}(x::AbstractArray{T})
    x[isnan(x)]        = zero(T)
    x[abs(x) .== abs(Inf)]  = zero(T)
    x
end

function binave{T}(fk::Matrix{T}, kmag::Matrix, bin_mids::Range)
    fpwr = zeros(T, length(bin_mids))
    rtcuts  = collect(bin_mids + step(bin_mids) / 2)  
    lftcuts = collect(bin_mids - step(bin_mids) / 2)  
    lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
    for i in 1:length(bin_mids)
        ibin = lftcuts[i] .<= kmag .< rtcuts[i]
        fpwr[i] = fk[ibin] |> mean   
    end
    fpwr
end
function binsum{T}(fk::Matrix{T}, kmag::Matrix, bin_mids::Range)
    fpwr = zeros(T, length(bin_mids))
    rtcuts  = collect(bin_mids +  step(bin_mids) / 2)  
    lftcuts = collect(bin_mids -  step(bin_mids) / 2)  
    lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
    for i in 1:length(bin_mids)
        ibin = lftcuts[i] .<= kmag .< rtcuts[i]
        fpwr[i] = fk[ibin] |> sum   
    end
    fpwr
end
function bincount(kmag::Matrix, bin_mids::Range)
    bcounts = zeros(Int64, length(bin_mids))
    rtcuts  = collect(bin_mids +  step(bin_mids) / 2)  
    lftcuts = collect(bin_mids -  step(bin_mids) / 2)  
    lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
    for i in 1:length(bin_mids)
        ibin = lftcuts[i] .<= kmag .< rtcuts[i]
        bcounts[i] = sum(ibin)
    end
    bcounts
end


""" 
The functions have the same name as above but use the different 
type signature of bin_edg to specify bin edges rather than bin midpoints. 
The reason these dispatch to different methods is that when bin_edg is 
called with log-spacing, for example, there is no clear notion of bin midpoint.
"""
function binave{T}(fk::Matrix{T}, kmag::Matrix, bin_edg::Array{Float64,1})
    #when the binning is a float we work with the edges instead
    fpwr = zeros(T, length(bin_edg))
    for i = 1:length(bin_edg)
        rtcuts  = bin_edg[i] 
        lftcuts = (i == 1) ? 0.1 : bin_edg[i - 1] 
        ibin = lftcuts .<= kmag .< rtcuts
        fpwr[i] = fk[ibin] |> mean   
    end
    fpwr
end
function binsum{T}(fk::Matrix{T}, kmag::Matrix, bin_edg::Array{Float64,1})
    #when the binning is a float we work with the edges instead
    fpwr = zero(T, length(bin_edg))
    for i = 1:length(bin_edg)
        rtcuts  = bin_edg[i] 
        lftcuts = (i == 1) ? 0.1 : bin_edg[i - 1] 
        ibin = lftcuts .<= kmag .< rtcuts
        fpwr[i] = fk[ibin] |> sum   
    end
    fpwr
end
function bincount(kmag::Matrix, bin_edg::Array{Float64,1})
    bcounts = zeros(Int64, length(bin_edg))
    for i = 1:length(bin_edg)
        rtcuts  = bin_edg[i] 
        lftcuts = (i == 1) ? 0.1 : bin_edg[i - 1] 
        ibin = lftcuts .<= kmag .< rtcuts
        bcounts[i] = sum(ibin)  
    end
    bcounts
end







#=

#------------------------------------
# HMC sampler
#--------------------------------------------
function hmc!(phik_curr, tildetx_hr_curr, parlr, parhr, scale_hmc = 1.0e-3)
    ePs = scale_hmc * rand()
    ulim =  30 
    h0 = smooth_heavy(parlr.grd.magk, 0.5, 1, 1500, 1/200) .* parlr.cPP ./ (parlr.grd.deltk^2) 
    mk = 1.0e-2 ./ h0
    mk[parlr.pMaskBool] = 0.0

    phik_test = copy(phik_curr)
    rk   = white(parlr) .* sqrt(mk); # note that the variance of real(pk_init) and imag(pk_init) is mk/2
    grad, loglike   = ttk_grad_wlog(tildetx_hr_curr, phik_test, parlr, parhr)
    h_at_zero = 0.5 * sum( abs2(rk[!parlr.pMaskBool])./(2*mk[!parlr.pMaskBool]/2)) - loglike # the 0.5 is out front since only half the sum is unique
    
    for HMCCounter = 1:ulim 
        loglike = lfrog!(phik_test, rk, tildetx_hr_curr, parlr, parhr, ePs, mk)
    end
    
    h_at_end = 0.5 * sum( abs2(rk[!parlr.pMaskBool])./(2*mk[!parlr.pMaskBool]/2)) - loglike # the 0.5 is out front since only half the sum is unique
    prob_accept = minimum([1, exp(h_at_zero - h_at_end)])
    if rand() < prob_accept
        phik_curr[:] = phik_test
        println("Accept: prob_accept = $(round(prob_accept,4)), h_at_end = $(round(h_at_end)), h_at_zero = $(round(h_at_zero)), loglike = $(round(loglike))")
        return 1
    else
        println("Reject: prob_accept = $(round(prob_accept,4)), h_at_end = $(round(h_at_end)), h_at_zero = $(round(h_at_zero)), loglike = $(round(loglike))")
        return 0
    end
end


function lfrog!(phik_curr, rk, tildetx_hr_curr, parlr, parhr, ePs, mk)
    grad, loglike   = ttk_grad_wlog(tildetx_hr_curr, phik_curr, parlr, parhr)
    rk_halfstep =  rk +  ePs .* grad ./ 2.0
    inv_mk = 1./ (mk ./ 2.0)
    inv_mk[parlr.pMaskBool] = 0.0
    phik_curr[:] = phik_curr + ePs .* inv_mk .* rk_halfstep
    grad, loglike   = ttk_grad_wlog(tildetx_hr_curr, phik_curr, parlr, parhr)
    rk[:] = rk_halfstep + ePs .* grad ./ 2.0
    loglike
end



#----------------------------------
# miscilanious function
#----------------------------------
white(par::SpectrumGrids) = (par.grd.deltk/par.grd.deltx)*fft2(randn(size(par.grd.x)),par.grd.deltx)


=#
