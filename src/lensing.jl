# lensing code

"""
```
lense!{T}(qu::QUpartials, len::LenseDecomp, g::FFTgrid{2,T}, qu_storage::QUpartials)
```
"""
function lense!{T}(qu::QUpartials, len::LenseDecomp, g::FFTgrid{2,T}, qu_storage::QUpartials)
	ilense!(qu, len, qu_storage)
	rlense!(qu, len)
	update!(qu, g) # update the derivatives and EB modes in qu
end



"""
```
rlense!{T}(qu::QUpartials, len::LenseDecomp)
```
Subgridscale lensing. Just updates qu.Qx and qu.Ux.
"""
function rlense!(qu::QUpartials, len::LenseDecomp)
    row, col = size(len.ϕk)
    @inbounds for j = 1:col, i = 1:row
            qu.qx[i,j]    = qu.qx[i,j]
            qu.qx[i,j]   += (qu.∂1qx[i,j]  * len.rdisplx[i,j]) + (qu.∂2qx[i,j]  * len.rdisply[i,j])
            qu.qx[i,j]   += 0.5 * (len.rdisplx[i,j] * qu.∂11qx[i,j] * len.rdisplx[i,j])
            qu.qx[i,j]   +=       (len.rdisplx[i,j] * qu.∂12qx[i,j] * len.rdisply[i,j])
            qu.qx[i,j]   += 0.5 * (len.rdisply[i,j] * qu.∂22qx[i,j] * len.rdisply[i,j])
            qu.ux[i,j]    = qu.ux[i,j]
            qu.ux[i,j]   += (qu.∂1ux[i,j]  * len.rdisplx[i,j]) + (qu.∂2ux[i,j]  * len.rdisply[i,j])
            qu.ux[i,j]   += 0.5 * (len.rdisplx[i,j] * qu.∂11ux[i,j] * len.rdisplx[i,j])
            qu.ux[i,j]   +=       (len.rdisplx[i,j] * qu.∂12ux[i,j] * len.rdisply[i,j])
            qu.ux[i,j]   += 0.5 * (len.rdisply[i,j] * qu.∂22ux[i,j] * len.rdisply[i,j])
    end
    return Void
end



"""
```
ilense!(qu::QUpartials, len::LenseDecomp, qu_storage::QUpartials)
```
Grid to grid lensing. Note that the entries of qu are now not related by derivatives of each-other.
Instead, they correspond to integerLensing of the corresponding derivatives.
"""
function ilense!(qu::QUpartials, len::LenseDecomp, qu_storage::QUpartials)
    row, col = size(len.ϕk)
    @inbounds for j = 1:col, i = 1:row
            qu_storage.qx[i,j]    = qu.qx[len.indrow[i,j], len.indcol[i,j]]
            qu_storage.ux[i,j]    = qu.ux[len.indrow[i,j], len.indcol[i,j]]
            qu_storage.∂1qx[i,j]  = qu.∂1qx[len.indrow[i,j], len.indcol[i,j]]
            qu_storage.∂2qx[i,j]  = qu.∂2qx[len.indrow[i,j], len.indcol[i,j]]
            qu_storage.∂1ux[i,j]  = qu.∂1ux[len.indrow[i,j], len.indcol[i,j]]
            qu_storage.∂2ux[i,j]  = qu.∂2ux[len.indrow[i,j], len.indcol[i,j]]
            qu_storage.∂11qx[i,j] = qu.∂11qx[len.indrow[i,j], len.indcol[i,j]]
            qu_storage.∂12qx[i,j] = qu.∂12qx[len.indrow[i,j], len.indcol[i,j]]
            qu_storage.∂22qx[i,j] = qu.∂22qx[len.indrow[i,j], len.indcol[i,j]]
            qu_storage.∂11ux[i,j] = qu.∂11ux[len.indrow[i,j], len.indcol[i,j]]
            qu_storage.∂12ux[i,j] = qu.∂12ux[len.indrow[i,j], len.indcol[i,j]]
            qu_storage.∂22ux[i,j] = qu.∂22ux[len.indrow[i,j], len.indcol[i,j]]
    end
	replace!(qu, qu_storage)
    return Void
end
