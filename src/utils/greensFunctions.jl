"""
r = createGridGreensFunction(nCells,res)\n
Computes the radius for the domain given by nCells.\n
This radius is used to compute the weakened Greens functions.\n
"""
function createGridGreensFunction(nCells,res)
    #get max dimensions
    #d = 2^(ceil(Int,log2(findmax(nCells.+2)[1])+1))
    dx,dy,dz = 2 .^(ceil.(Int,log2.(nCells.+2).+1))
    lx = res[1].*collect(-dx/2:dx/2-1)
    ly = res[2].*collect(-dy/2:dy/2-1)
    lz = res[3].*collect(-dz/2:dz/2-1)

    # repeat the axes along the number corresponding dimensions
    X = repeat(lx,outer=(1,dy,dz))
    Y = repeat(reshape(ly,1,:),outer=(dx,1,dz))
    Z = repeat(reshape(lz,1,1,:),outer=(dx,dy,1))
    # compute the radius
    r = @. sqrt(X^2 + Y^2 + Z^2)
    return r
end


"""
Internal function that computes the greens function.
"""
function _createGreensFunction(r,kb,a)
    # sommerfeld radiation condition
    scalarG = 3/(kb*a)^2*(sin(kb*a)/(kb*a)-cos(kb*a))

    # Green's function at r != 0
    G = @. exp.(-1im*kb*r)/(4*π*r)*scalarG

    # Green's function at r = 0
    G₀ = 3/(4*π*kb^2*a^3)*((1+1im*kb*a)*exp(-1im*kb*a)-1)

    G = circshift(G,floor.(Int,size(G)./2))

    G[1,1,1] = G₀
    return fft(G)
end

"""
G = createGreensFunctions(numCells,res,kb).\n
creates the green function on the larger grid.\n
"""
function createGreensFunctions(numCells,res,kb)
    a = findmin(res)[1]/2
    r = createGridGreensFunction(numCells,res)
    G = _createGreensFunction(r,kb,a)
    return prod(res).*G
end

"""
_greensFunctionTimesContrastSource!(a,G,χ,E,w,A,Ig,IgT,tmp,planfft,planifft)\n
Using the Green's function and contrast source we compute the vector potential.\n
a = ∫GχEdV which is computed using an FFT.\n
"""
function _greensFunctionTimesContrastSource!(a,G,χ,E,A,Ig,Er,planfft,planifft)
    mul!(Er,Ig.T,E) # move E to larger domain for fft
    A .= χ.*reshape(Er,size(G)...,3) # multiply with the contrast
    planfft*A # do the fft of the contrast source
    @. A = G*A # multiply the green's function with the contrast source in freq domain
    planifft*A # ifft
    mul!(a,Ig.N,vec(A)) # map to smaller domain again
end

"""
JIncToEInc!(jI,a,tmp,w,A,G,χ,Ig,AToE,efft,planfft,planifft,divωϵim)\n
Compute the electric field in a vacuum given an initial set of current sources (jI)\n
!!!jI will be overwritten to the incident electric field!!!\n
"""
function JIncToEInc!(jI,a,A,G,χ,Ig,AToE,efft,planfft,planifft,divωϵim)
    _greensFunctionTimesContrastSource!(a,G,χ.vac,jI,A,Ig,efft,planfft,planifft)
    mul!(jI,AToE,a)
    @. jI *= divωϵim
end


"""
ETotalMinEScattered!(v,eT,tmp,a,w,A,G,χ,Ig,AToE,efft,planfft,planifft)\n
calculate the incident electric field from the total electric field.\n
This can also be seen as Ax in the context of ||b-Ax||_2^2\n
"""
ETotalMinEScattered!(v,eT,a,A,G,χ,Ig,AToE,efft,planfft,planifft) = ETotalMinEScattered!(v,eT,a,A,G,χ,Ig,AToE,efft,planfft,planifft,1)
function ETotalMinEScattered!(v,eT,a,A,G,χ,Ig,AToE,efft,planfft,planifft,scalar)
    _greensFunctionTimesContrastSource!(a,G,χ.patient,eT,A,Ig,efft,planfft,planifft)
    copyto!(v,eT)
    mul!(v,AToE,a,-scalar,scalar)
end


"""
ETotalMinEScatteredForResidual!(v,eT,tmp,a,w,A,G,χ,Ig,AToE,efft,planfft,planifft,scalar)\n
similar to ETotalMinEScattered!(), but now r-AΔx is computed\n
That is the change in the residual as a result of the update in x\n
"""
function ETotalMinEScatteredForResidual!(v,eT,a,A,G,χ,Ig,AToE,efft,planfft,planifft,scalar)
    _greensFunctionTimesContrastSource!(a,G,χ.patient,eT,A,Ig,efft,planfft,planifft)
    mul!(eT,AToE,a,-scalar,scalar)
    axpy!(one(ComplexF32),eT,v)
end
