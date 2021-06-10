"""
r = createGridGreensFunction(nCells,res)\n
Computes the radius for the domain given by nCells.\n
This radius is used to compute the weakened Greens functions.\n
"""
function createGridGreensFunction(nCells,res)
    #get max dimensions
    dx,dy,dz = 2 .^(ceil.(Int,log2.(nCells.+2).+1))
    lx = res[1].*collect(-dx/2f0:dx/2f0-1)
    ly = res[2].*collect(-dy/2f0:dy/2f0-1)
    lz = res[3].*collect(-dz/2f0:dz/2f0-1)

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
G = createGreensFunctions(nCells,res,kb).\n
creates the green function on the larger grid.\n
"""
function createGreensFunctions(nCells,res,kb)
    a = findmin(res)[1]/2
    r = createGridGreensFunction(nCells,res)
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

"""
libraryMatrixTimesCurrentDensity!(JContrast,nUpdates,jv,eI,S,tmp,efft,G,A,w,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq).\n
calculate the scattered electric field given the current density.\n
"""
function libraryMatrixTimesCurrentDensity!(JContrast,jv,eI,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq)
    vInputTup = -JContrast |> cu 

    VIE.setSourceValuesInDomain!(jv,eI,vInputTup,S)
    #from source in E incident to the actual incident electric field
    VIE.JIncToEInc!(eI,a,A,G,χ,Ig,AToE,efft,pfft,pifft,scaling)
    #compute total electric field
    VIE.cgs_efield!(eI,efft,G,A,χ,a,AToE,Ig,pfft,pifft,x,p,r,rt,u,v,q,uq)
    return x
end #TODO: check if cpu version/collect are required


"""
setSourceValuesInDomain!(jv,eI,v,S)\n
will set the current density values (jv's) to -v\n
Using the S matrix these are placed into the incident electric field.\n
"""
function setSourceValuesInDomain!(jv,eI,v,S)
    # set source values to the vector we are multiplying A with
    jv .= -one(ComplexF32).*v
    # set the source values into the incident electric field (will be equal to Jinc before the function JIncToEinc!)
    mul!(eI,S.N,jv)
end


"""
wrapperVIE!(output,jv,eI,C,vInput,S,tmp,efft,G,A,w,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq,nc)\n
In place version to calculate Av = (I-CSZ)v = v - CSZv.\n
-CSZv is calculated using the VIE method by setting the sources to -v and multiplying the edges with the corresponding C value.
"""
function wrapperVIE!(output,jv,eI,C,vInput,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq)
    # set sources in the vacuum to the correct values 
    setSourceValuesInDomain!(jv,eI,vInput,S)
    # calculate the incident electric field inside the vacuum
    JIncToEInc!(eI,a,A,G,χ,Ig,AToE,efft,pfft,pifft,scaling)
    # solve for the field inside the patient anatomy
    cgs_efield!(eI,efft,G,A,χ,a,AToE,Ig,pfft,pifft,x,p,r,rt,u,v,q,uq)
    # calculate the (v - CSZv) vector for the minimization process
    
    mul!(output,S.T,x)
    @. output *= C
    
    axpy!(one(ComplexF32),vInput,output) # output = vInput - CSZvInput
    #output += vInput
end