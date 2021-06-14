"""
getNUpdates(logicLocations).\n
get the number of edges that are updated as [Nx,Ny,Nz]\n
sum([Nx,Ny,Nz]) == size of the matrix inverse.\n
"""
function getNUpdates(logicLocations)
    return count(logicLocations[1]),count(logicLocations[2]),count(logicLocations[3])
end

"""
χ = setDielectric(BGDielectric,m,nCells).\n
given the background dielectric make the dielectric used for the VIE method.\n
it has the same size as the Green's function operator and is zero padded.\n
for now the background medium is vacuum, but this could be changed to some homogenous dielectric.\n
"""
function setDielectric(BGDielectric,m,nCells)
    T = ComplexF32
    dx,dy,dz = 2 .^(ceil.(Int,log2.(nCells.+2).+1))
    p,q,r = nCells
    #dielectric vacuum
    vac = one(T)
    # dielectric background
    tmpx = (BGDielectric[:σˣ]./(1im*m.ω*m.ϵ₀) + BGDielectric[:ϵˣ] .-1)[:,2:end-1,2:end-1]
    tmpy = (BGDielectric[:σʸ]./(1im*m.ω*m.ϵ₀) + BGDielectric[:ϵʸ] .-1)[2:end-1,:,2:end-1]
    tmpz = (BGDielectric[:σᶻ]./(1im*m.ω*m.ϵ₀) + BGDielectric[:ϵᶻ] .-1)[2:end-1,2:end-1,:]
    
    patient = zeros(T,dx,dy,dz,3)
    patient[1:p+2,1:q+1,1:r+1,1] = tmpx
    patient[1:p+1,1:q+2,1:r+1,2] = tmpy
    patient[1:p+1,1:q+1,1:r+2,3] = tmpz

    return (;vac,patient)
end

"""
χ = setDielectric_gpu(BGDielectric,m,nCells).\n
given the background dielectric make the dielectric used for the VIE method on the gpu.\n
it has the same size as the Green's function operator and is zero padded.\n
for now the background medium is vacuum, but this could be changed to some homogenous dielectric.\n
"""
function setDielectric_gpu(BGDielectric,m,nCells)
    T = ComplexF32
    dx,dy,dz = 2 .^(ceil.(Int,log2.(nCells.+2).+1))
    p,q,r = nCells
    #dielectric vacuum
    vac = one(T)
    # dielectric
    tmpx = (BGDielectric[:σˣ]./(1im*m.ω*m.ϵ₀) + BGDielectric[:ϵˣ] .-1)[:,2:end-1,2:end-1]
    tmpy = (BGDielectric[:σʸ]./(1im*m.ω*m.ϵ₀) + BGDielectric[:ϵʸ] .-1)[2:end-1,:,2:end-1]
    tmpz = (BGDielectric[:σᶻ]./(1im*m.ω*m.ϵ₀) + BGDielectric[:ϵᶻ] .-1)[2:end-1,2:end-1,:]
    
    pat = zeros(T,dx,dy,dz,3)
    pat[1:p+2,1:q+1,1:r+1,1] = tmpx
    pat[1:p+1,1:q+2,1:r+1,2] = tmpy
    pat[1:p+1,1:q+1,1:r+2,3] = tmpz
    patient = pat |> cu 
    
    return (;vac,patient)
end

"""
catForVIE(sol,jv,eI,C,vecxyz,S,tmp,efft,G,A,w,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq).\n
function that is used to make a linear map of f(x) = x-CS^TZx.\n
Where Zx is calculated using the VIE method.\n
"""
function catForVIE(sol,jv,eI,C,vecxyz,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq)
  wrapperVIE!(sol,jv,eI,C,vecxyz,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq)
  return sol
end

function catForVIE_gpu(sol,jv,eI,C,vecxyz,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq) # TODO: if outer solver is on gpu the collect should be removed
  vInput = vecxyz |> cu 
  wrapperVIE!(sol,jv,eI,C,vInput,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq)
  return collect(sol)
end

function catForVIE_outer_gpu(sol,jv,eI,C,vecxyz,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq,cInput,connect_op,cOutput)
  mul!(cInput,connect_op.N,complex(vecxyz))
  wrapperVIE!(sol,jv,eI,C,cInput,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq)
  mul!(cOutput,connect_op.T,sol)
  return real(cOutput)
end

"""
getLinearMap(BGDielectric,CVector,logicLocations,m,resolution).\n
returns the linear map (f) that that replaces A in ||b-Ax||_2^2 in order to do ||b-f(x)||_2^2.\n
also returns the linearmap that will take the result from min ||b-f(x)||_2^2 to compute the scattered electric field.\n
"""
function getLinearMap(BGDielectric,CVector,logicLocations,m,resolution)
    p,q,r = size(BGDielectric[:σˣ]) .- [0,1,1];
    nCells = [p-2,q-2,r-2];
    a = resolution[1]/2; #resolution/2
    scaling = -1im*m.μ₀*m.ω/(m.kb^2);
    
    #get the number of edges that need to be updated
    nUpdates = getNUpdates(logicLocations);
    
    #operators
    AToE = createSparseDifferenceOperators(nCells,resolution,m.kb);
    Ig   = createGreensFunctionsRestrictionOperators(nCells);
    G    = createGreensFunctions(nCells,resolution,m.kb);

    #dielectric
    χ = setDielectric(BGDielectric,m,nCells);
    
    #allocate memory for inplace operations
    jv,eI,a,A,efft,pfft,pifft = allocateSpaceVIE(nCells,nUpdates);
    x,p,r,rt,u,v,q,uq = allocateCGSVIE(nCells);

    #get the sparse S matrix for each field component
    S = getS(logicLocations); 

    outputVIE = zeros(ComplexF32, sum(nUpdates));
   
    LinearOp =  LinearMap{ComplexF32}(vecxyz-> catForVIE(outputVIE,jv,eI,CVector,vecxyz,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq),sum(nUpdates))
    totalFieldsOP = LinearMap{ComplexF32}(J -> libraryMatrixTimesCurrentDensity!(J,jv,eI,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq),sum(nUpdates))

    return LinearOp,totalFieldsOP
end

"""
getLinearMap_gpu(BGDielectric,CVector,logicLocations,m,resolution).\n
returns the linear map (f) that that replaces A in ||b-Ax||_2^2 in order to do ||b-f(x)||_2^2.\n
also returns the linearmap that will take the result from min ||b-f(x)||_2^2 to compute the scattered electric field.\n
The operators are used for a GPU calculation of the VIE.\n
"""
function getLinearMap_gpu(BGDielectric,CVector,logicLocations,m,resolution)
    p,q,r = size(BGDielectric[:σˣ]) .- [0,1,1];
    nCells = [p-2,q-2,r-2];
    a = resolution[1]/2; #resolution/2
    scaling = -1im*m.μ₀*m.ω/(m.kb^2);
    
    #get the number of edges that need to be updated
    nUpdates = getNUpdates(logicLocations);
    
    #operators
    AToE = createSparseDifferenceOperators(nCells,resolution,m.kb) |> CUDA.CUSPARSE.CuSparseMatrixCSC;
    Ig   = createGreensFunctionsRestrictionOperators_gpu(nCells);
    G    = createGreensFunctions(nCells,resolution,m.kb) |> cu;

    #dielectric
    χ = setDielectric_gpu(BGDielectric,m,nCells);
    
    #allocate memory for inplace operations
    jv,eI,a,A,efft,pfft,pifft = allocateSpaceVIE_gpu(nCells,nUpdates);
    x,p,r,rt,u,v,q,uq = allocateCGSVIE_gpu(nCells);
    #get the sparse S matrix for each field component
    S = getS_gpu(logicLocations); 
    C = CVector |> cu

    outputVIE = CUDA.zeros(ComplexF32, sum(nUpdates));

    LinearOp =  LinearMap{ComplexF32}(vecxyz-> catForVIE_gpu(outputVIE,jv,eI,C,vecxyz,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq),sum(nUpdates))
    totalFieldsOP = LinearMap{ComplexF32}(J -> libraryMatrixTimesCurrentDensity!(J,jv,eI,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq),sum(nUpdates))

    return LinearOp,totalFieldsOP
end

"""
getLinearMap_outer_gpu(BGDielectric,CVector,logicLocations,m,resolution).\n
returns the linear map (f) that that replaces A in ||b-Ax||_2^2 in order to do ||b-f(x)||_2^2.\n
also returns the linearmap that will take the result from min ||b-f(x)||_2^2 to compute the scattered electric field.\n
The operators when th entire pipeline is on the gpu calculation.\n
NOTE: the iterative solvers that have been tested that work on the GPU either don't converge or are slower than IterativeSolvers.jl's gmres.\n
"""
function getLinearMap_outer_gpu(BGDielectric,CVector,logicLocations,m,resolution)
    p,q,r = size(BGDielectric[:σˣ]) .- [0,1,1];
    nCells = [p-2,q-2,r-2];
    a = resolution[1]/2; #resolution/2
    scaling = -1im*m.μ₀*m.ω/(m.kb^2);
    
    #get the number of edges that need to be updated
    nUpdates = getNUpdates(logicLocations);
    
    #operators
    AToE = createSparseDifferenceOperators(nCells,resolution,m.kb) |> CUDA.CUSPARSE.CuSparseMatrixCSC;
    Ig   = createGreensFunctionsRestrictionOperators_gpu(nCells);
    G    = createGreensFunctions(nCells,resolution,m.kb) |> cu;
    connect_op = createOuterOps(nUpdates);

    #dielectric
    χ = setDielectric_gpu(BGDielectric,m,nCells);
    
    #allocate memory for inplace operations
    jv,eI,a,A,efft,pfft,pifft = allocateSpaceVIE_gpu(nCells,nUpdates);
    x,p,r,rt,u,v,q,uq = allocateCGSVIE_gpu(nCells);
    #for connections 
    cInput = CUDA.zeros(ComplexF32,sum(nUpdates));
    cOutput = CUDA.zeros(ComplexF32,2*sum(nUpdates));
    #get the sparse S matrix for each field component
    S = getS_gpu(logicLocations); 
    C = CVector |> cu

    outputVIE = CUDA.zeros(ComplexF32, sum(nUpdates));

    LinearOp =  LinearMap{Float32}(vecxyz-> catForVIE_outer_gpu(outputVIE,jv,eI,C,vecxyz,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq,cInput,connect_op,cOutput),2*sum(nUpdates))
    totalFieldsOP = LinearMap{ComplexF32}(J -> libraryMatrixTimesCurrentDensity!(J,jv,eI,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq),sum(nUpdates))

    return LinearOp,totalFieldsOP
end

"""
b,norm(b) = getb(CVector,SVector,Einc).\n
Calculates the vector b for the minimaztion of ||b-Ax||_2^2.\n
Einc is the electric field on the yee grid ordered in a vector by Einc_x,Einc_y,Einc_z.\n
Returns the normalized vector b and a float to rescale the output of the minimization.\n
"""
function getb(CVector,SVector,Einc)
    b = CVector.*Einc[SVector[1]];
    nb = norm(b);
    bnormalized = b./nb;
    return bnormalized,nb
end

"""
jv,eI,a,A,efft,pfft,pifft = allocateSpaceVIE(ncells,nupdates)\n
allocates arrays that are used for the volume integral equation method.\n
"""
function allocateSpaceVIE(nCells,nupdates)
    T = ComplexF32
    p,q,r = nCells
    dx,dy,dz = 2 .^(ceil.(Int,log2.(nCells.+2).+1))

    xl = (p+2)*(q+1)*(r+1)
    yl = (p+1)*(q+2)*(r+1)
    zl = (p+1)*(q+1)*(r+2)
    tl = xl + yl + zl 

    #source values
    jv = zeros(T,sum(nupdates)) 
    #incident electric fields and vector potential    
    eI,a = (zeros(T,tl) for _ in 1:2)
    #fft temporaries
    efft,A = (zeros(T,dx*dy*dz*3) for _ in 1:2) 
    A = reshape(A,dx,dy,dz,3)
    #planning for fft and ifft
    pfft  = FFTW.plan_fft!(A, (1,2,3))
    pifft = FFTW.plan_ifft!(A, (1,2,3))
    
    return jv,eI,a,A,efft,pfft,pifft
end

"""
jv,eI,a,A,efft,pfft,pifft = allocateSpaceVIE_gpu(ncells,nupdates)\n
allocates arrays that are used for the volume integral equation method on the gpu.\n
"""
function allocateSpaceVIE_gpu(nCells,nupdates)
    T = ComplexF32
    p,q,r = nCells
    dx,dy,dz = 2 .^(ceil.(Int,log2.(nCells.+2).+1))

    xl = (p+2)*(q+1)*(r+1)
    yl = (p+1)*(q+2)*(r+1)
    zl = (p+1)*(q+1)*(r+2)
    tl = xl + yl + zl 

    #source values
    jv = CUDA.zeros(T,sum(nupdates)) 
    #incident electric fields and vector potential    
    eI,a = (CUDA.zeros(T,tl) for _ in 1:2)
    #fft temporaries
    efft,A = (CUDA.zeros(T,dx*dy*dz*3) for _ in 1:2) 
    #planning for fft and ifft
    pfft  = FFTW.plan_fft!(A, (1,2,3))
    pifft = FFTW.plan_ifft!(A, (1,2,3))
    
    return jv,eI,a,A,efft,pfft,pifft
end

"""
x,p,r,rt,u,v,q,uq = allocateCGSVIE(nCells)\n
allocates the memory for the conjugate gradient squared method to solve the VIE.\n
"""
function allocateCGSVIE(nCells)
    T = ComplexF32
    #vectors
    np,nq,nr = nCells
    xl = (np+2)*(nq+1)*(nr+1)
    yl = (np+1)*(nq+2)*(nr+1)
    zl = (np+1)*(nq+1)*(nr+2)
    tl = xl + yl + zl 

    x,p,r,rt,u,v,q,uq = (zeros(T,tl) for _ in 1:8)
    return x,p,r,rt,u,v,q,uq
end

"""
x,p,r,rt,u,v,q,uq = allocateCGSVIE_gpu(nCells)\n
allocates the memory on the gpu for the conjugate gradient squared method to solve the VIE.\n
"""
function allocateCGSVIE_gpu(nCells)
    T = ComplexF32
    #vectors
    np,nq,nr = nCells
    xl = (np+2)*(nq+1)*(nr+1)
    yl = (np+1)*(nq+2)*(nr+1)
    zl = (np+1)*(nq+1)*(nr+2)
    tl = xl + yl + zl 

    x,p,r,rt,u,v,q,uq = (CUDA.zeros(T,tl) for _ in 1:8)
    return x,p,r,rt,u,v,q,uq
end
