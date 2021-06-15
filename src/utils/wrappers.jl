function create_problem(BG_σ,BG_ϵ,Update_σ,Update_ϵ,Einc,frequency,resolution)
    m = getConstants(frequency)

    p,q,r = size(BG_σ);
    nCells = [p-2,q-2,r-2];
    a = minimum(resolution)/2;
    scaling = -1im*m.μ₀*m.ω/(m.kb^2);
    

    x_axis = 0f0:resolution[1]:p*resolution[1];
    y_axis = 0f0:resolution[2]:q*resolution[2];
    z_axis = 0f0:resolution[3]:r*resolution[3];

    BGDielectric = cellToYeeDielectric(BG_σ, BG_ϵ, x_axis, y_axis, z_axis);
    UpdateDielectric = cellToYeeDielectric(Update_σ, Update_ϵ, x_axis, y_axis, z_axis);
    DiffDielectric, logicLocations = computeUpdateMaps(BGDielectric, UpdateDielectric);

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
    C = getC(DiffDielectric,m);

    b,nb = getb(C,S,Einc)
    
    outputVIE = zeros(ComplexF32, sum(nUpdates));

    LinearOp =  LinearMap{ComplexF32}(vecxyz-> catForVIE(outputVIE,jv,eI,C,vecxyz,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq),sum(nUpdates))
    totalFieldsOP = LinearMap{ComplexF32}(J -> libraryMatrixTimesCurrentDensity!(J,jv,eI,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq),sum(nUpdates))

    return (g_map = LinearOp, b_vector = b, norm_b = nb, scattered_electric_field_map = totalFieldsOP)
end

function create_problem_gpu(BG_σ,BG_ϵ,Update_σ,Update_ϵ,Einc,frequency,resolution)
    p,q,r = size(BG_σ);
    nCells = [p-2,q-2,r-2];
    a = minimum(resolution)/2;
    scaling = -1im*m.μ₀*m.ω/(m.kb^2);
    
    m = getConstants(frequency)

    x_axis = 0f0:resolution[1]:p*resolution[1];
    y_axis = 0f0:resolution[2]:q*resolution[2];
    z_axis = 0f0:resolution[3]:r*resolution[3];

    BGDielectric = cellToYeeDielectric(BG_σ, BG_ϵ, x_axis, y_axis, z_axis);
    UpdateDielectric = cellToYeeDielectric(Update_σ, Update_ϵ, x_axis, y_axis, z_axis);
    DiffDielectric, logicLocations = computeUpdateMaps(BGDielectric, UpdateDielectric);

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
    C = getC(DiffDielectric,m) |> cu;

    b,nb = getb(C,S,Einc)
    
    outputVIE = CUDA.zeros(ComplexF32, sum(nUpdates));

    LinearOp =  LinearMap{ComplexF32}(vecxyz-> catForVIE_gpu(outputVIE,jv,eI,C,vecxyz,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq),sum(nUpdates))
    totalFieldsOP = LinearMap{ComplexF32}(J -> libraryMatrixTimesCurrentDensity_gpu!(J,jv,eI,S,efft,G,A,χ,a,AToE,Ig,scaling,pfft,pifft,x,p,r,rt,u,v,q,uq),sum(nUpdates))

    return (g_map = LinearOp, b_vector = b, norm_b = nb, scattered_electric_field_map = totalFieldsOP)
end

"""
scattered_current_density, scattered_electric_field = solve_problem(problem; verbose = true, max_iterations = 100, restart = 50, tol = 1f-4).\n
the problem is created by calling the create_problem function.\n 
the problem is solved using IterativeSolvers gmres implementation.\n 
it can also be solved by other solvers if you believe they are more accurate or faster.\n 
personally, I found this solver to be both fast, stable, and finding the correct solution.\n 
Some standard arguments are supplied, but can be tweaked according to preference/complexity of the problem.\n
"""
function solve_problem(problem; verbose = true, max_iterations = 100, restart = 50, tol = 1f-4)
    # solve the matrix inverse to obtain the scattered current density
    scattered_current_density = IterativeSolvers.gmres(problem.g_map,problem.b_vector,verbose=verbose,initially_zero=true,reltol=tol,restart = restart,maxiter=max_iterations);
    # the vector b is normalized to a norm of 1, thus we rescale the solution since the problem is linear.
    scattered_current_density .*= problem.norm_b
    # we calculate the scattered electric field that is created by the dielectric update that is performed
    scattered_electric_field  = problem.scattered_electric_field_map*scattered_current_density
    return scattered_current_density, scattered_electric_field
end