"""
prob = create_problem(BG_σ,BG_ϵ,Update_σ,Update_ϵ,Einc,frequency,resolution).\n
supplying the background and desired dielectrics, the incident electric field, frequency and resolution\n 
this will create a problem tuple that can be solved by solve_problem().\n
"""
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

"""
prob = create_problem(BG_σ,BG_ϵ,Update_σ,Update_ϵ,Einc,frequency,resolution).\n
supplying the background and desired dielectrics, the incident electric field, frequency and resolution\n 
this will create a problem tuple that can be solved using a GPU by solve_problem().\n
"""
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

"""
electric_field = calculate_electric_field_vacuum_to_dielectric(σ,ϵr,res,source,frequency).\n
this will calculate an electric field distribution using the VIE method.\n 
Warning: this is not stable for very high conductivities or high permittivities.\n
"""
function calculate_electric_field_vacuum_to_dielectric(σ,ϵr,res,source,frequency)
    m = getConstants(frequency);
    np,nq,nr   = size(σ);
    nCells  = [np-2,nq-2,nr-2];
    a       = minimum(res)/2f0;
    divωϵim = 1/(1im*m.ω*m.ϵ₀);

    x_axis = 0:res[1]:np*res[1]
    y_axis = 0:res[2]:nq*res[2]
    z_axis = 0:res[3]:nr*res[3]
    
    Dielectric = cellToYeeDielectric(σ,ϵr,x_axis,y_axis,z_axis);

    #dielectric
    χ = setDielectric(Dielectric,m,nCells);
    # operators
    AToE = createSparseDifferenceOperators(nCells,res,m.kb);
    Ig   = createGreensFunctionsRestrictionOperators(nCells);
    G = createGreensFunctions(nCells,res,m.kb);
    #allocate memory for the VIE method
    _,_,a,A,efft,pfft,pifft = allocateSpaceVIE(nCells,[1,0,0]);
    #x,p,r,rt,u,v,q,uq = allocateCGSVIE(nCells);
    
    eI = copy(source);
    #from source in E incident to the actual incident electric field
    JIncToEInc!(eI,a,A,G,χ,Ig,AToE,efft,pfft,pifft,divωϵim);
    
    #create map for VIE 
    v_out = zeros(ComplexF32,length(eI))
    VIE_map = LinearMap{ComplexF32}(x_in -> ETotalMinEScattered_map!(v_out,x_in,a,A,G,χ,Ig,AToE,efft,pfft,pifft,one(ComplexF32)), length(v_out))
    
    #compute total electric field
    x = IterativeSolvers.gmres(VIE_map,eI,verbose=true,restart = 50,maxiter=50)

    return x
end