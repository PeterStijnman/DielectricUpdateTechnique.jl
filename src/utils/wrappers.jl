"""
prob = create_problem(background_σ, background_ϵ, desired_σ, desired_ϵ, incident_electric_field, frequency, resolution).\n
supplying the background and desired dielectrics, the incident electric field, frequency and resolution\n 
this will create a problem tuple that can be solved by solve_problem().\n
"""
function create_problem(
    background_σ,
    background_ϵ,
    desired_σ,
    desired_ϵ,
    incident_electric_field,
    frequency,
    resolution
)

    constants = get_constants(frequency)

    p, q, r = size(background_σ);
    number_of_cells = [p-2, q-2, r-2];
    a = minimum(resolution)/2;
    scaling = -1im*constants.μ₀*constants.ω/(constants.kb^2);
    

    x_axis = 0f0:resolution[1]:p*resolution[1];
    y_axis = 0f0:resolution[2]:q*resolution[2];
    z_axis = 0f0:resolution[3]:r*resolution[3];

    background_dielectric = cell_to_yee_dielectric(background_σ, background_ϵ, x_axis, y_axis, z_axis);
    desired_dielectric = cell_to_yee_dielectric(desired_σ, desired_ϵ, x_axis, y_axis, z_axis);
    difference_in_dielectric, locations_of_change = get_update_maps(background_dielectric, desired_dielectric);

    #get the number of edges that need to be updated
    number_of_changes = _get_number_of_updated_edges(locations_of_change);
    
    #operators
    AToE = _electric_vector_potential_to_electric_field_operator(number_of_cells, resolution, constants.kb);
    Ig   = _create_Greens_functions_restriction_operators(number_of_cells);
    G    = create_Greens_function(number_of_cells, resolution, constants.kb);


    dielectric = _set_dielectric(background_dielectric, constants, number_of_cells);
    
    #allocate memory for inplace operations
    sources, vacuum_electric_field, vector_potential_1d, vector_potential_3d, fft_tmp, pfft, pifft = _allocate_memory_for_volume_integral_equation(number_of_cells, number_of_changes);
    x, p, r, rt, u, v, q, uq = _allocate_memory_cgs(number_of_cells);
    #get the sparse S matrix for each field component
    S = get_S_matrix(locations_of_change); 
    C = get_C_matrix(difference_in_dielectric, constants);

    b, nb = get_b_vector(C, S, incident_electric_field)
    
    output_volume_integral_equation = zeros(ComplexF32, sum(number_of_changes));

    volume_integral_equation_map =  LinearMap{ComplexF32}(vecxyz-> _volume_integral_equation_wrapper_explicit_return(output_volume_integral_equation, sources, vacuum_electric_field, C, vecxyz, S, fft_tmp, G, vector_potential_3d, dielectric, vector_potential_1d, AToE, Ig, scaling, pfft, pifft, x, p, r, rt, u, v, q, uq), sum(number_of_changes))
    scattered_field_map = LinearMap{ComplexF32}(current_density -> library_matrix_times_current_density!(current_density, sources, vacuum_electric_field, S, fft_tmp, G, vector_potential_3d, dielectric, vector_potential_1d, AToE, Ig, scaling, pfft, pifft, x, p, r, rt, u, v, q, uq), sum(number_of_changes))

    return (g_map = volume_integral_equation_map, b_vector = b, norm_b = nb, scattered_electric_field_map = scattered_field_map)
end

"""
prob = create_problem(background_σ, background_ϵ, desired_σ, desired_ϵ, incident_electric_field, frequency, resolution).\n
supplying the background and desired dielectrics, the incident electric field, frequency and resolution\n 
this will create a problem tuple that can be solved using a GPU by solve_problem().\n
"""
function create_problem_gpu(
    background_σ,
    background_ϵ,
    desired_σ,
    desired_ϵ,
    incident_electric_field,
    frequency,
    resolution
)

    constants = get_constants(frequency)
    
    p, q, r = size(background_σ);
    number_of_cells = [p-2, q-2, r-2];
    a = minimum(resolution)/2;
    
    scaling = -1im*constants.μ₀*constants.ω/(constants.kb^2);

    x_axis = 0f0:resolution[1]:p*resolution[1];
    y_axis = 0f0:resolution[2]:q*resolution[2];
    z_axis = 0f0:resolution[3]:r*resolution[3];

    background_dielectric = cell_to_yee_dielectric(background_σ, background_ϵ, x_axis, y_axis, z_axis);
    desired_dielectric = cell_to_yee_dielectric(desired_σ, desired_ϵ, x_axis, y_axis, z_axis);
    difference_in_dielectric, locations_of_change = get_update_maps(background_dielectric, desired_dielectric);

    #get the number of edges that need to be updated
    number_of_changes = _get_number_of_updated_edges(locations_of_change);
    
    #operators
    AToE = _electric_vector_potential_to_electric_field_operator(number_of_cells, resolution, constants.kb) |> CUDA.CUSPARSE.CuSparseMatrixCSC;
    Ig   = _create_Greens_functions_restriction_operators_gpu(number_of_cells);
    G    = create_Greens_function(number_of_cells, resolution, constants.kb) |> cu;


    dielectric = _set_dielectric_gpu(background_dielectric, constants, number_of_cells);
    
    #allocate memory for inplace operations
    sources, vacuum_electric_field, vector_potential_1d, vector_potential_3d, fft_tmp, pfft, pifft = _allocate_memory_for_volume_integral_equation_gpu(number_of_cells, number_of_changes);
    x, p, r, rt, u, v, q, uq = _allocate_memory_cgs_gpu(number_of_cells);
    #get the sparse S matrix for each field component
    S = get_S_matrix_gpu(locations_of_change); 
    C = get_C_matrix(difference_in_dielectric, constants) |> cu;

    b, nb = get_b_vector(C, S, incident_electric_field)
    
    outputVIE = CUDA.zeros(ComplexF32, sum(number_of_changes));

    volume_integral_equation_map =  LinearMap{ComplexF32}(vecxyz-> _volume_integral_equation_wrapper_explicit_return_gpu(outputVIE, sources, vacuum_electric_field, C, vecxyz, S, fft_tmp, G, vector_potential_3d, dielectric, vector_potential_1d, AToE, Ig, scaling, pfft, pifft, x, p, r, rt, u, v, q, uq), sum(number_of_changes))
    scattered_field_map = LinearMap{ComplexF32}(current_density -> library_matrix_times_current_density_gpu!(current_density, sources, vacuum_electric_field, S, fft_tmp, G, vector_potential_3d, dielectric, vector_potential_1d, AToE, Ig, scaling, pfft, pifft, x, p, r, rt, u, v, q, uq), sum(number_of_changes))

    return (g_map = volume_integral_equation_map, b_vector = b, norm_b = nb, scattered_electric_field_map = scattered_field_map)
end

"""
scattered_current_density, scattered_electric_field = solve_problem(problem; verbose = true, max_iterations = 100, restart = 50, tol = 1f-4).\n
the problem is created by calling the create_problem function.\n 
the problem is solved using IterativeSolvers gmres implementation.\n 
it can also be solved by other solvers if you believe they are more accurate or faster.\n 
personally, I found this solver to be both fast, stable, and finding the correct solution.\n 
Some standard arguments are supplied, but can be tweaked according to preference/complexity of the problem.\n
"""
function solve_problem(
    problem;
    verbose = true,
    max_iterations = 100,
    restart = 50,
    tol = 1f-4
)
    # solve the matrix inverse to obtain the scattered current density
    scattered_current_density = IterativeSolvers.gmres(problem.g_map, problem.b_vector, verbose=verbose, initially_zero=true, reltol=tol, restart = restart, maxiter=max_iterations);
    # the vector b is normalized to a norm of 1, thus we rescale the solution since the problem is linear.
    scattered_current_density .*= problem.norm_b
    # we calculate the scattered electric field that is created by the dielectric update that is performed
    scattered_electric_field  = problem.scattered_electric_field_map*scattered_current_density
    return scattered_current_density, scattered_electric_field
end

"""
electric_field = calculate_electric_field_vacuum_to_dielectric(σ, ϵr, resolution, source, frequency)\n
this will calculate an electric field distribution using the VIE method.\n 
Warning: this is not stable for very high conductivities or high permittivities.\n
"""
function calculate_electric_field_vacuum_to_dielectric(
    σ,
    ϵr,
    resolution,
    source,
    frequency
)
    constants = get_constants(frequency);
    np, nq, nr   = size(σ);
    number_of_cells  = [np-2, nq-2, nr-2];
    a       = minimum(resolution)/2f0;
    divωϵim = 1/(1im*constants.ω*constants.ϵ₀);

    x_axis = 0:resolution[1]:np*resolution[1]
    y_axis = 0:resolution[2]:nq*resolution[2]
    z_axis = 0:resolution[3]:nr*resolution[3]
    
    yee_dielectric = cell_to_yee_dielectric(σ, ϵr, x_axis, y_axis, z_axis);

    dielectric = _set_dielectric(yee_dielectric, constants, number_of_cells);
    # operators
    AToE = _electric_vector_potential_to_electric_field_operator(number_of_cells, resolution, constants.kb);
    Ig   = _create_Greens_functions_restriction_operators(number_of_cells);
    G = create_Greens_function(number_of_cells, resolution, constants.kb);
    #allocate memory for the VIE method
    _, _, vector_potential_1d, vector_potential_3d, fft_tmp, pfft, pifft = _allocate_memory_for_volume_integral_equation(number_of_cells, [1,0,0]);
    
    incident_electric_field = copy(source);
    #from source in E incident to the actual incident electric field
    source_to_incident_field!(incident_electric_field, vector_potential_1d, vector_potential_3d, G, dielectric, Ig, AToE, fft_tmp, pfft, pifft, divωϵim);
    
    #create map for VIE 
    v_out = zeros(ComplexF32, length(incident_electric_field))
    volume_integral_equation_map = LinearMap{ComplexF32}(x_in -> total_minus_scattered_electric_field_map!(v_out, x_in, vector_potential_1d, vector_potential_3d, G, dielectric, Ig, AToE, fft_tmp, pfft, pifft, one(ComplexF32)), length(v_out))
    
    #solve for the total electric field
    total_electric_field = IterativeSolvers.gmres(volume_integral_equation_map, incident_electric_field, verbose=true, restart = 50, maxiter=50)

    return total_electric_field
end