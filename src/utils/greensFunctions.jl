"""
r = _create_grid_Greens_function(number_of_cells, resolution)\n
Computes the radius for the domain given by number_of_cells.\n
This radius is used to compute the weakened Greens functions.\n
"""
function _create_grid_Greens_function(number_of_cells, resolution)
    #get max dimensions
    dx,dy,dz = 2 .^(ceil.(Int, log2.(number_of_cells.+2).+1))
    lx = resolution[1].*collect(-dx/2f0:dx/2f0-1)
    ly = resolution[2].*collect(-dy/2f0:dy/2f0-1)
    lz = resolution[3].*collect(-dz/2f0:dz/2f0-1)

    # repeat the axes along the number corresponding dimensions
    X = repeat(lx, outer=(1, dy, dz))
    Y = repeat(reshape(ly,1,:), outer=(dx, 1, dz))
    Z = repeat(reshape(lz,1,1,:), outer=(dx, dy, 1))

    radius = @. sqrt(X^2 + Y^2 + Z^2)

    return radius
end


"""
Internal function that computes the greens function.
"""
function _create_Greens_function(radius, k_b, a)
    # sommerfeld radiation condition
    scalar_G = 3/(k_b*a)^2*(sin(k_b*a)/(k_b*a)-cos(k_b*a))
    # Green's function at r != 0
    G = @. exp.(-1im*k_b*radius)/(4*π*radius)*scalar_G
    # Green's function at r = 0
    G₀ = 3/(4*π*k_b^2*a^3)*((1+1im*k_b*a)*exp(-1im*k_b*a)-1)
    G = circshift(G, floor.(Int, size(G)./2))
    G[1,1,1] = G₀

    return fft(G)
end

"""
G = create_Greens_function(number_of_cells, resolution, k_b).\n
creates the green function on the larger grid.\n
"""
function create_Greens_function(number_of_cells, resolution, k_b)
    a = findmin(resolution)[1]/2
    radius = _create_grid_Greens_function(number_of_cells, resolution)
    G = _create_Greens_function(radius, k_b, a)

    return prod(resolution).*G
end

"""
_Greens_function_times_contrast_source!(vector_potential_1d, G, dielectric, E, w, vector_potential_3d, Ig, fft_tmp, planfft, planifft)\n
Using the Green's function and contrast source we compute the vector potential.\n
a = ∫GdielectricEdV is computed using an FFT.\n
or f = ∫GKdV is computed using an FFT.\n
"""
function _Greens_function_times_contrast_source!(
    vector_potential_1d,
    G,
    dielectric,
    electric_field,
    vector_potential_3d,
    Ig,
    fft_tmp,
    planfft,
    planifft
)
    # move E to larger domain for fft
    mul!(fft_tmp, Transpose(Ig), electric_field) 
    # multiply with the contrast
    vector_potential_3d .= dielectric.*reshape(fft_tmp, size(G)..., 3) 
    # do the fft of the contrast source (dielectric.*E)
    planfft*vector_potential_3d 
    # multiply the green's function with the contrast source in freq domain
    @. vector_potential_3d = G*vector_potential_3d 
    planifft*vector_potential_3d
    # map to smaller domain again
    mul!(vector_potential_1d, Ig, vec(vector_potential_3d)) 
    
    return nothing
end

"""
source_to_incident_field!(source_to_field, vector_potential_1d, vector_potential_3d, G, dielectric, Ig, operator, fft_tmp, planfft, planifft, scaling)\n
Compute the incident field in a vacuum given a source distribution\n
source_to_field will contain the initial source distribution and will contain the incident field when the function is finished\n
For an electric current density distribution to incident electric field:\n
    source_to_field: J\n
    vector_potential_1d: a\n
    vector_potential_3d: A\n
    operator: AtoE\n
    scaling: 1/(1im*ω*ϵ₀)\n
For an electric current density distribution to incident magnetic field:\n
    source_to_field: J\n
    vector_potential_1d: a\n
    vector_potential_3d: A\n 
    operator: AtoH\n
    scaling: 1/(1im*ω*ϵ₀)\n
For a magnetic source distribution to incident electric field:\n
    source_to_field: K\n
    vector_potential_1d: f\n
    vector_potential_3d: F\n
    operator: FtoE\n
    scaling: 1/(1im*ω*μ₀)\n
For a magnetic source distribution to incident magnetic field:\n
    source_to_field: K\n
    vector_potential_1d: f\n
    vector_potential_3d: F\n
    operator: FtoH\n
    scaling: 1/(1im*ω*μ₀)\n
"""
function source_to_incident_field!(
    source_to_field,
    vector_potential_1d,
    vector_potential_3d,
    G,
    dielectric,
    Ig,
    operator,
    fft_tmp,
    planfft,
    planifft,
    scaling
)
    _Greens_function_times_contrast_source!(vector_potential_1d, G, dielectric.vac, source_to_field, vector_potential_3d, Ig, fft_tmp, planfft, planifft)
    mul!(source_to_field, operator, vector_potential_1d)
    @. source_to_field *= scaling

    return nothing
end 

"""
total_minus_scattered_electric_field!(v,total_electric_field,tmp,a,w,A,G,dielectric,Ig,AToE,fft_tmp,planfft,planifft)\n
calculate the incident electric field from the total electric field.\n
This can also be seen as Ax in the context of ||b-Ax||_2^2\n
"""
_total_minus_scattered_electric_field!(v, total_electric_field, vector_potential_1d, vector_potential_3d, G, dielectric, Ig, AToE, fft_tmp, planfft, planifft) = _total_minus_scattered_electric_field!(v, total_electric_field, vector_potential_1d, vector_potential_3d, G, dielectric, Ig, AToE, fft_tmp, planfft, planifft, 1)
function _total_minus_scattered_electric_field!(
    v,
    total_electric_field,
    vector_potential_1d,
    vector_potential_3d,
    G,
    dielectric,
    Ig,
    AToE,
    fft_tmp,
    planfft,
    planifft,
    scalar
)
    _Greens_function_times_contrast_source!(vector_potential_1d, G, dielectric.patient, total_electric_field, vector_potential_3d, Ig, fft_tmp, planfft, planifft)
    copyto!(v, total_electric_field)
    mul!(v, AToE, vector_potential_1d, -scalar, scalar)

    return nothing
end

"""
_total_minus_scattered_electric_field_for_residual!(v,total_electric_field,tmp,vector_potential_3d,w,vector_potential_3d,G,dielectric,Ig,AToE,fft_tmp,planfft,planifft,scalar)\n
similar to total_minus_scattered_electric_field!(), but now r-AΔx is computed\n
That is the change in the residual as a result of the update in x\n
"""
function _total_minus_scattered_electric_field_for_residual!(
    v,
    total_electric_field,
    vector_potential_1d,
    vector_potential_3d,
    G,
    dielectric,
    Ig,
    AToE,
    fft_tmp,
    planfft,
    planifft,
    scalar
)
    _Greens_function_times_contrast_source!(vector_potential_1d, G, dielectric.patient, total_electric_field, vector_potential_3d, Ig, fft_tmp, planfft, planifft)
    mul!(total_electric_field, AToE, vector_potential_1d, -scalar, scalar)
    axpy!(one(ComplexF32), total_electric_field, v)

    return nothing
end

"""
_set_source_values_in_domain!(sources, incident_electric_field, v, S)\n
will set the current density values to -source_values\n
Using the S matrix these are placed into the incident electric field.\n
"""
function _set_source_values_in_domain!(sources, incident_electric_field, source_values, S)
    # set source values to the vector we are multiplying A with
    sources .= -one(ComplexF32).*source_values
    # set the source values into the incident electric field (will be equal to Jinc before the function JIncToEinc!)
    mul!(incident_electric_field, S, sources)

    return nothing
end

"""
library_matrix_times_current_density!(current_density, nUpdates, sources, incident_electric_field, S, tmp, fft_tmp, G, vector_potential_3d, w, dielectric, vector_potential_3d, AToE, Ig, scaling, pfft, pifft, x, p, r, rt, u, v, q, uq).\n
calculate the scattered electric field given the current density.\n
"""
function library_matrix_times_current_density!(
    current_density,
    sources,
    incident_electric_field,
    S,
    fft_tmp,
    G,
    vector_potential_3d,
    dielectric,
    vector_potential_1d,
    AToE,
    Ig,
    scaling,
    pfft,
    pifft,
    x,
    p,
    r,
    rt,
    u,
    v,
    q,
    uq
)
    source_values_input = -current_density

    _set_source_values_in_domain!(sources, incident_electric_field, source_values_input, S)
    #from source in E incident to the actual incident electric field
    source_to_incident_field!(incident_electric_field, vector_potential_1d, vector_potential_3d, G, dielectric, Ig, AToE, fft_tmp, pfft, pifft, scaling)
    #compute total electric field
    cgs_efield!(incident_electric_field, fft_tmp, G, vector_potential_3d, dielectric, vector_potential_1d, AToE, Ig, pfft, pifft, x, p, r, rt, u, v, q, uq)

    return x
end 

"""
library_matrix_times_current_density_gpu!(current_density, nUpdates, sources, incident_electric_field, S, tmp, fft_tmp, G, vector_potential_3d, w, dielectric, vector_potential_3d, AToE, Ig, scaling, pfft, pifft, x, p, r, rt, u, v, q, uq).\n
calculate the scattered electric field given the current density.\n
"""
function library_matrix_times_current_density_gpu!(
    current_density,
    sources,
    incident_electric_field,
    S,
    fft_tmp,
    G,
    vector_potential_3d,
    dielectric,
    vector_potential_1d,
    AToE,
    Ig,
    scaling,
    pfft,
    pifft,
    x,
    p,
    r,
    rt,
    u,
    v,
    q,
    uq
)
    source_values_input = -current_density |> cu 

    _set_source_values_in_domain!(sources, incident_electric_field, source_values_input, S)
    #from source in E incident to the actual incident electric field
    source_to_incident_field!(incident_electric_field, vector_potential_1d, vector_potential_3d, G, dielectric, Ig, AToE, fft_tmp, pfft, pifft, scaling)
    #compute total electric field
    cgs_efield!(incident_electric_field, fft_tmp, G, vector_potential_3d, dielectric, vector_potential_1d, AToE, Ig, pfft, pifft, x, p, r, rt, u, v, q, uq)

    return x
end #TODO: check if cpu version/collect are required



"""
_volume_integral_equation_wrapper!(output, sources, incident_electric_field, C, input_vector, S, tmp, fft_tmp, G, vector_potential_3d, w, dielectric, vector_potential_3d, AToE, Ig, scaling, pfft, pifft, x, p, r, rt, u, v, q, uq)\n
In place version to calculate Av = (I-CSZ)v = v - CSZv.\n
-CSZv is calculated using the VIE method by setting the sources to -v and multiplying the edges with the corresponding C value.
"""
function _volume_integral_equation_wrapper!(
    output,
    sources,
    incident_electric_field,
    C,
    input_vector,
    S,
    fft_tmp,
    G,
    vector_potential_3d,
    dielectric,
    vector_potential_1d,
    AToE,
    Ig,
    scaling,
    pfft,
    pifft,
    x,
    p,
    r,
    rt,
    u,
    v,
    q,
    uq
)
    # set sources in the vacuum to the correct values 
    _set_source_values_in_domain!(sources, incident_electric_field, input_vector, S)
    # calculate the incident electric field inside the vacuum
    source_to_incident_field!(incident_electric_field, vector_potential_1d, vector_potential_3d, G, dielectric, Ig, AToE, fft_tmp, pfft, pifft, scaling)
    # solve for the field inside the patient anatomy
    cgs_efield!(incident_electric_field, fft_tmp, G, vector_potential_3d, dielectric, vector_potential_1d, AToE, Ig, pfft, pifft, x, p, r, rt, u, v, q, uq)
    # calculate the (v - CSZv) vector for the minimization process    
    mul!(output, Transpose(S), x)
    @. output *= C
    # output = input_vector - CSZinput_vector
    axpy!(one(ComplexF32), input_vector, output)
    
    return nothing 
end

"""
total_minus_scattered_electric_field_map!(v, total_electric_field, vector_potential_3d, vector_potential_3d, G, dielectric, Ig, AToE, fft_tmp, planfft, planifft, scalar).\n
used as a map to solve for the total electric field.\n
"""
function total_minus_scattered_electric_field_map!(
    v,
    total_electric_field,
    vector_potential_1d,
    vector_potential_3d,
    G,
    dielectric,
    Ig,
    AToE,
    fft_tmp,
    planfft,
    planifft,
    scalar
)
    copyto!(v, total_electric_field)
    _Greens_function_times_contrast_source!(vector_potential_1d, G, dielectric.patient, v, vector_potential_3d, Ig, fft_tmp, planfft, planifft)
    mul!(v, AToE, vector_potential_1d, -scalar, scalar)
    
    return v
end
