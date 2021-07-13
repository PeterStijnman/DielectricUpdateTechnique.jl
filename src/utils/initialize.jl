"""
_get_number_of_updated_edges(locations_of_change).\n
get the number of edges that are updated as [Nx,Ny,Nz]\n
sum([Nx,Ny,Nz]) == size of the matrix inverse.\n
"""
function _get_number_of_updated_edges(locations_of_change)
    return count(locations_of_change[1]), count(locations_of_change[2]), count(locations_of_change[3])
end

"""
dielectric = _set_dielectric(background_dielectric, constants, number_of_cells).\n
given the background dielectric make the dielectric used for the VIE method.\n
it has the same size as the Green's function operator and is zero padded.\n
for now the background medium is vacuum, but this could be changed to some homogenous dielectric.\n
"""
function _set_dielectric(background_dielectric, constants, number_of_cells)
    T = ComplexF32
    dx, dy, dz = 2 .^(ceil.(Int, log2.(number_of_cells.+2).+1))
    p, q, r = number_of_cells
    #dielectric vacuum
    vac = one(T)
    # dielectric background
    tmpx = (background_dielectric[:σˣ]./(1im*constants.ω*constants.ϵ₀) + background_dielectric[:ϵˣ] .-1)[:,2:end-1,2:end-1]
    tmpy = (background_dielectric[:σʸ]./(1im*constants.ω*constants.ϵ₀) + background_dielectric[:ϵʸ] .-1)[2:end-1,:,2:end-1]
    tmpz = (background_dielectric[:σᶻ]./(1im*constants.ω*constants.ϵ₀) + background_dielectric[:ϵᶻ] .-1)[2:end-1,2:end-1,:]
    
    patient = zeros(T, dx, dy, dz, 3)
    patient[1:p+2,1:q+1,1:r+1,1] = tmpx
    patient[1:p+1,1:q+2,1:r+1,2] = tmpy
    patient[1:p+1,1:q+1,1:r+2,3] = tmpz

    return (;vac, patient)
end

"""
dielectric = _set_dielectric_gpu(background_dielectric, constants, number_of_cells).\n
given the background dielectric make the dielectric used for the VIE method on the gpu.\n
it has the same size as the Green's function operator and is zero padded.\n
for now the background medium is vacuum, but this could be changed to some homogenous dielectric.\n
"""
function _set_dielectric_gpu(background_dielectric, constants, number_of_cells)
    T = ComplexF32
    dx, dy, dz = 2 .^(ceil.(Int, log2.(number_of_cells.+2).+1))
    p, q, r = number_of_cells
    #dielectric vacuum
    vac = one(T)
    # dielectric
    tmpx = (background_dielectric[:σˣ]./(1im*constants.ω*constants.ϵ₀) + background_dielectric[:ϵˣ] .-1)[:,2:end-1,2:end-1]
    tmpy = (background_dielectric[:σʸ]./(1im*constants.ω*constants.ϵ₀) + background_dielectric[:ϵʸ] .-1)[2:end-1,:,2:end-1]
    tmpz = (background_dielectric[:σᶻ]./(1im*constants.ω*constants.ϵ₀) + background_dielectric[:ϵᶻ] .-1)[2:end-1,2:end-1,:]
    
    pat = zeros(T, dx, dy, dz, 3)
    pat[1:p+2,1:q+1,1:r+1,1] = tmpx
    pat[1:p+1,1:q+2,1:r+1,2] = tmpy
    pat[1:p+1,1:q+1,1:r+2,3] = tmpz
    patient = pat |> cu 
    
    return (;vac, patient)
end

"""
_volume_integral_equation_wrapper_explicit_return(sol, sources, vacuum_electric_field, C, vector_to_set_current_density, S, tmp, fft_tmp, G, A, w, dielectric, a, AToE, Ig, scaling, planfft, planifft, x, p, r, rt, u, v, q, uq).\n
function that is used to make a linear map of f(x) = x-CS^TZx.\n
Where Zx is calculated using the VIE method.\n
"""
function _volume_integral_equation_wrapper_explicit_return(
    sol,
    sources,
    vacuum_electric_field,
    C,
    vector_to_set_current_density,
    S,
    fft_tmp,
    G,
    vector_potential_3d,
    dielectric,
    vector_potential_1d,
    AToE,
    Ig,
    scaling,
    planfft,
    planifft,
    x,
    p,
    r,
    rt,
    u,
    v,
    q,
    uq
)
  _volume_integral_equation_wrapper!(sol, sources, vacuum_electric_field, C, vector_to_set_current_density, S, fft_tmp, G, vector_potential_3d, dielectric, vector_potential_1d, AToE, Ig, scaling, planfft, planifft, x, p, r, rt, u, v, q, uq)

  return sol
end

function _volume_integral_equation_wrapper_explicit_return_gpu(
    sol,
    sources,
    vacuum_electric_field,
    C,
    vector_to_set_current_density,
    S,
    fft_tmp,
    G,
    vector_potential_3d,
    dielectric,
    vector_potential_1d,
    AToE,
    Ig,
    scaling,
    planfft,
    planifft,
    x,
    p,
    r,
    rt,
    u,
    v,
    q,
    uq
) 
    input_vector = vector_to_set_current_density |> cu 
    _volume_integral_equation_wrapper!(sol, sources, vacuum_electric_field, C, input_vector, S, fft_tmp, G, vector_potential_3d, dielectric, vector_potential_1d, AToE, Ig, scaling, planfft, planifft, x, p, r, rt, u, v, q, uq)

return collect(sol)
end

"""
b, norm(b) = get_b_vector(CVector, SVector, Einc).\n
Calculates the vector b for the minimaztion of ||b-Ax||_2^2.\n
Einc is the electric field on the yee grid ordered in a vector by Einc_x, Einc_y, Einc_z.\n
Returns the normalized vector b and a float to rescale the output of the minimization.\n
"""
function get_b_vector(C, S, incident_electric_field)
    b = C.*(Transpose(S)*incident_electric_field);
    nb = norm(b);
    bnormalized = b./nb;

    return bnormalized, nb
end

"""
sources, vacuum_electric_field, a, A, fft_tmp, planfft, planifft = _allocate_memory_for_volume_integral_equation(number_of_cells, number_of_updates)\n
allocates arrays that are used for the volume integral equation method.\n
"""
function _allocate_memory_for_volume_integral_equation(
    number_of_cells,
    number_of_updates
)
    T = ComplexF32
    p, q, r = number_of_cells
    dx, dy, dz = 2 .^(ceil.(Int, log2.(number_of_cells.+2).+1))

    xl = (p+2)*(q+1)*(r+1)
    yl = (p+1)*(q+2)*(r+1)
    zl = (p+1)*(q+1)*(r+2)
    tl = xl + yl + zl 

    #source values
    sources = zeros(T, sum(number_of_updates)) 
    #incident electric fields and vector potential    
    vacuum_electric_field, vector_potential_1d = (zeros(T, tl) for _ in 1:2)
    #fft temporaries
    fft_tmp, vector_potential_3d = (zeros(T, dx*dy*dz*3) for _ in 1:2) 
    vector_potential_3d = reshape(vector_potential_3d, dx, dy, dz, 3)
    #planning for fft and ifft
    planfft  = FFTW.plan_fft!(vector_potential_3d, (1, 2, 3))
    planifft = FFTW.plan_ifft!(vector_potential_3d, (1, 2, 3))
    
    return sources, vacuum_electric_field, vector_potential_1d, vector_potential_3d, fft_tmp, planfft, planifft
end

"""
sources, vacuum_electric_field, a, A, fft_tmp, planfft, planifft = _allocate_memory_for_volume_integral_equation_gpu(number_of_cells, number_of_updates)\n
allocates arrays that are used for the volume integral equation method on the gpu.\n
"""
function _allocate_memory_for_volume_integral_equation_gpu(
    number_of_cells,
    number_of_updates
)
    T = ComplexF32
    p, q, r = number_of_cells
    dx, dy, dz = 2 .^(ceil.(Int, log2.(number_of_cells.+2).+1))

    xl = (p+2)*(q+1)*(r+1)
    yl = (p+1)*(q+2)*(r+1)
    zl = (p+1)*(q+1)*(r+2)
    tl = xl + yl + zl 

    #source values
    sources = CUDA.zeros(T, sum(number_of_updates)) 
    #incident electric fields and vector potential    
    vacuum_electric_field, vector_potential_1d = (CUDA.zeros(T, tl) for _ in 1:2)
    #fft temporaries
    fft_tmp, vector_potential_3d = (CUDA.zeros(T, dx*dy*dz*3) for _ in 1:2)
    vector_potential_3d = reshape(vector_potential_3d, dx, dy, dz, 3)
    #planning for fft and ifft
    planfft  = FFTW.plan_fft!(vector_potential_3d, (1, 2, 3))
    planifft = FFTW.plan_ifft!(vector_potential_3d, (1, 2, 3))
    
    return sources, vacuum_electric_field, vector_potential_1d, vector_potential_3d, fft_tmp, planfft, planifft
end

"""
x, p, r, rt, u, v, q, uq = _allocate_memory_cgs(number_of_cells)\n
allocates the memory for the conjugate gradient squared method to solve the VIE.\n
"""
function _allocate_memory_cgs(number_of_cells)
    T = ComplexF32
    #vectors
    np, nq, nr = number_of_cells
    xl = (np+2)*(nq+1)*(nr+1)
    yl = (np+1)*(nq+2)*(nr+1)
    zl = (np+1)*(nq+1)*(nr+2)
    tl = xl + yl + zl 

    x, p, r, rt, u, v, q, uq = (zeros(T, tl) for _ in 1:8)

    return x, p, r, rt, u, v, q, uq
end

"""
x, p, r, rt, u, v, q, uq = _allocate_memory_cgs_gpu(number_of_cells)\n
allocates the memory on the gpu for the conjugate gradient squared method to solve the VIE.\n
"""
function _allocate_memory_cgs_gpu(number_of_cells)
    T = ComplexF32
    #vectors
    np, nq, nr = number_of_cells
    xl = (np+2)*(nq+1)*(nr+1)
    yl = (np+1)*(nq+2)*(nr+1)
    zl = (np+1)*(nq+1)*(nr+2)
    tl = xl + yl + zl 

    x, p, r, rt, u, v, q, uq = (CUDA.zeros(T, tl) for _ in 1:8)

    return x, p, r, rt, u, v, q, uq
end
