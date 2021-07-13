"""
S = get_S_matrix(locations_of_change).\n
given a boolean map with all the x, y, z edges that are updated it returns the sparse support matrix S.\n
this is used to map quantities from the large domain (enitre simulation domain) to the update domain.\n
S = (N = S, T = transpose of S).\n
WARNING: any updates on the edges are not taken into account.\n
"""
function get_S_matrix(locations_of_change)
    T = ComplexF32

    Sxloc = vec(locations_of_change[1][:,2:end-1,2:end-1])
    m  = length(Sxloc)
    n  = count(Sxloc)
    x  = sparse(findall(Sxloc), collect(1:n), ones(T, n), m, n)
    
    Syloc = vec(locations_of_change[2][2:end-1,:,2:end-1])
    m  = length(Syloc)
    n  = count(Syloc)
    y  = sparse(findall(Syloc), collect(1:n), ones(T, n), m, n)
    
    Szloc = vec(locations_of_change[3][2:end-1,2:end-1,:])
    m  = length(Szloc)
    n  = count(Szloc)
    z  = sparse(findall(Szloc), collect(1:n), ones(T, n), m, n)
    
    return blockdiag(x, y, z)
end

"""
S = get_S_matrix_gpu(locations_of_change).\n
given a boolean map with all the x, y, z edges that are updated it returns the sparse support matrix S on the GPU.\n
this is used to map quantities from the large domain (enitre simulation domain) to the update domain.\n
S = (N = S, T = transpose of S).\n
WARNING: any updates on the edges are not taken into account.\n
"""
function get_S_matrix_gpu(locations_of_change)
    T = ComplexF32

    Sxloc = vec(locations_of_change[1][:,2:end-1,2:end-1])
    m  = length(Sxloc)
    n  = count(Sxloc)
    x  = sparse(findall(Sxloc), collect(1:n), ones(T, n), m, n)
    
    Syloc = vec(locations_of_change[2][2:end-1,:,2:end-1])
    m  = length(Syloc)
    n  = count(Syloc)
    y  = sparse(findall(Syloc), collect(1:n), ones(T, n), m, n)
    
    Szloc = vec(locations_of_change[3][2:end-1,2:end-1,:])
    m  = length(Szloc)
    n  = count(Szloc)
    z  = sparse(findall(Szloc), collect(1:n), ones(T, n), m, n)
    
    S  = blockdiag(x, y, z) |> CUDA.CUSPARSE.CuSparseMatrixCSC
    return S
end

"""
C = get_C_matrix(difference_in_dielectric, constants)\n.
returns the diagonal matrix C as a vector.\n 
difference_in_dielectric is the change in the dielectric (new dielectric - background dielectric).\n 
constants = get_constants(freq).\n 
"""
function get_C_matrix(difference_in_dielectric, constants)
    # dielectric update in the entire domain
    CxFull = vec(difference_in_dielectric[:σˣ] + constants[:ω]*constants[:ϵ₀]im*difference_in_dielectric[:ϵˣ])
    CyFull = vec(difference_in_dielectric[:σʸ] + constants[:ω]*constants[:ϵ₀]im*difference_in_dielectric[:ϵʸ])
    CzFull = vec(difference_in_dielectric[:σᶻ] + constants[:ω]*constants[:ϵ₀]im*difference_in_dielectric[:ϵᶻ])

    # find all non zeros
    Cxloc = findall(abs.(CxFull) .> 0.0)
    Cyloc = findall(abs.(CyFull) .> 0.0)
    Czloc = findall(abs.(CzFull) .> 0.0)

    # conversion to gpu
    x = ComplexF32.(CxFull[Cxloc])
    y = ComplexF32.(CyFull[Cyloc])
    z = ComplexF32.(CzFull[Czloc])

    return vcat(x, y, z)
end

