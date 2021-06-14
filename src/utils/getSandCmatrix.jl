"""
S = getS(logicLocations).\n
given a boolean map with all the x, y, z edges that are updated it returns the sparse support matrix S.\n
this is used to map quantities from the large domain (enitre simulation domain) to the update domain.\n
S = (N = S, T = transpose of S).\n
WARNING: any updates on the edges are not taken into account.\n
"""
function getS(logicLocations)
    T = ComplexF32

    Sxloc = vec(logicLocations[1][:,2:end-1,2:end-1])
    m  = length(Sxloc)
    n  = count(Sxloc)
    x  = sparse(findall(Sxloc),collect(1:n),ones(T,n),m,n)
    
    Syloc = vec(logicLocations[2][2:end-1,:,2:end-1])
    m  = length(Syloc)
    n  = count(Syloc)
    y  = sparse(findall(Syloc),collect(1:n),ones(T,n),m,n)
    
    Szloc = vec(logicLocations[3][2:end-1,2:end-1,:])
    m  = length(Szloc)
    n  = count(Szloc)
    z  = sparse(findall(Szloc),collect(1:n),ones(T,n),m,n)
    
    return blockdiag(x,y,z)
end

"""
S = getS_gpu(logicLocations).\n
given a boolean map with all the x, y, z edges that are updated it returns the sparse support matrix S on the GPU.\n
this is used to map quantities from the large domain (enitre simulation domain) to the update domain.\n
S = (N = S, T = transpose of S).\n
WARNING: any updates on the edges are not taken into account.\n
"""
function getS_gpu(logicLocations)
    T = ComplexF32

    Sxloc = vec(logicLocations[1][:,2:end-1,2:end-1])
    m  = length(Sxloc)
    n  = count(Sxloc)
    x  = sparse(findall(Sxloc),collect(1:n),ones(T,n),m,n)
    
    Syloc = vec(logicLocations[2][2:end-1,:,2:end-1])
    m  = length(Syloc)
    n  = count(Syloc)
    y  = sparse(findall(Syloc),collect(1:n),ones(T,n),m,n)
    
    Szloc = vec(logicLocations[3][2:end-1,2:end-1,:])
    m  = length(Szloc)
    n  = count(Szloc)
    z  = sparse(findall(Szloc),collect(1:n),ones(T,n),m,n)
    
    S  = blockdiag(x,y,z) |> CUDA.CUSPARSE.CuSparseMatrixCSC
    return S
end
#TODO: there is probably a type stable way to fuse getS and getS_gpu

"""
C = getC(UpdateDielectric,m)\n.
returns the diagonal matrix C as a vector.\n 
UpdateDielectric is the change in the dielectric (new dielectric - background dielectric).\n 
m = getConstants(freq).\n 
"""
function getC(UpdateDielectric,m)
    # dielectric update in the entire domain
    CxFull = vec(UpdateDielectric[:σˣ] + m[:ω]*m[:ϵ₀]im*UpdateDielectric[:ϵˣ])
    CyFull = vec(UpdateDielectric[:σʸ] + m[:ω]*m[:ϵ₀]im*UpdateDielectric[:ϵʸ])
    CzFull = vec(UpdateDielectric[:σᶻ] + m[:ω]*m[:ϵ₀]im*UpdateDielectric[:ϵᶻ])

    # find all non zeros
    Cxloc = findall(abs.(CxFull) .> 0.0)
    Cyloc = findall(abs.(CyFull) .> 0.0)
    Czloc = findall(abs.(CzFull) .> 0.0)

    # conversion to gpu
    x = ComplexF32.(CxFull[Cxloc])
    y = ComplexF32.(CyFull[Cyloc])
    z = ComplexF32.(CzFull[Czloc])

    return vcat(x,y,z)
end

