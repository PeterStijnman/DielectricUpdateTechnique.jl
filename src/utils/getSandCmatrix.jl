function getS(logicLocations,flagGPU)
    flagGPU ? T = ComplexF32 : T = ComplexF64

    Sxloc = vec(logicLocations[1][:,2:end-1,2:end-1])
    m  = length(Sxloc)
    n  = count(Sxloc)
    x  = sparse(findall(Sxloc),collect(1:n),ones(T,n),m,n)
    xT = sparse(collect(1:n),findall(Sxloc),ones(T,n),n,m)
    
    Syloc = vec(logicLocations[2][2:end-1,:,2:end-1])
    m  = length(Syloc)
    n  = count(Syloc)
    y  = sparse(findall(Syloc),collect(1:n),ones(T,n),m,n)
    yT = sparse(collect(1:n),findall(Syloc),ones(T,n),n,m)
    
    Szloc = vec(logicLocations[3][2:end-1,2:end-1,:])
    m  = length(Szloc)
    n  = count(Szloc)
    z  = sparse(findall(Szloc),collect(1:n),ones(T,n),m,n)
    zT = sparse(collect(1:n),findall(Szloc),ones(T,n),n,m)
    
    if flagGPU
        x  = CUDA.CUSPARSE.CuSparseMatrixCSC(x)
        xT = CUDA.CUSPARSE.CuSparseMatrixCSC(xT)
        y  = CUDA.CUSPARSE.CuSparseMatrixCSC(y)
        yT = CUDA.CUSPARSE.CuSparseMatrixCSC(yT)
        z  = CUDA.CUSPARSE.CuSparseMatrixCSC(z)
        zT = CUDA.CUSPARSE.CuSparseMatrixCSC(zT)
    end

    return (;x,xT,y,yT,z,zT)
end


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

    x = CUDA.CuArray(Cx)
    y = CUDA.CuArray(Cy)
    z = CUDA.CuArray(Cz)

    return (;x,y,z)
end
