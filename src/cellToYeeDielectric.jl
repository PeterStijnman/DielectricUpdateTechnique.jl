"""
yeeDielectric = cellToYeeDielectric(Cellσ,Cellϵ,xAxis,yAxis,zAxis).\n

Input the conductivity and permittivity on a cell grid (3D Array).\n
And input the node coordinates that make of the grid (x,y,z 1D Vectors).\n

The function calculates what the dielectric properties are on the edges of each cell.\n

returns yeeDielectric = (;σˣ, ϵˣ, σʸ, ϵʸ, σᶻ, ϵᶻ).\n
"""
function cellToYeeDielectric(Cellσ,Cellϵ,x,y,z)
    #get voxel lengths
    xLengths = diff(vec(x))
    yLengths = diff(vec(y))
    zLengths = diff(vec(z))

    #get surface areas
    xS = yLengths*zLengths'
    yS = xLengths*zLengths'
    zS = xLengths*yLengths'

    #translate the voxel dielectric to the yee cell grid
    σˣ = _xDielectric(Cellσ,xS)
    ϵˣ = _xDielectric(Cellϵ,xS)
    σʸ = _yDielectric(Cellσ,yS)
    ϵʸ = _yDielectric(Cellϵ,yS)
    σᶻ = _zDielectric(Cellσ,zS)
    ϵᶻ = _zDielectric(Cellϵ,zS)
    
    return (;σˣ, ϵˣ, σʸ, ϵʸ, σᶻ, ϵᶻ)
end

function _xDielectric(A,xS)
    typeToUse = eltype(A)
    dimensions = size(A)
    Aˣ = zeros(typeToUse, dimensions[1], dimensions[2]+1, dimensions[3]+1)
    Aˣ[:,2:end-1,2:end-1]   = permutedims((permutedims(A[:,1:end-1,1:end-1], [2,3,1]).*xS[1:end-1,1:end-1]+
                                              permutedims(A[:,1:end-1,2:end], [2,3,1]).*xS[1:end-1,2:end]+
                                              permutedims(A[:,2:end,1:end-1], [2,3,1]).*xS[2:end,1:end-1]+
                                              permutedims(A[:,2:end,2:end], [2,3,1]).*xS[2:end,2:end])./
                                              (xS[1:end-1,1:end-1]+xS[1:end-1,2:end]+
                                               xS[2:end,1:end-1]+xS[2:end,2:end]), [3,1,2])
    
    Aˣ[:,1,2:end-1]      = permutedims((permutedims(A[:,1,1:end-1], [2,1]).*xS[1,1:end-1]+
        permutedims(A[:,1,2:end], [2,1]).*xS[1,2:end])./
        (xS[1,1:end-1]+xS[1,2:end]), [2,1])

    Aˣ[:,end,2:end-1]    = permutedims((permutedims(A[:,end,1:end-1], [2,1]).*xS[end,1:end-1]+
                                            permutedims(A[:,end,2:end], [2,1]).*xS[end,2:end])./
                                            (xS[end,1:end-1]+xS[end,2:end]), [2,1])

    Aˣ[:,2:end-1,1]      = permutedims((permutedims(A[:,1:end-1,1], [2,1]).*xS[1:end-1,1]+
                                            permutedims(A[:,2:end,1], [2,1]).*xS[2:end,1])./
                                            (xS[1:end-1,1]+xS[2:end,1]), [2,1])

    Aˣ[:,2:end-1,end]    = permutedims((permutedims(A[:,1:end-1,end], [2,1]).*xS[1:end-1,end]+
                                            permutedims(A[:,2:end,end], [2,1]).*xS[2:end,end])./
                                            (xS[1:end-1,end]+xS[2:end,end]), [2,1])
    #edges
    Aˣ[:,1,1]               = A[:,1,1]
    Aˣ[:,end,1]             = A[:,end,1]
    Aˣ[:,1,end]             = A[:,1,end]
    Aˣ[:,end,end]           = A[:,end,end]
    return Aˣ
end

function _yDielectric(A, yS)
    typeToUse = eltype(A)
    dimensions = size(A)
    Aʸ = zeros(typeToUse, dimensions[1]+1, dimensions[2], dimensions[3]+1)
    Aʸ[2:end-1,:,2:end-1]   =   permutedims((permutedims(A[1:end-1,:,1:end-1], [1,3,2]).*yS[1:end-1,1:end-1]+
                                permutedims(A[1:end-1,:,2:end], [1,3,2]).*yS[1:end-1,2:end]+
                                permutedims(A[2:end,:,1:end-1], [1,3,2]).*yS[2:end,1:end-1]+
                                permutedims(A[2:end,:,2:end], [1,3,2]).*yS[2:end,2:end])./
                                (yS[1:end-1,1:end-1]+yS[1:end-1,2:end]+
                                yS[2:end,1:end-1]+yS[2:end,2:end]), [1,3,2])

    Aʸ[1,:,2:end-1] =   permutedims((permutedims(A[1,:,1:end-1], [2,1]).*yS[1,1:end-1]+
                        permutedims(A[1,:,2:end], [2,1]).*yS[1,2:end])./
                        (yS[1,1:end-1]+yS[1,2:end]), [2,1])

    Aʸ[end,:,2:end-1]   =   permutedims((permutedims(A[end,:,1:end-1], [2,1]).*yS[end,1:end-1]+
                            permutedims(A[end,:,2:end], [2,1]).*yS[end,2:end])./
                            (yS[end,1:end-1]+yS[end,2:end]), [2,1])

    Aʸ[2:end-1,:,1] =   (A[1:end-1,:,1].*yS[1:end-1,1]+
                        A[2:end,:,1].*yS[2:end,1])./
                        (yS[1:end-1,1]+yS[2:end,1])

    Aʸ[2:end-1,:,end]   =   (A[1:end-1,:,end].*yS[1:end-1,end]+
                            A[2:end,:,end].*yS[2:end,end])./
                            (yS[1:end-1,end]+yS[2:end,end])
    #edges
    Aʸ[1,:,1]               = A[1,:,1]
    Aʸ[end,:,1]             = A[end,:,1]
    Aʸ[1,:,end]             = A[1,:,end]
    Aʸ[end,:,end]           = A[end,:,end]

    return Aʸ
end

function _zDielectric(A,zS)
    typeToUse = eltype(A)
    dimensions = size(A)
    Aᶻ = zeros(typeToUse, dimensions[1]+1, dimensions[2]+1, dimensions[3])
    
    Aᶻ[2:end-1,2:end-1,:]   = (A[1:end-1,1:end-1,:].*zS[1:end-1,1:end-1]+
                                  A[1:end-1,2:end,:].*zS[1:end-1,2:end]+
                                  A[2:end,1:end-1,:].*zS[2:end,1:end-1]+
                                  A[2:end,2:end,:].*zS[2:end,2:end])./
                                  (zS[1:end-1,1:end-1]+zS[2:end,2:end]+
                                  zS[1:end-1,2:end]+zS[2:end,1:end-1])
    #side squares
    Aᶻ[1,2:end-1,:]         = (A[1,1:end-1,:].*zS[1,1:end-1]+
                                    A[1,2:end,:].*zS[1,2:end])./
                                    (zS[1,1:end-1] + zS[1,2:end])

    Aᶻ[end,2:end-1,:]       = (A[end,1:end-1,:].*zS[end,1:end-1]+
                                    A[end,2:end,:].*zS[end,2:end])./
                                    (zS[end,1:end-1] + zS[end,2:end])

    Aᶻ[2:end-1,1,:]         = (A[1:end-1,1,:].*zS[1:end-1,1]+
                                    A[2:end,1,:].*zS[2:end,1])./
                                    (zS[1:end-1,1] + zS[2:end,1])

    Aᶻ[2:end-1,end,:]       = (A[1:end-1,end,:].*zS[1:end-1,end]+
                                    A[2:end,end,:].*zS[2:end,end])./
                                    (zS[1:end-1,end] + zS[2:end,end])

    #edges
    Aᶻ[1,1,:]               = A[1,1,:]
    Aᶻ[end,1,:]             = A[end,1,:]
    Aᶻ[1,end,:]             = A[1,end,:]
    Aᶻ[end,end,:]           = A[end,end,:]

    return Aᶻ
end