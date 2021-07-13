"""
yeeDielectric = cellToYeeDielectric(cell_σ, cell_ϵr, x_axis, y_axis, z_axis).\n

Input the conductivity and permittivity on a cell grid (3D Array).\n
And input the node (corners of the cells) coordinates that make of the grid (x,y,z 1D Vectors).\n

The function calculates what the dielectric properties are on the edges of each cell.\n

returns yeeDielectric = (;σˣ, ϵˣ, σʸ, ϵʸ, σᶻ, ϵᶻ).\n
"""
function cell_to_yee_dielectric(cell_σ, cell_ϵr, x, y, z)
    #get voxel lengths
    xLengths = diff(vec(x))
    yLengths = diff(vec(y))
    zLengths = diff(vec(z))

    #get surface areas to average the dielectric over
    x_surface = yLengths*zLengths'
    y_surface = xLengths*zLengths'
    z_surface = xLengths*yLengths'

    #translate the voxel dielectric to the yee cell grid
    σˣ = _x_dielectric(cell_σ, x_surface)
    ϵˣ = _x_dielectric(cell_ϵr, x_surface)
    σʸ = _y_dielectric(cell_σ, y_surface)
    ϵʸ = _y_dielectric(cell_ϵr, y_surface)
    σᶻ = _z_dielectric(cell_σ, z_surface)
    ϵᶻ = _z_dielectric(cell_ϵr, z_surface)
    
    return (;σˣ, ϵˣ, σʸ, ϵʸ, σᶻ, ϵᶻ)
end

function _x_dielectric(A, x_surface)
    typeToUse = eltype(A)
    dimensions = size(A)

    Aˣ = zeros(typeToUse, dimensions[1], dimensions[2]+1, dimensions[3]+1)
    #average the center of the grid 
    Aˣ[:,2:end-1,2:end-1]   = permutedims((permutedims(A[:,1:end-1,1:end-1], [2,3,1]).*x_surface[1:end-1,1:end-1]+
                                            permutedims(A[:,1:end-1,2:end], [2,3,1]).*x_surface[1:end-1,2:end]+
                                            permutedims(A[:,2:end,1:end-1], [2,3,1]).*x_surface[2:end,1:end-1]+
                                            permutedims(A[:,2:end,2:end], [2,3,1]).*x_surface[2:end,2:end])./
                                            (x_surface[1:end-1,1:end-1]+x_surface[1:end-1,2:end]+
                                            x_surface[2:end,1:end-1]+x_surface[2:end,2:end]), [3,1,2])
    #average the sides
    Aˣ[:,1,2:end-1]      = permutedims((permutedims(A[:,1,1:end-1], [2,1]).*x_surface[1,1:end-1]+
                                        permutedims(A[:,1,2:end], [2,1]).*x_surface[1,2:end])./
                                        (x_surface[1,1:end-1]+x_surface[1,2:end]), [2,1])

    Aˣ[:,end,2:end-1]    = permutedims((permutedims(A[:,end,1:end-1], [2,1]).*x_surface[end,1:end-1]+
                                        permutedims(A[:,end,2:end], [2,1]).*x_surface[end,2:end])./
                                        (x_surface[end,1:end-1]+x_surface[end,2:end]), [2,1])

    Aˣ[:,2:end-1,1]      = permutedims((permutedims(A[:,1:end-1,1], [2,1]).*x_surface[1:end-1,1]+
                                        permutedims(A[:,2:end,1], [2,1]).*x_surface[2:end,1])./
                                        (x_surface[1:end-1,1]+x_surface[2:end,1]), [2,1])

    Aˣ[:,2:end-1,end]    = permutedims((permutedims(A[:,1:end-1,end], [2,1]).*x_surface[1:end-1,end]+
                                        permutedims(A[:,2:end,end], [2,1]).*x_surface[2:end,end])./
                                        (x_surface[1:end-1,end]+x_surface[2:end,end]), [2,1])
    #set the edges of the grid
    Aˣ[:,1,1]     = A[:,1,1]
    Aˣ[:,end,1]   = A[:,end,1]
    Aˣ[:,1,end]   = A[:,1,end]
    Aˣ[:,end,end] = A[:,end,end]
    return Aˣ
end

function _y_dielectric(A, y_surface)
    typeToUse = eltype(A)
    dimensions = size(A)

    Aʸ = zeros(typeToUse, dimensions[1]+1, dimensions[2], dimensions[3]+1)
    #average the center of the grid 
    Aʸ[2:end-1,:,2:end-1] = permutedims((permutedims(A[1:end-1,:,1:end-1], [1,3,2]).*y_surface[1:end-1,1:end-1]+
                                         permutedims(A[1:end-1,:,2:end], [1,3,2]).*y_surface[1:end-1,2:end]+
                                         permutedims(A[2:end,:,1:end-1], [1,3,2]).*y_surface[2:end,1:end-1]+
                                         permutedims(A[2:end,:,2:end], [1,3,2]).*y_surface[2:end,2:end])./
                                         (y_surface[1:end-1,1:end-1]+y_surface[1:end-1,2:end]+
                                         y_surface[2:end,1:end-1]+y_surface[2:end,2:end]), [1,3,2])
    #average the sides
    Aʸ[1,:,2:end-1] = permutedims((permutedims(A[1,:,1:end-1], [2,1]).*y_surface[1,1:end-1]+
                                   permutedims(A[1,:,2:end], [2,1]).*y_surface[1,2:end])./
                                   (y_surface[1,1:end-1]+y_surface[1,2:end]), [2,1])

    Aʸ[end,:,2:end-1] = permutedims((permutedims(A[end,:,1:end-1], [2,1]).*y_surface[end,1:end-1]+
                                     permutedims(A[end,:,2:end], [2,1]).*y_surface[end,2:end])./
                                     (y_surface[end,1:end-1]+y_surface[end,2:end]), [2,1])

    Aʸ[2:end-1,:,1] = (A[1:end-1,:,1].*y_surface[1:end-1,1]+
                       A[2:end,:,1].*y_surface[2:end,1])./
                       (y_surface[1:end-1,1]+y_surface[2:end,1])

    Aʸ[2:end-1,:,end] = (A[1:end-1,:,end].*y_surface[1:end-1,end]+
                         A[2:end,:,end].*y_surface[2:end,end])./
                         (y_surface[1:end-1,end]+y_surface[2:end,end])
    #set the edges of the grid
    Aʸ[1,:,1]     = A[1,:,1]
    Aʸ[end,:,1]   = A[end,:,1]
    Aʸ[1,:,end]   = A[1,:,end]
    Aʸ[end,:,end] = A[end,:,end]

    return Aʸ
end

function _z_dielectric(A, z_surface)
    typeToUse = eltype(A)
    dimensions = size(A)

    Aᶻ = zeros(typeToUse, dimensions[1]+1, dimensions[2]+1, dimensions[3])
    #average the center of the grid 
    Aᶻ[2:end-1,2:end-1,:] = (A[1:end-1,1:end-1,:].*z_surface[1:end-1,1:end-1]+
                             A[1:end-1,2:end,:].*z_surface[1:end-1,2:end]+
                             A[2:end,1:end-1,:].*z_surface[2:end,1:end-1]+
                             A[2:end,2:end,:].*z_surface[2:end,2:end])./
                             (z_surface[1:end-1,1:end-1]+z_surface[2:end,2:end]+
                              z_surface[1:end-1,2:end]+z_surface[2:end,1:end-1])
    #side squares
    Aᶻ[1,2:end-1,:] = (A[1,1:end-1,:].*z_surface[1,1:end-1]+
                       A[1,2:end,:].*z_surface[1,2:end])./
                       (z_surface[1,1:end-1] + z_surface[1,2:end])

    Aᶻ[end,2:end-1,:] = (A[end,1:end-1,:].*z_surface[end,1:end-1]+
                         A[end,2:end,:].*z_surface[end,2:end])./
                         (z_surface[end,1:end-1] + z_surface[end,2:end])

    Aᶻ[2:end-1,1,:] = (A[1:end-1,1,:].*z_surface[1:end-1,1]+
                       A[2:end,1,:].*z_surface[2:end,1])./
                       (z_surface[1:end-1,1] + z_surface[2:end,1])

    Aᶻ[2:end-1,end,:] = (A[1:end-1,end,:].*z_surface[1:end-1,end]+
                         A[2:end,end,:].*z_surface[2:end,end])./
                         (z_surface[1:end-1,end] + z_surface[2:end,end])

    #edges
    Aᶻ[1,1,:]     = A[1,1,:]
    Aᶻ[end,1,:]   = A[end,1,:]
    Aᶻ[1,end,:]   = A[1,end,:]
    Aᶻ[end,end,:] = A[end,end,:]

    return Aᶻ
end