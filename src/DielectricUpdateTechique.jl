module DielectricUpdateTechique

#using CUDA
using FFTW
using LinearAlgebra 
using LinearMaps
using SparseArrays


"""
caxpy!(c, a, x, y).\n
calculates c = a*x + y inplace.\n
"""
function caxpy!(c,a,x,y)
    copyto!(c,y)
    axpy!(a,x,c)
end

include("utils/getConstants.jl")
include("utils/computeUpdateMaps.jl")
include("utils/getSandCmatrix.jl")
include("utils/operators.jl")
include("utils/GreensFunctions.jl")
include("utils/initialize.jl")
include("utils/CGSforVIE.jl")
include("utils/cellToYeeDielectric.jl")

end
