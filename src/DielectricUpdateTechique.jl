module DielectricUpdateTechique

using CUDA
using FFTW
using LinearAlgebra 
using LinearMaps
using SparseArrays
using IterativeSolvers

"""
caxpy!(c, a, x, y).\n
calculates c = a*x + y inplace.\n
"""
function caxpy!(c,a,x,y)
    copyto!(c,y)
    axpy!(a,x,c)
end

include("utils/cellToYeeDielectric.jl")
include("utils/CGSforVIE.jl")
include("utils/computeUpdateMaps.jl")
include("utils/getConstants.jl")
include("utils/getSandCmatrix.jl")
include("utils/greensFunctions.jl")
include("utils/initialize.jl")
include("utils/operators.jl")
include("utils/wrappers.jl")

end
