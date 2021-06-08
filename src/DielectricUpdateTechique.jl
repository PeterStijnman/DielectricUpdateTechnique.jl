module DielectricUpdateTechique

using LinearAlgebra, SparseArrays

"""
caxpy!(c, a, x, y).\n
calculates c = a*x + y inplace.\n
"""
function caxpy!(c,a,x,y)
    copyto!(c,y)
    axpy!(a,x,c)
end

# some basic utilities
include("utils/getConstants.jl")
include("utils/computeUpdateMaps.jl")
include("utils/getSandCmatrix.jl")

# inputs for method are bg Dielectric, new dielectric, location of new dielectric, incident electric field
include("cellToYeeDielectric.jl")

end
