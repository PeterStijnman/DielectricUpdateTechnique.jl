module DielectricUpdateTechique

using LinearAlgebra
# Write your package code here.

# some basic utilities
include("utils/vectorComponentsOperations.jl")
include("utils/getConstants.jl")
include("utils/computeUpdateMaps.jl")

# inputs for method are bg Dielectric, new dielectric, location of new dielectric, incident electric field
include("cellToYeeDielectric.jl")

end
