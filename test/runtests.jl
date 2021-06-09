using Core: ComplexF32
using DielectricUpdateTechique
using Test

@testset "DielectricUpdateTechique.jl" begin
    # Write your tests here.
    include("testRemainingUtils.jl")
    include("testGetConstants.jl")
    include("testCellToYeeDielectric.jl")
    include("testComputeUpdateMaps.jl")
    include("testOperators.jl")
end

