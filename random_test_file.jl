using Revise
using BenchmarkTools
import DielectricUpdateTechique as DUTCH

bg = (σˣ = ones(3,3,3),σʸ = ones(3,3,3),σᶻ= ones(3,3,3),ϵˣ = ones(3,3,3),ϵʸ = ones(3,3,3),ϵᶻ= ones(3,3,3));
imp = (σˣ = ones(3,3,3),σʸ = ones(3,3,3),σᶻ= 2 .* ones(3,3,3),ϵˣ = ones(3,3,3),ϵʸ = ones(3,3,3),ϵᶻ= ones(3,3,3));

Δχ, loc = DUTCH.computeUpdateMaps(bg,imp);



@code_warntype DUTCH.computeUpdateMaps(bg,imp);

@btime DUTCH.computeUpdateMaps(bg,imp);

using CUDA
using LinearAlgebra
using SparseArrays

A = spdiagm(10,10,0=>ones(10))

C = (;A)
typeof(C)
CUDA.CUSPARSE.CuSparseMatrixCSC.(C)