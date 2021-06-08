import DielectricUpdateTechique as DUTCH
using Test

x = collect(0:1e-3:3e-3)
y = collect(0:1e-3:3e-3)
z = collect(0:1e-3:3e-3)

vσ = rand(3, 3, 3);
vϵ = rand(3, 3, 3);

bg = DUTCH.cellToYeeDielectric(vσ, vϵ, x, y, z)

vσ[1:2,2,2] = 2 .* vσ[1:2,2,2];

imp = DUTCH.cellToYeeDielectric(vσ, vϵ, x, y, z)

Δ, loc = DUTCH.computeUpdateMaps(bg,imp)

@test round(Δ[:σˣ][2,2,2], digits=3) == round(imp[:σˣ][2,2,2] - bg[:σˣ][2,2,2], digits=3)
@test round(Δ[:σˣ][1,2,2], digits=3) == round(imp[:σˣ][1,2,2] - bg[:σˣ][1,2,2], digits=3)

@test round(Δ[:σʸ][2,2,2], digits=3) == round(imp[:σʸ][2,2,2] - bg[:σʸ][2,2,2], digits=3)
@test round(Δ[:σʸ][1,2,2], digits=3) == round(imp[:σʸ][1,2,2] - bg[:σʸ][1,2,2], digits=3)

@test round(Δ[:σᶻ][2,2,2], digits=3) == round(imp[:σᶻ][2,2,2] - bg[:σᶻ][2,2,2], digits=3)
@test round(Δ[:σᶻ][1,2,2], digits=3) == round(imp[:σᶻ][1,2,2] - bg[:σᶻ][1,2,2], digits=3)

# there are 20 edges to be updated for 2 neighbouring cells
@test sum(vec(loc[:updateLocˣ])) + sum(vec(loc[:updateLocʸ])) + sum(vec(loc[:updateLocᶻ])) == 20


vσ = rand(3, 3, 3);
vϵ = rand(3, 3, 3);

bg = DUTCH.cellToYeeDielectric(vσ, vϵ, x, y, z)

vσ[1,2,2] = 2 .* vσ[1,2,2];
vσ[3,2,2] = 2 .* vσ[3,2,2];

imp = DUTCH.cellToYeeDielectric(vσ, vϵ, x, y, z)

Δ, loc = DUTCH.computeUpdateMaps(bg,imp)
# there are 24 edges to be updated for 2 non neighbouring cells
@test sum(vec(loc[:updateLocˣ])) + sum(vec(loc[:updateLocʸ])) + sum(vec(loc[:updateLocᶻ])) == 24

# test for getS(_gpu) and getC 
S = DUTCH.getS(loc)
size(S.N)
