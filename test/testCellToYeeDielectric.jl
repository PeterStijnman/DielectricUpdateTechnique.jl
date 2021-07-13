import DielectricUpdateTechique as DUTCH
using Test

x = collect(0:1e-3:3e-3)
y = collect(0:1e-3:3e-3)
z = collect(0:1e-3:3e-3)

vσ = rand(3, 3, 3);
vϵ = rand(3, 3, 3);

yeeTup = DUTCH.cell_to_yee_dielectric(vσ, vϵ, x, y, z)

@test size(yeeTup[:σˣ]) == (3, 4, 4)
@test size(yeeTup[:σʸ]) == (4, 3, 4)
@test size(yeeTup[:σᶻ]) == (4, 4, 3)

@test round(yeeTup[:σˣ][2,2,2], digits=5) == round(sum(vec(vσ[2,1:2,1:2]))/4, digits=5)
@test round(yeeTup[:σˣ][2,1,3], digits=5) == round(sum(vec(vσ[2,1,2:3]))/2, digits=5)
@test round(yeeTup[:σˣ][2,1,4], digits=5) == round(vσ[2,1,3], digits=5)

@test round(yeeTup[:ϵˣ][2,2,2], digits=5) == round(sum(vec(vϵ[2,1:2,1:2]))/4, digits=5)
@test round(yeeTup[:ϵˣ][2,1,3], digits=5) == round(sum(vec(vϵ[2,1,2:3]))/2, digits=5)
@test round(yeeTup[:ϵˣ][2,1,4], digits=5) == round(vϵ[2,1,3], digits=5)


@test round(yeeTup[:σʸ][2,2,2], digits=5) == round(sum(vec(vσ[1:2,2,1:2]))/4, digits=5)
@test round(yeeTup[:σʸ][1,2,3], digits=5) == round(sum(vec(vσ[1,2,2:3]))/2, digits=5)
@test round(yeeTup[:σʸ][1,2,4], digits=5) == round(vσ[1,2,3], digits=5)

@test round(yeeTup[:ϵʸ][2,2,2], digits=5) == round(sum(vec(vϵ[1:2,2,1:2]))/4, digits=5)
@test round(yeeTup[:ϵʸ][1,2,3], digits=5) == round(sum(vec(vϵ[1,2,2:3]))/2, digits=5)
@test round(yeeTup[:ϵʸ][1,2,4], digits=5) == round(vϵ[1,2,3], digits=5)

@test round(yeeTup[:σᶻ][2,2,2], digits=5) == round(sum(vec(vσ[1:2,1:2,2]))/4, digits=5)
@test round(yeeTup[:σᶻ][3,1,2], digits=5) == round(sum(vec(vσ[2:3,1,2]))/2, digits=5)
@test round(yeeTup[:σᶻ][4,1,2], digits=5) == round(vσ[3,1,2], digits=5)

@test round(yeeTup[:ϵᶻ][2,2,2], digits=5) == round(sum(vec(vϵ[1:2,1:2,2]))/4, digits=5)
@test round(yeeTup[:ϵᶻ][3,1,2], digits=5) == round(sum(vec(vϵ[2:3,1,2]))/2, digits=5)
@test round(yeeTup[:ϵᶻ][4,1,2], digits=5) == round(vϵ[3,1,2], digits=5)