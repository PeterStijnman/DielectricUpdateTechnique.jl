import DielectricUpdateTechique as DUTCH
using Test

BG_σ = zeros(Float32,13,13,13);
BG_ϵr = ones(Float32,13,13,13);

radius_sphere = 5f-3;
x_axis = -6f-3:1f-3:6f-3 |> collect |> s -> repeat(s,outer=(1,13,13));
y_axis = -6f-3:1f-3:6f-3 |> collect |> s -> reshape(s,(1,:))|> s -> repeat(s,outer=(13,1,13));
z_axis = -6f-3:1f-3:6f-3 |> collect |> s -> reshape(s,(1,1,:))|> s -> repeat(s,outer=(13,13,1));
r = sqrt.(x_axis.^2 + y_axis.^2 + z_axis.^2);

BG_σ[r .< radius_sphere] .= 0.25;
BG_ϵr[r .< radius_sphere] .= 78;

Update_σ = copy(BG_σ);
Update_ϵr = copy(BG_ϵr);

radius_sphere = 2f-3;
Update_σ[r .< radius_sphere] .= 25;
Update_ϵr[r .< radius_sphere] .= 1;

x_axis = -6.5f-3:1f-3:6.5f-3
y_axis = -6.5f-3:1f-3:6.5f-3
z_axis = -6.5f-3:1f-3:6.5f-3

BGDielectric = DUTCH.cellToYeeDielectric(BG_σ,BG_ϵr,x_axis,y_axis,z_axis);
UpdateDielectric = DUTCH.cellToYeeDielectric(Update_σ,Update_ϵr,x_axis,y_axis,z_axis);

frequency = 300f6;
m = DUTCH.getConstants(frequency);

# incident electric field 
p,q,r   = size(BG_σ);
nCells  = [p-2,q-2,r-2];
res     = [1f-3,1f-3,1f-3];
a       = 5f-4 #resolution/2
divωϵim = 1/(1im*m.ω*m.ϵ₀);
#dielectric
χ = DUTCH.setDielectric(BGDielectric,m,nCells);
# operators
AToE = DUTCH.createSparseDifferenceOperators(nCells,res,m.kb);
Ig   = DUTCH.createGreensFunctionsRestrictionOperators(nCells);
G = DUTCH.createGreensFunctions(nCells,res,m.kb);
#allocate memory for the VIE method
jv,eI,a,A,efft,pfft,pifft = DUTCH.allocateSpaceVIE(nCells,[1,0,0]);
eI[5] = 1f0 + 0f0im; 
#allocate cgs memory
x,p,r,rt,u,v,q,uq = DUTCH.allocateCGSVIE(nCells);
#from source in E incident to the actual incident electric field
DUTCH.JIncToEInc!(eI,a,A,G,χ,Ig,AToE,efft,pfft,pifft,divωϵim);
#compute total electric field
DUTCH.cgs_efield!(eI,efft,G,A,χ,a,AToE,Ig,pfft,pifft,x,p,r,rt,u,v,q,uq,tol=1f-18,maxit = 18)

Einc = x;

# total electric field old fashioned method 
p,q,r   = size(BG_σ);
nCells  = [p-2,q-2,r-2];
res     = [1f-3,1f-3,1f-3];
a       = 5f-4 #resolution/2
divωϵim = 1/(1im*m.ω*m.ϵ₀);
#dielectric
χ = DUTCH.setDielectric(UpdateDielectric,m,nCells);
# operators
AToE = DUTCH.createSparseDifferenceOperators(nCells,res,m.kb);
Ig   = DUTCH.createGreensFunctionsRestrictionOperators(nCells);
G = DUTCH.createGreensFunctions(nCells,res,m.kb);
#allocate memory for the VIE method
jv,eI,a,A,efft,pfft,pifft = DUTCH.allocateSpaceVIE(nCells,[1,0,0]);
eI[5] = 1f0 + 0f0im; 
#allocate cgs memory
x,p,r,rt,u,v,q,uq = DUTCH.allocateCGSVIE(nCells);
#from source in E incident to the actual incident electric field
DUTCH.JIncToEInc!(eI,a,A,G,χ,Ig,AToE,efft,pfft,pifft,divωϵim);
#compute total electric field
DUTCH.cgs_efield!(eI,efft,G,A,χ,a,AToE,Ig,pfft,pifft,x,p,r,rt,u,v,q,uq,tol=1f-18,maxit = 18)

Etot = x;

# total electric field with update 
prob = DUTCH.create_problem(BG_σ,BG_ϵr,Update_σ,Update_ϵr,Einc,frequency,res);

Jsc,Esc = DUTCH.solve_problem(prob);

Etot_update = Einc + Esc;


