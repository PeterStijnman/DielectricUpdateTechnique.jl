using Core: ComplexF32
import DielectricUpdateTechique as DUTCH
using Test

frequency = 300f6;

# create grid for the problem
res = [1f-3,1f-3,1f-3];
x_axis = -6*res[1]:res[1]:6*res[1] |> collect |> s -> repeat(s,outer=(1,13,13));
y_axis = -6*res[2]:res[2]:6*res[2] |> collect |> s -> reshape(s,(1,:))|> s -> repeat(s,outer=(13,1,13));
z_axis = -6*res[3]:res[3]:6*res[3] |> collect |> s -> reshape(s,(1,1,:))|> s -> repeat(s,outer=(13,13,1));
radius = sqrt.(x_axis.^2 + y_axis.^2 + z_axis.^2);

# create background dielectric (sphere within vacuum)
BG_σ = zeros(Float32,13,13,13);
BG_ϵr = ones(Float32,13,13,13);

# create the sphere with σ = 0.25 S/m and ϵr = 78
radius_sphere = 5f-3;
BG_σ[radius .< radius_sphere] .= 0.25;
BG_ϵr[radius .< radius_sphere] .= 78;

# create the scenario we want to obtain 
# a smaller sphere within the background dielectric
Update_σ = copy(BG_σ);
Update_ϵr = copy(BG_ϵr);

# the smaller sphere has σ = 4 S/m and ϵr = 20
radius_sphere = 2f-3;
Update_σ[radius .< radius_sphere] .= 4;
Update_ϵr[radius .< radius_sphere] .= 20;

# define the source to calculate the incidenct electric field
p,q,r   = size(BG_σ);
source = zeros(ComplexF32,p*(q-1)*(r-1)+(p-1)*q*(r-1)+(p-1)*(q-1)*r);
source[941] = 1f0 + 0f0im;

# calculate the incident and total electric field starting from vacuum surrounding.
Einc = DUTCH.calculate_electric_field_vacuum_to_dielectric(BG_σ,BG_ϵr,res,source,frequency);
Etot = DUTCH.calculate_electric_field_vacuum_to_dielectric(Update_σ,Update_ϵr,res,source,frequency);

# calculate the total electric field starting from the background dielectric 
prob = DUTCH.create_problem(BG_σ,BG_ϵr,Update_σ,Update_ϵr,Einc,frequency,res);
Jsc,Esc = DUTCH.solve_problem(prob,tol=1f-6);
Etot_update = Einc + Esc;

# check mismatch between scattered e-field from 2 VIE simulations and the update calculation.
err = DUTCH.norm(Etot-Einc-Esc);
norm_Esc = DUTCH.norm(Esc);
@test err/norm_Esc < 0.07

# set on more sources
source[2822] = 1f0 + 0f0im;

# calculate the incident and total electric field starting from vacuum surrounding.
Einc = DUTCH.calculate_electric_field_vacuum_to_dielectric(BG_σ,BG_ϵr,res,source,frequency);
Etot = DUTCH.calculate_electric_field_vacuum_to_dielectric(Update_σ,Update_ϵr,res,source,frequency);

# calculate the total electric field starting from the background dielectric 
prob = DUTCH.create_problem(BG_σ,BG_ϵr,Update_σ,Update_ϵr,Einc,frequency,res);
Jsc,Esc = DUTCH.solve_problem(prob,tol=1f-6);
Etot_update = Einc + Esc;

# check mismatch between scattered e-field from 2 VIE simulations and the update calculation.
err = DUTCH.norm(Etot-Einc-Esc);
norm_Esc = DUTCH.norm(Esc);
@test err/norm_Esc < 0.07