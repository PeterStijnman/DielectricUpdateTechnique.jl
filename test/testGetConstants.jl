import DielectricUpdateTechique as DUTCH
using Test

f = 300e6; #300MHz
m = DUTCH.get_constants(f);

@test m[:c]  == 299792458
@test m[:μ₀] == 4f-7*π
@test m[:ϵ₀] == 1/(m[:c]^2*m[:μ₀])
@test m[:f₀] == f
@test m[:ω]  == 2*π*f
@test m[:kb] == 2*π*f/m[:c]
@test m[:λ]  == m[:c]/f

f = 1337e6; #1337MHz
m = DUTCH.get_constants(f);
@test m[:f₀] == f
@test m[:ω]  == 2*π*f
@test m[:kb] == 2*π*f/m[:c]
@test m[:λ]  == m[:c]/f
