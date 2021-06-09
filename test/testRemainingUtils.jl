import DielectricUpdateTechique as DUTCH
using Test


a = rand(9);
b = rand(9);
c = zeros(9);

scalar = rand(1)[1]

DUTCH.caxpy!(c, scalar, a, b);
@test c == scalar.*a + b
