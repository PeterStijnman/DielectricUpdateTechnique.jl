import DielectricUpdateTechique as DUTCH
using Test


a = (x = rand(3), y = rand(3), z = rand(3));
b = (x = rand(3), y = rand(3), z = rand(3));
c = (x = zeros(3), y = zeros(3), z = zeros(3));


DUTCH.copyVec!(c, a);
@test (c.x == a.x) && (c.y == a.y) && (c.z == a.z)
@test DUTCH.lengthVec(a) == 9

DUTCH.copyVec!(c, b);
DUTCH.axpyVec!(2.0, a, b);
@test (b.x == a.x.*2.0 + c.x) && (b.y == a.y.*2.0 + c.y) && (b.z == a.z.*2.0 + c.z)

DUTCH.copyVec!(c, b);
DUTCH.axpbyVec!(2.0, a, 1.5, b);
@test (b.x == a.x.*2.0 + c.x.*1.5) && (b.y == a.y.*2.0 + c.y.*1.5) && (b.z == a.z.*2.0 + c.z.*1.5)

DUTCH.copyVec!(c, b);
DUTCH.scaleVec!(3.2, b);
@test (b.x == 3.2.*c.x) && (b.y == 3.2.*c.y) && (b.z == 3.2.*c.z)

DUTCH.caxpyVec!(c, 4.1, a, b);
@test (c.x == 4.1.*a.x + b.x) && (c.y == 4.1.*a.y + b.y) && (c.z == 4.1.*a.z + b.z)
