// Gmsh project created on Sun Dec 06 22:04:40 2020
shift = 100;
ms = 2.0;

Point(1) = {0, 0, 0, ms};
Point(2) = {1, 0, 0, ms};
Point(3) = {0, 1, 0, ms};
Line(1) = {3, 1};
Line(2) = {1, 2};
Line(3) = {2, 3};

Point(4) = {0+shift, 0, 0, ms};
Point(5) = {1+shift, 0, 0, ms};
Point(6) = {0+shift, 1, 0, ms};
Line(4) = {6, 4};
Line(5) = {4, 5};
Line(6) = {5, 6};

Curve Loop(1) = {1, 2, 3};
Curve Loop(2) = {4, 5, 6};
Surface(1) = {1};
Surface(2) = {2};
