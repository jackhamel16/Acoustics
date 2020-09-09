Point(1) = {0, 0, 0, 1};
Point(2) = {0, 1, 0, 1};
Line(1) = {2, 1};
Extrude {1, 0, 0} {
  Curve{1};Layers{1}; 
}
Physical Surface(1) = {5};
Physical Curve(333) = {3, 4};
Physical Curve(666) = {1};
Physical Curve(444) = {2};
