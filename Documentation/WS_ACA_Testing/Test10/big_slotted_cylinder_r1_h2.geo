// Gmsh project created on Thu Aug 05 11:16:25 2021
SetFactory("OpenCASCADE");

height = 2;
outer_r = 1;
inner_r = 0.95;
wall_thickness = outer_r-inner_r;
slot_angle = Pi/6;
slot_height=0.4;

Cylinder(1) = {0, 0, -height/2, 0, 0, height, outer_r, 2*Pi};
Cylinder(2) = {0, 0, -height/2+wall_thickness, 0, 0, height-2*wall_thickness, inner_r, 2*Pi};
cyl_vol = BooleanDifference {Volume{1}; Delete;} {Volume{2}; Delete;};

Rectangle(1) = {inner_r, 0, 0, wall_thickness, slot_height, 0};
Rotate {{inner_r, 0, 0}, {0, 0, 0}, Pi/2} { Surface{1}; }
Translate {0, 0, -slot_height/2} { Surface{1}; }
Rotate {{0, 0, 1}, {0, 0, 0}, slot_angle/2} { Surface{1}; }
slot_ids[] = Extrude {{0, 0, 1}, {0, 0, 0}, -slot_angle} { Surface{1}; };

slotted_cyl_vol = BooleanDifference {Volume{cyl_vol}; Delete;} {Volume{slot_ids[1]}; Delete;};