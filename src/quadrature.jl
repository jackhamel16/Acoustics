using LinearAlgebra

include("mesh.jl")

# The gauss7 rule comes from:
#    https://www.math.unipd.it/~alvise/SETS_CUBATURE_TRIANGLE/dunavant/set_dunavant_barycentric.m
#    Note: ordering of the triangle nodes counter-clockwise
const gauss1rule = [3.33333333333333314829616256247391e-01 3.33333333333333314829616256247391e-01 3.33333333333333314829616256247391e-01 1.00000000000000000000000000000000e+00]
const gauss7rule = [3.33333333333333314829616256247391e-01 3.33333333333333314829616256247391e-01 3.33333333333333314829616256247391e-01 2.25000000000000255351295663786004e-01
                    4.70142064105100010440452251714305e-01 4.70142064105100010440452251714305e-01 5.97158717897999791190954965713900e-02 1.32394152788506136442236993389088e-01
                    4.70142064105100010440452251714305e-01 5.97158717897999791190954965713900e-02 4.70142064105100010440452251714305e-01 1.32394152788506136442236993389088e-01
                    5.97158717897999791190954965713900e-02 4.70142064105100010440452251714305e-01 4.70142064105100010440452251714305e-01 1.32394152788506136442236993389088e-01
                    1.01286507323455995943639607048681e-01 1.01286507323455995943639607048681e-01 7.97426985353087980357145170273725e-01 1.25939180544827139529573400977824e-01
                    1.01286507323455995943639607048681e-01 7.97426985353087980357145170273725e-01 1.01286507323455995943639607048681e-01 1.25939180544827139529573400977824e-01
                    7.97426985353087980357145170273725e-01 1.01286507323455995943639607048681e-01 1.01286507323455995943639607048681e-01 1.25939180544827139529573400977824e-01]
const gauss13rule = [3.33333333333333314829616256247391e-01 3.33333333333333314829616256247391e-01 3.33333333333333314829616256247391e-01 -1.49570044467682294886401450639823e-01
                     2.60345966079039981000420311829657e-01 2.60345966079039981000420311829657e-01 4.79308067841920037999159376340685e-01 1.75615257433208354909126569509681e-01
                     2.60345966079039981000420311829657e-01 4.79308067841920037999159376340685e-01 2.60345966079039981000420311829657e-01 1.75615257433208354909126569509681e-01
                     4.79308067841920037999159376340685e-01 2.60345966079039981000420311829657e-01 2.60345966079039981000420311829657e-01 1.75615257433208354909126569509681e-01
                     6.51301029022160055115264754022064e-02 6.51301029022160055115264754022064e-02 8.69739794195568016732522664824501e-01 5.33472356088381047256596900751902e-02
                     6.51301029022160055115264754022064e-02 8.69739794195568016732522664824501e-01 6.51301029022160055115264754022064e-02 5.33472356088381047256596900751902e-02
                     8.69739794195568016732522664824501e-01 6.51301029022160055115264754022064e-02 6.51301029022160055115264754022064e-02 5.33472356088381047256596900751902e-02
                     4.86903154253160025399793653377856e-02 3.12865496004874010793628258397803e-01 6.38444188569810000544180184078868e-01 7.71137608902571491942268266939209e-02
                     4.86903154253160025399793653377856e-02 6.38444188569810000544180184078868e-01 3.12865496004874010793628258397803e-01 7.71137608902571491942268266939209e-02
                     3.12865496004874010793628258397803e-01 4.86903154253160025399793653377856e-02 6.38444188569810000544180184078868e-01 7.71137608902571491942268266939209e-02
                     3.12865496004874010793628258397803e-01 6.38444188569810000544180184078868e-01 4.86903154253160025399793653377856e-02 7.71137608902571491942268266939209e-02
                     6.38444188569810000544180184078868e-01 4.86903154253160025399793653377856e-02 3.12865496004874010793628258397803e-01 7.71137608902571491942268266939209e-02
                     6.38444188569810000544180184078868e-01 3.12865496004874010793628258397803e-01 4.86903154253160025399793653377856e-02 7.71137608902571491942268266939209e-02]
const gauss79rule = [3.33333333333333314829616256247391e-01 3.33333333333333314829616256247391e-01 3.33333333333333314829616256247391e-01 3.30570555416238795465311284260679e-02
                     5.00950464352200031115103229240049e-01 5.00950464352200031115103229240049e-01 -1.90092870440006223020645848009735e-03 8.67019185662996926845791367810534e-04
                     5.00950464352200031115103229240049e-01 -1.90092870440006223020645848009735e-03 5.00950464352200031115103229240049e-01 8.67019185662996926845791367810534e-04
                     -1.90092870440006223020645848009735e-03 5.00950464352200031115103229240049e-01 5.00950464352200031115103229240049e-01 8.67019185662996926845791367810534e-04
                     4.88212957934729019360275970029761e-01 4.88212957934729019360275970029761e-01 2.35740841305419612794480599404778e-02 1.16600527164479588621004424453531e-02
                     4.88212957934729019360275970029761e-01 2.35740841305419612794480599404778e-02 4.88212957934729019360275970029761e-01 1.16600527164479588621004424453531e-02
                     2.35740841305419612794480599404778e-02 4.88212957934729019360275970029761e-01 4.88212957934729019360275970029761e-01 1.16600527164479588621004424453531e-02
                     4.55136681950283006337087954307208e-01 4.55136681950283006337087954307208e-01 8.97266360994339873258240913855843e-02 2.28769363564209210482047751611390e-02
                     4.55136681950283006337087954307208e-01 8.97266360994339873258240913855843e-02 4.55136681950283006337087954307208e-01 2.28769363564209210482047751611390e-02
                     8.97266360994339873258240913855843e-02 4.55136681950283006337087954307208e-01 4.55136681950283006337087954307208e-01 2.28769363564209210482047751611390e-02
                     4.01996259318288973183541656908346e-01 4.01996259318288973183541656908346e-01 1.96007481363422053632916686183307e-01 3.04489826739378910414046686128131e-02
                     4.01996259318288973183541656908346e-01 1.96007481363422053632916686183307e-01 4.01996259318288973183541656908346e-01 3.04489826739378910414046686128131e-02
                     1.96007481363422053632916686183307e-01 4.01996259318288973183541656908346e-01 4.01996259318288973183541656908346e-01 3.04489826739378910414046686128131e-02
                     2.55892909759420972282129014274688e-01 2.55892909759420972282129014274688e-01 4.88214180481158055435741971450625e-01 3.06248917253548920414107925580538e-02
                     2.55892909759420972282129014274688e-01 4.88214180481158055435741971450625e-01 2.55892909759420972282129014274688e-01 3.06248917253548920414107925580538e-02
                     4.88214180481158055435741971450625e-01 2.55892909759420972282129014274688e-01 2.55892909759420972282129014274688e-01 3.06248917253548920414107925580538e-02
                     1.76488255995106008144901466039300e-01 1.76488255995106008144901466039300e-01 6.47023488009788039221348299179226e-01 2.43680576767999132470343681688973e-02
                     1.76488255995106008144901466039300e-01 6.47023488009788039221348299179226e-01 1.76488255995106008144901466039300e-01 2.43680576767999132470343681688973e-02
                     6.47023488009788039221348299179226e-01 1.76488255995106008144901466039300e-01 1.76488255995106008144901466039300e-01 2.43680576767999132470343681688973e-02
                     1.04170855336758003129027372324344e-01 1.04170855336758003129027372324344e-01 7.91658289326483965986369639722398e-01 1.59974320320239449255694808016415e-02
                     1.04170855336758003129027372324344e-01 7.91658289326483965986369639722398e-01 1.04170855336758003129027372324344e-01 1.59974320320239449255694808016415e-02
                     7.91658289326483965986369639722398e-01 1.04170855336758003129027372324344e-01 1.04170855336758003129027372324344e-01 1.59974320320239449255694808016415e-02
                     5.30689638409300029620041527778085e-02 5.30689638409300029620041527778085e-02 8.93862072318139966320416078815470e-01 7.69830181560197251977584187443426e-03
                     5.30689638409300029620041527778085e-02 8.93862072318139966320416078815470e-01 5.30689638409300029620041527778085e-02 7.69830181560197251977584187443426e-03
                     8.93862072318139966320416078815470e-01 5.30689638409300029620041527778085e-02 5.30689638409300029620041527778085e-02 7.69830181560197251977584187443426e-03
                     4.16187151960289991592389924335293e-02 4.16187151960289991592389924335293e-02 9.16762569607942001681522015132941e-01 -6.32060497487997672433346352249828e-04
                     4.16187151960289991592389924335293e-02 9.16762569607942001681522015132941e-01 4.16187151960289991592389924335293e-02 -6.32060497487997672433346352249828e-04
                     9.16762569607942001681522015132941e-01 4.16187151960289991592389924335293e-02 4.16187151960289991592389924335293e-02 -6.32060497487997672433346352249828e-04
                     1.15819214068219999286268873106565e-02 1.15819214068219999286268873106565e-02 9.76836157186355968917723657796159e-01 1.75113430119299361663320890869500e-03
                     1.15819214068219999286268873106565e-02 9.76836157186355968917723657796159e-01 1.15819214068219999286268873106565e-02 1.75113430119299361663320890869500e-03
                     9.76836157186355968917723657796159e-01 1.15819214068219999286268873106565e-02 1.15819214068219999286268873106565e-02 1.75113430119299361663320890869500e-03
                     3.44855770229000990756418332239264e-01 6.06402646106160014838337701803539e-01 4.87415836648390499163951972150244e-02 1.64658391895759412260069609601487e-02
                     3.44855770229000990756418332239264e-01 4.87415836648390499163951972150244e-02 6.06402646106160014838337701803539e-01 1.64658391895759412260069609601487e-02
                     6.06402646106160014838337701803539e-01 3.44855770229000990756418332239264e-01 4.87415836648390499163951972150244e-02 1.64658391895759412260069609601487e-02
                     6.06402646106160014838337701803539e-01 4.87415836648390499163951972150244e-02 3.44855770229000990756418332239264e-01 1.64658391895759412260069609601487e-02
                     4.87415836648390499163951972150244e-02 3.44855770229000990756418332239264e-01 6.06402646106160014838337701803539e-01 1.64658391895759412260069609601487e-02
                     4.87415836648390499163951972150244e-02 6.06402646106160014838337701803539e-01 3.44855770229000990756418332239264e-01 1.64658391895759412260069609601487e-02
                     3.77843269594853981008242271855124e-01 6.15842614456540982104115755646490e-01 6.31411594860509239879320375621319e-03 4.83903354048498268724642912275158e-03
                     3.77843269594853981008242271855124e-01 6.31411594860509239879320375621319e-03 6.15842614456540982104115755646490e-01 4.83903354048498268724642912275158e-03
                     6.15842614456540982104115755646490e-01 3.77843269594853981008242271855124e-01 6.31411594860509239879320375621319e-03 4.83903354048498268724642912275158e-03
                     6.15842614456540982104115755646490e-01 6.31411594860509239879320375621319e-03 3.77843269594853981008242271855124e-01 4.83903354048498268724642912275158e-03
                     6.31411594860509239879320375621319e-03 3.77843269594853981008242271855124e-01 6.15842614456540982104115755646490e-01 4.83903354048498268724642912275158e-03
                     6.31411594860509239879320375621319e-03 6.15842614456540982104115755646490e-01 3.77843269594853981008242271855124e-01 4.83903354048498268724642912275158e-03
                     3.06635479062356997026483895751880e-01 5.59048000390295007910879121482139e-01 1.34316520547347995062636982765980e-01 2.58049065346499101325505876047828e-02
                     3.06635479062356997026483895751880e-01 1.34316520547347995062636982765980e-01 5.59048000390295007910879121482139e-01 2.58049065346499101325505876047828e-02
                     5.59048000390295007910879121482139e-01 3.06635479062356997026483895751880e-01 1.34316520547347995062636982765980e-01 2.58049065346499101325505876047828e-02
                     5.59048000390295007910879121482139e-01 1.34316520547347995062636982765980e-01 3.06635479062356997026483895751880e-01 2.58049065346499101325505876047828e-02
                     1.34316520547347995062636982765980e-01 3.06635479062356997026483895751880e-01 5.59048000390295007910879121482139e-01 2.58049065346499101325505876047828e-02
                     1.34316520547347995062636982765980e-01 5.59048000390295007910879121482139e-01 3.06635479062356997026483895751880e-01 2.58049065346499101325505876047828e-02
                     2.49419362774741998345362503641809e-01 7.36606743262866014987366725108586e-01 1.39738939623920144228463868785184e-02 8.47109105444097086612398328497875e-03
                     2.49419362774741998345362503641809e-01 1.39738939623920144228463868785184e-02 7.36606743262866014987366725108586e-01 8.47109105444097086612398328497875e-03
                     7.36606743262866014987366725108586e-01 2.49419362774741998345362503641809e-01 1.39738939623920144228463868785184e-02 8.47109105444097086612398328497875e-03
                     7.36606743262866014987366725108586e-01 1.39738939623920144228463868785184e-02 2.49419362774741998345362503641809e-01 8.47109105444097086612398328497875e-03
                     1.39738939623920144228463868785184e-02 2.49419362774741998345362503641809e-01 7.36606743262866014987366725108586e-01 8.47109105444097086612398328497875e-03
                     1.39738939623920144228463868785184e-02 7.36606743262866014987366725108586e-01 2.49419362774741998345362503641809e-01 8.47109105444097086612398328497875e-03
                     2.12775724802801990964695733055123e-01 7.11675142287434003840473906166153e-01 7.55491329097640607059815920365509e-02 1.83549141062799327228649559629048e-02
                     2.12775724802801990964695733055123e-01 7.55491329097640607059815920365509e-02 7.11675142287434003840473906166153e-01 1.83549141062799327228649559629048e-02
                     7.11675142287434003840473906166153e-01 2.12775724802801990964695733055123e-01 7.55491329097640607059815920365509e-02 1.83549141062799327228649559629048e-02
                     7.11675142287434003840473906166153e-01 7.55491329097640607059815920365509e-02 2.12775724802801990964695733055123e-01 1.83549141062799327228649559629048e-02
                     7.55491329097640607059815920365509e-02 2.12775724802801990964695733055123e-01 7.11675142287434003840473906166153e-01 1.83549141062799327228649559629048e-02
                     7.55491329097640607059815920365509e-02 7.11675142287434003840473906166153e-01 2.12775724802801990964695733055123e-01 1.83549141062799327228649559629048e-02
                     1.46965436053239001390480211739487e-01 8.61402717154986952152739831944928e-01 -8.36815320822592578764442805550061e-03 7.04404677907997478318591344503830e-04
                     1.46965436053239001390480211739487e-01 -8.36815320822592578764442805550061e-03 8.61402717154986952152739831944928e-01 7.04404677907997478318591344503830e-04
                     8.61402717154986952152739831944928e-01 1.46965436053239001390480211739487e-01 -8.36815320822592578764442805550061e-03 7.04404677907997478318591344503830e-04
                     8.61402717154986952152739831944928e-01 -8.36815320822592578764442805550061e-03 1.46965436053239001390480211739487e-01 7.04404677907997478318591344503830e-04
                     -8.36815320822592578764442805550061e-03 1.46965436053239001390480211739487e-01 8.61402717154986952152739831944928e-01 7.04404677907997478318591344503830e-04
                     -8.36815320822592578764442805550061e-03 8.61402717154986952152739831944928e-01 1.46965436053239001390480211739487e-01 7.04404677907997478318591344503830e-04
                     1.37726978828923013464802238559059e-01 8.35586957912363037515035557589727e-01 2.66860632587139212645865882223006e-02 1.01126849274619633883842695354360e-02
                     1.37726978828923013464802238559059e-01 2.66860632587139212645865882223006e-02 8.35586957912363037515035557589727e-01 1.01126849274619633883842695354360e-02
                     8.35586957912363037515035557589727e-01 1.37726978828923013464802238559059e-01 2.66860632587139212645865882223006e-02 1.01126849274619633883842695354360e-02
                     8.35586957912363037515035557589727e-01 2.66860632587139212645865882223006e-02 1.37726978828923013464802238559059e-01 1.01126849274619633883842695354360e-02
                     2.66860632587139212645865882223006e-02 1.37726978828923013464802238559059e-01 8.35586957912363037515035557589727e-01 1.01126849274619633883842695354360e-02
                     2.66860632587139212645865882223006e-02 8.35586957912363037515035557589727e-01 1.37726978828923013464802238559059e-01 1.01126849274619633883842695354360e-02
                     5.96961091490069983844790613147779e-02 9.29756171556852972770457199658267e-01 1.05477192941400010894881233980414e-02 3.57390938594998736066443711933971e-03
                     5.96961091490069983844790613147779e-02 1.05477192941400010894881233980414e-02 9.29756171556852972770457199658267e-01 3.57390938594998736066443711933971e-03
                     9.29756171556852972770457199658267e-01 5.96961091490069983844790613147779e-02 1.05477192941400010894881233980414e-02 3.57390938594998736066443711933971e-03
                     9.29756171556852972770457199658267e-01 1.05477192941400010894881233980414e-02 5.96961091490069983844790613147779e-02 3.57390938594998736066443711933971e-03
                     1.05477192941400010894881233980414e-02 5.96961091490069983844790613147779e-02 9.29756171556852972770457199658267e-01 3.57390938594998736066443711933971e-03
                     1.05477192941400010894881233980414e-02 9.29756171556852972770457199658267e-01 5.96961091490069983844790613147779e-02 3.57390938594998736066443711933971e-03]
function gaussQuadrature(scale_factor::Float64, func::Function, points::Array{Float64, 2}, weights::Array{Float64, 1})
    num_points = length(weights)
    quadrature_sum = 0
    for sum_idx in 1:num_points
        x, y, z = points[sum_idx,:]
        quadrature_sum += weights[sum_idx] * func(x, y, z)
    end
    scale_factor * quadrature_sum
end

function integrateTriangle(nodes::Array{Float64, 2}, func::Function, points::Array{Float64, 2}, weights::Array{Float64, 1})
    num_points = size(points)[1]
    triangle_area = norm(cross(nodes[2,:]-nodes[1,:], nodes[3,:]-nodes[2,:]))/2
    quadrature_points = Array{Float64, 2}(undef, num_points, 3)
    for point_idx in 1:num_points
        quadrature_points[point_idx,:] = barycentric2Cartesian(nodes, points[point_idx,:])
    end
    gaussQuadrature(triangle_area, func, quadrature_points, weights)
end
