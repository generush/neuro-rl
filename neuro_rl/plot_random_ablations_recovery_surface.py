import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming you already have a 2D NumPy array called 'data'
# Replace 'data' with your actual array




prob_of_recovery_given_neuron_ablated = np.array([[0,0,0.0416666666666667,0,0,0,0,0,0,0.0344827586206897,0,0,0,0,0,0,0.0294117647058823,0,0.0714285714285714,0,0,0,0,0,0.0416666666666667,0,0,0,0,0,0.037037037037037,0.0555555555555556,0,0.0434782608695652,0,0,0,0,0.0344827586206897,0,0,0,0,0,0,0,0,0,0,0,0,0,0.04,0,0,0,0,0,0,0,0,0,0,0,0,0.0434782608695652,0,0,0,0,0,0,0,0.103448275862069,0.05,0.05,0.0333333333333333,0,0,0,0,0,0.0434782608695652,0,0,0,0,0,0,0,0,0.0344827586206897,0,0,0.0333333333333333,0,0,0,0,0,0,0,0.037037037037037,0,0,0,0,0,0,0.03125,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0434782608695652,0.032258064516129,0],
[0.113636363636364,0.0833333333333333,0.0851063829787234,0.0517241379310345,0.0208333333333333,0.0655737704918033,0.0392156862745098,0.0285714285714286,0.0408163265306122,0.0408163265306122,0.0652173913043478,0.0727272727272727,0.0363636363636364,0.108108108108108,0.0508474576271187,0.0408163265306122,0.0555555555555556,0.0357142857142857,0.130434782608696,0.0555555555555556,0.0357142857142857,0.113207547169811,0.0681818181818182,0.113636363636364,0.0681818181818182,0.0714285714285714,0.0483870967741936,0.0384615384615385,0.130434782608696,0,0.108695652173913,0,0.0338983050847458,0.075,0.0425531914893617,0.0545454545454545,0.0714285714285714,0.0655737704918033,0.0357142857142857,0.0833333333333333,0.0476190476190476,0.0833333333333333,0.0434782608695652,0.037037037037037,0.0892857142857143,0,0.0188679245283019,0.0416666666666667,0.0535714285714286,0.0909090909090909,0.0681818181818182,0.0847457627118644,0.104166666666667,0.04,0,0.0227272727272727,0.0444444444444444,0.0652173913043478,0.0256410256410256,0.0540540540540541,0.0625,0.04,0.0508474576271187,0.0425531914893617,0.0238095238095238,0.0625,0.025,0.0181818181818182,0.046875,0.0512820512820513,0.0392156862745098,0.0185185185185185,0.0512820512820513,0.153846153846154,0.142857142857143,0.0588235294117647,0.0535714285714286,0.0172413793103448,0.0425531914893617,0.0943396226415094,0,0.0338983050847458,0.0204081632653061,0.0888888888888889,0.105263157894737,0.0508474576271187,0.04,0.015625,0.0169491525423729,0.0851063829787234,0.0344827586206897,0.0357142857142857,0.037037037037037,0.0444444444444444,0.160714285714286,0.06,0.0816326530612245,0.0384615384615385,0.025,0.0681818181818182,0.0566037735849057,0.0526315789473684,0.0576923076923077,0,0.0740740740740741,0.08,0.0212765957446808,0.024390243902439,0.0535714285714286,0,0.0208333333333333,0.0377358490566038,0.0851063829787234,0.0181818181818182,0,0.0727272727272727,0.0175438596491228,0.0746268656716418,0.0196078431372549,0.0588235294117647,0.0232558139534884,0,0.0888888888888889,0,0.0350877192982456,0.0212765957446808,0.0555555555555556,0.0545454545454545],
[0.0757575757575758,0.12,0.108108108108108,0.13953488372093,0.09375,0.112359550561798,0.2,0.078125,0.144736842105263,0.102564102564103,0.114285714285714,0.108108108108108,0.114942528735632,0.188405797101449,0.1125,0.125,0.0875,0.0843373493975904,0.151515151515152,0.141025641025641,0.0898876404494382,0.131578947368421,0.147058823529412,0.0909090909090909,0.09375,0.125,0.0786516853932584,0.103896103896104,0.1,0.072463768115942,0.140845070422535,0.101449275362319,0.0941176470588235,0.21875,0.101449275362319,0.0493827160493827,0.051948051948052,0.142857142857143,0.0921052631578947,0.0641025641025641,0.0909090909090909,0.111111111111111,0.128205128205128,0.0952380952380952,0.0886075949367089,0.101449275362319,0.0789473684210526,0.103896103896104,0.133333333333333,0.108108108108108,0.0821917808219178,0.123287671232877,0.105263157894737,0.118421052631579,0.253731343283582,0.0847457627118644,0.0746268656716418,0.0789473684210526,0.0882352941176471,0.0615384615384615,0.109090909090909,0.0958904109589041,0.0930232558139535,0.0735294117647059,0.0945945945945946,0.118421052631579,0.0428571428571429,0.1125,0.126315789473684,0.0344827586206897,0.166666666666667,0.102564102564103,0.107692307692308,0.219512195121951,0.161764705882353,0.151898734177215,0.0857142857142857,0.157894736842105,0.0641025641025641,0.0769230769230769,0.0394736842105263,0.0465116279069768,0.0704225352112676,0.10958904109589,0.102941176470588,0.158536585365854,0.111111111111111,0.0823529411764706,0.0886075949367089,0.0434782608695652,0.0476190476190476,0.0595238095238095,0.0731707317073171,0.125,0.227848101265823,0.126760563380282,0.0379746835443038,0.0540540540540541,0.078125,0.144927536231884,0.156626506024096,0.111111111111111,0.155844155844156,0.0625,0.0875,0.128205128205128,0.106060606060606,0.121212121212121,0.116883116883117,0.0666666666666667,0.144736842105263,0.0704225352112676,0.114285714285714,0.0921052631578947,0.0579710144927536,0.146666666666667,0.0759493670886076,0.103448275862069,0.0853658536585366,0.0759493670886076,0.0735294117647059,0.0405405405405405,0.140845070422535,0.10958904109589,0.0779220779220779,0.0895522388059701,0.1375,0.0595238095238095],
[0.257731958762887,0.25,0.186274509803922,0.113207547169811,0.127906976744186,0.0973451327433628,0.24468085106383,0.121951219512195,0.125,0.121212121212121,0.170212765957447,0.152380952380952,0.0900900900900901,0.252747252747253,0.142857142857143,0.147368421052632,0.122641509433962,0.153846153846154,0.235955056179775,0.123711340206186,0.162162162162162,0.21,0.191489361702128,0.131868131868132,0.136842105263158,0.193877551020408,0.173913043478261,0.171717171717172,0.168224299065421,0.155555555555556,0.185567010309278,0.245098039215686,0.178571428571429,0.157303370786517,0.171717171717172,0.16504854368932,0.155339805825243,0.133928571428571,0.166666666666667,0.213592233009709,0.141304347826087,0.188118811881188,0.135416666666667,0.135922330097087,0.132075471698113,0.21505376344086,0.122448979591837,0.123711340206186,0.173913043478261,0.177777777777778,0.189473684210526,0.13,0.142857142857143,0.177570093457944,0.348314606741573,0.178571428571429,0.141304347826087,0.156862745098039,0.10752688172043,0.153061224489796,0.269230769230769,0.13,0.18348623853211,0.184782608695652,0.168421052631579,0.144230769230769,0.0824742268041237,0.0849056603773585,0.243243243243243,0.2,0.257731958762887,0.144329896907216,0.191919191919192,0.327102803738318,0.222222222222222,0.206896551724138,0.178947368421053,0.146788990825688,0.221052631578947,0.15,0.198019801980198,0.155963302752294,0.111111111111111,0.159574468085106,0.172043010752688,0.153846153846154,0.174311926605505,0.182608695652174,0.184466019417476,0.16304347826087,0.132075471698113,0.129310344827586,0.0952380952380952,0.212121212121212,0.273584905660377,0.188888888888889,0.137254901960784,0.122448979591837,0.195652173913043,0.168316831683168,0.171428571428571,0.101851851851852,0.15625,0.141509433962264,0.118811881188119,0.19047619047619,0.186046511627907,0.221052631578947,0.218181818181818,0.158333333333333,0.192982456140351,0.152173913043478,0.118811881188119,0.193877551020408,0.129032258064516,0.211538461538462,0.130841121495327,0.160377358490566,0.188679245283019,0.17,0.115789473684211,0.153846153846154,0.204301075268817,0.176470588235294,0.153061224489796,0.106382978723404,0.158415841584158,0.116504854368932],
[0.358974358974359,0.301724137931034,0.333333333333333,0.221374045801527,0.361111111111111,0.226277372262774,0.433333333333333,0.263157894736842,0.253731343283582,0.272727272727273,0.314049586776859,0.256,0.227586206896552,0.419354838709677,0.288590604026846,0.264,0.303030303030303,0.285714285714286,0.43859649122807,0.3,0.290780141843972,0.304,0.285714285714286,0.256637168141593,0.254098360655738,0.286885245901639,0.269503546099291,0.3,0.277372262773723,0.243697478991597,0.305084745762712,0.364341085271318,0.26865671641791,0.301724137931034,0.266129032258064,0.317073170731707,0.34375,0.278195488721804,0.328125,0.296296296296296,0.258333333333333,0.314049586776859,0.296875,0.298387096774194,0.253968253968254,0.35042735042735,0.276422764227642,0.223214285714286,0.254545454545454,0.298245614035088,0.305785123966942,0.301587301587302,0.268292682926829,0.28030303030303,0.431034482758621,0.25,0.273504273504273,0.294573643410853,0.254545454545454,0.245762711864407,0.406779661016949,0.298387096774194,0.317829457364341,0.264462809917355,0.319672131147541,0.28,0.180327868852459,0.341085271317829,0.266187050359712,0.284210526315789,0.401639344262295,0.28125,0.341463414634146,0.48062015503876,0.352459016393443,0.326086956521739,0.330508474576271,0.338582677165354,0.277310924369748,0.310077519379845,0.220472440944882,0.297101449275362,0.279411764705882,0.275,0.341880341880342,0.317829457364341,0.32824427480916,0.297101449275362,0.264705882352941,0.290909090909091,0.181818181818182,0.216783216783217,0.271317829457364,0.2890625,0.376,0.264150943396226,0.285714285714286,0.245762711864407,0.324561403508772,0.237410071942446,0.333333333333333,0.270491803278689,0.333333333333333,0.295454545454545,0.305785123966942,0.353383458646616,0.256880733944954,0.31304347826087,0.353846153846154,0.284722222222222,0.283582089552239,0.288135593220339,0.279661016949152,0.221374045801527,0.275,0.384615384615385,0.323308270676692,0.316176470588235,0.287878787878788,0.266129032258064,0.296610169491525,0.265151515151515,0.247787610619469,0.264,0.223140495867769,0.24390243902439,0.30952380952381,0.221374045801527],
[0.442857142857143,0.431506849315069,0.447204968944099,0.303225806451613,0.422535211267606,0.329268292682927,0.493243243243243,0.330985915492958,0.33974358974359,0.410596026490066,0.356643356643357,0.344155844155844,0.311111111111111,0.482993197278912,0.38150289017341,0.394366197183099,0.372549019607843,0.370629370629371,0.468085106382979,0.333333333333333,0.353658536585366,0.397435897435897,0.302013422818792,0.357142857142857,0.297297297297297,0.394736842105263,0.345454545454545,0.420382165605096,0.356687898089172,0.337837837837838,0.428571428571429,0.397350993377483,0.347826086956522,0.364285714285714,0.333333333333333,0.398692810457516,0.402597402597403,0.339506172839506,0.412162162162162,0.358974358974359,0.302013422818792,0.373239436619718,0.320261437908497,0.346666666666667,0.379084967320261,0.354609929078014,0.380281690140845,0.304347826086957,0.356060606060606,0.3828125,0.35031847133758,0.306122448979592,0.321917808219178,0.414012738853503,0.526315789473684,0.424242424242424,0.36551724137931,0.357142857142857,0.32824427480916,0.337837837837838,0.450704225352113,0.371794871794872,0.368098159509202,0.394366197183099,0.361842105263158,0.339869281045752,0.333333333333333,0.358620689655172,0.382165605095541,0.361344537815126,0.375,0.357615894039735,0.356164383561644,0.51875,0.371621621621622,0.436708860759494,0.410958904109589,0.39751552795031,0.371428571428571,0.356687898089172,0.3,0.341935483870968,0.326923076923077,0.344827586206897,0.398601398601399,0.416107382550336,0.397350993377483,0.392156862745098,0.363636363636364,0.355072463768116,0.297101449275362,0.349397590361446,0.339869281045752,0.368421052631579,0.503355704697987,0.369230769230769,0.357615894039735,0.323308270676692,0.338461538461538,0.352941176470588,0.445859872611465,0.324503311258278,0.384105960264901,0.414634146341463,0.342281879194631,0.378205128205128,0.31578947368421,0.387755102040816,0.4,0.383720930232558,0.364779874213836,0.347826086956522,0.340277777777778,0.372670807453416,0.349315068493151,0.389261744966443,0.346153846153846,0.408805031446541,0.352201257861635,0.358108108108108,0.308219178082192,0.329113924050633,0.355072463768116,0.357615894039735,0.352564102564103,0.38255033557047,0.353333333333333,0.254777070063694],
[0.51219512195122,0.526627218934911,0.538043478260869,0.48314606741573,0.475903614457831,0.46031746031746,0.571428571428571,0.5,0.480446927374302,0.5,0.490909090909091,0.474285714285714,0.494949494949495,0.573863636363636,0.5,0.49390243902439,0.438502673796791,0.467836257309942,0.582352941176471,0.454545454545455,0.510752688172043,0.51123595505618,0.452380952380952,0.467948717948718,0.43859649122807,0.497237569060774,0.527173913043478,0.519125683060109,0.489247311827957,0.433526011560694,0.526315789473684,0.548022598870056,0.464864864864865,0.508982035928144,0.491803278688525,0.510869565217391,0.49171270718232,0.483695652173913,0.514450867052023,0.533333333333333,0.494444444444444,0.467065868263473,0.460227272727273,0.531428571428571,0.464088397790055,0.5375,0.508771929824561,0.522292993630573,0.45679012345679,0.506666666666667,0.515625,0.50887573964497,0.394444444444444,0.491525423728814,0.571428571428571,0.5375,0.48,0.48936170212766,0.463414634146341,0.471264367816092,0.576470588235294,0.5,0.548913043478261,0.490683229813665,0.5,0.508021390374332,0.380681818181818,0.525423728813559,0.477272727272727,0.540540540540541,0.523529411764706,0.483333333333333,0.548571428571429,0.619565217391304,0.464285714285714,0.51063829787234,0.554913294797688,0.530386740331492,0.49375,0.523255813953488,0.44311377245509,0.502824858757062,0.489130434782609,0.44047619047619,0.465909090909091,0.558139534883721,0.491428571428571,0.528735632183908,0.471590909090909,0.445121951219512,0.448484848484848,0.459016393442623,0.454545454545455,0.51123595505618,0.526627218934911,0.481481481481481,0.508571428571429,0.434782608695652,0.493421052631579,0.505263157894737,0.533333333333333,0.477777777777778,0.514619883040936,0.513661202185792,0.519553072625698,0.482954545454545,0.4375,0.511627906976744,0.525139664804469,0.505102040816326,0.477272727272727,0.461538461538462,0.487951807228916,0.497175141242938,0.470588235294118,0.420454545454545,0.508108108108108,0.502673796791444,0.486338797814208,0.5,0.436046511627907,0.45945945945946,0.429411764705882,0.470238095238095,0.486187845303867,0.50561797752809,0.485380116959064,0.456521739130435]])

x = np.arange(128)
y = np.array([8,16,24,32,40,48,56])
X, Y = np.meshgrid(x, y)

# Create the surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

orig_map=plt.cm.get_cmap('plasma')
reversed_map = orig_map.reversed()

# Reverse the y-axis
ax.invert_yaxis()

surf = ax.plot_surface(X, Y, prob_of_recovery_given_neuron_ablated, cmap=reversed_map, linewidth=0, edgecolor='none', alpha=0.8)

# Customize the plot as needed
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
fig.colorbar(surf)

# Show the plot
plt.title('3D Surface Plot')
plt.show()