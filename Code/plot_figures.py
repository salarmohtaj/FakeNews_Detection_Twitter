import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
####SVM Linear
# dic = {
#     "Ordinary Text" : {"F1": 0.9393802381181292, "SD": 0.0068027946574182275},
# 'Replace URLs with constant' : {"F1": 0.9402055085804921, "SD": 0.0069012586041174925},
# 'Replace URLs with text' : {"F1": 0.9608612077072178, "SD": 0.0044641332918426094},
# 'Replace Username' : {"F1": 0.9375936850284765, "SD": 0.00733812229766244},
# 'Replace Emoji' : {"F1": 0.9393161164203707, "SD": 0.0065662548447577365},
# }

####SVM Sigmoid
dic = {
    "Ordinary Text" : {"F1": 0.8987749859031087, "SD": 0.005688641808923174},
'Replacing URLs with \n a special token' : {"F1": 0.8987134542495447, "SD": 0.007255653933336379},
"Replacing URLs with \n the webpage's text" : {"F1": 0.9285891187585158, "SD": 0.006727727428293432},
'Replacing Twitter handles \n with a special token' : {"F1": 0.8948657523758502, "SD": 0.007340451160241443},
'Replacing Emojis with the \n expression they represent' : {"F1": 0.898651765191444, "SD": 0.0068357576611648265},
}


####LSTM
# dic = {
#     "Ordinary Text" : {"F1": 0.9122725939878971, "SD": 0.02186913936428959},
# 'Replacing URLs with \n a special token' : {"F1": 0.9098153218772028, "SD": 0.01258744449780861},
# "Replacing URLs with \n the webpage's text" : {"F1": 0.9530781148331844, "SD": 0.009020343548651942},
# 'Replacing Twitter handles \n with a special token' : {"F1": 0.8968180871494076, "SD": 0.017873236241410736},
# 'Replacing Emojis with the \n expression they represent' : {"F1": 0.9159579192709065, "SD": 0.014667590348088962},
# }
####Bert
# dic = {
#     "Ordinary Text" : {"F1": 0.9044922295722886, "SD": 0.015663747144179677},
# 'Replacing URLs with \n a special token' : {"F1": 0.9235593477529405, "SD": 0.011739872055585231},
# "Replacing URLs with \n the webpage's text" : {"F1": 0.941631367516378, "SD": 0.008341664284516886},
# 'Replacing Twitter handles \n with a special token' : {"F1": 0.9176184715472401, "SD": 0.009702299020903707},
# 'Replacing Emojis with the \n expression they represent' : {"F1": 0.9024946573133772, "SD": 0.01174156713258706},
# }

output_dir = "../Data/CodaLab_Data/Figures"
color_1 = "darkorange"
color_2 = "royalblue"
colors = [color_1, color_2]

def plot_F1(dic, colors, output_dir = ""):
    plot_name = "F1_SVM.pdf"
    x, y, sd = [], [], []
    for index, key in enumerate(dic.keys()):
        x.append(index + 1)
        y.append(dic[key]["F1"])
        sd.append(dic[key]["SD"])

    plt.errorbar(x, y, sd, linestyle='None', fmt='o', markersize=6, capsize=4, color=colors[0], markeredgecolor=colors[1],
                 markerfacecolor=colors[1])
    for i in range(len(x)):
        plt.text(x[i]+0.08,y[i],float("{:.4f}".format(y[i])),color=colors[1])
    plt.xticks([1, 2, 3, 4, 5], dic.keys(), rotation=30)
    plt.axis([0.5, 5.5, 0.84, 1])
    plt.xlabel("Preprocessing task")
    plt.ylabel("F1 score")
    plt.title("SVM")
    plt.gcf().subplots_adjust(left=0.15)
    plt.tight_layout()
    #plt.savefig(os.path.join(output_dir, plot_name), format='eps')
    plt.savefig(os.path.join(output_dir, plot_name))
    plt.show()

plot_F1(dic,colors,output_dir)