import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

dic = {
    "Ordinary Text" : {"F1": 0.9393802381181292, "SD": 0.0068027946574182275},
'Replace URLs with constant' : {"F1": 0.9402055085804921, "SD": 0.0069012586041174925},
'Replace URLs with text' : {"F1": 0.9608612077072178, "SD": 0.0044641332918426094},
'Replace Username' : {"F1": 0.9375936850284765, "SD": 0.00733812229766244},
'Replace Emoji' : {"F1": 0.9393161164203707, "SD": 0.0065662548447577365},
}


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
    plt.xticks([1, 2, 3, 4, 5], dic.keys(), rotation=20)
    plt.axis([0.5, 5.5, 0.9, 1])
    plt.xlabel("Preprocessing task")
    plt.ylabel("F1 score")
    plt.title("SVM")
    plt.gcf().subplots_adjust(left=0.15)
    plt.tight_layout()
    #plt.savefig(os.path.join(output_dir, plot_name), format='eps')
    plt.savefig(os.path.join(output_dir, plot_name))
    plt.show()

plot_F1(dic,colors,output_dir)