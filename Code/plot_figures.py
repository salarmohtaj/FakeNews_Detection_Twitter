import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

dic = {
    "Normal Text" : {"F1": 0.3180765401461694, "SD": 0.020604953234156818},
'Replace URLs with constant' : {"F1": 0.30976339617963566, "SD": 0.019779024689548248},
'Replace URLs with text' : {"F1": 0.3169065053560062, "SD": 0.024132644777348104},
'Replace Username' : {"F1": 0.29809144457679787, "SD": 0.019562213566060173},
'Replace Emoji' : {"F1": 0.3240282757443207, "SD": 0.019955753258774835},
}
# x = np.array([1, 2, 3, 4, 5])
# #y = np.power(x, 5) # Effectively y = x**2
# y = np.array([0.3180765401461694, 0.30976339617963566, 0.3169065053560062, 0.29809144457679787, 0.3240282757443207])
# sd = np.array([0.020604953234156818, 0.019779024689548248, 0.024132644777348104, 0.019562213566060173, 0.019955753258774835])



output_dir = "../../../Data/US_Election_Data/Figures"
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

    plt.errorbar(x, y, sd, linestyle='None', fmt='o', markersize=8, capsize=4, color=colors[0], markeredgecolor=colors[1],
                 markerfacecolor=colors[1])
    for i in range(len(x)):
        plt.text(x[i]+0.08,y[i],float("{:.4f}".format(y[i])),color=colors[1])
    plt.xticks([1, 2, 3, 4, 5], dic.keys(), rotation=30)
    # plt.set_ylim([0.25, 0.35])
    plt.axis([0.5, 5.5, 0.25, 0.4])
    plt.xlabel("Preprocessing")
    plt.ylabel("F1 Score")
    plt.title("SVM")
    plt.gcf().subplots_adjust(left=0.15)
    plt.tight_layout()
    #plt.savefig(os.path.join(output_dir, plot_name), format='eps')
    plt.savefig(os.path.join(output_dir, plot_name))
    plt.show()

plot_F1(dic,colors,output_dir)