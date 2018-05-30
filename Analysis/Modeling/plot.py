import numpy as np
## %matplotlib inline
import functions.plot as plot

data = np.load('UbelixData/data/each_class_s_r7320712from0to16.npz')
print(data.files)

num_plot = len(np.unique(data["aucs_all"]))

print(len(np.unique(data["class_tested"])))

classes = [['Face', 'Art', 'Landscape'],
           ['sienna', 'b', 'g'],
           ['sandybrown', 'lightskyblue', 'mediumseagreen']]

for k in range(int(num_plot/3)):
    for i in range(3):
        plt = plot.each_class(data['tprs_all'][k*3+i],
                              data['mean_fpr_all'][k*3+i],
                              data['aucs_all'][k*3+i],
                              str(k),
                              classes, i)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    plt.show()