import numpy as np
## %matplotlib inline
import functions.plot as plot
from functions.stats import bootstrap
data = np.load('each_class_s_r7320713from0to16.npz')

print(data.files)

num_plot = len(np.unique(data["aucs_all"]))

print(len(np.unique(data["class_tested"])))
np.savetxt("classificationbyimg.csv", data['aucs_all'], delimiter=",")
classes = [['Face', 'Art', 'Landscape'],
           ['sienna', 'b', 'g'],
           ['sandybrown', 'lightskyblue', 'mediumseagreen']]
n_classes = 2
samples=len(data['aucs_all'])
for k in range(int(n_classes)):
    #print('Iteration: ', k)
    bootstrap(np.asarray(data['aucs_all'][(120*k)+60:(120*k)+120]).ravel()) #* n_classes:k * n_classes + n_classes

for k in range(int(num_plot/n_classes)):
    for i in range(n_classes):
        plt = plot.each_class(data['tprs_all'][k*n_classes+i],
                              data['mean_fpr_all'][k*n_classes+i],
                              data['aucs_all'][k*n_classes+i],
                              str(k),
                              classes, i)
    print('Iteration: ', k )
    bootstrap(data['aucs_all'][k*n_classes:k*n_classes+n_classes].ravel())
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    plt.show()