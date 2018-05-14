import numpy as np
## %matplotlib inline
import functions.plot as plot
data=np.load('tmp_each_class.npz')
num_plot=len(np.unique(data["comparison"]))*len(np.unique(data["class_tested"]))
print(data["class_tested"])
print(len(np.unique(data["class_tested"])))
for i in range(1):
      plot.each_class(data['tprs_all'][i],
               data['mean_fpr_all'][i],
               data['aucs_all'][i])