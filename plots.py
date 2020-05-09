import numpy as np
import pickle
import matplotlib.pyplot as plt


path = "save_states//"
filenames = [ "linear_rank_len5_plot.p",  "linear_rank_len10_plot.p",  "linear_rank_len15_plot.p",
                 "sin_rank_len5_plot.p",     "sin_rank_len10_plot.p",     "sin_rank_len15_plot.p",
             "relu_rank_len5_plot.p", "relu_rank_len10_plot.p", "relu_rank_len15_plot.p"  ]


name = filenames[0]              

with open(path+name, "rb") as f:
    x = pickle.load(f)

loss_list, testloss_list, accu_list = x
accu_list = (np.array(accu_list)/100.0).tolist()

print(len(accu_list))
print(len(loss_list))


plt.plot( range(1,len(loss_list)+1), accu_list, 'b' )
plt.plot( range(1,len(loss_list)+1), loss_list, 'r' )
plt.plot( range(1,len(loss_list)+1), testloss_list, 'g' )
plt.ylim((-0.1,1.1))
plt.show()
