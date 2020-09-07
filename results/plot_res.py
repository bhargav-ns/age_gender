
import matplotlib.pyplot as plt
import numpy as np
import csv
from numpy import genfromtxt
import pdb

add_98_pre_list = np.delete(np.ndarray.tolist(np.diagonal(genfromtxt('prob_conf_mat5_98.csv', delimiter=','))),0)
add_98_post_list = np.delete(np.ndarray.tolist(np.diagonal(genfromtxt('prob_conf_mat_post.csv', delimiter=','))),0)
stage4_pre_list = np.delete(np.ndarray.tolist(np.diagonal(genfromtxt('prob_conf_mat5_4.csv', delimiter=','))),0)
stage3_pre_list = np.delete(np.ndarray.tolist(np.diagonal(genfromtxt('prob_confs_mat5_3.csv', delimiter=','))),0)
pdb.set_trace()

data = [
    [0.88,0.688,0.844,0.728,0.633],
    add_98_post_list,
    add_98_pre_list,
    stage3_pre_list,
    stage4_pre_list
]


N = 5
ind = np.arange(N) 
width = 0.1       
plt.bar(ind, data[0], width, 
    label='ResNet50 - Previous Benchmark')
plt.bar(ind + width, data[1], width,
    label='AGV Pipeline - Add98 Post-facto')
plt.bar(ind + 2*width, data[2], width,
    label='AGV Pipeline - Add98 Pre-facto')
plt.bar(ind + 3*width, data[3], width,
    label='AGV Pipeline - Stage3_unit15 Pre-facto')
plt.bar(ind + 4*width, data[4], width,
    label='AGV Pipeline - Stage4_unit1 Pre-facto')


plt.ylabel('Normalized frequency of correct classification')
plt.title('Confusion Matrix Scores for Various Networks')

plt.xticks(ind + width / 2, ('Kids', 'Teens', 'Adults', 'Middle-aged', 'Old'))
plt.legend(loc='best')
plt.show()