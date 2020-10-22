import os
import scipy.io as sio
import numpy as np
import time
import sys
sys.path.append("/home/pei/CLR_SC/RCAE_code/")
import matplotlib.pyplot as plt
from RCAE import RCAE
import funs as Ifuns

knn = 20
graph_way = "t_free"

print("knn = ", knn)

lam = 1

#  for lam_s in [0.01, 0.1, 1, 10, 100]:
lam_s = 10

X, y_true, N, dim, c_true = Ifuns.load_mat("/home/pei/DATA/BinaryAlpha_20200916.mat")
print(N)

obj = RCAE(X, c_true)
obj.clu(graph_knn=knn, graph_way=graph_way, lam=lam, lam_s = lam_s)
y = obj.y_pre

acc = Ifuns.accuracy(y_true, y)
print("{:.3f}".format(acc))
