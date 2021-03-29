import os
import scipy.io as sio
import numpy as np
import time
import sys
from RCAE_pack.RCAE import RCAE
from RCAE_pack.Public import Ifuns, Gfuns, Mfuns

knn = 20
graph_way = "t_free"

lam = 1
#  for lam_s in [0.01, 0.1, 1, 10, 100]:
lam_s = 10

X, y_true, N, dim, c_true = Ifuns.load_mat("/home/pei/DATA/BinaryAlpha_20200916.mat")

G = Gfuns.kng(X, knn=knn, way=graph_way, isSym=True)
obj = RCAE(G, c_true)
obj.clu(lam=lam, lam_s = lam_s)
y = obj.y_pre

acc = Mfuns.accuracy(y_true, y)
print("{:.3f}".format(acc))

#  paper: BinAlaph acc = 0.499
#  run :  Binalaph acc = 0.489
