from libcpp.vector cimport vector

cdef extern from "fun.cpp":
    pass

cdef extern from "fun.hpp":
    #  vector[double] EProjSimplex_new_cpp(vector[double] &v, int k);
    void EProjSimplex_new_cpp(vector[double] &v, int k, vector[double] &x);
    vector[vector[double]] EProjSimplex_new_M(vector[vector[double]] &V, int k);
