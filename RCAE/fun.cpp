#include "fun.hpp"

std::vector<std::vector<double>> EProjSimplex_new_M( std::vector<std::vector<double>> & V, int k){
    std::vector<std::vector<double>> Ret(V.size(), std::vector<double>(V[0].size(), 0));
    #pragma omp parallel for
    for (int i = 0; i < V.size(); i++){
	EProjSimplex_new_cpp(V[i], k, Ret[i]);
    }
    return Ret;
}
	
void EProjSimplex_new_cpp(std::vector<double> &v, int k, std::vector<double> &x){

    int ft = 1;
    int n = v.size();
    double f = 1;
    double lam_m = 0;
    int npos = 0;

    double v_mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
    double kdn = (double)k/n;
    for (auto &ele : v){
	ele += kdn - v_mean;
    }
    double vmin = *std::min_element(v.begin(), v.end());

    if (vmin < 0){
	while (fabs(f) > 1e-10){
	    npos = 0;
	    f = 0;
	    for (int i = 0; i < n; i++){
		x[i] = v[i] - lam_m;
		if (x[i] > 0){
		    f += x[i];
		    npos += 1;
		}
	    }
	    f -= k;
	    lam_m += f/npos;
	    ft += 1;

	    if (ft > 100){
		break;
	    }
	}
	for (auto &ele : x){
            if (ele < 0){
		ele = 0;
	    }
	}
    }else{
	for (int i = 0; i < n; i++){
            x[i] = v[i];
	}
    }
}

// std::vector<double> EProjSimplex_new_cpp(std::vector<double> &v, int k){
//
//     int ft = 1;
//     int n = v.size();
//     double f = 1;
//     double lam_m = 0;
//     int npos = 0;
//
//     double v_mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
//     double kdn = (double)k/n;
//     for (auto &ele : v){
//         ele += kdn - v_mean;
//     }
//
//     std::vector<double> x(v);
//
//     double vmin = *std::min_element(v.begin(), v.end());
//
//     if (vmin < 0){
//         while (fabs(f) > 1e-10){
//             npos = 0;
//             f = 0;
//             for (int i = 0; i < n; i++){
//                 x[i] = v[i] - lam_m;
//                 if (x[i] > 0){
//                     f += x[i];
//                     npos += 1;
//                 }
//             }
//             f -= k;
//             lam_m += f/npos;
//             ft += 1;
//
//             if (ft > 100){
//                 break;
//             }
//         }
//         for (auto &ele : x){
//             if (ele < 0){
//                 ele = 0;
//             }
//         }
//     }else{
//         for (int i = 0; i < n; i++){
//             x[i] = v[i];
//         }
//     }
//
//     return x;
// }
