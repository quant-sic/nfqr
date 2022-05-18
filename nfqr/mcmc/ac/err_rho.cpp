#include <cmath>
#include <memory>
#include <torch/extension.h>

#include <vector>

torch::Tensor err_rho(const int N,const int t_max, int w_opt, torch::Tensor rho){

    torch::Tensor ext_rho = torch::zeros({2 * t_max + w_opt + 1},torch::kFloat32);
    torch::Tensor err_rho_out = torch::zeros({t_max +1},torch::kFloat32);
    
    using namespace torch::indexing;
    ext_rho.index_put_({ Slice(None,t_max+1) },rho.index({Slice()}));

    float* err_rho_ptr = err_rho_out.data_ptr<float>();
    float* ext_rho_ptr = ext_rho.data_ptr<float>();

    for(int w=0;w<t_max+1;w++){
        for( int k=std::max(1,w - w_opt);k<w + w_opt + 1;k++){
            const int k_w= abs(k-w);
            err_rho_ptr[w] += pow((ext_rho_ptr[k + w] + ext_rho_ptr[k_w] - 2.0 * ext_rho_ptr[w] * ext_rho_ptr[k]),2);
        }
        err_rho_ptr[w] = sqrt(err_rho_ptr[w] / N);
    }
    return err_rho_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("err_rho", &err_rho);
}
