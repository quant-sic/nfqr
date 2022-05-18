#include "rotor.h"
#include <cmath>

double QR::evaluate(std::shared_ptr<SampleState> config) {

  double S;

  // S = beta*(1.0 - cos(config->data - config->data(circular_shift_left{config->size}))).sum();
  
  S = beta*(1.0 - cos(config->data - config->data(config->_circular_shift_left))).sum();

  return S;

  // int size = config->size;
  // double x_diff = config->data[0] - config->data[size - 1];
  // double S = 1. - cos(x_diff);
  // for (unsigned int j = 1; j < size; ++j) {
  //   double x_diff = config->data[j] - config->data[j - 1];
  //   S += 1. - cos(x_diff);
  // }
  // return beta * S;
};

double test_action(torch::Tensor config, std::shared_ptr<Action> action) {

  auto state = std::make_shared<SampleState>(config);

  return action->evaluate(state);
}



void QR::force(std::shared_ptr<SampleState> config,
               std::shared_ptr<SampleState> force_out) {

    // std::cout<< "array"<< config->data<< "\n";
    // std::cout<< "shift left"<<config->data(circular_shift_left{config->size})<< "\n";
    
    // std::cout<< "shift right"<<config->data(circular_shift_right{config->size})<< "\n";

    if (config->size > 75){
      force_out->data = beta * (sin(config->data - config->data(config->_circular_shift_right)) + sin(config->data - config->data(config->_circular_shift_left)));
    }else{
      const int size = config->size;
      // // Left boundary
      double x_m = config->data[size - 1];
      double x = config->data[0];
      double x_p = config->data[1];
      force_out->data[0] = beta * (sin(x - x_m) + sin(x - x_p));

      // Interior points
      for (unsigned int j = 1; j < size - 1; ++j) {
        x_m = config->data[j - 1];
        x = config->data[j];
        x_p = config->data[j + 1];
        force_out->data[j] = beta * (sin(x - x_m) + sin(x - x_p));
      }
      // Right boundary
      x_m = config->data[size - 2];
      x = config->data[size - 1];
      x_p = config->data[0];
      force_out->data[size - 1] = beta * (sin(x - x_m) + sin(x - x_p));
    }




    // using Eigen::last;

    // force_out->data(Eigen::seq(1,last-1) = beta*(sin(config->data(Eigen::seq(1,last-1))-config->data(Eigen::seq(0,last-2)))+ sin(config->data(Eigen::seq(1,last-1))-config->data(Eigen::seq(2,last))));
    // force_out->data[0] += beta*(sin(config->data[0]-config->data(last)) + sin(config->data[0]-config->data[1]));
    // force_out->data(last) += beta*(sin(config->data(last)-config->data(last-1)) + sin(config->data(last)-config->data[0]));
};

void QR::evaluate(std::shared_ptr<torch::Tensor> config,std::shared_ptr<torch::Tensor> action_out){

    *action_out = beta * (1.0 - torch::cos(torch::roll(*config, 1, -1) - *config)).sum(-1);
};


void QR::force(std::shared_ptr<torch::Tensor> config,std::shared_ptr<torch::Tensor> force_out){

    *force_out=beta * (
            torch::sin(*config - torch::roll(*config, -1, -1))
            - torch::sin(torch::roll(*config, 1, -1) - *config)
        );

};

torch::Tensor QR::map_to_range(torch::Tensor config){

    return config%(2*M_PI);
};





double test_sus(torch::Tensor config,
                std::shared_ptr<TopologicalSusceptibility> sus) {

  auto state = std::make_shared<SampleState>(config);

  return sus->evaluate(state);
}

double TopologicalSusceptibility::evaluate(std::shared_ptr<SampleState> config) {

  double charge_value = charge.evaluate(config);
  return charge_value * charge_value;
}

double inline mod_2pi(const double x) {
  return x - 2. * M_PI * floor(0.5 * (x + M_PI) / M_PI);
}

double TopologicalCharge::evaluate(std::shared_ptr<SampleState> config) {

  double diff = config->data[config->size - 1] - config->data[0];
  double charge = mod_2pi(diff);

  for (unsigned int j = 0; j < config->size - 1; ++j) {
    double diff = config->data[j] - config->data[j + 1];
    charge += mod_2pi(diff);
  }

  return charge / (2 * M_PI);
}

torch::Tensor TopologicalSusceptibility::evaluate(torch::Tensor config){

    return torch::pow(charge.evaluate(config),2);
}

torch::Tensor TopologicalCharge::evaluate(torch::Tensor config){

    return ((torch::roll(config, 1, -1) - config + M_PI) % (2 * M_PI) - M_PI).sum(-1) / (2 * M_PI);
}
