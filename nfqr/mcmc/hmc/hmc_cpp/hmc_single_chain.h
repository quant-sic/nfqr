#ifndef _HMC_SINGLE_CHAIN_H_
#define _HMC_SINGLE_CHAIN_H_

#include <random>
#include "observable.h"
#include "action.h"
#include <torch/extension.h>
#include <memory>

class HMC_Single_Config {

public:
  HMC_Single_Config(std::shared_ptr<Observable> _observable, std::shared_ptr<Action> _action,
      int _size) {
    action = _action;
    size = _size;
    observable = _observable;
  }

  void initialize() {

    n_accepted = 0;
    n_steps_taken = 0;

    std::random_device rd;
    engine = std::mt19937_64(rd());
    normal_dist = std::normal_distribution<double>(0.0, 1.0);
    uniform_dist = std::uniform_real_distribution<double>(0.0, 1.0);

    current_config = std::make_shared<SampleState>(size);

    std::generate(current_config->data.data(),
                  current_config->data.data() + size,
                  [this]() { return uniform_dist(engine) * 2 * M_PI; });

    q = std::make_shared<SampleState>(size);
    p = std::make_shared<SampleState>(size);
    force = std::make_shared<SampleState>(size);

  }

  void set_n_accepted(int _n_accepted) { n_accepted = _n_accepted; }
  const int get_n_accepted() const { return n_accepted; }

  void set_n_steps_taken(int _n_steps_taken) { n_steps_taken = _n_steps_taken; }
  const int get_n_steps_taken() const { return n_steps_taken; }

  const float get_acceptance_rate() const { 
      if(n_steps_taken==0){return 0.0;} 
      else {return ((float)n_accepted) / ((float)n_steps_taken); }}

  void set_current_config(torch::Tensor _current_config) {
    current_config = std::make_shared<SampleState>(_current_config);
  }
  torch::Tensor get_current_config() const {
    return torch::from_blob(current_config->data.data(), {size},
                            torch::kFloat64)
        .clone();
  }

  void reset_expectation_values() {
    expectation_values = std::vector<double>();
  }
  std::vector<double> get_expectation_values() const {
    return expectation_values;
  }

  void advance(int n_steps, int n_traj_steps, double step_size);

  void burnin(int n_burnin_steps, int n_traj_steps, double step_size){
      this->advance(n_burnin_steps, n_traj_steps, step_size);
      this->reset_expectation_values();
      this->set_n_steps_taken(0);
      this->set_n_accepted(0);
  }

  void inline leapfrog(std::shared_ptr<SampleState> q, std::shared_ptr<SampleState> p,
              double step_size, int n_traj_steps,
              std::shared_ptr<Action> action,
              std::shared_ptr<SampleState> force);


private:
  unsigned int size;
  std::shared_ptr<Action> action;
  std::shared_ptr<SampleState> current_config;
  unsigned int n_accepted;
  unsigned int n_steps_taken;
  float acceptance_rate;
  std::vector<double> expectation_values;

  std::mt19937_64 engine;
  std::uniform_real_distribution<double> uniform_dist;
  std::normal_distribution<double> normal_dist;

  std::shared_ptr<SampleState> q;
  std::shared_ptr<SampleState> p;
  std::shared_ptr<SampleState> force;

  std::shared_ptr<Observable> observable;
};


#endif //_HMC_SINGLE_CHAIN_H_