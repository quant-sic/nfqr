#ifndef _HMC_N_CHAINS_H_
#define _HMC_N_CHAINS_H_

#include <random>
#include "observable.h"
#include "action.h"
#include <torch/extension.h>
#include <memory>

class HMC_Batch {

public:
  HMC_Batch(std::shared_ptr<Observable> _observable, std::shared_ptr<Action> _action,
      int _size,int _batch_size) {
    action = _action;
    size = _size;
    observable = _observable;
    batch_size = _batch_size;
  }

  void initialize() {

    n_accepted = torch::zeros({batch_size},torch::kInt64);
    n_steps_taken = 0;

    current_config = std::make_shared<torch::Tensor>(torch::rand({batch_size,size},torch::kFloat64));
    *current_config *= 2*M_PI;

    q = std::make_shared<torch::Tensor>(torch::empty_like(*current_config));
    p = std::make_shared<torch::Tensor>(torch::empty_like(*current_config));
    force = std::make_shared<torch::Tensor>(torch::empty_like(*current_config));

    h = torch::empty({batch_size},torch::kFloat64);
    hp = torch::empty({batch_size},torch::kFloat64);
    log_ratio = torch::empty({batch_size},torch::kFloat64);
    accept_mask = torch::empty({batch_size},torch::kFloat64);

    action_out = std::make_shared<torch::Tensor>(torch::empty({batch_size},torch::kFloat64));

  }

  void set_n_accepted(torch::Tensor _n_accepted) { n_accepted = _n_accepted; }
  const torch::Tensor get_n_accepted() const { return n_accepted; }

  void set_n_steps_taken(int _n_steps_taken) { n_steps_taken = _n_steps_taken; }
  const int get_n_steps_taken() const { return n_steps_taken; }

  const torch::Tensor get_acceptance_rate() const { 
      if(n_steps_taken==0){return torch::zeros({batch_size});} 
      else {return n_accepted.to(torch::kFloat32) / (float)n_steps_taken; }}

  void set_current_config(torch::Tensor _current_config) {
    current_config = std::make_shared<torch::Tensor>(_current_config);
  }
  torch::Tensor get_current_config() const {
    return *current_config;
  }

  void reset_expectation_values() {
    expectation_values = std::vector<torch::Tensor>();
  }
  std::vector<torch::Tensor> get_expectation_values() const {
    return expectation_values;
  }

  void advance(int n_steps, int n_traj_steps, double step_size);

  void burnin(int n_burnin_steps, int n_traj_steps, double step_size){
      this->advance(n_burnin_steps, n_traj_steps, step_size);
      this->reset_expectation_values();
      this->set_n_steps_taken(0);
      this->set_n_accepted(torch::zeros({batch_size}));
  }

  void inline leapfrog(std::shared_ptr<torch::Tensor> q, std::shared_ptr<torch::Tensor> p,
              double step_size, int n_traj_steps,
              std::shared_ptr<Action> action,
              std::shared_ptr<torch::Tensor> force);


private:
  unsigned int size;
  unsigned int batch_size;
  std::shared_ptr<Action> action;
  std::shared_ptr<torch::Tensor> current_config;
  unsigned int n_steps_taken;
  float acceptance_rate;

  std::vector<torch::Tensor> expectation_values;
  torch::Tensor n_accepted;

  std::shared_ptr<torch::Tensor> q;
  std::shared_ptr<torch::Tensor> p;
  std::shared_ptr<torch::Tensor> force;
  std::shared_ptr<torch::Tensor> action_out;


  torch::Tensor h; 
  torch::Tensor hp;
  torch::Tensor log_ratio;
  torch::Tensor accept_mask;

  std::shared_ptr<Observable> observable;
};

#endif //_HMC_N_CHAINS_H_