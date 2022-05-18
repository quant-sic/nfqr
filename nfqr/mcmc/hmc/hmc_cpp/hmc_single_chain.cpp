#include "hmc_single_chain.h"
#include <memory>
#include "rotor.h"

void inline HMC_Single_Config::leapfrog(std::shared_ptr<SampleState> q, std::shared_ptr<SampleState> p,
              double step_size, int n_traj_steps,
              std::shared_ptr<Action> action,
              std::shared_ptr<SampleState> force) {

  for (unsigned int traj_step = 0; traj_step < n_traj_steps; traj_step++) {

    action->force(q, force);

    p->data += step_size / 2 * force->data;
    q->data += p->data * step_size;

    action->force(q, force);
    p->data += step_size / 2 * force->data;
  }
}

void HMC_Single_Config::advance(int n_steps, int n_traj_steps,double step_size) {

  double h;
  double hp;
  double log_ratio;
  double exp_value = observable->evaluate(current_config);

  for (unsigned int step = 0; step < n_steps; step++) {

    q->data = current_config->data;

    std::generate(p->data.data(), p->data.data() + size,
                  [this]() { return normal_dist(engine); });


    h = 0.5 * (p->data * p->data).sum() + action->evaluate(q);

    this->leapfrog(q, p, step_size, n_traj_steps, action,force);

    hp = 0.5 * (p->data * p->data).sum() + action->evaluate(q);

    log_ratio = h - hp;

    if ((log_ratio >= 0) | (log(uniform_dist(engine)) < log_ratio)) {
      n_accepted++;
      current_config->data = q->data;
      exp_value = observable->evaluate(current_config);
    }

    n_steps_taken++;
    expectation_values.push_back(exp_value);

  };
};




std::vector<torch::Tensor> test_leapfrog(torch::Tensor q, torch::Tensor p,
                                         double step_size, int n_traj_steps,
                                         std::shared_ptr<Action> action) {

  auto state_q_ptr = std::make_shared<SampleState>(q);
  auto state_p_ptr = std::make_shared<SampleState>(p);
  auto force_ptr = std::make_shared<SampleState>(p);

  HMC_Single_Config hmc (std::make_shared<TopologicalCharge> (),action,q.size(0));

  hmc.leapfrog(state_q_ptr, state_p_ptr, step_size, n_traj_steps, action,force_ptr);

  auto q_clone =
      torch::from_blob((state_q_ptr->data).data(), {q.size(0)}, torch::kFloat64)
          .clone();
  auto p_clone =
      torch::from_blob((state_p_ptr->data).data(), {p.size(0)}, torch::kFloat64)
          .clone();
  // auto p_clone = torch::empty({p.size(0)});
  // std::memcpy(p_clone.data_ptr(),(state_p_ptr->data).data(),
  // sizeof(double)*q.size(0));

  return {q_clone, p_clone};
}