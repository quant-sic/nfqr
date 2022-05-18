#include "hmc_n_chains.h"

void inline HMC_Batch::leapfrog (std::shared_ptr<torch::Tensor> q,std::shared_ptr<torch::Tensor> p, double step_size,int traj_steps,std::shared_ptr<Action> action,std::shared_ptr<torch::Tensor> force){

    for(int traj_step=0; traj_step<traj_steps;traj_step++){
        action->force(q,force);

        *p += step_size / 2 * (*force);
        *q += *p * step_size;
        
        action->force(q,force);
        *p += step_size / 2 * (*force);
    }
}

void HMC_Batch::advance(int n_steps,int n_traj_steps,double step_size){
    torch::NoGradGuard no_grad;

    for(int step=0;step<n_steps;step++){

        (*q).copy_(*current_config);
        action->evaluate(q,action_out);


        (*p) = torch::randn_like(*p);
        h = 0.5 * torch::pow(*p,2).sum(-1) + *action_out;

        leapfrog(q,p,step_size,n_traj_steps,action,force);

        action->evaluate(q,action_out);

        hp = 0.5 * torch::pow(*p,2).sum(-1) + *action_out;

        log_ratio = h - hp;

        accept_mask = (log_ratio >= 0) | (torch::log(torch::rand_like(log_ratio)) < log_ratio);

        using namespace torch::indexing;

        // std::cout << accept_mask << "\n";
        // std::cout << log_ratio << "\n";

        n_accepted += accept_mask;
        
        (*current_config).index_put_({accept_mask,"..."},(*q).index({accept_mask,"..."}));
        // _current_config = (~accept_mask) * _current_config + accept_mask*(*q_ptr); 
        
        // _current_config = action->map_to_range(_current_config);
        
        expectation_values.push_back(observable->evaluate(*current_config).clone());

        n_steps_taken++;
    };
};
    






