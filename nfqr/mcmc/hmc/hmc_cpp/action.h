#ifndef _ACTION_H_
#define _ACTION_H_

#include <torch/extension.h>
#include "sample_state.h"
#include <memory>

class Action{

    public:
        virtual void evaluate(std::shared_ptr<torch::Tensor> config,std::shared_ptr<torch::Tensor> action_out) =0;
        virtual void force(std::shared_ptr<torch::Tensor> config,std::shared_ptr<torch::Tensor> force_out) =0;
        virtual torch::Tensor map_to_range(torch::Tensor config) =0;
        virtual double evaluate(std::shared_ptr<SampleState> config) = 0;
        virtual void force(std::shared_ptr<SampleState> config,
                            std::shared_ptr<SampleState> force_out) = 0;
};



#endif //_ACTION_H_