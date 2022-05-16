#ifndef _ACTION_H_
#define _ACTION_H_

#include <torch/types.h>
#include 

class Action{

    public:
        virtual torch::Tensor evaluate(torch::Tensor config) =0;
        virtual torch::Tensor force(torch::Tensor config) =0;
        virtual torch::Tensor map_to_range(torch::Tensor config) =0;
};

class Action {

    public:
        virtual double evaluate(std::shared_ptr<SampleState> config) = 0;
        virtual void force(std::shared_ptr<SampleState> config,
                            std::shared_ptr<SampleState> force_out) = 0;
        // virtual torch::Tensor map_to_range(torch::Tensor config) =0;
};

// class pyAction :public Action{

//     public:
//         using Action::Action;

//         /* Trampoline (need one for each virtual function) */
//         torch::Tensor evaluate(torch::Tensor config) override {
//             PYBIND11_OVERRIDE_PURE(
//                 torch::Tensor, /* Return type */
//                 Action,      /* Parent class */
//                 evaluate,          /* Name of function in C++ (must match Python name) */
//                 config      /* Argument(s) */
//             );
//         } 

//         torch::Tensor force(torch::Tensor config) override {
//             PYBIND11_OVERRIDE_PURE(
//                 torch::Tensor, /* Return type */
//                 Action,      /* Parent class */
//                 force,          /* Name of function in C++ (must match Python name) */
//                 config      /* Argument(s) */
//             );
//         } 


// };

#endif //_ROTOR_H_