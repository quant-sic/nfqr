#include <torch/extension.h>
#include "hmc_n_chains.h"
#include "hmc_single_chain.h"
#include "action.h"
#include "rotor.h"
#include "observable.h"
#include <memory>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  py::class_<Action, std::shared_ptr<Action> /* <- holder type */>(m, "Action");


  py::class_<QR, std::shared_ptr<QR> /* <- holder type */, Action>(m, "QR")
      .def(py::init<double>())
      
      .def("evaluate_batch", [](QR &self, torch::Tensor config){
        auto config_ptr = std::make_shared<torch::Tensor>(config);
        auto action_out = std::make_shared<torch::Tensor>(torch::empty({config.size(0)}));
        self.evaluate(config_ptr,action_out);

        return *action_out;
      }, "evaluate torch batch action")
      
      .def("evaluate_single_config", [](QR &self, torch::Tensor config){
        auto config_state = std::make_shared<SampleState>(config);
        return self.evaluate(config_state);
      }, "evaluate single config action")

      .def("force_batch", [](QR &self, torch::Tensor config){
        auto config_ptr = std::make_shared<torch::Tensor>(config);
        auto force_out = std::make_shared<torch::Tensor>(torch::empty({config.size(0)}));
        self.force(config_ptr,force_out);

        return *force_out;
      }, "evaluate torch batch force")
      
      .def("force_single_config", [](QR &self, torch::Tensor config){
        auto config_state = std::make_shared<SampleState>(config);
        auto force_state = std::make_shared<SampleState>(config.size(0));
        self.force(config_state,force_state);

        auto out_tensor = torch::from_blob(force_state->data.data(),{config.size(0)},torch::kFloat64).clone();
        return out_tensor;
      }, "evaluate single config force");


  py::class_<HMC_Single_Config>(m, "HMC_Single_Config")
      .def(
          py::init<std::shared_ptr<Observable>, std::shared_ptr<Action>, int>(),py::arg("observable"),py::arg("action"),py::arg("size"))
      .def_property("n_accepted", &HMC_Single_Config::get_n_accepted, &HMC_Single_Config::set_n_accepted)
      .def_property("current_config", &HMC_Single_Config::get_current_config,
                    &HMC_Single_Config::set_current_config)
      .def_property("n_steps_taken", &HMC_Single_Config::get_n_steps_taken,
                    &HMC_Single_Config::get_n_steps_taken)
      .def_property_readonly("expectation_values", &HMC_Single_Config::get_expectation_values)
      .def_property_readonly("acceptance_rate", &HMC_Single_Config::get_acceptance_rate)
      .def("reset_expectation_values", &HMC_Single_Config::reset_expectation_values)
      .def("burnin", &HMC_Single_Config::burnin,py::arg("n_steps"),py::arg("n_traj_steps"),py::arg("step_size"))
      .def("initialize", &HMC_Single_Config::initialize)
      .def("advance", &HMC_Single_Config::advance,py::arg("n_steps"),py::arg("n_traj_steps"),py::arg("step_size"));

  py::class_<HMC_Batch>(m, "HMC_Batch")
      .def(
          py::init<std::shared_ptr<Observable>, std::shared_ptr<Action>, int,int>(),py::arg("observable"),py::arg("action"),py::arg("size"),py::arg("batch_size"))
      .def_property("n_accepted", &HMC_Batch::get_n_accepted, &HMC_Batch::set_n_accepted)
      .def_property("current_config", &HMC_Batch::get_current_config,
                    &HMC_Batch::set_current_config)
      .def_property("n_steps_taken", &HMC_Batch::get_n_steps_taken,
                    &HMC_Batch::get_n_steps_taken)
      .def_property_readonly("expectation_values", &HMC_Batch::get_expectation_values)
      .def_property_readonly("acceptance_rate", &HMC_Batch::get_acceptance_rate)
      .def("reset_expectation_values", &HMC_Batch::reset_expectation_values)
      .def("burnin", &HMC_Batch::burnin,py::arg("n_steps"),py::arg("n_traj_steps"),py::arg("step_size"))
      .def("initialize", &HMC_Batch::initialize)
      .def("advance", &HMC_Batch::advance,py::arg("n_steps"),py::arg("n_traj_steps"),py::arg("step_size"));


  py::class_<Observable, std::shared_ptr<Observable> /* <- holder type */>(
      m, "Observable");

  py::class_<TopologicalSusceptibility,
             std::shared_ptr<TopologicalSusceptibility> /* <- holder type */,
             Observable>(m, "TopologicalSusceptibility")
      .def(py::init<>())
      .def("evaluate_batch", py::overload_cast<torch::Tensor>(&TopologicalSusceptibility::evaluate))
      .def("evaluate_single_config", [](TopologicalSusceptibility &self, torch::Tensor config){
        auto config_state = std::make_shared<SampleState>(config);
        return self.evaluate(config_state);
      }, "evaluate single config action");
      // .def("evaluate", &TopologicalSusceptibility::evaluate,
      //      "evaluate susceptibility");

  py::class_<TopologicalCharge,
             std::shared_ptr<TopologicalCharge> /* <- holder type */,
             Observable>(m, "TopologicalCharge")
      .def(py::init<>())
      .def("evaluate_batch", py::overload_cast<torch::Tensor>(&TopologicalCharge::evaluate))
      .def("evaluate_single_config", [](TopologicalCharge &self, torch::Tensor config){
        auto config_state = std::make_shared<SampleState>(config);
        return self.evaluate(config_state);
      }, "evaluate single config action");
      // .def("evaluate", &TopologicalCharge::evaluate, "evaluate charge");
}