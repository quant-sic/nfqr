#ifndef _ROTOR_H_
#define _ROTOR_H_

#include <torch/extension.h>
#include "sample_state.h"
#include "observable.h"
#include <memory>
#include "action.h"

class QR : public Action {

  private:
    double beta;

  public:
    QR(double beta) : beta(beta){};

    double evaluate(std::shared_ptr<SampleState> config);
    void force(std::shared_ptr<SampleState> config,
              std::shared_ptr<SampleState> force_out);
    void evaluate(std::shared_ptr<torch::Tensor> config,std::shared_ptr<torch::Tensor> action_out);
    void force(std::shared_ptr<torch::Tensor> config,std::shared_ptr<torch::Tensor> force_out);
    torch::Tensor map_to_range(torch::Tensor config);
};

class TopologicalCharge : public Observable {

  public:
    double evaluate(std::shared_ptr<SampleState> config);
    torch::Tensor evaluate(torch::Tensor config);

};

class TopologicalSusceptibility : public Observable {

private:
  TopologicalCharge charge;

public:
  TopologicalSusceptibility() { charge = TopologicalCharge(); }

  double evaluate(std::shared_ptr<SampleState> config);
  torch::Tensor evaluate(torch::Tensor config);

};


#endif //_ROTOR_H_