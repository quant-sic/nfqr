#ifndef _OBSERVABLE_H_
#define _OBSERVABLE_H_

#include <torch/extension.h>
#include "sample_state.h"

class Observable {

public:
  virtual double evaluate(std::shared_ptr<SampleState> config) = 0;
  virtual torch::Tensor evaluate(torch::Tensor)=0;
};

#endif // _OBSERVABLE_H_