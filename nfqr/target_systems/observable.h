#ifndef _OBSERVABLE_H_
#define _OBSERVABLE_H_

#include <torch/types.h>

class Observable{

    public:
        virtual torch::Tensor evaluate(torch::Tensor)=0;

};

#endif // _OBSERVABLE_H_