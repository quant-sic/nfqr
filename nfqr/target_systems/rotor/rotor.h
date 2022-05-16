#ifndef _ROTOR_H_
#define _ROTOR_H_


#include <torch/types.h>

#include <iostream>
#include <vector>

#include "observable.h"
#include "action.h"

class QR : public Action {

    private:
        double beta;

    public:
        QR(double beta):beta(beta){};

        torch::Tensor evaluate(torch::Tensor config);
        torch::Tensor force(torch::Tensor config);
        torch::Tensor map_to_range(torch::Tensor config);

};



class TopologicalCharge : public Observable {

    public:

        torch::Tensor evaluate(torch::Tensor config);

};

class TopologicalSusceptibility : public Observable {

    private:
        TopologicalCharge charge;

    public:
        TopologicalSusceptibility(){
            charge = TopologicalCharge();
        }
        
        torch::Tensor evaluate(torch::Tensor config);

};





#endif //_ROTOR_H_