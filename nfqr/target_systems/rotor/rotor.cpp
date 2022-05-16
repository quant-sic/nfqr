#include <iostream>
#include <vector>
#include <cmath>

#include <torch/extension.h>

#include "rotor.h"


torch::Tensor QR::evaluate(torch::Tensor config){

    auto _config = beta * (1.0 - torch::cos(torch::roll(config, 1, -1) - config)).sum(-1);

    return _config;
};


torch::Tensor QR::force(torch::Tensor config){

    auto _force = beta * (torch::sin(config - torch::roll(config, -1, -1)) - torch::sin(torch::roll(config, 1,-1) - config));

    return _force;
};


torch::Tensor QR::map_to_range(torch::Tensor config){

    return config%(2*M_PI);
};


torch::Tensor TopologicalSusceptibility::evaluate(torch::Tensor config){

    return torch::pow(charge.evaluate(config),2);
}

torch::Tensor TopologicalCharge::evaluate(torch::Tensor config){

    return ((torch::roll(config, 1, -1) - config + M_PI) % (2 * M_PI) - M_PI).sum(-1) / (2 * M_PI);
}

