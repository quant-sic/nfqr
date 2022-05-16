#include <cmath>
#include <memory>
#include <random>
#include <torch/extension.h>

#include <Eigen/Core>
#include <vector>

struct circular_shift_left {
  Eigen::Index size() const { return _size; }
  Eigen::Index operator[] (Eigen::Index i) const { return (i+1)%_size;}
  Eigen::Index _size;
};

struct circular_shift_right {
  Eigen::Index size() const { return _size; }
  Eigen::Index operator[] (Eigen::Index i) const { return (i+_size - 1) % _size;}
  Eigen::Index _size;
};

struct mapped_index {
  Eigen::Index size() const { return _size; }
  Eigen::Index operator[] (Eigen::Index i) const { return (*map)[i];}
  std::shared_ptr<Eigen::ArrayXi> map; 
  Eigen::Index _size;
};

class SampleState {
    public:
        SampleState(torch::Tensor tensor) {
            size = tensor.size(0);
            data = Eigen::ArrayXd(size);
            std::memcpy(data.data(), tensor.data_ptr<double>(), sizeof(double) * size);
            
            _circular_shift_left = {size};
            _circular_shift_right = {size};
        }
        SampleState(Eigen::ArrayXd _data, int _size) {
            size = _size;
            data = _data;

            _circular_shift_left = {size};
            _circular_shift_right = {size};
        }
        SampleState(int _size) {
            size = _size;
            data = Eigen::ArrayXd(_size);
            std::memset(data.data(), 0, sizeof(double) * size);

            _circular_shift_left = {size};
            _circular_shift_right = {size};
        }


        Eigen::ArrayXd data;
        int size;

        circular_shift_left _circular_shift_left;
        circular_shift_right _circular_shift_right;

};



class Action{

    public:
        virtual void evaluate(std::shared_ptr<torch::Tensor> config,std::shared_ptr<torch::Tensor> action_out) =0;
        virtual void force(std::shared_ptr<torch::Tensor> config,std::shared_ptr<torch::Tensor> force_out) =0;
        virtual torch::Tensor map_to_range(torch::Tensor config) =0;
        virtual double evaluate(std::shared_ptr<SampleState> config) = 0;
        virtual void force(std::shared_ptr<SampleState> config,
                            std::shared_ptr<SampleState> force_out) = 0;
};


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

double QR::evaluate(std::shared_ptr<SampleState> config) {

  double S;

  // S = beta*(1.0 - cos(config->data - config->data(circular_shift_left{config->size}))).sum();
  
  S = beta*(1.0 - cos(config->data - config->data(config->_circular_shift_left))).sum();

  return S;

  // int size = config->size;
  // double x_diff = config->data[0] - config->data[size - 1];
  // double S = 1. - cos(x_diff);
  // for (unsigned int j = 1; j < size; ++j) {
  //   double x_diff = config->data[j] - config->data[j - 1];
  //   S += 1. - cos(x_diff);
  // }
  // return beta * S;
};

double test_action(torch::Tensor config, std::shared_ptr<Action> action) {

  auto state = std::make_shared<SampleState>(config);

  return action->evaluate(state);
}



void QR::force(std::shared_ptr<SampleState> config,
               std::shared_ptr<SampleState> force_out) {

    // std::cout<< "array"<< config->data<< "\n";
    // std::cout<< "shift left"<<config->data(circular_shift_left{config->size})<< "\n";
    
    // std::cout<< "shift right"<<config->data(circular_shift_right{config->size})<< "\n";

    if (config->size > 75){
      force_out->data = beta * (sin(config->data - config->data(config->_circular_shift_right)) + sin(config->data - config->data(config->_circular_shift_left)));
    }else{
      const int size = config->size;
      // // Left boundary
      double x_m = config->data[size - 1];
      double x = config->data[0];
      double x_p = config->data[1];
      force_out->data[0] = beta * (sin(x - x_m) + sin(x - x_p));

      // Interior points
      for (unsigned int j = 1; j < size - 1; ++j) {
        x_m = config->data[j - 1];
        x = config->data[j];
        x_p = config->data[j + 1];
        force_out->data[j] = beta * (sin(x - x_m) + sin(x - x_p));
      }
      // Right boundary
      x_m = config->data[size - 2];
      x = config->data[size - 1];
      x_p = config->data[0];
      force_out->data[size - 1] = beta * (sin(x - x_m) + sin(x - x_p));
    }




    // using Eigen::last;

    // force_out->data(Eigen::seq(1,last-1) = beta*(sin(config->data(Eigen::seq(1,last-1))-config->data(Eigen::seq(0,last-2)))+ sin(config->data(Eigen::seq(1,last-1))-config->data(Eigen::seq(2,last))));
    // force_out->data[0] += beta*(sin(config->data[0]-config->data(last)) + sin(config->data[0]-config->data[1]));
    // force_out->data(last) += beta*(sin(config->data(last)-config->data(last-1)) + sin(config->data(last)-config->data[0]));
};

void QR::evaluate(std::shared_ptr<torch::Tensor> config,std::shared_ptr<torch::Tensor> action_out){

    *action_out = beta * (1.0 - torch::cos(torch::roll(*config, 1, -1) - *config)).sum(-1);
};


void QR::force(std::shared_ptr<torch::Tensor> config,std::shared_ptr<torch::Tensor> force_out){

    *force_out=beta * (
            torch::sin(*config - torch::roll(*config, -1, -1))
            - torch::sin(torch::roll(*config, 1, -1) - *config)
        );

};

torch::Tensor QR::map_to_range(torch::Tensor config){

    return config%(2*M_PI);
};




























class Observable {

public:
  virtual double evaluate(std::shared_ptr<SampleState> config) = 0;
  virtual torch::Tensor evaluate(torch::Tensor)=0;
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

double test_sus(torch::Tensor config,
                std::shared_ptr<TopologicalSusceptibility> sus) {

  auto state = std::make_shared<SampleState>(config);

  return sus->evaluate(state);
}

double
TopologicalSusceptibility::evaluate(std::shared_ptr<SampleState> config) {

  double charge_value = charge.evaluate(config);
  return charge_value * charge_value;
}

double inline mod_2pi(const double x) {
  return x - 2. * M_PI * floor(0.5 * (x + M_PI) / M_PI);
}

double TopologicalCharge::evaluate(std::shared_ptr<SampleState> config) {

  double diff = config->data[config->size - 1] - config->data[0];
  double charge = mod_2pi(diff);

  for (unsigned int j = 0; j < config->size - 1; ++j) {
    double diff = config->data[j] - config->data[j + 1];
    charge += mod_2pi(diff);
  }

  return charge / (2 * M_PI);
}

torch::Tensor TopologicalSusceptibility::evaluate(torch::Tensor config){

    return torch::pow(charge.evaluate(config),2);
}

torch::Tensor TopologicalCharge::evaluate(torch::Tensor config){

    return ((torch::roll(config, 1, -1) - config + M_PI) % (2 * M_PI) - M_PI).sum(-1) / (2 * M_PI);
}











class HMC_Single_Config {

public:
  HMC_Single_Config(std::shared_ptr<Observable> _observable, std::shared_ptr<Action> _action,
      int _size) {
    action = _action;
    size = _size;
    observable = _observable;
  }

  void initialize() {

    n_accepted = 0;
    n_steps_taken = 0;

    std::random_device rd;
    engine = std::mt19937_64(rd());
    normal_dist = std::normal_distribution<double>(0.0, 1.0);
    uniform_dist = std::uniform_real_distribution<double>(0.0, 1.0);

    current_config = std::make_shared<SampleState>(size);

    std::generate(current_config->data.data(),
                  current_config->data.data() + size,
                  [this]() { return uniform_dist(engine) * 2 * M_PI; });

    q = std::make_shared<SampleState>(size);
    p = std::make_shared<SampleState>(size);
    force = std::make_shared<SampleState>(size);

  }

  void set_n_accepted(int _n_accepted) { n_accepted = _n_accepted; }
  const int get_n_accepted() const { return n_accepted; }

  void set_n_steps_taken(int _n_steps_taken) { n_steps_taken = _n_steps_taken; }
  const int get_n_steps_taken() const { return n_steps_taken; }

  const float get_acceptance_rate() const { 
      if(n_steps_taken==0){return 0.0;} 
      else {return ((float)n_accepted) / ((float)n_steps_taken); }}

  void set_current_config(torch::Tensor _current_config) {
    current_config = std::make_shared<SampleState>(_current_config);
  }
  torch::Tensor get_current_config() const {
    return torch::from_blob(current_config->data.data(), {size},
                            torch::kFloat64)
        .clone();
  }

  void reset_expectation_values() {
    expectation_values = std::vector<double>();
  }
  std::vector<double> get_expectation_values() const {
    return expectation_values;
  }

  void advance(int n_steps, int n_traj_steps, double step_size);

  void burnin(int n_burnin_steps, int n_traj_steps, double step_size){
      this->advance(n_burnin_steps, n_traj_steps, step_size);
      this->reset_expectation_values();
      this->set_n_steps_taken(0);
      this->set_n_accepted(0);
  }

  void inline leapfrog(std::shared_ptr<SampleState> q, std::shared_ptr<SampleState> p,
              double step_size, int n_traj_steps,
              std::shared_ptr<Action> action,
              std::shared_ptr<SampleState> force);


private:
  unsigned int size;
  std::shared_ptr<Action> action;
  std::shared_ptr<SampleState> current_config;
  unsigned int n_accepted;
  unsigned int n_steps_taken;
  float acceptance_rate;
  std::vector<double> expectation_values;

  std::mt19937_64 engine;
  std::uniform_real_distribution<double> uniform_dist;
  std::normal_distribution<double> normal_dist;

  std::shared_ptr<SampleState> q;
  std::shared_ptr<SampleState> p;
  std::shared_ptr<SampleState> force;

  std::shared_ptr<Observable> observable;
};

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
  double exp_value;

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









class HMC_Batch {

public:
  HMC_Batch(std::shared_ptr<Observable> _observable, std::shared_ptr<Action> _action,
      int _size,int _batch_size) {
    action = _action;
    size = _size;
    observable = _observable;
    batch_size = _batch_size;
  }

  void initialize() {

    n_accepted = torch::zeros({batch_size},torch::kInt64);
    n_steps_taken = 0;

    current_config = std::make_shared<torch::Tensor>(torch::rand({batch_size,size},torch::kFloat64));
    *current_config *= 2*M_PI;

    q = std::make_shared<torch::Tensor>(torch::empty_like(*current_config));
    p = std::make_shared<torch::Tensor>(torch::empty_like(*current_config));
    force = std::make_shared<torch::Tensor>(torch::empty_like(*current_config));

    h = torch::empty({batch_size},torch::kFloat64);
    hp = torch::empty({batch_size},torch::kFloat64);
    log_ratio = torch::empty({batch_size},torch::kFloat64);
    accept_mask = torch::empty({batch_size},torch::kFloat64);

    action_out = std::make_shared<torch::Tensor>(torch::empty({batch_size},torch::kFloat64));

  }

  void set_n_accepted(torch::Tensor _n_accepted) { n_accepted = _n_accepted; }
  const torch::Tensor get_n_accepted() const { return n_accepted; }

  void set_n_steps_taken(int _n_steps_taken) { n_steps_taken = _n_steps_taken; }
  const int get_n_steps_taken() const { return n_steps_taken; }

  const torch::Tensor get_acceptance_rate() const { 
      if(n_steps_taken==0){return torch::zeros({batch_size});} 
      else {return n_accepted.to(torch::kFloat32) / (float)n_steps_taken; }}

  void set_current_config(torch::Tensor _current_config) {
    current_config = std::make_shared<torch::Tensor>(_current_config);
  }
  torch::Tensor get_current_config() const {
    return *current_config;
  }

  void reset_expectation_values() {
    expectation_values = std::vector<torch::Tensor>();
  }
  std::vector<torch::Tensor> get_expectation_values() const {
    return expectation_values;
  }

  void advance(int n_steps, int n_traj_steps, double step_size);

  void burnin(int n_burnin_steps, int n_traj_steps, double step_size){
      this->advance(n_burnin_steps, n_traj_steps, step_size);
      this->reset_expectation_values();
      this->set_n_steps_taken(0);
      this->set_n_accepted(torch::zeros({batch_size}));
  }

  void inline leapfrog(std::shared_ptr<torch::Tensor> q, std::shared_ptr<torch::Tensor> p,
              double step_size, int n_traj_steps,
              std::shared_ptr<Action> action,
              std::shared_ptr<torch::Tensor> force);


private:
  unsigned int size;
  unsigned int batch_size;
  std::shared_ptr<Action> action;
  std::shared_ptr<torch::Tensor> current_config;
  unsigned int n_steps_taken;
  float acceptance_rate;

  std::vector<torch::Tensor> expectation_values;
  torch::Tensor n_accepted;

  std::shared_ptr<torch::Tensor> q;
  std::shared_ptr<torch::Tensor> p;
  std::shared_ptr<torch::Tensor> force;
  std::shared_ptr<torch::Tensor> action_out;


  torch::Tensor h; 
  torch::Tensor hp;
  torch::Tensor log_ratio;
  torch::Tensor accept_mask;

  std::shared_ptr<Observable> observable;
};

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
    
























PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("test_leapfrog", &test_leapfrog);
  m.def("test_action", &test_action);
  m.def("test_sus", &test_sus);

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