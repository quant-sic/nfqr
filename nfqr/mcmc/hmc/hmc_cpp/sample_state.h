#ifndef _SAMPLE_STATE_H_
#define _SAMPLE_STATE_H_

#include <Eigen/Core>
#include <torch/extension.h>

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


#endif //_SAMPLE_STATE_H_