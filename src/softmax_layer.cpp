#include "RNN/softmax_layer.hpp"
#include "RNN/utils.hpp"

using namespace RNN::Utils;

namespace RNN {

  namespace layers {

    template <class T>
    Softmax<T>::Softmax(size_t vec_len, int axis) {
      this->axis = axis;
      this->vec_len = vec_len;
      this->probs = new T[vec_len];
      this->y_hat.clear();
    }

    template <class T>
    Softmax<T>::~Softmax() {
      delete [] probs;
    }

    template <class T>
    void Softmax<T>::forward(T *vec, T *labels) {
      T max_val = LONG_MIN;
      T denom_sum = 0.0;
      for (size_t i = 0; i < vec_len; ++i) {
        if (vec[i] > max_val) {
          max_val = vec[i];
        }
      }
      assert(max_val != LONG_MIN);
      for (size_t i = 0; i < vec_len; ++i) {
        probs[i] = vec[i] - max_val;
        probs[i] = utils<T>::Exp(probs[i]);
        denom_sum += probs[i];
      }
      for (size_t i = 0; i < vec_len; ++i) {
        this->probs[i] /= denom_sum; 
      }

      // Compute loss
      bool flag = false;
      for (size_t i = 0; i < vec_len; ++i) {
        if (labels[i] == 1) {
          this->_loss = utils<T>::Log(probs[i]);
          flag = true;
          break;
        }
      }
      if (!flag) {
        std::cerr << "Seems like ground truth doesn't contain correct label\n";
        exit(0);
      }
    }

    template <class T>
    void Softmax<T>::backward(T *grads, T *labels) {
      for (size_t i = 0; i < vec_len; ++i) {
        T grad;
        if (labels[i] == 1) {
          grad = probs[i] - 1;
        } else {
          grad = probs[i];
        }
        grads[i] = grad;
      }
    }

  }  // namespace layers

}  // namespace RNN