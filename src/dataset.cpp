#include "RNN/dataset.hpp"

namespace RNN {

  namespace Utils {

    template <class T>
    Dataset<T>::Dataset(std::string root_path) {
      this->root_path = root_path;
      this->pre_process();
    }

    template <class T>
    void Dataset<T>::load_data(size_t start_idx, T *batch_data, phase PHASE) {

    }

    template <class T>
    void Dataset<T>::pre_process() {

    }

    template <class T>
    void Dataset<T>::generate_target(T *input, T *target) {

    }

    template <class T>
    void Dataset<T>::tokenize_chars(T *input, T *tokenized) {

    }

  }  // namespace Utils

}  // namespace RNN