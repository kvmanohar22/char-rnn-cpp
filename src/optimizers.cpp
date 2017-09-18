#include "RNN/optimizers.hpp"
#include "RNN/hyperparameters.hpp"

#include <vector>

namespace RNN {

  namespace Optimizers {

    template <class T>
    Optimizer<T>::Optimizer(Utils::phase PHASE, int decay_epochs, int display,
                            T base_lr, int epochs, T momentum, T gamma,
                            size_t rnn_size, size_t batch_size, size_t seq_len) {
      this->PHASE = PHASE;
      this->decay_epochs = decay_epochs;
      this->display = display;
      this->base_lr = base_lr;
      this->epochs = epochs;
      this->momentum = momentum;
      this->gamma = gamma;
      this->rnn_size = rnn_size;
      this->batch_size = batch_size;
      this->seq_len = seq_len;

      // Set data
      dataset = new Utils::Dataset<T>(Utils::Hyperparams::root_path);
      this->current_idx = 0;
      
      // Initialize the Network
      cell = new RNN::vanilla_cell<T>(dataset->get_voc_len(), batch_size);
    }

    template <class T>
    Optimizer<T>::~Optimizer() {
      delete dataset;
      delete cell;
    }

    template <class T>
    void Optimizer<T>::step(T *input, T *target) {
      std::vector<T> loss_t;

      // Forward pass
      for (size_t dt = 0; dt < dataset->get_voc_len(); ++dt) {
        T loss = cell->forward(input);
        loss_t.push_back(loss);
      }

      // Backward pass and update weights
      cell->backward();

    }

    template <class T>
    void Optimizer<T>::steps(int num_epochs) {

      T *batch_input = new T[batch_size * seq_len];
      T *batch_target = new T[batch_size * seq_len];
      T *batch_tokenized_input = new T[batch_size * seq_len * dataset->get_voc_len()];
      T *batch_tokenized_target = new T[batch_size * seq_len * dataset->get_voc_len()];
      for (int epoch = 0; epoch < num_epochs; ++epoch) {
        if (current_idx > dataset->get_max_len(PHASE)) {
          this->current_idx = 0;
        }
        dataset->load_data(current_idx, batch_input, this->PHASE);
        dataset->generate_target(batch_input, batch_target);

        dataset->tokenize_chars(batch_input, batch_tokenized_input);
        dataset->tokenize_chars(batch_target, batch_tokenized_target);

        this->step(batch_tokenized_input, batch_tokenized_target);

        current_idx += batch_size;
      }
    
    }

  }  // namespace Optimizers

}  // namespace RNN