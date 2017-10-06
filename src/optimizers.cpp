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

      current_idx = 0;
      
      // Set data
      dataset = new Utils::Dataset<T>(Utils::Hyperparams::root_path);
      // Initialize the Network
      cell = new RNN::vanilla_cell<T>(dataset->get_voc_len(), batch_size);
      // Softmax layer to compute the loss of the model at a given time step
      soft_layer = new RNN::layers::Softmax<T>(dataset->get_voc_len());
    }

    template <class T>
    Optimizer<T>::~Optimizer() {
      delete dataset;
      delete cell;
      delete soft_layer;
    }

    template <class T>
    void Optimizer<T>::step(T *input, T *target) {
      std::vector<T> loss_t;
      T *output_dt = new T[dataset->get_voc_len()];
      T *input_dt = new T[dataset->get_voc_len()];
      T *target_dt = new T[dataset->get_voc_len()];

      // Refresh the internal state of the network
      cell->refresh();
      soft_layer->refresh();

      // Forward pass
      for (size_t dt = 0; dt < seq_len; ++dt) {
        // Slice the data for a single time step
        for (size_t j = 0; j < dataset->get_voc_len(); ++j) {
          input_dt[j] = input[dataset->get_voc_len() * dt + j];
          target_dt[j] = target[dataset->get_voc_len() * dt + j];
        }
        cell->forward(input_dt);
        cell->get_o_dt(output_dt);
        soft_layer->forward(output_dt, target_dt);

        // Store the intermediate activations for use in the backprop
        cell->accumulate_o_dt();
        cell->accumulate_h_dt();
        soft_layer->accumulate_yhat();

        // Store Cross Entropy loss for debugging
        T loss = soft_layer->loss();
        loss_t.push_back(loss);
      }

      // Backward pass and update weights
      for (size_t dt = seq_len; dt > 0; --dt) {

      }

      delete output_dt;
      delete input_dt;
      delete target_dt;
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
        dataset->load_data(current_idx, batch_input);
        dataset->generate_target(batch_input, batch_target);

        dataset->tokenize_chars(batch_input, batch_tokenized_input);
        dataset->tokenize_chars(batch_target, batch_tokenized_target);

        this->step(batch_tokenized_input, batch_tokenized_target);

        current_idx += batch_size;
      }
    
    }

  }  // namespace Optimizers

}  // namespace RNN