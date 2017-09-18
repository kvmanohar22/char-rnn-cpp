#ifndef _RNN_OPTIMIZERS_HPP_
#define _RNN_OPTIMIZERS_HPP_

#include "RNN/dataset.hpp"
#include "RNN/rnn_cell.hpp"

#include <cstddef>

namespace RNN {

  namespace Optimizers {

    template <class T>
    class Optimizer {
      private:
       /**
        * Base Learning Rate at which training begins
        */
        T base_lr;
       /**
        * Decay base_lr after `decay_epochs` epochs
        */
        int decay_epochs;
       /**
        * Display log after `display` epochs
        */
        int display;
       /**
        * Total number of epochs to train the model
        */
        int epochs;
       /**
        * Momentum value to update the weights during backprop
        */
        T momentum;
       /**
        * `gamma` used during step decay
        */
        T gamma;

        size_t rnn_size;
        size_t batch_size;
        size_t seq_len;

        Utils::Dataset<T> *dataset;
        Utils::phase PHASE;

        RNN::vanilla_cell<T> *cell;


       /**
        * current idx of the data to be loaded
        */
        size_t current_idx;

      public:
       /**
        * Default Constructer
        */
        Optimizer(Utils::phase PHASE, int decay_epochs=10, int display=1,
                  T base_lr=0.001, int epochs=100, T momentum=0.99, T gamma=0.5,
                  size_t rnn_size=512, size_t batch_size=1, size_t seq_len=100);

        ~Optimizer();

       /**
        * Perform one forward, one backward, and one weight update
        * This is done once for the entire time span
        * @param input  Input for one complete time sequence
        * @param target Target for one complete time  sequence
        */
        void step(T *input, T *target);

       /**
        * Perform `num_steps` and update weights in each of the step
        * @param num_steps Number of steps
        */
        void steps(int num_epochs=100);
    };

  }  // namespace Optimizers

}  // namespace RNN

#endif