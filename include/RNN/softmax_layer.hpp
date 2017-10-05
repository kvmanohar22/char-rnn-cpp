#ifndef _RNN_SOFTMAX_LAYER_HPP_
#define _RNN_SOFTMAX_LAYER_HPP_

#include <memory>
#include <iostream>
#include <cassert>
#include <climits>
#include <cstddef>
#include <algorithm>


namespace RNN {

  namespace layers {
   /**
    * Softmax Layer
    * Applies stable softmax to a vector on inputs and then computes the
    * cross entropy loss for multi-class classification
    */

    template <class T>
    class Softmax {
      
      private:
       /**
        * axis along which softmax is applied
        */
        int axis;
       /**
        * Cross-Entropy loss computed during forward propagation
        */
        T _loss;
       /**
        * Length of the input vector
        */
        size_t vec_len;
       /**
        * Probability values which are calculated during forward propagation
        * Store them for use during the backward pass
        */
        T *probs;

      public:
       /**
        * Constructer
        * @param vec_len Length of input vector
        * @param axis Axis along which softmax is applied
        */
        Softmax(size_t vec_len, int axis = 0);
       /**
        * Default Destructer
        */
        ~Softmax();
       /**
        * Forward propagation
        * @param vec    Input values to the softmax
        * @param labels One hot vector of labels
        */
        void forward(T *vec, T *labels);
       /**
        * Compute the gradients during backpropagation with respect
        * to loss function
        * @param grads  Vector of gradients wrt loss function
        * (loss function defaulted to cross entropy)
        * @param labels One hot vector of labels
        */
        void backward(T *grads, T *labels);

       /**
        * Return loss
        */
        inline T loss() const {
          return this->_loss;
        }
    };

  }  // namespace layers

}  // namespace RNN

#endif