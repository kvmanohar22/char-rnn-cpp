#ifndef _RNN_RNN_CELL_HPP_
#define _RNN_RNN_CELL_HPP_

#include <cstddef>
#include <vector>

namespace RNN {

  template <class T>
  class vanilla_cell {

    private:

      T *W_xh, *dW_xh;
      T *W_hh, *dW_hh;
      T *W_hy, *dW_hy;

      T *b_xh, *db_xh;
      T *b_hy, *db_hy;

      T *h;
      T *_output;
      std::vector<T *> _h;
      std::vector<T *> _o;

      size_t rnn_size;
      size_t seq_len;
      size_t voc_len;
      size_t batch_size;

    protected:
      void update_W_hh();
      void update_W_xh();
      void update_W_hy();

      void update_b_xh();
      void update_b_hy();

      void update_h(T *new_h=NULL, bool all_zero=false);

      void set_memory();

     /**
      * Accumulate the gradients at each `dt` of time step
      * Once the `time step` is complete, it updates the weights
      */
      void accumulate_dbhy();
      void accumulate_dbxh();
      void accumulate_dWhy();
      void accumulate_dWhh();
      void accumulate_dWxh();

    public:
      vanilla_cell() {}
      vanilla_cell(size_t voc_len, size_t batch_size);
      vanilla_cell(size_t rnn_size, size_t seq_len, size_t voc_len, size_t batch_size);
      ~vanilla_cell();

      T forward(T *x);
      void backward();

     /**
      * Return the output of the cell
      * The values are raw values to which softmax is applied
      */
      void get_o_dt(T *_o);

     /**
      * Return the h^(t) of the cell
      */
      void get_h_dt(T *_h_dt);

     /**
      * Set the internal hypothesis to zero
      * This occurs when at the beginning of each step
      */
      inline void refresh() {
        this->update_h(NULL, true);
        this->_o.clear();
        this->_h.clear();
      }

     /**
      * Save the weights of the network
      */
      void checkpoint();

     /**
      * Accumulate gradients after each `dt` in one single time step
      * These gradients will be finally used to update the weights using
      * gradient descent
      */
      void accumulate_gradients(int w_idx, T *grads);

     /**
      * Accumulate output after each `dt` in time step
      * @params out_dt Output of RNN Cell at each step of `dt`
      */
      inline void accumulate_o_dt() {
        this->_o.push_back(this->_output);
      }

     /**
      * Accumulate h after each `dt` in time step
      * @params h_dt Output of RNN Cell at each step of `dt`
      */
      inline void accumulate_h_dt() {
        this->_h.push_back(this->h);
      }
  };

}

#endif