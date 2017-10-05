#ifndef _RNN_RNN_CELL_HPP_
#define _RNN_RNN_CELL_HPP_

#include <cstddef>

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

      void update_h(T *new_h);

      void set_memory();

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
      void output(T *_o);

      void checkpoint();
  };

}

#endif