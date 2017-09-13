#ifndef _RNN_RNN_CELL_HPP_
#define _RNN_RNN_CELL_HPP_

#include <cstddef>

namespace RNN {

  template <class T>
  class RNN_Cell {

    private:

      T *W_xh, *dW_xh;
      T *W_hh, *dW_hh;
      T *W_hy, *dW_hy;

      T *b_xh, *db_xh;
      T *b_hy, *db_hy;

      T *h;

      size_t rnn_size;
      size_t seq_len;
      size_t voc_len;

    protected:
      void update_W_hh();
      void update_W_xh();
      void update_W_hy();

      void update_b_xh();
      void update_b_hy();

    public:
      RNN_Cell(size_t voc_len);
      RNN_Cell(size_t rnn_size, size_t seq_len, size_t voc_len);
      ~RNN_Cell();

      void forward(T *x);
      void backward();

      void checkpoint();
  };

}

#endif