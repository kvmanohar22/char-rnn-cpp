#include "RNN/rnn_cell.hpp"
#include "RNN/utils.hpp"
#include "RNN/softmax_layer.hpp"

#include <memory>
#include <iostream>
#include <cassert>
#include <algorithm>

using namespace RNN::Utils;

namespace RNN {

  template <class T>
  vanilla_cell<T>::vanilla_cell(size_t voc_len, size_t batch_size) {
    this->rnn_size = 512;
    this->seq_len = 100;
    this->voc_len = voc_len;
    this->batch_size = batch_size;

    set_memory();
  }

  template <class T>
  vanilla_cell<T>::vanilla_cell(size_t rnn_size, size_t seq_len,
    size_t voc_len, size_t batch_size) {
    this->rnn_size = rnn_size;
    this->seq_len = seq_len;
    this->voc_len = voc_len;
    this->batch_size = batch_size;

    set_memory();
  }

  template <class T>
  void vanilla_cell<T>::set_memory() {
    W_xh = new T[voc_len * rnn_size];
    dW_xh = new T[voc_len * rnn_size];

    W_hh = new T[rnn_size * rnn_size];
    dW_hh = new T[rnn_size * rnn_size];

    W_hy = new T[rnn_size * voc_len];
    dW_hy = new T[rnn_size * voc_len];

    b_xh = new T[rnn_size];
    db_xh = new T[rnn_size];

    b_hy = new T[voc_len];
    db_hy = new T[voc_len];

    h = new T[rnn_size];
    _output = new T[voc_len];
  }

  template <class T>
  vanilla_cell<T>::~vanilla_cell() {
    delete [] W_xh;
    delete [] dW_xh;

    delete [] W_hh;
    delete [] dW_hh;

    delete [] W_hy;
    delete [] dW_hy;

    delete [] b_xh;
    delete [] db_xh;

    delete [] b_hy;
    delete [] db_hy;
  }

  template <class T>
  T vanilla_cell<T>::forward(T *x) {
    T *result_xh = new T[rnn_size];
    T *result_hh = new T[rnn_size];
    T *result_ad = new T[rnn_size];
    T *result_bb = new T[rnn_size];

    // Compute $$ a^(t) $$
    utils<T>::mul_mat_vec(result_xh, W_xh, x, rnn_size, voc_len);
    utils<T>::mul_mat_vec(result_hh, W_hh, h, rnn_size, rnn_size);
    utils<T>::add_vec_vec(result_ad, result_xh, result_hh, rnn_size);
    utils<T>::add_vec_vec(result_ad, result_ad, b_xh, rnn_size);

    // Perform tanh nonlinearity and update the hidden state
    utils<T>::Tanh(result_ad, rnn_size);
    update_h(result_ad);

    // Compute the output of hidden state
    utils<T>::mul_mat_vec(_output, W_hy, this->h, voc_len, rnn_size);
    utils<T>::add_vec_vec(_output, _output, b_hy, voc_len);

    delete [] result_xh;
    delete [] result_hh;
    delete [] result_ad;
    delete [] result_bb;
  }

  template <class T>
  void vanilla_cell<T>::backward() {

  }

  template <class T>
  void vanilla_cell<T>::output(T *_o) {
    for (size_t i = 0; i < voc_len; ++i) {
      _o[i] = this->_output[i];
    }
  }

  template <class T>
  void vanilla_cell<T>::update_W_hh() {
    
  }

  template <class T>
  void vanilla_cell<T>::update_h(T *new_h) {
    for (size_t i = 0; i < rnn_size; ++i) {
      try {
        this->h[i] = new_h[i];
      } catch (std::exception &exp) {
        std::cerr << "Exception occured in updating hypothesis: "
                  << exp.what() << std::endl;
      }
    }
  }

  template <class T>
  void vanilla_cell<T>::update_W_xh() {
    
  }

  template <class T>
  void vanilla_cell<T>::update_W_hy() {
    
  }

  template <class T>
  void vanilla_cell<T>::update_b_xh() {
    
  }

  template <class T>
  void vanilla_cell<T>::update_b_hy() {

  }

 template <class T>
  void vanilla_cell<T>::checkpoint() {

  }

}