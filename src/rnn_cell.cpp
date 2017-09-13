#include "RNN/rnn_cell.hpp"

#include <memory>
#include <algorithm>

namespace RNN {

  template <class T>
  RNN_Cell<T>::RNN_Cell(size_t voc_len) {
    this->rnn_size = 512;
    this->seq_len = 100;
    this->voc_len = voc_len;

    this->W_xh = new T[seq_len * rnn_size];
    this->W_hh = new T[rnn_size * rnn_size];
    this->W_hy = new T[rnn_size * voc_len];

    this->b_xh = new T[rnn_size];
    this->b_hy = new T[voc_len];
  }

  template <class T>
  RNN_Cell<T>::RNN_Cell(size_t rnn_size, size_t seq_len, size_t voc_len) {
    this->rnn_size = rnn_size;
    this->seq_len = seq_len;
    this->voc_len = voc_len;

    this->W_xh = new T[seq_len * rnn_size];
    this->W_hh = new T[rnn_size * rnn_size];
    this->W_hy = new T[rnn_size * voc_len];

    this->b_xh = new T[rnn_size];
    this->b_hy = new T[voc_len];
  }

  template <class T>
  RNN_Cell<T>::~RNN_Cell() {
    delete [] this->W_xh;
    delete [] this->W_hh;
    delete [] this->W_hy;

    delete [] this->b_xh;
    delete [] this->b_hy;
  }

  template <class T>
  void RNN_Cell<T>::forward(T *x) {

  }

  template <class T>
  void RNN_Cell<T>::backward() {

  }

  template <class T>
  void RNN_Cell<T>::update_W_hh() {
    
  }

  template <class T>
  void RNN_Cell<T>::update_W_xh() {
    
  }

  template <class T>
  void RNN_Cell<T>::update_W_hy() {
    
  }

  template <class T>
  void RNN_Cell<T>::update_b_xh() {
    
  }

  template <class T>
  void RNN_Cell<T>::update_b_hy() {

  }

 template <class T>
  void RNN_Cell<T>::checkpoint() {

  }

}