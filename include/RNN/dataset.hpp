#ifndef _RNN_DATASET_HPP_
#define _RNN_DATASET_HPP_

#include <string>

namespace RNN {

  namespace Utils {

    enum phase {TRAIN, VALIDATE, TEST};

    template <class T>
    class Dataset {

      private:
        /**
         * Base path in which the dataset is persent
         */
        std::string root_path;
      
        size_t voc_len;
        size_t dataset_size;
        size_t train_size;
        size_t valid_size;
        size_t test_size;

        /**
         * Entire data
         * Total number of characters in the training set
         * Well, we can afford that amount of memory !
         */
        T *data;

      public:

        Dataset() {}
        Dataset(std::string root_path);
        ~Dataset();
        /**
         * Preprocesses the dataset
         */
        void pre_process();

        /**
         * Loads chunk of data
         */
        void load_data(size_t start_idx, T *batch_data, phase PHASE=TRAIN);

        /**
         * Given input, this generates the target for that input
         * @param input  Vector of input characters whose size is `seq_len`
         * @param target Vector of target characters whose size is `seq_len`
         */
        void generate_target(T *input, T *target);

        /**
         * Converts each character into one-hot vector
         * @param input     Vector of input characters whose size is `seq_len`
         * @param tokenized Vector of target characters as one-hot vectors
         *                  whose size is `seq_len * voc_len` 
         */
        void tokenize_chars(T *input, T *tokenized);

        inline size_t get_voc_len() {
          return this->voc_len;
        }

        inline size_t get_dataset_size() {
          return this->dataset_size;
        }


        /**
         * Returns the number of `batch_size * seq_len` chunks
         */
        inline size_t get_max_len(phase PHASE) {
          if (PHASE == TRAIN) {
            return train_size;
          } else if (PHASE == VALIDATE) {
            return valid_size;
          } else {
            return test_size;
          }
        }

    };

  }  // namespace Utils

}  // namespace RNN

#endif