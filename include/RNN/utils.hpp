#ifndef _RNN_UTILS_HPP_
#define _RNN_UTILS_HPP_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <map>
#include <math.h>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace RNN {

	namespace Utils {

   /**
    * Utility functions
    */

		template <class T>
		class utils {
			protected:
          /**
           * xavier initializer
           * @param mat Matrix of randomly initialized values
           * @param h   Height of matrix
           * @param w   Width of matrix
           */
				void xavier_init(T *mat, size_t &h, size_t &w = 0);
          /**
           * Gaussian initializer
           * @param mean Mean of distribution
           * @param std  Std deviation of distribution
           * @param mat  Matrix of randomly initialized values
           * @param h    Height of matrix
           * @param w    Width of matrix
           */
				void gaussian_init(T mean, T std, T *mat, size_t &h, size_t &w = 0);
          /**
           * Constant Initializer
           * @param vec     Vector of randomly initialized values
           * @param vec_len Length of input vector
           * @param val     Constant value to be initialized with
           */
				void constant_init(T *vec, size_t &vec_len, T &val = 0);

			public:
				static const std::map<int, std::string> init_type;

				static void mul_mat_mat(T *result, T *m1, T *m2, size_t &r1, size_t &h1, size_t &r2);
				static void mul_mat_vec(T *result, T *mat, T *vec, size_t &r1, size_t &h1);
				static void add_vec_vec(T *result, T *v1, T *v2, size_t &vec_len);
		
				static void weight_init(T *mat, size_t &h, size_t &w, int &type = 1);
				static void bias_init(T *vec, size_t &vec_len, T &val, int &type = 2);

				static T Exp(T &arg);
				static T Log(T &arg);

            static T Tanh(T &arg);
            static void Tanh(T *arg, size_t len);
		};

	}

}

#endif