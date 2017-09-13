#ifndef _RNN_UTILS_HPP_
#define _RNN_UTILS_HPP_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace Utils {

	template <class T>
	class utils {
		protected:
			void xavier_init(T *mat, size_t &h, size_t &w = 0);
			void gaussian_init(T *mat, size_t &h, size_t &w = 0);
			void constant_init(T *vec, size_t &vec_len, T &val);

		public:
			static const std::map<int, std::string> init_type;

			static void mul_mat_mat(T *result, T *m1, T *m2, size_t &r1, size_t &h1, size_t &r2);
			static void mul_mat_vec(T *result, T *mat, T *vec, size_t &r1, size_t &h1);
			static void add_vec_vec(T *result, T *v1, T *v2, size_t &vec_len);
	
			static void weight_init(T *mat, size_t &h, size_t &w, int &type = 1);
			static void bias_init(T *vec, size_t &vec_len, T &val, int &type = 2);
	};
}

#endif