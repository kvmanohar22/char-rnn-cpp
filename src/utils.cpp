#include "RNN/utils.hpp"

namespace RNN {

	namespace Utils {

		template <class T>
		const std::map<int, std::string> 
			utils<T>::init_type = {{0, "gaussian"}, 
			{1, "xavier"}, {2, "constant"}};

		template <class T>
		void utils<T>::mul_mat_mat(T *result, T *m1, T *m2,
			size_t &h1, size_t &w1, size_t &w2) {
			for (size_t i = 0; i < h1; ++i) {
				for (size_t j = 0; j < w2; ++j) {
					try {
						result[i * w2 + j] = 0;
					} catch (std::exception &exp) {
						std::cerr << "Exception occured in matrix multiplication: "
									 << exp.what() << std::endl;
					}
					for (size_t k = 0; k < w1; ++k) {
						try {
							result[i * w2 + j] += m1[i * w1 + k] * m2[k * w2 + j];
						} catch (std::exception &exp) {
						std::cerr << "Exception occured in matrix multiplication: "
									 << exp.what() << std::endl;						
						}
					}
				}
			}
		}

		template <class T>
		void utils<T>::mul_mat_vec(T *result, T *mat, T *vec,
			size_t &r1, size_t &h1) {
			for (size_t i = 0; i < h1; ++i) {
				T result_i = 0;
				for (size_t j = 0; j < r1; ++j) {
					try {
						result_i += mat[i * r1 + j] * vec[j];
					} catch (std::exception &exp) {
						std::cerr << "Exception occured in matrix vector multiplication: "
									 << exp.what() << std::endl;
					}
				}
				result[i] = result_i;
			}
		}

		template <class T>
		void utils<T>::add_vec_vec(T *result, T *v1, T *v2, size_t &vec_len) {
			for (size_t i = 0; i < vec_len; ++i) {
				try {
					result[i] = v1[i] + v2[i];
				} catch (std::exception &exp) {
					std::cerr << "Exception occured in matrix vector multiplication: "
								 << exp.what() << std::endl;				
				}
			}
		}

	  template <class T>
	  void utils<T>::weight_init(T *mat, size_t &h, size_t &w, int &type) {
		  	switch(type) {
		  		case 0:
		  			gaussian_init(mat, h, w);
		  			break;
		  		case 1:
		  			xavier_init(mat, h, w);
		  		default:
		  			std::cerr << "No such initializer\n";
		  	}
	  }

	  template <class T>
	  void utils<T>::bias_init(T *vec, size_t &vec_len, T &val, int &type) {
		  	switch(type) {
		  		case 0:
		  			gaussian_init(vec, vec_len);
		  		case 1:
		  			xavier_init(vec, vec_len);
		  		case 2:
		  			constant_init(vec, vec_len, val);
		  		default:
		  			std::cerr << "No such initializer\n";
		  	}
	  }

	  template <class T>
	  void utils<T>::xavier_init(T *mat, size_t &h, size_t &w) {
	 		
	  }

	  template <class T>
	  void utils<T>::gaussian_init(T mean, T std, T *mat, size_t &h, size_t &w) {

	  }

	  template <class T>
	  void utils<T>::constant_init(T *vec, size_t &vec_len, T &val) {

	  }

	  template <class T>
	  T utils<T>::Exp(T &arg) {
	  		return exp(arg);
	  }

	  template <class T>
	  T utils<T>::Log(T &arg) {
	  		return log(arg);
	  }

	  template <class T>
	  T utils<T>::Tanh(T &arg) {
	  		return tanh(arg);
	  }

	  template <class T>
	  void utils<T>::Tanh(T *arg, size_t len) {
	  		for (size_t i = 0; i < len; ++i) {
	  			arg[i] = Tanh(arg[i]);
	  		}
	  }

	}  // namespace Utils

}
