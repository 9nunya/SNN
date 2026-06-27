%%writefile include/util.h
#ifndef SNN_UTIL_H
#define SNN_UTIL_H

#include <vector>
#include <random>
#include <type_traits>

namespace snn {

  template<typename T>
  struct range {
    T min, max;
  };

  namespace util {
    // Random uniform distribution for floating point and integers
    template<typename T>
    T random_uniform(T min, T max);

    // Random choice from a vector
    template<typename T>
    T random_choice(const std::vector<T>& choices);
  }
}

#endif