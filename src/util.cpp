#include "util.h"
#include <stdexcept>
#include <chrono>

namespace snn {
  namespace util {
    static std::mt19937 gen(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));

    template<typename T>
    T random_uniform(T min, T max) {
      if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dis(min, max);
        return dis(gen);
      } else {
        std::uniform_int_distribution<T> dis(min, max);
        return dis(gen);
      }
    }

    template<typename T>
    T random_choice(const std::vector<T>& choices) {
      if (choices.empty()) {
        throw std::runtime_error("random_choice called with empty vector");
      }
      std::uniform_int_distribution<size_t> dis(0, choices.size() - 1);
      return choices[dis(gen)];
    }

    // Explicit instantiations for common types
    template float random_uniform<float>(float, float);
    template double random_uniform<double>(double, double);
    template int random_uniform<int>(int, int);

    template float random_choice<float>(const std::vector<float>&);
    template int random_choice<int>(const std::vector<int>&);
  }
}