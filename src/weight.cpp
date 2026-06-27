#include "weight.h"
#include <cmath>

namespace snn {
    template<typename T>
    weight<T> weight_init(T tau_plus, T tau_minus, T a_plus, T a_minus, T g_min, T g_max) {
        weight<T> w;
        w.tau_plus = tau_plus;
        w.tau_minus = tau_minus;
        w.a_plus = a_plus;
        w.a_minus = a_minus;
        w.g_min = g_min;
        w.g_max = g_max;
        w.x = T{};
        w.y = T{};
        w.g = (g_min + g_max) / T{2};
        return w;
    }

    template<typename T>
    void weight_forward(weight<T>* w, T dt) {
        w->x *= std::exp(-dt / w->tau_plus);
        w->y *= std::exp(-dt / w->tau_minus);
    }

    template<typename T>
    void weight_update(weight<T>* w, T dt, bool pre_spike, bool post_spike) {
        if (pre_spike)
            w->x += w->a_plus;
        if (post_spike)
            w->y -= w->a_minus;
        T alpha_g = w->g_max - w->g_min;
        if (pre_spike) {
          w->g += alpha_g * w->y;
          w->g = w->g > w->g_max ? w->g_max : w->g < w->g_min ? w->g_min : w->g;
        }
        if (post_spike) {
          w->g += alpha_g * w->x;
          w->g = w->g > w->g_max ? w->g_max : w->g < w->g_min ? w->g_min : w->g;
        }
    }

    template weight<float> weight_init<float>(float, float, float, float, float, float);
    template void weight_forward<float>(weight<float>*, float);
    template void weight_update<float>(weight<float>*, float, bool, bool);
}