#ifndef SNN_WEIGHT_H
#define SNN_WEIGHT_H

namespace snn {
    template<typename T>
    struct weight {
        T tau_plus, tau_minus;
        T a_plus, a_minus;
        T g_min, g_max;
        T x, y;
        T g;
    };

    template<typename T>
    weight<T> weight_init(T tau_plus, T tau_minus, T a_plus, T a_minus, T g_min, T g_max);

    template<typename T>
    void weight_forward(weight<T>* w, T dt);

    template<typename T>
    void weight_update(weight<T>* w, T dt, bool pre_spike, bool post_spike);
}

#endif // SNN_WEIGHT_H