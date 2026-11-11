/*
 * Host test of the exported learned get-up network (tars_rl_getup_nn_weights.h, the file that goes into the firmware): the C forward pass must reproduce the
 * PyTorch actor on the observations it saw in simulation rollouts (getup_nn_vectors.h, written by tars_rl/rl/export_getup_c.py).
 */
#include <math.h>
#include <stdio.h>

#include "tars_rl_core.h"
#include "tars_rl_getup_nn_weights.h"
#include "getup_nn_vectors.h"

int main(void)
{
    double worst = 0.0;
    int bad = 0;
    for (int i = 0; i < TARS_GETUP_NN_NVEC; i++) {
        float out[3];
        tars_mlp_forward(&tars_getup_nn_mlp, &tars_getup_nn_test_obs[28 * i], out);
        for (int k = 0; k < 3; k++) {
            float want = tars_getup_nn_test_act[3 * i + k];
            want = want > 1.0f ? 1.0f : (want < -1.0f ? -1.0f : want);          /* the C network clips its output to [-1, 1] like the environment does */
            const double e = fabs((double)out[k] - (double)want);
            if (e > worst) worst = e;
            if (e > 2e-3) bad++;
        }
    }
    for (int i = 0; i < TARS_GETUP_NN_NVEC; i++) {                    /* the static-scratch variant (used on the robot) is bit for bit the same */
        float a[3], b[3];
        tars_mlp_forward(&tars_getup_nn_mlp, &tars_getup_nn_test_obs[28 * i], a);
        tars_mlp_forward_static(&tars_getup_nn_mlp, &tars_getup_nn_test_obs[28 * i], b);
        for (int k = 0; k < 3; k++) if (a[k] != b[k]) bad++;
    }
    printf("test_getup_nn: %d vectors, worst abs error %.2e, %d values off by more than 2e-3\n", TARS_GETUP_NN_NVEC, worst, bad);
    return bad ? 1 : 0;
}
