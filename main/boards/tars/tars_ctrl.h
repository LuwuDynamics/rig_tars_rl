/*
 * tars_ctrl.h - portable C99 supervisor for the learned Tars behaviours (no ESP-IDF dependency).
 *
 *   STAND  : nominal pose (hips 0, lift 0)
 *   WALK   : periodic hip/lift pattern + IMU pitch feedback   (gait parameters found by CMA-ES in simulation);
 *            forward / backward walks also keep their heading with a gyro heading hold (unequal hip strokes)
 *   FALLEN : detects a fall from the IMU "up" vector, waits until the robot is still
 *   GETUP_NN: a neural network (PPO in simulation) flips the robot up and brings the arms back to the nominal pose by itself (the first attempt by default)
 *   GETUP  : CMA-ES keyframe motion  -> (upright detected) -> retract legs -> arms down -> extend legs   (the fallback / the 2nd attempt)
 *   FAILED : gave up (attempts used up / lying on its side); relaxes the legs; leaves this state when somebody picks the robot up and
 *            puts it down again (still) - it then looks at the situation afresh (STAND if upright, else a new get-up with a fresh counter)
 *   MOTION : a pre-computed show motion (nod, bow, head-bang, the emotions ... tars_rl/FUN_MOTIONS.md) played from a table of servo targets; only entered from a still,
 *            upright STAND on request (tars_ctrl_motion_start), ends by itself, on tars_ctrl_motion_stop(), on a walking command, or as a fall
 *
 * Frame/sign conventions are those of the simulator (see tars_rl_core.h):
 *   hip q>0 = foot forward [rad],  lift z>0 = legs retract upward [m],  up[] = world-up in the body frame.
 */
#ifndef TARS_CTRL_H
#define TARS_CTRL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float q0, a1, b1, a2, b2;      /* hip pattern  [rad]                       */
    float z0, c1, d1, c2, d2;      /* lift pattern [m]                         */
    float f;                       /* gait frequency [Hz]                      */
    float kp_h, kd_h;              /* hip  feedback  [rad/rad], [rad/(rad/s)]  */
    float kp_z, kd_z;              /* lift feedback  [m/rad],   [m/(rad/s)]    */
} tars_gait_t;

typedef struct {
    int k;                         /* number of keyframes (14)                 */
    float tend;                    /* seconds covered by the keyframes (7.0)   */
    float hip_range;               /* rad,  keyframe value 1.0 -> hip_range (2.4)   */
    float lift_range;              /* m,    keyframe value 1.0 -> lift_range (0.014) */
    float rate;                    /* max hip speed during stage 1 [rad/s]     */
    const float *hip;              /* [k] normalised keyframes                 */
    const float *lift;             /* [k]                                       */
} tars_getup_t;

/* a pre-computed show motion (generated into tars_motions.h by tars_rl/firmware/gen_motions.py): servo targets every `dt` seconds, linearly interpolated.
 * The table starts and ends at the nominal pose (0, 0, 0). */
typedef struct {
    const char *name;
    int n;                         /* samples                                                                          */
    float dt;                      /* s between samples (0.02)                                                         */
    float tilt_limit_deg;          /* the torso tilting further than this for 0.2 s ends the motion as a fall (the 60 deg walking fall limit is not used during a motion) */
    const int16_t *hl, *hr;        /* hips [1e-3 rad], q>0 = foot forward  (hr may be the same array as hl)           */
    const int16_t *z;              /* lift [1e-4 m],   z>0 = arms retract up                                          */
    const char *face;              /* emoji of the screen that goes with the motion (an emotion), NULL = none          */
} tars_motion_t;

typedef enum {
    TARS_S_STAND = 0, TARS_S_WALK_F, TARS_S_WALK_B, TARS_S_TURN_L, TARS_S_TURN_R,
    TARS_S_FALLEN, TARS_S_GETUP1, TARS_S_GETUP2A, TARS_S_GETUP2B, TARS_S_GETUP2C, TARS_S_FAILED,
    TARS_S_GETUP_NN,               /* the learned (neural network) get-up                                                                              */
    TARS_S_MOTION                  /* appended last: a show motion from a table; the numbers of the other states did not change                        */
} tars_state_t;

typedef struct {
    /* configuration */
    const tars_gait_t *gait_f, *gait_b, *gait_turn, *gait_turn_r;   /* NULL -> not supported -> stand; gait_turn_r NULL -> mirror of gait_turn */
    const tars_getup_t *getup;
    float fall_up_thresh;          /* up[2] below this => tilted >60 deg (0.5)           */
    float fall_time;               /* s the tilt must persist (0.3)                       */
    float still_time;              /* s of low gyro before a get-up starts (0.6)         */
    float lift_up;                 /* leg retraction used while the arms go down (0.013 m)*/
    float hip_rate_down;           /* arm lowering speed in stage 2 (1.2 rad/s)          */
    int   max_attempts;            /* get-up attempts before FAILED (3)                  */
    float ramp_time;               /* gait amplitude ramp-in/out (1.0 s)                 */
    float cmd_deadband;            /* |cmd| (servo counts) below which the robot stands  */
    /* runtime tuning of the walking feedback (defaults 1 / 1 / 0 / 1 = the gait exactly as searched in simulation) */
    float fb_kp_scale;             /* scale of the pitch feedback            (kp_h, kp_z)                              */
    float fb_kd_scale;             /* scale of the pitch-rate feedback       (kd_h, kd_z)                              */
    float fb_lpf_hz;               /* first-order low-pass [Hz] on the pitch rate used by the feedback, 0 = off        */
    float amp_scale;               /* scale of the periodic hip / lift pattern (not of q0 / z0 / the feedback)         */
    /* speed command: the magnitude of the walking command (servo counts - the MCP / BLE "speed" parameter, 0..23) scales the periodic gait pattern:
     * speed_kmin for |cmd| <= speed_lo, speed_kmax for |cmd| >= speed_hi, linear in between (the calibrated simulator: 70 % of the pattern = about half the speed,
     * more than 100 % adds little speed and many falls; the top speed is set to 90 % because 100 % fell too easily on the real robot).
     * speed_map = 0: always the full gait (the old behaviour). */
    int   speed_map;
    float speed_lo, speed_hi, speed_kmin, speed_kmax;
    float k_cmd;                   /* the slew-limited scale in use */
    /* walking straight (forward / backward walks only): the two hips get unequal strokes, ql = q (1 + eps), qr = q (1 - eps), which turns the robot a little;
     * eps = +-(trim + hold_kp * heading + hold_kd * yaw rate)  (heading = the yaw angle integrated from the gyro since the walk began, + = counter-clockwise = to the left).
     * The sign is chosen so that a positive value turns the robot to the RIGHT in both directions.  trim: a constant correction for the robot's own drift;
     * hold_*: keep the heading of the walk's start.  straight = 0: both hips get the same stroke (the old behaviour, bit for bit). */
    int   straight;                /* 1 = trim and heading hold active (the default since the real-robot test; 0 = off)     */
    float trim;                    /* constant correction, + = turn right (fraction of the stroke)                         */
    float hold_kp;                 /* per rad of heading error                                                             */
    float hold_kd;                 /* per rad/s of yaw rate                                                                */
    float eps_max;                 /* |eps| limit (0.3)                                                                    */
    float heading_max;             /* rad, limit of the remembered heading error (0.26 = 15 deg; no wind-up when the robot is turned by hand) */
    float heading;                 /* rad, integrated while walking forward / backward                                    */
    float eps;                     /* the stroke difference in use                                                         */
    /* standing assist hook (the push-recovery experiment of tars_rl/REPORT.md sections 13 / 16 - NOT used by the Tars firmware, kept for the simulator tests):
     * while the robot stands (STAND state, no walking command, up[2] > stand_up_min) the callback decides the hip /
     * lift targets instead of the plain "slew to the nominal pose".  stand_on = 0 or no callback: the plain stand.  stand_reset runs whenever the assist
     * (re)starts so that its memory is cleared. */
    void (*stand_reset)(void *ctx);
    void (*stand_step)(void *ctx, const float up[3], const float gyro[3], float dt, float q[2], float *z);
    void *stand_ctx;
    int   stand_on;
    float stand_up_min;            /* the assist only runs while up[2] is above this (0.8)                              */
    int   stand_active;            /* the assist ran in the previous step                                                */
    /* learned get-up (a small neural network trained with PPO in simulation, see tars_rl/REPORT.md section 21): while the robot lies still on its back / front the
     * network, called every 0.1 s with attitude, gyro, its own previous commands and the elapsed time, flips it up and brings the arms back to the nominal pose.
     * getup_nn_mode: 0 = the keyframe get-up only, 1 = the network on the 1st and 3rd attempt and the keyframes on the 2nd, 2 = the network only.
     * getup_nn_fn == NULL (no network compiled in) = keyframes.  A failed network attempt (timeout) goes through FALLEN like a failed keyframe attempt. */
    int   getup_nn_mode;
    void (*getup_nn_fn)(void *ctx, const float obs[28], float act[3]);
    void *getup_nn_ctx;
    float nn_timeout;              /* s, 7: attempt given up                                                           */
    float nn_hip_rate;             /* rad/s, 8: speed limit of the commanded hips (as in the training)               */
    float nn_lift_rate;            /* m/s, 0.25                                                                        */
    float nn_acc;                  /* s since the last decision                                                        */
    int   nn_have_pf;              /* nn_pf holds the frame of 20 ms before the next decision                          */
    float nn_pf[12];
    float nn_a[3];                 /* the held action (normalised)                                                    */
    float nn_q[3];                 /* applied targets: left hip [rad], right hip [rad], lift [m]                        */
    /* show motions (tars_rl/FUN_MOTIONS.md): tars_ctrl_motion_start() only files a request; it starts when the robot stands still and upright (the supervisor waits up to
     * motion_wait_max for a walk to end) and then plays the table at `speed` (0.3..1) times its real time.  The torso pitches on the hips while the arms stand on the floor,
     * so the usual 60 deg fall detection is replaced by the motion's own tilt limit, and a second guard watches the legs: while the lift is extended the torso pitch must follow the hips
 * (real robot: within 6 deg, 14 deg at the fastest head-bang); more than 30 deg apart for 0.15 s means the whole robot is rocking over its toes (the full-speed sneeze did, and was dropped).
 * Ending (finished, stopped, a walking command, a fall): the hips go home first, then the lift. */
    float motion_wait_max;         /* s a request may wait for the robot to be standing (3)                           */
    float motion_start_up;         /* up[2] needed to start (0.96 = 16 deg)                                           */
    float motion_home_rate_hip;    /* rad/s of the hips going home when a motion is cut short (2)                     */
    float motion_home_rate_lift;   /* m/s of the lift going home (0.05, the stand's own lift speed)                   */
    const tars_motion_t *motion_req;   /* pending request (set last by tars_ctrl_motion_start)                        */
    float motion_req_speed, motion_wait;
    const tars_motion_t *motion;   /* the motion in progress                                                          */
    float motion_speed, motion_t;  /* time scale, position in the table [s of table time]                             */
    int   motion_phase;            /* 0 playing, 1 hips going home, 2 lift going home                                  */
    int   motion_abort;            /* 1: cut short                                                                     */
    float motion_tilt_t;           /* s the tilt limit has been exceeded                                               */
    float motion_leg_t;            /* s the legs have been tilted (torso pitch far from the hip angle while the legs stand on the floor) */
    float motion_q[3];             /* the targets in force: left hip [rad], right hip [rad], lift [m]                  */
    int   motion_end;              /* how the last motion ended: 1 played to the end, 2 stopped, 3 walking command, 4 fall (tilt limit), 5 request timed out, 6 the legs tilted (the whole robot rocked) */
    /* state */
    tars_state_t state;
    float t_state, t_gait, gain;
    float fall_t, still_t, stable_t;
    float moved_t;                 /* FAILED: s the robot has been handled (picked up / moved) since it gave up */
    float gy_f;                    /* low-passed pitch rate (see fb_lpf_hz) */
    int   gy_ok;                   /* gy_f holds a value */
    int   attempts;
    float hc, zc;                  /* slew-rate limited hip / lift commands */
    int   getup_sign;              /* +1 lying on back, -1 lying on front   */
} tars_ctrl_t;

/* 1 if every parameter of a gait lies inside the ranges the gait search used (+2 % of the range); a gait that arrives over the serial line must pass */
int tars_gait_in_range(const tars_gait_t *g);

void tars_ctrl_init(tars_ctrl_t *c);
void tars_ctrl_reset(tars_ctrl_t *c);

/*
 * One control step: dt = 2 ms at the default 500 Hz (the simulator steps it at 20 ms).
 *   up         : world-up in body frame (Mahony)             gyro : body rates [rad/s] (robot frame)
 *   cmd_vx/yaw : firmware command units (servo counts): vx>0 forward, vyaw>0 turn left
 *   out_q[2]   : hip targets {left, right} [rad], foot-forward positive
 *   out_z      : lift target [m]
 */
void tars_ctrl_step(tars_ctrl_t *c, const float up[3], const float gyro[3], float dt,
                    float cmd_vx, float cmd_vyaw, float out_q[2], float *out_z);

/* ask for a show motion.  0 = accepted (it starts once the robot stands still and upright, within motion_wait_max: stop the walking command first), -1 = bad arguments,
 * -3 = busy (falling, getting up, failed, or another motion is requested / running).  speed: 0.3 .. 1 (clamped).  Safe to call from another task than tars_ctrl_step() (it only sets the request). */
int tars_ctrl_motion_start(tars_ctrl_t *c, const tars_motion_t *m, float speed);
/* cut a motion short (or withdraw a request); the hips and the lift go home by themselves */
void tars_ctrl_motion_stop(tars_ctrl_t *c);

const char *tars_state_name(tars_state_t s);

#ifdef __cplusplus
}
#endif
#endif
