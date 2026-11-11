# Host-side tests of the Tars log tool: the portable firmware sources are compiled unchanged.
FW = ../../../main/boards/tars
CC ?= cc
CFLAGS = -std=c99 -O1 -g -Wall -Wextra -Wno-unused-parameter -I$(FW)
SRC = $(FW)/tars_log_core.c $(FW)/tars_log_shell.c

NNTEST = $(wildcard $(FW)/tars_rl_getup_nn_weights.h)
NNVEC = $(wildcard getup_nn_vectors.h)
all: tars_log_sim test_core test_gyro_bias test_ctrl $(if $(and $(NNTEST),$(NNVEC)),test_getup_nn)

tars_log_sim: sim_main.c $(SRC) $(FW)/tars_log_shell.h $(FW)/tars_log_port.h $(FW)/tars_log_core.h
	$(CC) -std=gnu99 -O1 -g -Wall -Wextra -Wno-unused-parameter -I$(FW) -D_GNU_SOURCE -o $@ sim_main.c $(SRC) -lm

test_core: test_core.c $(FW)/tars_log_core.c $(FW)/tars_log_core.h
	$(CC) $(CFLAGS) -o $@ test_core.c $(FW)/tars_log_core.c -lm -lpthread

test_gyro_bias: test_gyro_bias.c $(FW)/tars_rl_core.c $(FW)/tars_rl_core.h
	$(CC) $(CFLAGS) -o $@ test_gyro_bias.c $(FW)/tars_rl_core.c -lm

test_ctrl: test_ctrl.c $(FW)/tars_ctrl.c $(FW)/tars_ctrl.h $(FW)/tars_motions.h $(FW)/tars_gait_params.h $(FW)/tars_gait_params_v5.h $(FW)/tars_gait_params_v7.h $(FW)/tars_gait_params_v8.h $(FW)/tars_gait_params_v9.h
	$(CC) $(CFLAGS) -o $@ test_ctrl.c $(FW)/tars_ctrl.c -lm

test_getup_nn: test_getup_nn.c $(FW)/tars_rl_core.c $(FW)/tars_rl_core.h $(FW)/tars_rl_getup_nn_weights.h getup_nn_vectors.h
	$(CC) $(CFLAGS) -I. -o $@ test_getup_nn.c $(FW)/tars_rl_core.c -lm

test: all
	./test_core
	./test_gyro_bias
	./test_ctrl
	$(if $(and $(NNTEST),$(NNVEC)),./test_getup_nn)
	python3 test_protocol.py

clean:
	rm -rf tars_log_sim test_core test_gyro_bias test_ctrl test_getup_nn *.dSYM
