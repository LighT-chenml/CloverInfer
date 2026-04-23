BUILD_DIR ?= build
NR_TASKLETS ?= 16
NUM_DPUS ?= 2
HEAD_DIM ?= 64
KEYS_PER_DPU ?= 8

DPU_CC ?= dpu-upmem-dpurte-clang
HOST_CC ?= cc
PKG_CONFIG ?= dpu-pkg-config

DPU_TARGET := $(BUILD_DIR)/dpu_qk
HOST_TARGET := $(BUILD_DIR)/host_qk

.PHONY: all run clean

all: $(DPU_TARGET) $(HOST_TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(DPU_TARGET): dpu_qk.c common.h | $(BUILD_DIR)
	$(DPU_CC) -O2 -DNR_TASKLETS=$(NR_TASKLETS) -I. -o $@ dpu_qk.c

$(HOST_TARGET): host_qk.c common.h $(DPU_TARGET) | $(BUILD_DIR)
	$(HOST_CC) -O2 -Wall -Wextra -I. -DDPU_BINARY=\"$(abspath $(DPU_TARGET))\" -o $@ host_qk.c $(shell $(PKG_CONFIG) --cflags --libs dpu)

run: all
	$(HOST_TARGET) $(NUM_DPUS) $(HEAD_DIM) $(KEYS_PER_DPU)

clean:
	rm -rf $(BUILD_DIR)
