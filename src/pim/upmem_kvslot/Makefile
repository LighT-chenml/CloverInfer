BUILD_DIR ?= build
NR_TASKLETS ?= 16
NUM_DPUS ?= 2

DPU_CC ?= dpu-upmem-dpurte-clang
HOST_CC ?= cc
PKG_CONFIG ?= dpu-pkg-config

DPU_TARGET := $(BUILD_DIR)/dpu_kvslot
HOST_TARGET := $(BUILD_DIR)/host_kvslot

.PHONY: all clean

all: $(DPU_TARGET) $(HOST_TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(DPU_TARGET): dpu_kvslot.c common.h | $(BUILD_DIR)
	$(DPU_CC) -O2 -DNR_TASKLETS=$(NR_TASKLETS) -I. -o $@ dpu_kvslot.c

$(HOST_TARGET): host_kvslot.c common.h $(DPU_TARGET) | $(BUILD_DIR)
	$(HOST_CC) -O2 -Wall -Wextra -I. -DDPU_BINARY=\"$(abspath $(DPU_TARGET))\" -o $@ host_kvslot.c $(shell $(PKG_CONFIG) --cflags --libs dpu)

clean:
	rm -rf $(BUILD_DIR)
