HOST_CC = g++
HOST_CC_FLAGS = -g -O2

GEM5_CC = aarch64-linux-g++
GEM5_CC_FLAGS = -g -O2 -static

GEM5_CC_FLAGS_0 = -g -O -static
GEM5_CC_FLAGS_1 = -g -O1 -static
GEM5_CC_FLAGS_3 = -g -O3 -static

DATA_DIR = ../data
GEM5_DIR = $(HOME)/gem5

RM=rm -f

KERNEL = kernel

OBJDIR = build
SOBJ_host = hotspot_host.o
SOBJ_gem5 = hotspot_gem5.o

OBJ_host := $(patsubst %.o,$(OBJDIR)/%.o,$(SOBJ_host))
OBJ_gem5 := $(patsubst %.o,$(OBJDIR)/%.o,$(SOBJ_gem5))


compile_all: hotspot_neon 

compile_neon: hotspot_neon


link_objs_gem5 = $(GEM5_CC) $(GEM5_CC_FLAGS) -o $@ $(OBJ_gem5) $(OBJDIR)/$(KERNEL)_$@.o $(LIBS)


run_neon: hotspot_neon
	@echo "======================================================================"
	@echo "NEON done by us"
	@echo "problem size () with 1 iterations... "
	@echo "======================================================================"
	$(GEM5_DIR)/build/ARM/gem5.opt $(GEM5_DIR)/configs/example/se.py --cpu-type=ex5_LITTLE --caches --bp-type=LocalBP --param='system.cpu[0].branchPred.BTBEntries=1' --param='system.cpu[0].branchPred.localCtrBits=1' --svevl=1 -c hotspot_neon -o '64 64 1 1 $(DATA_DIR)/temp_64 $(DATA_DIR)/power_64 output_neon.out'
	

hotspot_neon: $(OBJ_gem5) $(KERNEL).c
	$(GEM5_CC) $(GEM5_CC_FLAGS) -march=armv8-a+simd -fno-tree-vectorize -o $(OBJDIR)/$(KERNEL)_$@.o -c $(KERNEL).c
	$(link_objs_gem5_0)
	


$(OBJDIR)/$(SOBJ_gem5) : hotspot.cpp
	$(GEM5_CC) $(GEM5_CC_FLAGS) -fno-tree-vectorize -c $< -o $@

clean :
	$(RM) hotspot_neon
	$(RM) -r build/

$(shell mkdir -p $(OBJDIR))
