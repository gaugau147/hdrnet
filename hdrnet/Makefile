TF_INC ?= `python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'`

SRC_DIR = ops

BUILD_DIR = build
LIB_DIR = lib

CC = gcc -std=c++17
CFLAGS = -fPIC -I$(TF_INC) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -O2

SRC = bilateral_cpu.cc
SRCS = $(addprefix $(SRC_DIR)/, $(SRC))

all: $(LIB_DIR)/hdrnet_ops.so

# Main library
$(LIB_DIR)/hdrnet_ops.so: $(LIB_DIR) $(SRCS)
	$(CC) -shared -o $@ $(SRCS) $(CFLAGS)

$(BUILD_DIR):
	mkdir -p $@


$(LIB_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)
