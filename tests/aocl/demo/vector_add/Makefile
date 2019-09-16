ifeq ($(VERBOSE),1)
ECHO := 
else
ECHO := @
endif

AOCL_COMPILE_CONFIG := $(shell aocl compile-config)
AOCL_LINK_CONFIG := $(shell aocl link-config)

ifeq ($(DEBUGF),1)
CXXFLAGS += -g
else
CXXFLAGS += -O2
endif


CXX := g++

TARGET := host
TARGET_DIR := bin

INCS := $(wildcard)
SRCS := $(wildcard host/*.cpp)
LIBS := rt pthread

all: $(TARGET_DIR)/$(TARGET)

$(TARGET_DIR)/$(TARGET): $(SRCS) $(TARGET_DIR)
	$(ECHO)$(CXX) $(CXXFLAGS) -fPIC $(AOCL_COMPILE_CONFIG) \
			$(SRCS) $(AOCL_LINK_CONFIG) \
			$(foreach L,$(LIBS),-l$L)\
			-o $(TARGET_DIR)/$(TARGET)

$(TARGET_DIR):
	$(ECHO)mkdir $(TARGET_DIR)

clean :
	$(ECHO)rm -rf $(TARGET_DIR)

.PHONY: all clean