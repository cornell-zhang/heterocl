ifeq ($(VERBOSE),1)
ECHO := 
else
ECHO := @
endif

AOCL_COMPILE_CONFIG := $(shell aocl compile-config)
AOCL_LINK_CONFIG := $(shell aocl link_config)

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