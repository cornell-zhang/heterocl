PKG_LEVEL = .
LEVEL = $(PKG_LEVEL)/..
include $(PKG_LEVEL)/Makefile.pkg.config

DIRS = 

# check llvm version
ifdef LLVM_CONFIG
	LLVM_VERSION = $(shell $(LLVM_CONFIG) --version| cut -b 1,3)
else
	LLVM_VERSION = 00
endif

ifdef CMAKE_CONFIG
	CMAKE_VERSION = $(shell $(CMAKE_CONFIG) --version | head -n 1 | cut -d ' ' -f 3 | cut -b 1,3)
else
	CMAKE_VERSION = 00
endif

ifneq (0, $(shell test $(CMAKE_VERSION) -ge 34; echo $$?))
	DIRS += cmake
else
	CMAKE_OK = yes
endif

ifeq ("", "$(findstring $(LLVM_VERSION), 40 50 60 70)")
	DIRS += llvm
endif

include $(PKG_LEVEL)/Makefile.pkg.common

