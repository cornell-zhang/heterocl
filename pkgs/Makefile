PKG_LEVEL = .
include $(PKG_LEVEL)/Makefile.pkg.common

DIRS = 

# check llvm version
ifdef LLVM_CONFIG
	LLVM_VERSION = $(shell $(LLVM_CONFIG) --version| cut -b 1,3)
else
	LLVM_VERSION = 00
endif

ifdef CMAKE_COONFIG
	CMAKE_VERSION = $(shell cmake --version| cut -b 1,3)
else
	CMAKE_VERSION = 00
endif

ifneq (0, $(shell test $(CMAKE_VERSION) -ge 34; $$?))
	DIRS += cmake
else
	CMAKE_OK = yes
endif

ifneq ("", $(findstring $(LLVM_VERSION), 40 50 60 70))
	DIRS += llvm
endif


