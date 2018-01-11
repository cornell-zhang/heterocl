PKG_LEVEL = .
include $(PKG_LEVEL)/Makefile.pkg.common

DIRS = 

# check llvm version
ifdef LLVM_CONFIG
	LLVM_VERSION = $(shell $(LLVM_CONFIG) --version| cut -b 1,3)
else
	LLVM_VERSION = 00
endif

DIRS += llvm
#ifneq ("", $(findstring $(LLVM_VERSION), 40 50 60 70))
#	DIRS += llvm
#endif





