.NOTPARALLEL:

PKG_LEVEL = ..
LEVEL = $(PKG_LEVEL)/..
include $(PKG_LEVEL)/Makefile.pkg.common

PkgBuildDir = $(HCL_PKG_BUILD)/llvm
LLVMSrcDir = $(PkgBuildDir)/src
LLVMBuildDir = $(PkgBuildDir)/build

LLVM_Release_Url := https://releases.llvm.org
LLVM_Version := 4.0.0

ifneq (no, $(CMAKE_OK))
	CMAKE = $(CMAKE_CONFIG)
else
	CMAKE = $(HCL_PKG_CMAKE_BIN)/cmake
endif

### Rules
all:: svn-llvm build-llvm 

svn-llvm: $(PkgBuildDir)/.dir $(PkgBuildDir)/.svn-llvm 

#=== Check out LLVM from SVN repo ===#

$(PkgBuildDir)/.svn-llvm:
	@echo "Downloading LLVM ..."
	@cd $(PkgBuildDir); \
	if [ ! -d src ]; then \
	  curl $(LLVM_Release_Url)/$(LLVM_Version)/llvm-$(LLVM_Version).src.tar.xz | \
		tar xJf - && mv llvm-$(LLVM_Version).src src;\
	fi
	@touch $@


#=== Patch LLVM ===#
patch-llvm: 

#=== Build LLVM ===#

build-llvm: $(LLVMBuildDir)/.built-llvm

$(LLVMBuildDir)/config.status: $(LLVMBuildDir)/.dir
	cd $(LLVMBuildDir); \
	$(CMAKE) -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=$(HCL_PKG_INSTALL) -DCMAKE_BUILD_TYPE=Release $(LLVMSrcDir)

$(LLVMBuildDir)/.built-llvm: $(LLVMBuildDir)/config.status 
	cd $(LLVMBuildDir); \
	$(MAKE)
	@touch $@
