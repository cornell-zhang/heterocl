.NOTPARALLEL:

PKG_LEVEL = ..
LEVEL = $(PKG_LEVEL)/..
include $(PKG_LEVEL)/Makefile.pkg.common

PkgBuildDir = $(HCL_PKG_BUILD)/llvm
LLVMSrcDir = $(PkgBuildDir)/src
LLVMBuildDir = $(PkgBuildDir)/build

LLVM_Repo := http://releases.llvm.org
LLVM_Ver := 6.0.0
LLVM_Src := llvm-$(LLVM_Ver).src
LLVM_Tar := $(LLVM_Src).tar.xz

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
	  wget $(LLVM_Repo)/$(LLVM_Ver)/$(LLVM_Tar); \
	  tar -xf $(LLVM_Tar); \
		rm $(LLVM_Tar); \
		mv $(LLVM_Src) src; \
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
