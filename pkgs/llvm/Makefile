.NOTPARALLEL:

PKG_LEVEL = ..
LEVEL = $(PKG_LEVEL)/..
include $(PKG_LEVEL)/Makefile.pkg.common

PkgBuildDir = $(HCL_PKG_BUILD)/llvm
LLVMSrcDir = $(PkgBuildDir)/src
LLVMBuildDir = $(PkgBuildDir)/build

LLVM_Repo := https://llvm.org/svn/llvm-project
LLVM_Tag := RELEASE_400/final

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
	@echo "Checking out LLVM ..." 
	@ cd $(PkgBuildDir); \
	if [ ! -d src ]; then \
	  svn co $(LLVM_Repo)/llvm/tags/$(LLVM_Tag) src; \
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
	$(MAKE) -j4
	@touch $@

