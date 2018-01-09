.NOTPARALLEL:

PKG_LEVEL = ..
include $(PKG_LEVEL)/Makefile.pkg.common

PkgBuildDir = $(HCL_PKG_BUILD)/llvm
LLVMSrcDir = $(PkgBuildDir)/src
LLVMBuildDir = $(PkgBuildDir)/build

LLVM_Repo := http://llvm.org/svn/llvm-project
LLVM_Tag := RELEASE_40/final

LLVM_Config_Option = --prefix=$(HCL_PKG_INSTALL);
LLVM_LDFlags := 

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

$(LLVMBuildDir)/config.status: $(LLVMBuildDir)/.dir $(LLVMSrcDir)/configure
	cd $(LLVMBuildDir); \
	$(LLVMSrcDir)/configure $(LLVM_Config_Option) 

$(LLVMBuildDir)/.built-llvm: $(LLVMBuildDir)/config.status 
	cd $(LLVMBuildDir); \
	$(MAKE); \
	$(MAKE) install 
	@touch $@

