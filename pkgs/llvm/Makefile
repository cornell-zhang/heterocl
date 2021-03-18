.NOTPARALLEL:

PKG_LEVEL = ..
LEVEL = $(PKG_LEVEL)/..
include $(PKG_LEVEL)/Makefile.pkg.common

PkgBuildDir = $(HCL_PKG_BUILD)/llvm
LLVMBuildDir = $(PkgBuildDir)/llvm-project

LLVM_Repo := https://github.com/llvm/llvm-project.git
LLVM_Ver := 12.x
LLVM_Proj := clang;lld;mlir

ifneq (no, $(CMAKE_OK))
	CMAKE = $(CMAKE_CONFIG)
else
	CMAKE = $(HCL_PKG_CMAKE_BIN)/cmake
endif

### Rules
all:: git-llvm build-llvm 

git-llvm: $(PkgBuildDir)/.dir $(PkgBuildDir)/.git-llvm 

#=== Check out LLVM from SVN repo ===#

$(PkgBuildDir)/.git-llvm:
	@echo "Checking out llvm-project ..."
	@cd $(PkgBuildDir); \
	if [ ! -d llvm-project ]; then \
	  git clone -b release/$(LLVM_Ver) $(LLVM_Repo); \
	fi
	@touch $@

#=== Build LLVM ===#

build-llvm: $(PkgBuildDir)/.build-llvm

$(PkgBuildDir)/.build-llvm:
	cd $(LLVMBuildDir); \
	mkdir -p build; cd build; \
	$(CMAKE) -G "Unix Makefiles" ../llvm \
		-DLLVM_ENABLE_PROJECTS="$(LLVM_Proj)" \
		-DCMAKE_INSTALL_PREFIX=$(HCL_PKG_INSTALL) \
		-DLLVM_TARGETS_TO_BUILD="X86" \
		-DCMAKE_BUILD_TYPE=Release; \
	$(CMAKE) --build . -- -j8
	@touch $@
