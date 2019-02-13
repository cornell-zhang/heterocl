.NOTPARALLEL:

PKG_LEVEL = ..
LEVEL = $(PKG_LEVEL)/..
include $(PKG_LEVEL)/Makefile.pkg.config

PkgBuildDir = $(HCL_PKG_BUILD)/cmake
CmakeBuildDir = $(PkgBuildDir)/build
CmakeBin = $(CmakeBuildDir)/cmake/bin

CmakeDistrib = https://cmake.org/files/v3.10/cmake-3.10.2-Linux-x86_64.tar.gz

### Rules
all:: build-cmake

#=== Build cmake ===#

build-cmake: $(CmakeBuildDir)/.built-cmake

$(CmakeBuildDir)/.built-cmake:
	@mkdir -p $(CmakeBuildDir); \
	cd $(CmakeBuildDir); \
	wget $(CmakeDistrib); \
	tar -zxf cmake-3.10.2-Linux-x86_64.tar.gz; \
	rm cmake-3.10.2-Linux-x86_64.tar.gz; \
	mv cmake-3.10.2-Linux-x86_64 cmake
	@touch $@

