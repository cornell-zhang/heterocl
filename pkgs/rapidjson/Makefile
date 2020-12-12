.NOTPARALLEL:

PKG_LEVEL = ..
LEVEL = $(PKG_LEVEL)/..
include $(PKG_LEVEL)/Makefile.pkg.config

PkgBuildDir = $(HCL_PKG_BUILD)/rapidjson
RapidjsonBuildDir = $(PkgBuildDir)/build

RapidjsonDistrib = https://github.com/Tencent/rapidjson.git

### Rules
all:: build-rapidjson

#=== Build rapidjson ===#

build-rapidjson: $(RapidjsonBuildDir)/.built-rapidjson

$(RapidjsonBuildDir)/.built-rapidjson:
	@mkdir -p $(RapidjsonBuildDir); \
	cd $(RapidjsonBuildDir); \
	git clone $(RapidjsonDistrib);
	@touch $@
