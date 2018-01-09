include Makefile.config

all: build

build: build-pkgs build-tvm

build-pkgs:
	$(MAKE) -C pkgs

build-tvm:
	$(MAKE) -C tvm

clean:
	rm -rf build
