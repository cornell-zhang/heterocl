.PHONY: all build-hcl build-tvm build-pkgs

include Makefile.config

all: build-hcl

build-pkgs:
	$(MAKE) -C pkgs

build-tvm: build-pkgs
	$(MAKE) -C tvm

build-hcl: build-tvm
	cd python; \
	python setup.py install --user --single-version-externally-managed; \
  	cd ../hlib/python; \
	python setup.py install --user --single-version-externally-managed;

build-python:
	cd python; \
	python setup.py install --user --single-version-externally-managed; \
	cd ../hlib/python; \
	python setup.py install --user --single-version-externally-managed;

clean:
	rm -rf build
	$(MAKE) clean -C tvm
