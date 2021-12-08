.PHONY: all build-hcl build-tvm build-pkgs

include Makefile.config

all: build-hcl

build-pkgs:
	$(MAKE) -C pkgs

build-tvm: build-pkgs
	$(MAKE) -C tvm

build-hcl: build-tvm
	cd python; \
	python setup.py install; \
	cd ../hlib/python; \
	python setup.py install;

build-python:
	cd python; \
	python setup.py install; \
	cd ../hlib/python; \
	python setup.py install;

clean:
	rm -rf build
	$(MAKE) clean -C tvm
