.PHONY: all build-hcl build-tvm build-pkgs

include Makefile.config

all: build-hcl
dev: develop-hcl

build-pkgs:
	$(MAKE) -C pkgs

build-tvm: build-pkgs
	$(MAKE) -C tvm

build-hcl: build-tvm
	cd python; \
	python setup.py install --user; \
	cd ../hlib/python; \
	python setup.py install --user;

develop-hcl: build-tvm
	cd python; \
	python setup.py develop --user; \
	cd ../hlib/python; \
	python setup.py develop --user;

build-python:
	cd python; \
	python setup.py install --user; \
	cd ../hlib/python; \
	python setup.py install --user;

clean:
	rm -rf build
	$(MAKE) clean -C tvm
