.PHONY: all build-hcl build-tvm build-pkgs

include Makefile.config

all: build-hcl

build-pkgs:
	$(MAKE) -C pkgs

build-src: build-pkgs
	$(MAKE) -C src -j$(MAKE_PARA)

build-hcl: build-src
	cd python; \
	python setup.py install --user; \
	cd ../hlib/python; \
	python setup.py install --user;

build-python:
	cd python; \
	python setup.py install --user; \
	cd ../hlib/python; \
	python setup.py install --user;

clean:
	rm -rf build
	$(MAKE) clean -C src
