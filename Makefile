include Makefile.config

all: build

build: build-pkgs build-tvm build-hcl

build-pkgs:
	$(MAKE) -C pkgs

build-tvm:
	$(MAKE) -C tvm

build-hcl:
	cd python; \
	python setup.py install --user; \
  cd ../hlib/python; \
	python setup.py install --user;

clean:
	rm -rf build
