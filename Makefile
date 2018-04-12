include Makefile.config

all: build

build: build-pkgs build-tvm build-hcl

build-pkgs:
	$(MAKE) -C pkgs

build-tvm:
	$(MAKE) -C tvm
	cd tvm/python; \
	python setup.py install --user

build-hcl:
	cd heterocl/python; \
	python setup.py install --user

clean:
	rm -rf build
