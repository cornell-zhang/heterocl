include Makefile.config

all: build

build: build-pkgs build-tvm build-hcl

build-pkgs:
	$(MAKE) -C pkgs

build-tvm:
	$(MAKE) -C tvm -j16
	cd tvm/python; \
	python setup.py install

build-hcl:
	cd heterocl/python; \
	python setup.py install --user

clean:
	rm -rf build
