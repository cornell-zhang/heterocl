CFLAGS += -Wall -g -std=c++11 -O2 -fPIC -I../src -I../../dmlc-core/include

.PHONY: all clean
.PRECIOUS: bin/% obj/%.o

all: $(patsubst %.cc,run_%,$(wildcard *.cc))

run_%: bin/%
	$^

bin/%: obj/%.o ../lib/libHalideIR.a
	@mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -o $@ $^

obj/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -c -o $@ $^

../lib/libHalideIR.a:
	$(MAKE) -C $@

clean:
	rm -rf bin/ obj/
