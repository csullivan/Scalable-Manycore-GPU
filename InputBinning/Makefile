NVCC=nvcc
FLAGS=-std=c++11  -I${WBPATH}/libwb -g -G
LDFLAGS=-L${WBPATH}/libwb/lib ${WBPATH}/libwb/vendor/json11.o -L/opt/software/cudatoolkit/7.5/lib64 -L/opt/software/cudatoolkit/7.5/lib -lwb -lcuda -lm -lrt -lstdc++

all: bin/ibin

run:
	./bin/ibin -i data/$(dataset)/mode.raw,data/$(dataset)/input0.raw,data/$(dataset)/input1.raw,data/$(dataset)/grid_size.raw -e data/$(dataset)/output.raw -t vector

bin/%:  build/%.o
	mkdir -p bin
	$(NVCC) $(FLAGS) $(LDFLAGS) $^ -o $@

build/%.o: %.cu
	mkdir -p $(@D)
	$(NVCC) $(FLAGS) -c  $< -o $@

clean:
	rm -rf bin build
