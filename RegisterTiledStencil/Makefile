NVCC=nvcc
FLAGS=-std=c++11  -I${WBPATH}/libwb -g -G
LDFLAGS=-L${WBPATH}/libwb/lib ${WBPATH}/libwb/vendor/json11.o -L/opt/software/cudatoolkit/7.5/lib64 -L/opt/software/cudatoolkit/7.5/lib -lwb -lcuda -lm -lrt -lstdc++

all: bin/stencil

run:
	./bin/stencil -i data/0/input.ppm -e data/0/output.ppm

bin/%:  build/%.o
	mkdir -p bin
	$(NVCC) $(FLAGS) $(LDFLAGS) $^ -o $@

build/%.o: %.cu
	mkdir -p $(@D)
	$(NVCC) $(FLAGS) -c  $< -o $@

clean:
	rm -rf bin build
