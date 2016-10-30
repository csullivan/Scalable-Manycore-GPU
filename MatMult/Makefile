NVCC=nvcc
FLAGS=-std=c++11  -I${WBPATH}/libwb -g -G
LDFLAGS=-L${WBPATH}/libwb/lib ${WBPATH}/libwb/vendor/json11.o -L/opt/software/cudatoolkit/7.5/lib64 -L/opt/software/cudatoolkit/7.5/lib -lwb -lcuda -lm -lrt -lstdc++

all: bin/mmult

run:
ifdef dataset
	./bin/mmult -i ./data/$(dataset)/input0.raw,./data/$(dataset)/input1.raw -e ./data/$(dataset)/output* -t matrix
else
	./bin/mmult -i ./data/0/input0.raw,./data/0/input1.raw -e ./data/0/output* -t matrix
endif

bin/%:  build/%.o
	mkdir -p bin
	$(NVCC) $(FLAGS) $(LDFLAGS) $^ -o $@

build/%.o: %.cu
	mkdir -p $(@D)
	$(NVCC) $(FLAGS) -c  $< -o $@

clean:
	rm -rf bin build
