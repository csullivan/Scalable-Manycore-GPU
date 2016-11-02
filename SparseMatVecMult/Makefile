NVCC=nvcc
FLAGS=-std=c++11  -I${WBPATH}/libwb -g -G
LDFLAGS=-L${WBPATH}/libwb/lib ${WBPATH}/libwb/vendor/json11.o -L/opt/software/cudatoolkit/7.5/lib64 -L/opt/software/cudatoolkit/7.5/lib -lwb -lcuda -lm -lrt -lstdc++

all: bin/mvmult

run:
ifdef dataset
	./bin/mvmult -i data/$(dataset)/mode.flag,data/$(dataset)/col.raw,data/$(dataset)/row.raw,data/$(dataset)/data.raw,data/$(dataset)/vec.raw -e data/$(dataset)/output.raw -t vector
else
	./bin/mvmult -i data/0/mode.flag,data/0/col.raw,data/0/row.raw,data/0/data.raw,data/0/vec.raw -e data/0/output.raw -t vector
endif

bin/%:  build/%.o
	mkdir -p bin
	$(NVCC) $(FLAGS) $(LDFLAGS) $^ -o $@

build/%.o: %.cu
	mkdir -p $(@D)
	$(NVCC) $(FLAGS) -c  $< -o $@

clean:
	rm -rf bin build
