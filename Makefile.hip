CC = /opt/rocm/bin/hipcc 

CCFLAGS :=  -g -D_GNU_SOURCE -std=c++11 -DHIP_MIOPEN_APP=1 -I/opt/rocm/miopen/include -I/opt/rocm/opencl/include
LDFLAGS := -O2 -L/opt/rocm/miopen/lib -L/opt/rocm/opencl/lib/ -lMIOpen

PROGRAMS := testReduceTensorHip

all: $(PROGRAMS)

# Step
testReduceTensorHip:  testReduceTensorHip.o
	$(CC) -o $@  $< $(LDFLAGS)

testReduceTensorHip.o: testReduceTensorHip.cpp appArgs.hpp appTimer.hpp
	$(CC) $(CCFLAGS) -c $<



clean:
	rm -f *.o $(PROGRAMS)
