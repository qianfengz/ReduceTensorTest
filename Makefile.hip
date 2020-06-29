CC = /opt/rocm/bin/hipcc 

CCFLAGS :=  -g -D_GNU_SOURCE -std=c++11 -DHIP_MIOPEN_APP=1 -I/data/install/include -I/opt/rocm/opencl/include
LDFLAGS := -O2 -L/data/install/lib -L/opt/rocm/opencl/lib/ -lMIOpen

PROGRAMS := testReduceTensorHip

all: $(PROGRAMS)

# Step
testReduceTensorHip:  testReduceTensorHip.o
	$(CC) -o $@  $< $(LDFLAGS)

testReduceTensorHip.o: testReduceTensorHip.cpp
	$(CC) $(CCFLAGS) -c $< 

clean:
	rm -f *.o $(PROGRAMS)
