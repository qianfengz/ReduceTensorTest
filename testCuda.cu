#include <cuda.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#define MY_CUDA_CHECK(flag)                                                                                                     \
    do  {                                                                                                                       \
        cudaError_t _tmpVal;                                                                                                    \
        if ( (_tmpVal = flag) != cudaSuccess )  {                                                                               \
            std::ostringstream ostr;                                                                                            \
            ostr << "CUDNN Function Failed (" <<  __FILE__ << "," <<  __LINE__ << ") " << cudaGetErrorString(_tmpVal);          \
            throw std::runtime_error(ostr.str());                                                                               \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)


template <typename T>
static T FRAND(void)
{
    double d = static_cast<double>(rand() / (static_cast<double>(RAND_MAX)));
    return static_cast<T>(d);
}

template <typename T>
static T RAN_GEN(T A, T B)
{
    T r = (FRAND<T>() * (B - A)) + A;
    return r;
}

int main()
{
    size_t szInData =  2000*64*sizeof(float); 
    size_t szOutData = 2000*sizeof(float);  

    void *inDevData; 
    void *outDevData; 

    MY_CUDA_CHECK( cudaMalloc(&inDevData, szInData) ); 
    MY_CUDA_CHECK( cudaMalloc(&outDevData, szOutData) ); 

    std::vector<float> inHostData; 
    std::vector<float> outHostData; 

    inHostData.resize( szInData/ sizeof(float) ); 
    outHostData.resize( szOutData/ sizeof(float) ); 

    for (auto& inVal : inHostData) 
	 inVal = RAN_GEN<float>(0.0f, 1.0f); 

    std::fill(outHostData.begin(), outHostData.end(), 0.0f); 

    MY_CUDA_CHECK( cudaMemcpy(inDevData, inHostData.data(), szInData, cudaMemcpyHostToDevice) ); 
    MY_CUDA_CHECK( cudaMemcpy(outDevData, outHostData.data(), szOutData, cudaMemcpyHostToDevice) ); 

    cudaDeviceSynchronize();  
}; 

