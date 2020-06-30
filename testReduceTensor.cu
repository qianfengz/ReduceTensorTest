#include <cuda.h>
#include <cudnn.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <array>
#include <vector>
#include <chrono>

#include "hostReduce.hpp"
#include "appArgs.hpp"

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

#define MY_CUDNN_CHECK(flag)                                                                                      \
    do  {                                                                                                         \
        cudnnStatus_t  _tmpVal;                                                                                   \
        if ( (_tmpVal = flag) != CUDNN_STATUS_SUCCESS )  {                                                        \
            std::ostringstream ostr;                                                                              \
            ostr << "CUDNN Function Failed (" <<  __FILE__ << "," <<  __LINE__ << "), error code = " << _tmpVal;  \
            throw std::runtime_error(ostr.str());                                                                 \
        }                                                                                                         \
    }                                                                                                             \
    while (0)

template <typename T>
static T FRAND(void)
{
    float d = static_cast<float>(rand() / (static_cast<float>(RAND_MAX)));
    return static_cast<T>(d);
}

template <typename T>
static T RAN_GEN(T A, T B)
{
    T r = (FRAND<T>() * (B - A)) + A;
    return r;
}

using std::chrono::system_clock;

class TestApp : public AppArgs 
{
public:
    TestApp() 
    {
        szInData = 0;
        szOutData = 0;

        inDevData = nullptr; 
        outDevData = nullptr; 

        szIndices = 0;
        szWorkspace = 0;

        indicesBuffer = nullptr; 
        workspaceBuffer = nullptr; 

        constructed = false; 
    }; 

    void prepare() 
    {
       MY_CUDNN_CHECK( cudnnCreate(&handle) );
       MY_CUDNN_CHECK( cudnnCreateReduceTensorDescriptor(&reduceDesc) );

       MY_CUDNN_CHECK( cudnnSetReduceTensorDescriptor(reduceDesc, op, compType, nanPropaOpt, indicesOpt, indicesType) );	    

       MY_CUDNN_CHECK( cudnnCreateTensorDescriptor(&inDesc) );
       MY_CUDNN_CHECK( cudnnCreateTensorDescriptor(&outDesc) );

       MY_CUDNN_CHECK( cudnnSetTensorNdDescriptor(inDesc, dataType, inLengths.size(), inLengths.data(), inStrides.data()) );
       MY_CUDNN_CHECK( cudnnSetTensorNdDescriptor(outDesc, dataType, outLengths.size(), outLengths.data(), outStrides.data()) );       

       MY_CUDNN_CHECK( cudnnGetTensorSizeInBytes(inDesc, &szInData) ); 
       MY_CUDNN_CHECK( cudnnGetTensorSizeInBytes(outDesc, &szOutData) ); 

       MY_CUDA_CHECK( cudaMalloc(&inDevData, szInData) ); 
       MY_CUDA_CHECK( cudaMalloc(&outDevData, szOutData) ); 

       MY_CUDNN_CHECK( cudnnGetReductionIndicesSize(handle, reduceDesc, inDesc, outDesc, &szIndices) ); 
       MY_CUDNN_CHECK( cudnnGetReductionWorkspaceSize(handle, reduceDesc, inDesc, outDesc, &szWorkspace) ); 

       if ( szIndices > 0 )
            MY_CUDA_CHECK( cudaMalloc(&indicesBuffer, szIndices) );

       if ( szWorkspace > 0 )
            MY_CUDA_CHECK( cudaMalloc(&workspaceBuffer, szWorkspace) );

       inHostData.resize( szInData/ sizeof(float) ); 
       outHostData.resize( szOutData/ sizeof(float) ); 
       outHostData2.resize( szOutData/ sizeof(float) ); 
       
       for (auto& inVal : inHostData) 
	    inVal = RAN_GEN<float>(-1.0f, 1.0f); 

       std::fill(outHostData.begin(), outHostData.end(), 0.0f);
       std::fill(outHostData2.begin(), outHostData2.end(), 0.0f);

       MY_CUDA_CHECK( cudaMemcpy(inDevData, inHostData.data(), szInData, cudaMemcpyHostToDevice) );
       MY_CUDA_CHECK( cudaMemcpy(outDevData, outHostData.data(), szOutData, cudaMemcpyHostToDevice) );

       constructed = true; 
    }; 

    void run() 
    {
       // run cudnnReduceTensor() the first time, the kernels could be compiled here, which consume unexpected time
       MY_CUDNN_CHECK( cudnnReduceTensor(handle, reduceDesc,
                                      szIndices? indicesBuffer : nullptr, szIndices,
                                      szWorkspace? workspaceBuffer : nullptr, szWorkspace,
                                      &alpha,
                                      inDesc,
                                      inDevData,
                                      &beta,
                                      outDesc,
                                      outDevData) );

       MY_CUDA_CHECK( cudaMemcpy(outHostData.data(), outDevData, szOutData, cudaMemcpyDeviceToHost) );

       if ( szIndices > 0 ) {
            std::vector<int> outIndices;
            outIndices.resize( szIndices/sizeof(int) );

            MY_CUDA_CHECK( cudaMemcpy(outIndices.data(), indicesBuffer, szIndices, cudaMemcpyDeviceToHost) );
       };

       // For the most common situtaion, we do verification
       if ( doVerify && (dataType == CUDNN_DATA_FLOAT && compType == CUDNN_DATA_FLOAT && op == CUDNN_REDUCE_TENSOR_ADD) ) {
            summationHost<float> summation(inLengths, outLengths, inStrides, outStrides, invariantDims, toReduceDims);

            summation.Run(alpha, inHostData.data(), beta, outHostData2.data());

            float max_error = 0.0f;
            for(int i=0; i < outHostData.size(); i++) {
                float error=std::abs(outHostData[i]-outHostData2[i]);

                if ( max_error < error )
                     max_error = error;
            };

            const float epsilon = 0.00002f;

            std::cout << "max_error = " << max_error << std::endl; 
         
            if ( max_error  < epsilon )
                 std::cout << "Verification succeeded!"  << std::endl;
       }

       execStart = system_clock::now();  

       // run cudnnReduceTensor() the second time 
       MY_CUDNN_CHECK( cudnnReduceTensor(handle, reduceDesc,
                                      indicesBuffer ? indicesBuffer : nullptr, szIndices,
                                      workspaceBuffer ? workspaceBuffer : nullptr, szWorkspace,
                                      &alpha,
                                      inDesc,
                                      inDevData,
                                      &beta,
                                      outDesc,
                                      outDevData) );

       MY_CUDA_CHECK( cudaMemcpy(outHostData.data(), outDevData, szOutData, cudaMemcpyDeviceToHost) );

       cudaStream_t  stream; 

       MY_CUDNN_CHECK( cudnnGetStream(handle, &stream) ); 

       MY_CUDA_CHECK( cudaStreamSynchronize(stream) ); 

       execEnd = system_clock::now(); 

       if ( szIndices > 0 ) {
            std::vector<int> outIndices;
            outIndices.resize( szIndices/sizeof(int) );

            MY_CUDA_CHECK( cudaMemcpy(outIndices.data(), indicesBuffer, szIndices, cudaMemcpyDeviceToHost) );
       };
    }; 

    void showTime()
    {
       std::chrono::nanoseconds tv = execEnd - execStart; 

       std::cout << "cudnnReduceTensor: " << std::endl; 
       std::cout << "Input tensor dimensions : "; 
       for (auto len : inLengths) 
	    std::cout << len << " "; 
       std::cout << std::endl; 

       std::cout << "To reduce dimensions : "; 
       for (auto ind : toReduceDims)
	    std::cout << ind << " "; 
       std::cout << std::endl; 

       std::cout << "The execution time for one call is " << tv.count() / 1000 << " microseconds. " << std::endl; 
    }; 

    ~TestApp() noexcept(false)
    {
       if ( constructed ) {
            MY_CUDNN_CHECK( cudnnDestroyReduceTensorDescriptor(reduceDesc) );

            MY_CUDNN_CHECK( cudnnDestroyTensorDescriptor(inDesc) );
            MY_CUDNN_CHECK( cudnnDestroyTensorDescriptor(outDesc) );

            MY_CUDA_CHECK( cudaFree(inDevData) );
            MY_CUDA_CHECK( cudaFree(outDevData) );

            if ( indicesBuffer )
                 MY_CUDA_CHECK( cudaFree(indicesBuffer) );

            if ( workspaceBuffer )
                 MY_CUDA_CHECK( cudaFree(workspaceBuffer) );
       };
    };

private: 
    bool constructed; 

    cudnnHandle_t handle;
    cudnnReduceTensorDescriptor_t reduceDesc;
 
    cudnnTensorDescriptor_t inDesc; 
    cudnnTensorDescriptor_t outDesc; 

    size_t szInData;
    size_t szOutData; 

    void *inDevData; 
    void *outDevData; 

    size_t szIndices; 
    size_t szWorkspace; 

    void *indicesBuffer; 
    void *workspaceBuffer; 

    std::vector<float> inHostData; 
    std::vector<float> outHostData; 
    std::vector<float> outHostData2; 

    const float alpha = 1.0f; 
    const float beta = 0.0f; 

    system_clock::time_point execStart; 
    system_clock::time_point execEnd; 
}; 

int main(int argc, char *argv[])
{
     TestApp myApp; 

     if ( myApp.processArgs(argc, argv) == 0 ) {
          myApp.prepare(); 
          myApp.run(); 
          myApp.showTime(); 
     }; 
}; 

