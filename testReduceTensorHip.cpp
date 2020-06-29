#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <array>
#include <vector>
#include <chrono>

#include "hostReduce.hpp"
#include "appArgs.hpp"

#define MY_HIP_CHECK(flag)                                                                                                     \
    do  {                                                                                                                       \
        hipError_t _tmpVal;                                                                                                    \
        if ( (_tmpVal = flag) != hipSuccess )  {                                                                               \
            std::ostringstream ostr;                                                                                            \
            ostr << "HIP Function Failed (" <<  __FILE__ << "," <<  __LINE__ << ") " << hipGetErrorString(_tmpVal);          \
            throw std::runtime_error(ostr.str());                                                                               \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define MY_MIOPEN_CHECK(flag)                                                                                     \
    do  {                                                                                                         \
        miopenStatus_t  _tmpVal;                                                                                  \
        if ( (_tmpVal = flag) != miopenStatusSuccess )  {                                                         \
            std::ostringstream ostr;                                                                              \
            ostr << "MIOPEN Function Failed (" <<  __FILE__ << "," <<  __LINE__ << "), error code = " << _tmpVal;  \
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
    TestApp() = default; 

    void prepare() 
    {
       MY_MIOPEN_CHECK( miopenCreate(&handle) );
       MY_MIOPEN_CHECK( miopenCreateReduceTensorDescriptor(&reduceDesc) );

       MY_MIOPEN_CHECK( miopenSetReduceTensorDescriptor(reduceDesc, op, compType, nanPropaOpt, indicesOpt, indicesType) );	    

       MY_MIOPEN_CHECK( miopenCreateTensorDescriptor(&inDesc) );
       MY_MIOPEN_CHECK( miopenCreateTensorDescriptor(&outDesc) );

       MY_MIOPEN_CHECK( miopenSetTensorDescriptor(inDesc, dataType, inLengths.size(), inLengths.data(), inStrides.data()) );
       MY_MIOPEN_CHECK( miopenSetTensorDescriptor(outDesc, dataType, outLengths.size(), outLengths.data(), outStrides.data()) );       

       MY_MIOPEN_CHECK( miopenGetTensorNumBytes(inDesc, &szInData) ); 
       MY_MIOPEN_CHECK( miopenGetTensorNumBytes(outDesc, &szOutData) ); 

       MY_HIP_CHECK( hipMalloc(&inDevData, szInData) ); 
       MY_HIP_CHECK( hipMalloc(&outDevData, szOutData) ); 

       MY_MIOPEN_CHECK( miopenGetReductionIndicesSize(handle, reduceDesc, inDesc, outDesc, &szIndices) ); 
       MY_MIOPEN_CHECK( miopenGetReductionWorkSpaceSize(handle, reduceDesc, inDesc, outDesc, &szWorkspace) ); 

       if ( szIndices > 0 )
            MY_HIP_CHECK( hipMalloc(&indicesBuffer, szIndices) );

       if ( szWorkspace > 0 )
            MY_HIP_CHECK( hipMalloc(&workspaceBuffer, szWorkspace) );

       inHostData.resize( szInData/ sizeof(float) ); 
       outHostData.resize( szOutData/ sizeof(float) ); 
       outHostData2.resize( szOutData/ sizeof(float) ); 
       
       for (auto& inVal : inHostData) 
	    inVal = RAN_GEN<float>(-1.0f, 1.0f); 

       std::fill(outHostData.begin(), outHostData.end(), 0.0f);
       std::fill(outHostData2.begin(), outHostData2.end(), 0.0f);

       MY_HIP_CHECK( hipMemcpy(inDevData, inHostData.data(), szInData, hipMemcpyHostToDevice) );
       MY_HIP_CHECK( hipMemcpy(outDevData, outHostData.data(), szOutData, hipMemcpyHostToDevice) );
    }; 

    void run() 
    {
       // run miopenReduceTensor() the first time, the kernels could be compiled here, which consume unexpected time
       MY_MIOPEN_CHECK( miopenReduceTensor(handle, reduceDesc,
                                      szIndices? indicesBuffer : nullptr, szIndices,
                                      szWorkspace? workspaceBuffer : nullptr, szWorkspace,
                                      &alpha,
                                      inDesc,
                                      inDevData,
                                      &beta,
                                      outDesc,
                                      outDevData) );

       MY_HIP_CHECK( hipMemcpy(outHostData.data(), outDevData, szOutData, hipMemcpyDeviceToHost) );

       if ( szIndices > 0 ) {
            std::vector<int> outIndices;
            outIndices.resize( szIndices/sizeof(int) );

            MY_HIP_CHECK( hipMemcpy(outIndices.data(), indicesBuffer, szIndices, hipMemcpyDeviceToHost) );
       };

       // For the most common situtaion, we do verification
       if ( doVerify && (dataType == miopenFloat && compType == miopenFloat && op == MIOPEN_REDUCE_TENSOR_ADD) ) {
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

       // run miopenReduceTensor() the second time 
       MY_MIOPEN_CHECK( miopenReduceTensor(handle, reduceDesc,
                                      szIndices? indicesBuffer : nullptr, szIndices,
                                      szWorkspace? workspaceBuffer : nullptr, szWorkspace,
                                      &alpha,
                                      inDesc,
                                      inDevData,
                                      &beta,
                                      outDesc,
                                      outDevData) );

       hipStream_t stream; 

       MY_MIOPEN_CHECK( miopenGetStream(handle,&stream) ); 

       MY_HIP_CHECK( hipStreamSynchronize(stream) ); 

       execEnd = system_clock::now(); 

       MY_HIP_CHECK( hipMemcpy(outHostData.data(), outDevData, szOutData, hipMemcpyDeviceToHost) );

       if ( szIndices > 0 ) {
            std::vector<int> outIndices;
            outIndices.resize( szIndices/sizeof(int) );

            MY_HIP_CHECK( hipMemcpy(outIndices.data(), indicesBuffer, szIndices, hipMemcpyDeviceToHost) );
       };

    }; 

    void showTime()
    {
       std::chrono::nanoseconds tv = execEnd - execStart; 

       std::cout << "miopenReduceTensor: " << std::endl; 
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

    ~TestApp() = default;  

private: 

    miopenHandle_t handle;
    miopenReduceTensorDescriptor_t reduceDesc;
 
    miopenTensorDescriptor_t inDesc; 
    miopenTensorDescriptor_t outDesc; 

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

