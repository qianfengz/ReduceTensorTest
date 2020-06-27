/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef _HOST_REDUCE_HPP_ 
#define _HOST_REDUCE_HPP_

#include <vector>
#include <cassert>

static inline void
get_all_indexes(const std::vector<int>& dimLengths, int dim, std::vector<std::vector<int>>& indexes)
{
    if(dim < dimLengths.size())
    {
        std::vector<std::vector<int>> updated_indexes;

        if(dim == 0)
        {
            for(int i = 0; i < dimLengths[dim]; i++)
            {
                std::vector<int> index = {i};

                updated_indexes.push_back(index);
            };
        }
        else
        {
            // go through all the current indexes
            for(const auto& index : indexes)
                for(int i = 0; i < dimLengths[dim]; i++)
                {
                    auto index_new = index; // explicit copying

                    index_new.push_back(i);

                    updated_indexes.push_back(index_new);
                };
        };

        // update to the indexes (output)
        indexes = updated_indexes;

        // further to construct the indexes from the updated status
        get_all_indexes(dimLengths, dim + 1, indexes);
    };
};

static inline int get_offset_from_index(const std::vector<int>& strides,
                                        const std::vector<int>& index)
{
    int offset = 0;

    assert(index.size() == strides.size());
    for(int i = 0; i < index.size(); i++)
        offset += strides[i] * index[i];

    return (offset);
};

static inline int get_flatten_offset(const std::vector<int>& lengths, const std::vector<int>& index)
{
    int offset = 0;

    assert(lengths.size() == index.size() && lengths.size() > 0);

    int len    = lengths.size();
    int stride = 1;

    // for len==1, the loop is not executed
    for(int i = len - 1; i > 0; i--)
    {
        offset += stride * index[i];

        stride *= lengths[i];
    };

    offset += stride * index[0];

    return (offset);
};

template <typename FloatType>
class summationHost
{
 public:
    summationHost() = default;
    summationHost(const std::vector<int> inLengths_,
		  const std::vector<int> outLengths_,
	          const	std::vector<int> inStrides_,
	          const	std::vector<int> outStrides_,
                  const std::vector<int>& invariantDims_,
                  const std::vector<int>& toReduceDims_)
    {
        this->inLengths  = inLengths_; 
        this->outLengths = outLengths_;
        this->inStrides  = inStrides_;
        this->outStrides = outStrides_;

        this->invariantDims = invariantDims_;
        this->toReduceDims  = toReduceDims_;

        assert(this->inLengths.size() == this->outLengths.size());
        assert(!this->toReduceDims.empty());

        for(const auto dim : this->invariantDims)
            this->invariantLengths.push_back(this->inLengths[dim]);

        for(const auto dim : this->toReduceDims)
            toReduceLengths.push_back(this->inLengths[dim]);

        this->reduceAllDims = this->invariantDims.empty();
    };

    ~summationHost(){};

    private:
    std::vector<int> inLengths;
    std::vector<int> outLengths;
    std::vector<int> inStrides;
    std::vector<int> outStrides;

    std::vector<int> invariantLengths;
    std::vector<int> toReduceLengths;

    std::vector<int> invariantDims;
    std::vector<int> toReduceDims;

    bool reduceAllDims;

 public:
    void Run(FloatType alpha, const FloatType* in_data, FloatType beta, FloatType* out_data)
    {
        if(reduceAllDims)
        {
            std::vector<std::vector<int>> indexes_1;

            get_all_indexes(inLengths, 0, indexes_1); // generate the input indexes space

            auto accuVal = static_cast<FloatType>(0.0f);  

            // go through indexes of the invariant dimensions
            for(const auto& src_index : indexes_1)
            {
                auto src_offset = get_offset_from_index(this->inStrides, src_index);

                auto currVal = in_data[src_offset];

                accuVal += currVal; 
            };

            // scale the accumulated value
            if( alpha != static_cast<FloatType>(1.0f) )
                accuVal *= alpha; 

            // scale the prior dst value and add it to the accumulated value
            if( beta != static_cast<FloatType>(0.0f) )
                accuVal += out_data[0] * beta;

            // store the reduced value to dst location
            out_data[0] = accuVal;
        }
        else
        {
            std::vector<std::vector<int>> indexes_1, indexes_2;

            get_all_indexes(
                this->invariantLengths, 0, indexes_1); // generate the invariant indexes space
            get_all_indexes(
                this->toReduceLengths, 0, indexes_2); // generate the toReduce indexes space

            // go through indexes of the invariant dimensions
            for(const auto& index_1 : indexes_1)
            {
                std::vector<int> src_index;
                std::vector<int> dst_index;

                src_index.resize(this->inLengths.size());
                dst_index.resize(this->inLengths.size());

                // initialize the src index
                std::fill(dst_index.begin(), dst_index.end(), 0);

                for(int k = 0; k < invariantDims.size(); k++)
                    dst_index[invariantDims[k]] = index_1[k];

                int dst_offset = get_offset_from_index(this->outStrides, dst_index);

                // generate the part of src index belonging to invariant dims
                for(int k = 0; k < invariantDims.size(); k++)
                    src_index[invariantDims[k]] = index_1[k];

                FloatType accuVal = static_cast<FloatType>(0.0f);

                // go through indexes of the toReduce dimensions
                for(const auto& index_2 : indexes_2)
                {
                    // generate the part of src index belonging to toReduce dims
                    for(int k = 0; k < toReduceDims.size(); k++)
                        src_index[toReduceDims[k]] = index_2[k];

                    auto src_offset = get_offset_from_index(this->inStrides, src_index);

                    auto currVal = in_data[src_offset];

		    accuVal += currVal; 
                };

                // scale the accumulated value
                if( alpha != static_cast<FloatType>(1.0f) )
                    accuVal *= alpha;

                // scale the prior dst value and add it to the accumulated value
                if( beta != static_cast<FloatType>(0.0f) )
                    accuVal += out_data[dst_offset] * beta;

                // store the reduced value to dst location
                out_data[dst_offset] = accuVal;
            };
        };
    }; 
};

#endif
