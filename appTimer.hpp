#ifndef _APP_TIMER_HPP_
#define _APP_TIMER_HPP_

#include <iostream>
#include <chrono>
#include <vector>

using std::chrono::steady_clock;

struct AppTimer 
{
    steady_clock::time_point zeroPoint;
    steady_clock::time_point execStart;
    steady_clock::time_point execEnd;

    void showSolvingTime(const std::vector<int>& inLengths, const std::vector<int>& toReduceDims)
    {
       std::chrono::nanoseconds tv0 = execStart - zeroPoint;
       std::chrono::nanoseconds tv1 = execEnd - execStart;

       std::cout << "ReduceTensor: " << std::endl;
       std::cout << "Input tensor dimensions : ";
       for (auto len : inLengths)
            std::cout << len << " ";
       std::cout << std::endl;

       std::cout << "To reduce dimensions : ";
       for (auto ind : toReduceDims)
            std::cout << ind << " ";
       std::cout << std::endl;
       std::cout << std::endl;

       std::cout << "The time spend before the last call of ReduceTensor is " << tv0.count() / 1000 << " microseconds. " << std::endl; 
       std::cout << "The execution time for one call is " << tv1.count() / 1000 << " microseconds. " << std::endl;
    };
}; 


#endif
