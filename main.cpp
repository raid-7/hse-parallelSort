#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <tbb/parallel_invoke.h>
#include <tbb/global_control.h>

struct Result
{
    double value;
    double variation;
};

std::ostream& operator<<(std::ostream& os, const Result& result)
{
    os << result.value << " +- " << result.variation;
    return os;
}

Result stats(const std::vector<double>& data) {
    double sum = 0.0;
    for (double x : data) {
        sum += x;
    }
    double mean = sum / data.size();
    double sqSum = 0.0;
    for (double x : data) {
        sqSum += x*x - mean * mean;
    }
    double stddev = data.size() < 2 ? 0.0 : std::sqrt(sqSum / (data.size() - 1));
    return {mean, stddev};
}

class Timer {
public:
    Timer() {
        Reset();
    }

    void Reset()
    {
        StartTime_ = std::chrono::high_resolution_clock::now();
    }

    double Nanoseconds()
    {
        auto finishTime = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(finishTime - StartTime_).count();
    }

    double Microseconds()
    {
        return Nanoseconds() / 1000.;
    }

    double Seconds()
    {
        return Microseconds() / 1'000'000.;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> StartTime_;
};

template <class V>
std::ostream& operator <<(std::ostream& stream, const std::vector<V>& v) {
    for (const auto& x : v) {
        stream << x << ' ';
    }
    return stream;
}

template <class It, class V>
It partition(It begin, It end, V ref) {
    if (begin >= end)
        return begin;
    It l = begin - 1;
    It r = end;
    while (true) {
        do ++l; while (*l < ref);
        do --r; while (*r > ref);
        if (l >= r)
            return r;
        std::iter_swap(l, r);
    }
}

template <class It>
void quicksort(It begin, It end) {
    if (begin < end) {
        It refIt = begin + (end - begin) / 2;
        It m = ::partition(begin, end, *refIt); // ADL sucks
        quicksort(begin, m);
        quicksort(m + 1, end);
    }
}

template <class It, size_t granularity=256>
void parallelQuicksort(It begin, It end) {
    if (begin < end) {
        if (end - begin <= granularity) {
            quicksort(begin, end);
            return;
        }

        It refIt = begin + (end - begin) / 2;
        It m = ::partition(begin, end, *refIt);
        tbb::parallel_invoke(
            [begin, m]() {
                parallelQuicksort(begin, m);
            },
            [m, end](){
                parallelQuicksort(m + 1, end);
            }
        );
    }
}


template <class F, class V>
Result time(size_t numRuns, F func, const V& arg) {
    std::vector<double> results;
    for (size_t i = 0; i < numRuns; ++i) {
        V copy = arg;
        Timer t;
        func(copy);
        results.push_back(t.Microseconds() / 1'000'000.0);
    }
    return stats(results);
}



int main() {
    tbb::global_control concurrencyLimit(tbb::global_control::max_allowed_parallelism, 4);

    auto nonParallel = [](auto& v) {
        quicksort(v.begin(), v.end());
    };
    auto parallel = [](auto& v) {
        parallelQuicksort(v.begin(), v.end());
    };

    std::vector<int> data(100'000'000);
    std::minstd_rand rand(42);
    std::uniform_int_distribution<int> dist(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    std::generate(data.begin(), data.end(), [&rand, &dist]() {
        return dist(rand);
    });

    std::cout << "Data generated." << std::endl;

    {
        std::cout << "Verifying correctness:" << std::endl;
        auto reference = data;
        std::sort(reference.begin(), reference.end());
        std::cout << "Reference sort finished." << std::endl;

        auto sorted = data;
        quicksort(sorted.begin(), sorted.end());
        std::cout << "Serial implementation correct: " << std::boolalpha << (sorted == reference) << std::endl;

        sorted = data;
        parallelQuicksort(sorted.begin(), sorted.end());
        std::cout << "Parallel implementation correct: " << std::boolalpha << (sorted == reference) << std::endl;
    }

    std::cout << "Running serial version:" << std::endl;
    std::cout << "Serial: " << time(5, nonParallel, data) << std::endl;
    std::cout << "Running parallel version:" << std::endl;
    std::cout << "Parallel: " << time(5, parallel, data) << std::endl;
    return 0;
}
