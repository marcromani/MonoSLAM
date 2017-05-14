#ifndef BUFFER_H
#define BUFFER_H

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

template <class T>
class Buffer {

private:

    int in;
    int out;
    int size;

    std::vector<T> data;

    std::atomic<int> numItems;
    std::mutex rwLock;

public:

    Buffer(int size_) : in(0), out(0), size(size_), data(size_), numItems(0) {}

    void push(T item) {

        std::lock_guard<std::mutex> lock(rwLock);

        data[in] = item;

        in++;
        if (in == size)
            in = 0;

        if (numItems < size)
            numItems++;
    }

    bool empty() {

        return (numItems == 0);
    }

    T pop() {

        while (numItems == 0)
            std::this_thread::yield();

        std::lock_guard<std::mutex> lock(rwLock);

        T result = data[out];

        out++;
        if (out == size)
            out = 0;

        numItems--;

        return result;
    }
};

#endif