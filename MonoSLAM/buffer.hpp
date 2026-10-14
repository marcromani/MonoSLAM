#ifndef MONOSLAM_BUFFER_H
#define MONOSLAM_BUFFER_H

#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <vector>

namespace monoslam {

template<class T>
class Buffer
{
  public:
    Buffer(size_t size)
      : data_(size)
      , in_(0)
      , out_(0)
      , numItems_(0)
      , closed_(false)
    {
        assert(size > 0);
    }

    void push(T item)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (closed_) {
                return;
            }

            data_[in_] = std::move(item);

            ++in_;
            if (in_ == data_.size()) {
                in_ = 0;
            }

            if (numItems_ < data_.size()) {
                ++numItems_;

            } else {
                out_ = in_;
            }
        }

        notEmpty_.notify_one();
    }

    bool pop(T& item)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        notEmpty_.wait(lock, [this] { return numItems_ > 0 || closed_; });

        if (numItems_ == 0) {
            return false;
        }

        item = std::move(data_[out_]);

        ++out_;
        if (out_ == data_.size()) {
            out_ = 0;
        }

        --numItems_;

        return true;
    }

    void close()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            closed_ = true;
        }

        notEmpty_.notify_all();
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        return numItems_ == 0;
    }

  private:
    std::vector<T> data_;

    size_t in_;
    size_t out_;
    size_t numItems_;

    mutable std::mutex mutex_;
    std::condition_variable notEmpty_;

    bool closed_;
};

}

#endif
