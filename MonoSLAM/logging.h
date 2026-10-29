#ifndef MONOSLAM_LOGGING_H
#define MONOSLAM_LOGGING_H

#include <fstream>
#include <iostream>

namespace monoslam::logging {

struct Logger
{
    static std::ofstream& file()
    {
        static std::ofstream f("monoslam.log", std::ios::app);
        return f;
    }
};

enum class LogLevel
{
    DEBUG,  // Log to file (only if NDEBUG is not defined)
    INFO,   // As DEBUG + always log to std::cout
    ERROR   // As DEBUG + always log to std::cerr
};

template<typename... Args>
void
log(LogLevel level, const char* filename, int line, Args&&... args)
{
    std::ofstream& f = Logger::file();

    if (f.is_open()) {
        f << '[' << filename << ':' << line << "]\n";
        (f << ... << std::forward<Args>(args));
    }

    if (level == LogLevel::INFO) {
        (std::cout << ... << std::forward<Args>(args));

    } else if (level == LogLevel::ERROR) {
        (std::cerr << ... << std::forward<Args>(args));
    }
}

#if !defined(NDEBUG)

#    define MONOSLAM_LOG(level, ...) monoslam::logging::log(level, __FILE__, __LINE__, __VA_ARGS__)

#else

#    define MONOSLAM_LOG(level, ...)                                            \
        do {                                                                    \
            if constexpr (level != monoslam::logging::LogLevel::DEBUG) {        \
                monoslam::logging::log(level, __FILE__, __LINE__, __VA_ARGS__); \
            }                                                                   \
                                                                                \
        } while (false)

#endif

}

#endif
