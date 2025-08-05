#pragma once

#ifndef ARM_UTIL_HPP
#define ARM_UTIL_HPP

#include <cstdint>
#include <string>
#include <vector>

#ifdef NDEBUG
#include <stdexcept>

#define assertx(msg) throw std::runtime_error(msg);
#define assertm(exp, msg)                  \
    do                                     \
    {                                      \
        if (!(exp))                        \
        {                                  \
            throw std::runtime_error(msg); \
        }                                  \
    } while (0)
#else
#include <cassert>

#define assertx(msg) assert((void(msg), false))
#define assertm(exp, msg) assert((void(msg), exp))

#endif

namespace util
{
std::vector<uint32_t> ReadFile(const std::string& path);
}

#endif