#pragma once

#ifndef ARM_UTIL_HPP
#define ARM_UTIL_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <cassert>

#define assertx(msg) assert((void(msg), false))
#define assertm(exp, msg) assert((void(msg), exp))

namespace util
{
std::vector<uint32_t> ReadFile(const std::string& path);
}

#endif