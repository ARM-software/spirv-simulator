#pragma once

#ifndef ARM_UTIL_HPP
#define ARM_UTIL_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace util
{
	std::vector<uint32_t> ReadFile(const std::string &path);
}

#endif