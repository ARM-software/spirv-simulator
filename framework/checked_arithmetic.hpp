#pragma once

#ifndef ARM_SPIRV_SIMULATOR_CHECKED_ARITHMETIC_HPP
#define ARM_SPIRV_SIMULATOR_CHECKED_ARITHMETIC_HPP

#include <cstdint>
#include <limits>

namespace SPIRVSimulator
{

inline constexpr bool CheckedAddU64(uint64_t lhs, uint64_t rhs, uint64_t& out) noexcept
{
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs)
    {
        return false;
    }

    out = lhs + rhs;
    return true;
}

inline constexpr bool CheckedSubU64(uint64_t lhs, uint64_t rhs, uint64_t& out) noexcept
{
    if (lhs < rhs)
    {
        return false;
    }

    out = lhs - rhs;
    return true;
}

inline constexpr bool CheckedMulU64(uint64_t lhs, uint64_t rhs, uint64_t& out) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    {
        return false;
    }

    out = lhs * rhs;
    return true;
}

inline constexpr uint64_t SaturatingMulU64(uint64_t lhs, uint64_t rhs) noexcept
{
    uint64_t result = 0;
    return CheckedMulU64(lhs, rhs, result) ? result : std::numeric_limits<uint64_t>::max();
}

inline constexpr uint64_t MagnitudeI64(int64_t value) noexcept
{
    const uint64_t unsigned_value = static_cast<uint64_t>(value);
    return value < 0 ? uint64_t{ 0 } - unsigned_value : unsigned_value;
}

inline constexpr bool CheckedAddI64(int64_t lhs, int64_t rhs, int64_t& out) noexcept
{
    if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
        (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
    {
        return false;
    }

    out = lhs + rhs;
    return true;
}

inline constexpr bool CheckedSubI64(int64_t lhs, int64_t rhs, int64_t& out) noexcept
{
    if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() + rhs) ||
        (rhs < 0 && lhs > std::numeric_limits<int64_t>::max() + rhs))
    {
        return false;
    }

    out = lhs - rhs;
    return true;
}

inline constexpr bool CheckedMulI64ByU64(int64_t lhs, uint64_t rhs, int64_t& out) noexcept
{
    if (rhs == 0)
    {
        out = 0;
        return true;
    }

    const bool     negative      = lhs < 0;
    const uint64_t lhs_magnitude = MagnitudeI64(lhs);

    // Positive results have a maximum magnitude of INT64_MAX.
    // Negative results may have magnitude 2^63, which represents INT64_MIN.
    constexpr uint64_t kInt64MinMagnitude =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + uint64_t{ 1 };
    const uint64_t result_magnitude_limit =
        negative ? kInt64MinMagnitude : static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

    if (lhs_magnitude > result_magnitude_limit / rhs)
    {
        return false;
    }

    const uint64_t result_magnitude = lhs_magnitude * rhs;
    out = negative ? -static_cast<int64_t>(result_magnitude - uint64_t{ 1 }) - int64_t{ 1 }
                   : static_cast<int64_t>(result_magnitude);
    return true;
}

inline constexpr bool CheckedExactDivideI64ByU64(int64_t numerator,
                                                 uint64_t denominator,
                                                 int64_t& out) noexcept
{
    if (denominator == 0)
    {
        return false;
    }

    const bool     negative            = numerator < 0;
    const uint64_t numerator_magnitude = MagnitudeI64(numerator);
    if ((numerator_magnitude % denominator) != 0)
    {
        return false;
    }

    const uint64_t quotient_magnitude = numerator_magnitude / denominator;
    out = negative ? -static_cast<int64_t>(quotient_magnitude - uint64_t{ 1 }) - int64_t{ 1 }
                   : static_cast<int64_t>(quotient_magnitude);
    return true;
}

inline constexpr bool CheckedExactDivideI64ByI64(int64_t numerator,
                                                 int64_t denominator,
                                                 int64_t& out) noexcept
{
    if (denominator == 0 ||
        (numerator == std::numeric_limits<int64_t>::min() && denominator == int64_t{ -1 }))
    {
        return false;
    }

    if ((numerator % denominator) != 0)
    {
        return false;
    }

    out = numerator / denominator;
    return true;
}

} // namespace SPIRVSimulator

#endif
