#pragma once

#ifndef ARM_SPIRV_SIMULATOR_HPP
#define ARM_SPIRV_SIMULATOR_HPP

#include <iostream>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <type_traits>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <queue>

#ifdef DEBUG_BUILD
#define MAX_LOOP_COUNT 100
#else
#define MAX_LOOP_COUNT 10000
#endif

//  Flip SPIRV_HEADERS_PRESENT to 1 to auto‑pull the SPIR‑V-Headers from the environment.
#define SPV_ENABLE_UTILITY_CODE 1

#ifndef SPIRV_HEADERS_PRESENT
#define SPIRV_HEADERS_PRESENT 0
#endif
#if SPIRV_HEADERS_PRESENT
#include <spirv/unified1/spirv.hpp>
#else
#include "spirv.hpp"
#endif

#include "spirv-tools/libspirv.h"

namespace SPIRVSimulator
{

// Any result ID or pointer object ID in this set, can be treated as if it has
// any valid value for the given type
#define SPS_FLAG_IS_ARBITRARY              1
// Used for values that are uninitialized due to simulator assumptions
#define SPS_FLAG_UNINITIALIZED             2
// Used to track pbuffer candidate values
#define SPS_FLAG_IS_CANDIDATE              4
// Used for values that are confirmed to be pbuffer pointer values
#define SPS_FLAG_IS_PBUFFER_PTR            8
// Used for values that contain or are affected by per-thread builtins (workgroup or vertex ids etc.)
#define SPS_FLAG_THREAD_SPECIFIC          16
// Used to track descriptor candidate values
#define SPS_FLAG_IS_DESCRIPTOR_CANDIDATE  32
// Used to track descriptor candidate values
#define SPS_FLAG_IS_DESCRIPTOR_BUFFER     64
// Used to mark pointer pointing to uninitialized bindings, these may be legal and needs special handling
#define SPS_FLAG_IS_UNINITIALIZED_BINDING 128

// Used to track metadata about value source chains
#define SPS_FLAG_IS_FLOAT_SOURCE        1
#define SPS_FLAG_IS_ARITHMETIC_SOURCE   2
#define SPS_FLAG_IS_INTERPOLATED_SOURCE 4


// Flags for the simulator, can be OR'ed into the flags argument on initialization

// If set, the simulator will crash when it encounters access to an incomplete buffer.
// If this is not set, access to such buffers will return the default value of the type.
#define ERROR_RAISE_ON_BUFFERS_INCOMPLETE 1


// Used for descriptor loop tracking
struct PhiIncoming {
    uint32_t value_id;
    uint32_t parent_label;
};

// Used by tracing tools to pass in potential pbuffer candidates.
// This is optional but allows for easier remapping user side in most cases
struct PhysicalAddressCandidate
{
    uint64_t address;
    uint64_t offset;

    void* payload;

    // The simulator will set this if it encounters a physical address pointer matching the
    // metadata contained in this struct, thereby confirming there is indeed a pbuffer pointer
    // in a given buffer with these properties/values.
    bool verified = false;
};

struct DescriptorCandidate
{
    uint64_t address;
    uint64_t offset;

    uint32_t size_value_id;

    void* payload;
};

// Used internally by the simulator, can be passed between invocations by copying it from one invocation to another to
// enable some optimizations.
struct InternalPersistentData
{
    // Any shader whose SimulationData shader ID is found here can be safely skipped
    std::set<uint64_t> uninteresting_shaders;
};

// ---------------------------------------------------------------------------
//  Input/output structure
//
//  This structure defines the shader inputs.
//  This must be populated and passed to the run(...) method to
//  populate the shader input values before and during execution.
//
//  The format of the data must be what the shader expects, eg. if a buffer is bound
//  to a binding with the std430 layout, the data in the byte vectors must obey the rules of
//  that layout

struct PhysicalAddressData;

struct SimulationData
{
    // The following are input values that should be populated by the user
    //////////////////////////////////////////////////////////////////////

    // The SpirV ID of the entry point to use
    uint32_t entry_point_id = 0;
    // The OpName label (function name) of the entry point to use, takes priority over entry_point_id if it is set.
    std::string entry_point_op_name = "";

    // Data block pointer -> byte_offset_to_array -> array length (in number of elements)
    // The Data block pointer should be the void* -> uint64_t bitcast of a pointer matching one of
    // the bound buffers/data blocks in bindings, push_constants, specialization_constants or physical_address_buffers
    std::unordered_map<uint64_t, std::unordered_map<size_t, size_t>> rt_array_lengths;

    // SpecId -> byte offset
    // For each SpecID this should give the offset (in bytes) to the given specialization constant in
    // specialization_constants
    std::unordered_map<uint32_t, size_t> specialization_constant_offsets;
    const void*                          specialization_constants = nullptr;

    // The full binary push_constant Data block
    const void* push_constants = nullptr;

    // DescriptorSet -> Binding -> Data block
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, void*>> bindings;

    // These can be provided by the user in order to properly initialize PhysicalStorageBuffer storage class values.
    // The keys here are uint64_t values who contain the bits in the physical address pointers seen on the GPU.
    // The value pair is the size of the buffer (in bytes) followed by the pointer to the Data block on the host side
    // (which should be a copy of the GPU side data)
    std::unordered_map<uint64_t, std::pair<size_t, void*>> physical_address_buffers;

    // Optional map of buffers to pbuffer candidates in said buffers.
    // If provided, the simulator will mark candidates in this when it finds a physical address pointer
    // and raise an error if it finds a physical address pointer with no candidate in this list.
    // If the map is empty all candidate related code and functionality will be skipped.
    std::unordered_map<const void*, std::vector<PhysicalAddressCandidate>> candidates;

    // Optional map of buffers to descriptor candidates in said buffers.
    // Any buffer that has the VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT set must have a corresponding entry here.
    // Unlike the PhysicalAddressCandidate map, the vector of DescriptorCandidate's can be empty, no errors will be raised if a descriptor is found without a candidate in this list.
    std::unordered_map<const void*, std::vector<DescriptorCandidate>> descriptor_candidates;

    // Optional value, a unique identifier for the input shader.
    // If provided, this can allow the simulator to massively speed up simulation time
    // for cases where the same shader is dispatched multiple times throughout a session.
    uint64_t shader_id = 0;


    // The following are output values that will be populated by the simulator
    //////////////////////////////////////////////////////////////////////////

    // Written to by the simulator
    std::unordered_map<const void*, std::vector<PhysicalAddressCandidate>> output_candidates;
    std::vector<PhysicalAddressData> physical_address_data;

    // Set to true if the simulator encountered a case that requires all threads in a dispatch to run in order to guarantee no pointers are missed
    bool full_dispatch_needed = false;

    // Set to true if the simulator encountered a physical address pointer that was not listed in the input candidates
    bool unlisted_candidate_found = false;

    // Set to true if the simulator encounters a loop lasting longer than MAX_LOOP_COUNT iterations, this will cause it to abort the loop and continue (simulator will assume a hang due to invalid inputs)
    bool aborted_long_loop = false;

    // Set to true if any arbitrary data was written to external memory
    bool had_arbitrary_write = false;

    // Used for internal optimization between dispatches, should never be touched by the user.
    InternalPersistentData persistent_data;
};

// For backwards compatibility
using InputData = SimulationData;

// ---------------------------------------------------------------------------
// Output structure

enum BitLocation
{
    Constant,     // Constant embedded in the shader, offsets will be relative to the start of the spirv binary code
    SpecConstant, // Spec constant, binding will be SpecId
    StorageClass  // storage_class specifies what block type we are dealing with
};

struct DataSourceBits
{
    /*
    Structure describing where a sequence of bits can be found that ended up being used to construct
    (or that was eventually interpreted as) a pointer pointing to data with the PhysicalStorageBuffer storage class
    */

    // Specifies where the data comes from
    BitLocation location;

    // If location is StorageClass, this holds the storage class
    spv::StorageClass storage_class;

    // The input source pointer this comes from
    const void* source_ptr = nullptr;

    // If the location is StorageClass and the pointer is located in a array of Block's
    // then this is the array index which holds the descriptor block containing the pointer
    uint64_t idx;

    // DescriptorSet ID when location is StorageClass. Unused otherwise
    uint64_t set_id;

    // Binding ID when location is StorageClass.
    // SpecId when location is SpecConstant.
    // Unused otherwise
    uint64_t binding_id;

    // Absolute byte offset into the containing buffer where the data is located
    // If location is Constant, then this will be the byte offset into the spirv shader words input
    uint64_t byte_offset;

    // The bit offset from the byte offset where the data was stored
    uint64_t bit_offset;

    // Number of bits in the bit sequence
    uint64_t bitcount;

    // Bit offset into the final pointer where this data ended up
    uint64_t val_bit_offset;
};

// We return a vector of these.
// The bit_components vector contain data on where the bits that eventually become the pointer were read from.
struct PhysicalAddressData
{
    std::vector<DataSourceBits> bit_components;
    uint64_t                    raw_pointer_value;
};

// ---------------------------------------------------------------------------

struct Instruction
{
    spv::Op opcode;

    // word_count is the total number of words this instruction is composed of,
    // including the word holding the opcode + wordcount value.
    // This (along with the opcode above) is a redundant value as the first uint32 in words will also hold it,
    // but it is included/decoded here for ease-of-use and clarity
    uint16_t                  word_count;
    std::span<const uint32_t> words;
};

struct Type
{
    enum class Kind
    {
        Void,
        BoolT, // Because x11 headers have a macro called Bool
        Int,
        Float,
        Vector,
        Matrix,
        Array,
        Struct,
        Pointer,
        RuntimeArray, // TODO: We can/probably should make these maps and use sparse access (eg. add a new map value for
                      // these and load during OpAccessChain)
        Image,
        Sampler,
        SampledImage,
        Opaque,
        NamedBarrier,
        AccelerationStructureKHR,
        RayQueryKHR
    } kind;

    struct ScalarTypeData
    {
        uint32_t width;
        bool     is_signed;
    };
    struct VectorTypeData
    {
        uint32_t elem_type_id;
        uint32_t elem_count;
    };
    struct MatrixTypeData
    {
        uint32_t col_type_id;
        uint32_t col_count;
    };
    struct ArrayTypeData
    {
        uint32_t elem_type_id;
        uint32_t length_id;
    };
    struct PointerTypeData
    {
        uint32_t storage_class;
        uint32_t pointee_type_id;
    };
    struct ImageTypeData
    {
        uint32_t sampled_type_id;
        uint32_t dim;
        uint32_t depth;
        uint32_t arrayed;
        uint32_t multisampled;
        uint32_t sampled;
        uint32_t image_format;
    };
    struct SampledImageTypeData
    {
        uint32_t image_type_id;
    };
    struct OpaqueTypeData
    {
        uint32_t name;
    };
    struct StructTypeData
    {
        uint32_t id; // The issue here is that even having the same data layout, structs are different types
    };

    union
    {
        ScalarTypeData       scalar;
        VectorTypeData       vector;
        MatrixTypeData       matrix;
        ArrayTypeData        array;
        PointerTypeData      pointer;
        ImageTypeData        image;
        SampledImageTypeData sampled_image;
        OpaqueTypeData       opaque;
        StructTypeData       structure;
    };
    Type() : kind(Kind::Void) { scalar = { 0, false }; }

    static Type BoolT()
    {
        Type t;
        t.kind   = Kind::BoolT;
        t.scalar = ScalarTypeData{ .width = 32, .is_signed = false };
        return t;
    }

    static Type Int(uint32_t width, bool is_signed)
    {
        Type t;
        t.kind   = Kind::Int;
        t.scalar = ScalarTypeData{ .width = width, .is_signed = is_signed };
        return t;
    }

    static Type Float(uint32_t width)
    {
        Type t;
        t.kind   = Kind::Float;
        t.scalar = ScalarTypeData{ .width = width, .is_signed = true };
        return t;
    }

    static Type Vector(uint32_t element_type_id, uint32_t element_count)
    {
        Type t;
        t.kind   = Kind::Vector;
        t.vector = VectorTypeData{ .elem_type_id = element_type_id, .elem_count = element_count };
        return t;
    }

    static Type Matrix(uint32_t column_type_id, uint32_t column_count)
    {
        Type t;
        t.kind   = Kind::Matrix;
        t.matrix = MatrixTypeData{ .col_type_id = column_type_id, .col_count = column_count };
        return t;
    }

    static Type Struct()
    {
        Type t;
        t.kind         = Kind::Struct;
        t.structure.id = 0;
        return t;
    }

    static Type Struct(uint32_t struct_id)
    {
        Type t;
        t.kind         = Kind::Struct;
        t.structure.id = struct_id;
        return t;
    }

    static Type Array(uint32_t elem_type_id, uint32_t length_id)
    {
        Type t;
        t.kind  = Kind::Array;
        t.array = ArrayTypeData{ .elem_type_id = elem_type_id, .length_id = length_id };
        return t;
    }

    static Type RuntimeArray(uint32_t elem_type_id)
    {
        Type t;
        t.kind  = Kind::RuntimeArray;
        t.array = ArrayTypeData{ .elem_type_id = elem_type_id, .length_id = 0 };
        return t;
    }

    static Type Pointer(uint32_t storage_class, uint32_t pointee_type_id)
    {
        Type t;
        t.kind    = Kind::Pointer;
        t.pointer = PointerTypeData{ .storage_class = storage_class, .pointee_type_id = pointee_type_id };
        return t;
    }
};

struct AggregateV;
struct PointerV;
struct VectorV;
struct MatrixV;
struct SampledImageV;

using Value = std::variant<std::monostate,
                           uint64_t,
                           int64_t,
                           double,
                           std::shared_ptr<VectorV>,
                           std::shared_ptr<MatrixV>,
                           std::shared_ptr<AggregateV>,
                           PointerV,
                           SampledImageV>;

struct ValueMetadata
{
    uint64_t flags = 0;
};

struct PointerV
{
    // Either an index (if the storage class is stored in internal heaps), or the actual pointer value
    // If it is a pointer, it always points to host memory, it must be remapped for pbuffer pointers to get the GPU
    // pointer
    uint64_t pointer_handle;

    // Flags related to the object this pointer points to, the full meta struct is redundant as the pointer keeps track of those values
    uint64_t pointee_flags;

    // The following two values refer to the base pointers type and result id.
    // Eg. a pointer created from OpAccessChain will keep these as they were in the base pointer.

    // The TypeID of this pointers base (not the pointee)
    uint32_t base_type_id;
    // The result ID of the instruction that made this pointer's base (if appliccable, this can be 0)
    uint32_t base_result_id;

    uint32_t storage_class;

    // If it points to a value inside a composite, aggregate or array value.
    // This is the indirection path within said value.
    std::vector<uint32_t> idx_path;
};

inline bool operator==(const PointerV& a, const PointerV& b)
{
  return a.pointer_handle == b.pointer_handle &&
         a.pointee_flags == b.pointee_flags && a.base_type_id == b.base_type_id &&
         a.base_result_id == b.base_result_id &&
         a.storage_class == b.storage_class && a.idx_path == b.idx_path;
}

struct SampledImageV
{
    uint64_t image_handle;
    uint64_t sampler_handle;
};

inline bool operator==(const SampledImageV& a, const SampledImageV& b)
{
    return a.image_handle == b.image_handle && a.sampler_handle == b.sampler_handle;
}

struct VectorV
{
    std::vector<Value> elems;
    VectorV() = default;

    template <typename T>
    explicit VectorV(std::initializer_list<T> initializer_list)
    {
        elems.reserve(initializer_list.size());
        for (const auto& item : initializer_list)
        {
            elems.push_back(Value(item));
        }
    }
};

inline bool operator==(const VectorV& a, const VectorV& b)
{
    if (a.elems.size() != b.elems.size())
    {
        return false;
    }
    for (size_t i = 0; i < a.elems.size(); ++i)
    {
        if (!(a.elems[i] == b.elems[i]))
        {
            return false;
        }
    }
    return true;
}

struct MatrixV
{
    std::vector<Value> cols;
    MatrixV() = default;

    explicit MatrixV(std::initializer_list<Value> initializer_list) :
        cols(initializer_list.begin(), initializer_list.end())
    {}

    template <typename T>
    MatrixV(std::initializer_list<T> initializer_list, uint32_t cols_count)
    {
        cols.reserve(cols_count);

        uint32_t rows = initializer_list.size() / cols_count;

        for (uint32_t i = 0; i < cols_count; ++i)
        {
            cols.emplace_back(std::make_shared<VectorV>());
        }

        auto it = initializer_list.begin();
        for (uint32_t c = 0; c < cols_count; ++c)
        {
            auto& v = std::get<std::shared_ptr<VectorV>>(cols[c]);
            for (uint32_t r = 0; r < rows; ++r)
            {
                v->elems.push_back(*(it + r * cols_count + c));
            }
        }
    }
};

inline bool operator==(const MatrixV& a, const MatrixV& b)
{
    if (a.cols.size() != b.cols.size())
    {
        return false;
    }
    for (size_t i = 0; i < a.cols.size(); ++i)
    {
        const auto& a_c = std::get<std::shared_ptr<VectorV>>(a.cols[i]);
        const auto& b_c = std::get<std::shared_ptr<VectorV>>(b.cols[i]);
        if (!(a_c && b_c))
        {
            return false;
        }
        if (!(*a_c == *b_c))
        {
            return false;
        }
    }
    return true;
}

struct AggregateV
{
    AggregateV() = default;
    explicit AggregateV(std::initializer_list<Value> initializer_list) :
        elems(initializer_list.begin(), initializer_list.end())
    {}
    std::vector<Value> elems;
}; // array or struct

inline bool operator==(const AggregateV& a, const AggregateV& b)
{
    if (a.elems.size() != b.elems.size())
    {
        return false;
    }
    for (size_t i = 0; i < a.elems.size(); ++i)
    {
        if (!(a.elems[i] == b.elems[i]))
        {
            return false;
        }
    }
    return true;
}

template <typename T>
concept Deref = requires(T t) { *t; };

template <typename T>
concept PointerT = std::is_pointer_v<T> || Deref<T>;

template <typename T>
concept ValueT = !PointerT<T>;

struct ValueComparator
{
    template <ValueT T>
    bool operator()(const T& a, const T& b) const
    {
        return a == b;
    }

    template <PointerT T>
    bool operator()(T a, T b) const
    {
        if (a && b)
            return *a == *b;
        return a == b;
    }

    template <typename A, typename B>
    bool operator()(const A&, const B&) const
    {
        return false;
    }
};

inline bool operator==(const Value& a, const Value& b)
{
    if (a.index() != b.index())
    {
        return false;
    }
    return std::visit(ValueComparator{}, a, b);
}

struct DecorationInfo
{
    spv::Decoration       kind;
    std::vector<uint32_t> literals;
};

void DecodeInstruction(std::span<const uint32_t>& program_words, Instruction& instruction);

template <class T>
void extract_bytes(std::vector<std::byte>& output, T input, size_t num_bits)
{
    if (sizeof(input) != 8)
    {
        throw std::runtime_error("SPIRV simulator: extract_bytes called on type that is not 8 bytes");
    }

    std::array<std::byte, sizeof(T)> arr;

    std::memcpy(arr.data(), &input, sizeof(T));
    if (num_bits > 32)
    {
        output.insert(output.end(), arr.begin(), arr.end());
    }
    else
    {
        output.insert(output.end(), arr.begin(), arr.begin() + 4);
    }
}

template <typename T>
T ReverseBits(T value, unsigned bitWidth)
{
    assert(std::is_unsigned<T>::value &&
           "SPIRV simulator: Can only reverse the bits in unsigned integer types, cast first");
    T result = 0;
    for (unsigned i = 0; i < bitWidth; ++i)
    {
        result <<= 1;
        result |= (value & 1);
        value >>= 1;
    }
    return result;
}

template <typename T>
T ArithmeticRightShiftUnsigned(T value, unsigned shift, unsigned bitWidth)
{
    assert(std::is_unsigned<T>::value &&
           "SPIRV simulator: Can only arith shift the bits in unsigned integer types, cast first");

    if (shift == 0 || shift >= bitWidth)
        return value;

    T msb     = (value >> (bitWidth - 1)) & 1;
    T shifted = value >> shift;

    if (msb)
    {
        T mask = ((T(1) << shift) - 1) << (bitWidth - shift);
        shifted |= mask;
    }

    return shifted;
}

size_t CountBitsUInt(uint64_t value);

// Bitcast can be very annoying to import on certain platforms, even if c++20 is supported
// Just do this for now, and we can replace this with the std::bit_cast version in the future
template <class To, class From>
typename std::enable_if_t<sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> &&
                              std::is_trivially_copyable_v<To>,
                          To>
bit_cast(const From& src) noexcept
{
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

// On x86 platforms, pointers are not 64bit
// This template should catch any reads from a pointer and safely convert it into x64 value
template <class To, class From>
typename std::enable_if_t<std::is_pointer_v<From> && std::is_integral_v<To> && sizeof(std::uintptr_t) < sizeof(To), To>
bit_cast(From p) noexcept
{
    return static_cast<To>(reinterpret_cast<std::uintptr_t>(p));
}

// On x86 platforms, pointers are not 64bit
// This template tries to convert uint64_t value to a readable pointer
// However, uint64_t value might not fit into a std::uintptr_t on x86
// In practice, should not happen
template <class To>
typename std::enable_if_t<std::is_pointer_v<To>, To> bit_cast(std::uint64_t v) noexcept
{
    if constexpr (sizeof(std::uintptr_t) < sizeof(std::uint64_t))
    {
        // On 32-bit, ensure no information would be lost.
        assert(v <= std::numeric_limits<uint32_t>::max() && "uint64_t value doesn't fit in uintptr_t");
    }
    return reinterpret_cast<To>(static_cast<std::uintptr_t>(v));
}

inline std::string read_instruction_literal(const Instruction& instruction,
                                             uint32_t start_word)
{
    std::vector<char> bytes;

    uint32_t word_index = start_word;
    while (word_index < instruction.word_count) {
        uint32_t word = instruction.words[word_index];

        for (int b = 0; b < 4; ++b) {
            uint8_t byte = static_cast<uint8_t>((word >> (8 * b)) & 0xFFu);
            if (byte == 0) {
                return std::string(bytes.begin(), bytes.end());
            }

            bytes.push_back(static_cast<char>(byte));
        }
        ++word_index;
    }

    std::cout << "SPIRV simulator: WARNING: read_instruction_literal reached end of range without seeing NUL terminator\n" << std::endl;

    return std::string(bytes.begin(), bytes.end());;
}


/// Operand chain related code
static inline bool IsIdKind(spv_operand_type_t t) {
    switch (t) {
        //case SPV_OPERAND_TYPE_TYPE_ID:
        case SPV_OPERAND_TYPE_RESULT_ID:
        //case SPV_OPERAND_TYPE_ID:
        //case SPV_OPERAND_TYPE_SCOPE_ID:
        //case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
            return true;
        default:
            return false;
    }
}

static inline uint32_t CountImageOperandsExtraWords(uint32_t mask) {
    uint32_t n = 0;
    if (mask & (1u << 0)) n += 1; // Bias
    if (mask & (1u << 1)) n += 1; // Lod
    if (mask & (1u << 2)) n += 2; // Grad
    if (mask & (1u << 3)) n += 1; // ConstOffset
    if (mask & (1u << 4)) n += 1; // Offset
    if (mask & (1u << 5)) n += 2; // ConstOffsets (minimum; adjust if you parse fully)
    if (mask & (1u << 6)) n += 1; // Sample
    if (mask & (1u << 7)) n += 1; // MinLod
    if (mask & (1u << 8)) n += 1; // MakeTexelAvailable
    if (mask & (1u << 9)) n += 1; // MakeTexelVisible
    return n; // others add 0
}

static inline uint32_t CountMemoryAccessExtraWords(uint32_t mask) {
    uint32_t n = 0;
    if (mask & (1u << 0)) n += 1; // Aligned -> alignment literal
    if (mask & (1u << 2)) n += 1; // MakePointerAvailable -> scope (literal or *_ID per grammar)
    if (mask & (1u << 3)) n += 1; // MakePointerVisible   -> scope (literal or *_ID per grammar)
    if (mask & (1u << 5)) n += 1; // KHR variants
    if (mask & (1u << 6)) n += 1;
    return n;
}

inline spv_result_t OnParsedHeader(void* user_data, spv_endianness_t endian, uint32_t magic, uint32_t version, uint32_t generator, uint32_t id_bound, uint32_t reserved) {
    (void)user_data;
    (void)endian;
    (void)magic;
    (void)version;
    (void)generator;
    (void)id_bound;
    (void)reserved;

    return SPV_SUCCESS;
}

struct ParseState {
    const uint32_t* module_words = nullptr;
    std::vector<std::vector<uint32_t>>*     out_table    = nullptr;
};

inline spv_result_t OnParsedInst(void* user, const spv_parsed_instruction_t* inst) {
    auto* st = static_cast<ParseState*>(user);

    std::vector<uint32_t> ids;

    for (uint16_t oi = 0; oi < inst->num_operands; ++oi) {
        const spv_parsed_operand_t& op = inst->operands[oi];
        const spv_operand_type_t& ot = op.type;

        uint32_t operand_offset = inst->type_id ? 1 : 0 + inst->result_id ? 1 : 0;
        uint32_t cursor = operand_offset;

        if (IsIdKind(ot)) {
            ids.push_back(inst->words[cursor++]);
            continue;
        }

        if (ot == SPV_OPERAND_TYPE_VARIABLE_ID) {
            while (cursor < inst->num_words) {
                ids.push_back(inst->words[cursor++]);
            }
            break;
        }

        if (ot == SPV_OPERAND_TYPE_OPTIONAL_IMAGE) {
            if (cursor < inst->num_words) {
                const uint32_t mask = inst->words[cursor++];
                cursor += CountImageOperandsExtraWords(mask);
            }
            continue;
        }

        if (ot == SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS) {
            if (cursor < inst->num_words) {
                const uint32_t mask = inst->words[cursor++];
                cursor += CountMemoryAccessExtraWords(mask);
            }
            continue;
        }

        ++cursor;
    }

    st->out_table->emplace_back(std::move(ids));
    return SPV_SUCCESS;
}


/// Control flow
struct BlockInfo {
    uint32_t label = 0;                   // %label (from OpLabel result-id)
    std::vector<uint32_t> succs;          // CFG successors (label ids)
    uint32_t loop_merge = 0;              // %merge if this is a loop header (0 otherwise)
    uint32_t loop_continue = 0;           // %continue if this is a loop header
    std::vector<uint32_t> instruction_indices;
};

struct CFG {
    // Block id -> BlockInfo
    std::unordered_map<uint32_t, BlockInfo> blocks;
    // For reverse edges (optional but useful)
    std::unordered_map<uint32_t, std::vector<uint32_t>> preds;
    // Maintain function boundaries if you have multiple functions
    std::vector<uint32_t> block_order;    // in module order (optional)
};

struct LoopInfo {
    uint32_t header = 0;
    uint32_t merge = 0;
    uint32_t cont = 0;
    std::vector<uint32_t> blocks;             // excludes merge
    std::set<uint32_t> block_set;
};

LoopInfo BuildLoopRegion(const CFG& cfg, uint32_t header);


/// Main class
class SPIRVSimulator
{
  public:
    explicit SPIRVSimulator(const std::vector<uint32_t>& program_words,
                            SimulationData&              input_data,
                            bool                         verbose = false,
                            uint64_t flags = 0);

    // Actually interpret the SPIRV. If we return true, then this means we have to execute
    // every thread of the invokation.
    bool Run();

    std::set<std::string> unsupported_opcodes;

    virtual ~SPIRVSimulator() = default;

  protected:
    SPIRVSimulator() = default;

    bool done_             = false;
    bool is_execution_fork = false;

    // If true, the simulated shader wrote something to non-image external memory, or to a non-interpolated output
    bool has_buffer_writes_ = false;

    uint32_t num_result_ids_     = 0;
    uint64_t current_fork_index_ = 0;
    uint32_t current_heap_index_ = 1;

    // Parsing artefacts
    SimulationData* input_data_;
    // Contains entry point ID -> entry point OpName labels (labels may be
    // non-existent/empty)
    std::unordered_map<uint32_t, std::string>           entry_points_;
    std::unordered_map<uint32_t, spv::ExecutionModel>   entry_point_models_;
    std::vector<uint32_t>                               program_words_;
    std::span<const uint32_t>                           stream_;
    std::vector<Instruction>                            instructions_;
    std::vector<std::vector<uint32_t>>                  ids_per_instruction_;
    std::vector<uint32_t>                               block_label_per_instruction_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> spec_instr_words_;
    std::unordered_map<uint32_t, Instruction>           spec_instructions_;
    std::unordered_map<uint32_t, size_t>                result_id_to_inst_index_;
    std::unordered_map<uint32_t, Type>                  types_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> struct_members_;
    std::unordered_map<uint32_t, uint32_t>              forward_type_declarations_; // Unused, consider removing this
    std::unordered_map<uint32_t, std::vector<DecorationInfo>>                               decorators_;
    // struct result_id -> struct member_literal -> array of Decoration
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::vector<DecorationInfo>>> struct_decorators_;
    std::unordered_map<uint32_t, std::string>                                               extended_imports_;
    std::unordered_map<uint32_t, std::string>                                               string_literals_;

    Type               void_type_;

    // This maps the result ID of pointers to the result ID of values stored
    // through them
    std::unordered_map<uint32_t, uint32_t> values_stored_;

    // Debug only
    bool verbose_;
    uint64_t flags_;

    // Counts how many times each branch instruction was taken, used to abort infinite loops
    std::unordered_map<uint32_t, uint64_t> branch_counters_;

    // These hold information about any pointers that reference physical storage
    // buffers
    std::vector<PointerV>                        physical_address_pointers_;
    std::vector<std::pair<PointerV, PointerV>>   pointers_to_physical_address_pointers_;
    std::unordered_map<uint32_t, DataSourceBits> data_source_bits_;

    // These hold information about descriptor buffer stuff
    // Descriptor writeout OpStore instruction id -> write counter for that writeout code
    std::unordered_map<uint32_t, uint32_t> merged_descriptor_write_count;

    // Control flow
    struct FunctionInfo
    {
        size_t                inst_index;
        size_t                first_inst_index;
        std::vector<uint32_t> parameter_ids_;
        std::vector<uint32_t> parameter_type_ids_;
    };
    struct ActiveLoop {
        uint32_t header;
        uint32_t merge;
    };

    uint32_t                                   prev_defined_func_id_;
    std::unordered_map<uint32_t, FunctionInfo> funcs_;

    // Control flow graph of the whole program, used for loop detection and analysis
    CFG cfg_;
    std::unordered_map<uint32_t, LoopInfo> loops_;
    std::vector<ActiveLoop> loop_stack_;

    uint32_t prev_block_id_             = 0;
    uint32_t current_block_id_          = 0;
    uint32_t current_merge_block_id_    = 0;
    uint32_t current_continue_block_id_ = 0;

    // Execution fork data, used to prevent infinte loops in SPIRV loop constructs and double forks
    std::set<uint32_t>* visisted_fork_branches_ = nullptr;
    std::set<uint32_t> forked_blocks_;

    // Heaps & frames
    struct Frame
    {
        size_t   pc;
        uint32_t result_id;
        uint32_t func_heap_index;
    };

    std::vector<Frame> call_stack_;

    // result_id -> Value
    // std::unordered_map<uint32_t, Value> globals_;
    std::vector<Value> values_;
    std::vector<ValueMetadata> value_meta_;
    std::vector<Value> function_heap_;

    // storage_class -> heap_index -> Heap Value for all non-function storage classes
    std::unordered_map<uint32_t, std::vector<Value>> heaps_;

    void BuildAllLoops()
    {
        for (auto [bid, bi] : cfg_.blocks) {
            if (bi.loop_merge) {
                loops_.emplace(bid, BuildLoopRegion(cfg_, bid));
            }
        }
    }

    virtual void on_loop_begin(uint32_t header);
    virtual void on_loop_exit (uint32_t header);
    virtual void on_loop_iteration(uint32_t header);
    virtual void OnEnterBlockHandleLoops();

    // Handlers used by the OpExtInst handler
    // Implementation of the operations in the GLSL extended set
    void GLSLExtHandler(uint32_t                         type_id,
                        uint32_t                         result_id,
                        uint32_t                         instruction_literal,
                        const std::span<const uint32_t>& operand_words);

    // Helpers
    // TODO: Many more of these can be const, fix
    virtual void DecodeHeader();
    virtual void ParseAll();
    virtual void Validate();
    virtual void BuildCFGFromWords();
    virtual void InitializeIdOpsTable() {
        ParseState parse_state = { program_words_.data(), &ids_per_instruction_ };

        spv_context   ctx  = spvContextCreate(SPV_ENV_UNIVERSAL_1_6);
        spv_diagnostic diag = nullptr;

        // calls OnParsedInst once per instruction.
        spvBinaryParse(
            ctx,
            &parse_state,
            program_words_.data(),
            program_words_.size(),
            OnParsedHeader,
            OnParsedInst,
            &diag
        );

        spvDiagnosticDestroy(diag);
        spvContextDestroy(ctx);
    };
    virtual bool ExecuteInstruction(const Instruction&, bool dummy_exec = false);
    virtual void ExecuteInstructions();
    virtual void CreateExecutionFork(const SPIRVSimulator& source, uint32_t branching_value_id, std::set<uint32_t>* visited_set, SimulationData* fork_input_data = nullptr);

    virtual std::string  GetValueString(const Value&);
    virtual std::string  GetTypeString(const Type&);
    virtual void         PrintInstruction(const Instruction&);
    virtual void         HandleUnimplementedOpcode(const Instruction&);
    virtual Value        MakeScalar(uint32_t type_id, const uint32_t*& words);
    virtual Value        MakeDefault(uint32_t type_id, const uint32_t** initial_data = nullptr);
    virtual uint64_t     RemapHostToClientPointer(uint64_t host_pointer) const;
    virtual void         WritePointer(const PointerV& ptr, const Value& value);
    virtual Value        ReadPointer(const PointerV& ptr);
    virtual const Value& GetValue(uint32_t result_id);
    virtual uint64_t     GetArrayLength(uint32_t length_id);
    virtual void         SetValue(uint32_t result_id, const Value& value, bool clear_meta=true);
    virtual const Type&  GetTypeByTypeId(uint32_t type_id) const;
    virtual const Type&  GetTypeByResultId(uint32_t result_id) const;
    virtual uint32_t     GetTypeID(uint32_t result_id) const;
    virtual void         WriteValue(std::byte* external_pointer, uint32_t type_id, const Value& value);
    virtual void         ReadWords(const std::byte* external_pointer, uint32_t type_id, std::vector<uint32_t>& buffer_data);
    virtual uint64_t GetPointerOffset(const PointerV& pointer_value) const;
    virtual size_t   CountSetBits(const Value& value, uint32_t type_id, bool* is_arbitrary);
    virtual size_t   GetBitizeOfType(uint32_t type_id);
    virtual uint32_t GetTargetPointerType(const PointerV& pointer);
    virtual size_t   GetBitizeOfTargetType(const PointerV& pointer);
    virtual void     GetBaseTypeIDs(uint32_t type_id, std::vector<uint32_t>& output);
    virtual bool     IsMemberOfStruct(uint32_t member_id, uint32_t& struct_id, uint32_t& member_literal) const;

    virtual std::vector<DataSourceBits> FindDataSourcesFromResultID(uint32_t result_id, uint32_t* property_flags = nullptr);
    virtual bool                        HasDecorator(uint32_t result_id, spv::Decoration decorator) const;
    virtual bool                        HasDecorator(uint32_t result_id, uint32_t member_id, spv::Decoration decorator) const;
    virtual uint32_t GetDecoratorLiteral(uint32_t result_id, spv::Decoration decorator, size_t literal_offset = 0) const;
    virtual uint32_t GetDecoratorLiteral(uint32_t result_id, uint32_t member_id, spv::Decoration decorator, size_t literal_offset = 0) const;

    virtual bool IsLoopCounterPhi(uint32_t candidate_id) const;
    virtual uint32_t DeriveDescriptorSizeID(const Instruction& condition_instruction) const;

    virtual bool ValueIsArbitrary(uint32_t result_id) const {
        return value_meta_[result_id].flags & SPS_FLAG_IS_ARBITRARY;
    };
    virtual bool PointeeValueIsArbitrary(const PointerV& pointer) const {
        return pointer.pointee_flags & SPS_FLAG_IS_ARBITRARY;
    };
    virtual bool ValueIsCandidate(uint32_t result_id) const {
        return value_meta_[result_id].flags & SPS_FLAG_IS_CANDIDATE;
    };
    virtual bool ValueIsThreadSpecific(uint32_t result_id) const {
        return value_meta_[result_id].flags & SPS_FLAG_THREAD_SPECIFIC;
    };

    virtual bool ValueHoldsPbufferPtr(uint32_t result_id) const {
        return value_meta_[result_id].flags & SPS_FLAG_IS_PBUFFER_PTR;
    };
    virtual bool PointeeValueHoldsPbufferPtr(const PointerV& pointer) const {
        return pointer.pointee_flags & SPS_FLAG_IS_PBUFFER_PTR;
    };

    virtual void SetHoldsPbufferPtr(uint32_t result_id) {
        value_meta_[result_id].flags |= SPS_FLAG_IS_PBUFFER_PTR;
    };
    virtual void ClearHoldsPbufferPtr(uint32_t result_id) {
        value_meta_[result_id].flags &= ~SPS_FLAG_IS_PBUFFER_PTR;
    };

    virtual void SetIsArbitrary(uint32_t result_id) {
        value_meta_[result_id].flags |= SPS_FLAG_IS_ARBITRARY;
    };
    virtual void ClearIsArbitrary(uint32_t result_id) {
        value_meta_[result_id].flags &= ~SPS_FLAG_IS_ARBITRARY;
    };

    virtual void SetIsThreadSpecific(uint32_t result_id) {
        value_meta_[result_id].flags |= SPS_FLAG_THREAD_SPECIFIC;
    };
    virtual void ClearIsThreadSpecific(uint32_t result_id) {
        value_meta_[result_id].flags &= ~SPS_FLAG_THREAD_SPECIFIC;
    };

    virtual void SetIsCandidate(uint32_t result_id) {
        value_meta_[result_id].flags |= SPS_FLAG_IS_CANDIDATE;
    };
    virtual void ClearIsCandidate(uint32_t result_id) {
        value_meta_[result_id].flags &= ~SPS_FLAG_IS_CANDIDATE;
    };

    virtual bool PointerIsCandidate(const void* potential_ptr, uint64_t offset) const {
        if (input_data_->candidates.find(potential_ptr) != input_data_->candidates.end())
        {
            for (const auto& candidate : input_data_->candidates.at(potential_ptr))
            {
                if (candidate.offset == offset)
                {
                    return true;
                }
            }
        }
        return false;
    };

    virtual bool PointerIsCandidate(const PointerV& pointer_value) const {
        const void* potential_raw_ptr = bit_cast<const void*>(pointer_value.pointer_handle);
        if (input_data_->candidates.find(potential_raw_ptr) != input_data_->candidates.end())
        {
            uint64_t pointee_offset = GetPointerOffset(pointer_value);
            for (const auto& candidate : input_data_->candidates.at(potential_raw_ptr))
            {
                if (candidate.offset == pointee_offset)
                {
                    return true;
                }
            }
        }
        return false;
    };

    virtual void ConfirmCandidate(const void* potential_ptr, uint64_t offset) {
        if (input_data_->candidates.find(potential_ptr) != input_data_->candidates.end())
        {
            for (auto& candidate : input_data_->candidates.at(potential_ptr))
            {
                if (candidate.offset == offset)
                {
                    candidate.verified = true;
                }
            }
        }
        // TODO: Report error
    };

    virtual void ConfirmCandidate(const PointerV& pointer_value) {
        const void* potential_raw_ptr = bit_cast<const void*>(pointer_value.pointer_handle);
        if (input_data_->candidates.find(potential_raw_ptr) != input_data_->candidates.end())
        {
            uint64_t pointee_offset = GetPointerOffset(pointer_value);
            for (auto& candidate : input_data_->candidates.at(potential_raw_ptr))
            {
                if (candidate.offset == pointee_offset)
                {
                    candidate.verified = true;
                }
            }
        }
        // TODO: Report error
    };

    virtual bool PointeeValueIsCandidate(const PointerV& pointer) const {
        return pointer.pointee_flags & SPS_FLAG_IS_CANDIDATE;
    };

    virtual void SetIsDescriptorCandidate(uint32_t result_id) {
        value_meta_[result_id].flags |= SPS_FLAG_IS_DESCRIPTOR_CANDIDATE;
    };
    virtual void ClearIsDescriptorCandidate(uint32_t result_id) {
        value_meta_[result_id].flags &= ~SPS_FLAG_IS_DESCRIPTOR_CANDIDATE;
    };

    virtual bool PointerIsDescriptorBuffer(const void* potential_ptr, uint64_t offset) const {
        (void)offset;
        if (input_data_->descriptor_candidates.find(potential_ptr) != input_data_->descriptor_candidates.end())
        {
            return true;
        }
        return false;
    };

    virtual bool PointerIsDescriptorBuffer(const PointerV& pointer_value) const {
        const void* potential_raw_ptr = bit_cast<const void*>(pointer_value.pointer_handle);
        if (input_data_->descriptor_candidates.find(potential_raw_ptr) != input_data_->descriptor_candidates.end())
        {
            return true;
        }
        return false;
    };

    virtual bool PointeeValueIsDescriptorBuffer(const PointerV& pointer) const {
        return pointer.pointee_flags & SPS_FLAG_IS_DESCRIPTOR_BUFFER;
    };

    virtual void TransferFlags(uint32_t target_rid, uint32_t source_rid) {
        value_meta_[target_rid].flags |= value_meta_[source_rid].flags;
    };

    virtual void TransferFlags(uint32_t result_id, uint64_t flags) {
      value_meta_[result_id].flags |= flags;
    };

    virtual void TransferFlagsToPointee(PointerV& pointer, uint32_t result_id) {
        pointer.pointee_flags |= value_meta_[result_id].flags;
    };
    virtual void TransferFlagsToPointee(uint32_t pointer_id, uint32_t result_id) {
        std::get<PointerV>(values_[pointer_id]).pointee_flags |= value_meta_[result_id].flags;
    };
    virtual void TransferFlagsFromPointee(uint32_t result_id, const PointerV& pointer) {
        value_meta_[result_id].flags |= pointer.pointee_flags;
    };

    virtual void ExtractFlags(uint32_t result_id, uint64_t& out_meta) {
      out_meta |= value_meta_[result_id].flags;
    };

    virtual void SetFlags(uint32_t result_id, uint64_t flag) {
        value_meta_[result_id].flags |= flag;
    };
    virtual void SetFlagsPointee(PointerV& pointer, uint64_t flags) {
        pointer.pointee_flags |= flags;
    };
    virtual void SetFlagsPointee(uint32_t pointer_id, uint64_t flags) {
        std::get<PointerV>(values_[pointer_id]).pointee_flags |= flags;
    };
    virtual uint64_t GetFlags(uint32_t result_id) const {
        return value_meta_[result_id].flags;
    };

    virtual void OverrideFlagsPointee(PointerV& pointer, uint32_t result_id) {
        pointer.pointee_flags = value_meta_[result_id].flags;
    };
    virtual void OverrideFlagsPointee(uint32_t pointer_id, uint32_t result_id) {
        std::get<PointerV>(values_[pointer_id]).pointee_flags = value_meta_[result_id].flags;
    };

    virtual bool HasFlags(uint32_t result_id, uint64_t flags) {
        return value_meta_[result_id].flags & flags;
    };
    virtual bool HasFlagsPointee(const PointerV& pointer, uint64_t flags) {
        return pointer.pointee_flags & flags;
    };

    virtual Value CopyValue(const Value& value) const;

    virtual std::vector<Value>& Heap(uint32_t sc)
    {
        if (sc == spv::StorageClass::StorageClassFunction)
        {
            return function_heap_;
        }
        else
        {
            return heaps_[sc];
        }
    };

    virtual uint32_t HeapAllocate(uint32_t sc, const Value& init)
    {
        auto& heap = Heap(sc);

        // Index 0 has special meaning, keep a dummy value there
        if (heap.size() == 0)
        {
            heap.push_back(std::monostate{});
        }

        uint32_t return_index = heap.size();
        if (sc == spv::StorageClass::StorageClassFunction)
        {
            return_index = current_heap_index_;

            if (heap.size() <= current_heap_index_)
            {
                heap.push_back(init);
            }
            else
            {
                heap[return_index] = init;
            }

            current_heap_index_ += 1;
        }
        else
        {
            heap.push_back(init);
        }

        return return_index;
    }

    // Opcode handlers, 96/498 implemented for SPIRV 1.6
    void T_Void(const Instruction&);
    void T_Bool(const Instruction&);
    void T_Int(const Instruction&);
    void T_Float(const Instruction&);
    void T_Vector(const Instruction&);
    void T_Matrix(const Instruction&);
    void T_Array(const Instruction&);
    void T_Struct(const Instruction&);
    void T_Pointer(const Instruction&);
    void T_ForwardPointer(const Instruction&);
    void T_RuntimeArray(const Instruction&);
    void T_Function(const Instruction&);
    void T_Image(const Instruction&);
    void T_Sampler(const Instruction&);
    void T_SampledImage(const Instruction&);
    void T_Opaque(const Instruction&);
    void T_NamedBarrier(const Instruction&);
    void T_AccelerationStructureKHR(const Instruction&);
    void T_RayQueryKHR(const Instruction&);
    void Op_EntryPoint(const Instruction&);
    void Op_ExtInstImport(const Instruction&);
    void Op_String(const Instruction&);
    void Op_Constant(const Instruction&);
    void Op_ConstantComposite(const Instruction&);
    void Op_CompositeConstruct(const Instruction&);
    void Op_Variable(const Instruction&);
    void Op_ImageTexelPointer(const Instruction&);
    void Op_Load(const Instruction&);
    void Op_CopyObject(const Instruction&);
    void Op_Store(const Instruction&);
    void Op_AccessChain(const Instruction&);
    void Op_Function(const Instruction&);
    void Op_FunctionEnd(const Instruction&);
    void Op_FunctionCall(const Instruction&);
    void Op_Label(const Instruction&);
    void Op_Branch(const Instruction&);
    void Op_BranchConditional(const Instruction&);
    void Op_Return(const Instruction&);
    void Op_ReturnValue(const Instruction&);
    void Op_INotEqual(const Instruction&);
    void Op_FAdd(const Instruction&);
    void Op_ExtInst(const Instruction&);
    void Op_SelectionMerge(const Instruction&);
    void Op_FMul(const Instruction&);
    void Op_LoopMerge(const Instruction&);
    void Op_IAdd(const Instruction&);
    void Op_ISub(const Instruction&);
    void Op_LogicalNot(const Instruction&);
    void Op_Capability(const Instruction&);
    void Op_Extension(const Instruction&);
    void Op_MemoryModel(const Instruction&);
    void Op_MemoryBarrier(const Instruction&);
    void Op_ExecutionMode(const Instruction&);
    void Op_Source(const Instruction&);
    void Op_SourceExtension(const Instruction&);
    void Op_Name(const Instruction&);
    void Op_MemberName(const Instruction&);
    void Op_Line(const Instruction&);
    void Op_Decorate(const Instruction&);
    void Op_MemberDecorate(const Instruction&);
    void Op_SpecConstant(const Instruction&);
    void Op_SpecConstantOp(const Instruction&);
    void Op_SpecConstantComposite(const Instruction&);
    void Op_SpecConstantFalse(const Instruction&);
    void Op_SpecConstantTrue(const Instruction&);
    void Op_ArrayLength(const Instruction&);
    void Op_UGreaterThanEqual(const Instruction&);
    void Op_Phi(const Instruction&);
    void Op_ConvertUToF(const Instruction&);
    void Op_ConvertSToF(const Instruction&);
    void Op_FDiv(const Instruction&);
    void Op_FSub(const Instruction&);
    void Op_Fwidth(const Instruction&);
    void Op_VectorTimesScalar(const Instruction&);
    void Op_SLessThan(const Instruction&);
    void Op_Dot(const Instruction&);
    void Op_FOrdGreaterThan(const Instruction&);
    void Op_FOrdGreaterThanEqual(const Instruction&);
    void Op_FOrdEqual(const Instruction&);
    void Op_FOrdNotEqual(const Instruction&);
    void Op_FUnordNotEqual(const Instruction&);
    void Op_CompositeExtract(const Instruction&);
    void Op_Bitcast(const Instruction&);
    void Op_IMul(const Instruction&);
    void Op_ConvertUToPtr(const Instruction&);
    void Op_UDiv(const Instruction&);
    void Op_UMod(const Instruction&);
    void Op_SMod(const Instruction&);
    void Op_SRem(const Instruction&);
    void Op_ULessThan(const Instruction&);
    void Op_ConstantTrue(const Instruction&);
    void Op_ConstantFalse(const Instruction&);
    void Op_ConstantNull(const Instruction&);
    void Op_AtomicIAdd(const Instruction&);
    void Op_AtomicISub(const Instruction&);
    void Op_AtomicExchange(const Instruction&);
    void Op_Select(const Instruction&);
    void Op_IEqual(const Instruction&);
    void Op_VectorShuffle(const Instruction&);
    void Op_CompositeInsert(const Instruction&);
    void Op_Transpose(const Instruction&);
    void Op_SampledImage(const Instruction&);
    void Op_ImageSampleImplicitLod(const Instruction&);
    void Op_ImageSampleDrefImplicitLod(const Instruction&);
    void Op_ImageSampleExplicitLod(const Instruction&);
    void Op_ImageFetch(const Instruction&);
    void Op_ImageGather(const Instruction&);
    void Op_ImageRead(const Instruction&);
    void Op_ImageWrite(const Instruction&);
    void Op_ImageQuerySize(const Instruction&);
    void Op_ImageQuerySizeLod(const Instruction&);
    void Op_FNegate(const Instruction&);
    void Op_MatrixTimesScalar(const Instruction&);
    void Op_MatrixTimesVector(const Instruction&);
    void Op_UGreaterThan(const Instruction&);
    void Op_FOrdLessThan(const Instruction&);
    void Op_FOrdLessThanEqual(const Instruction&);
    void Op_ShiftRightLogical(const Instruction&);
    void Op_ShiftLeftLogical(const Instruction&);
    void Op_BitwiseOr(const Instruction&);
    void Op_BitwiseAnd(const Instruction&);
    void Op_Not(const Instruction&);
    void Op_Switch(const Instruction&);
    void Op_All(const Instruction&);
    void Op_Any(const Instruction&);
    void Op_BitCount(const Instruction&);
    void Op_Kill(const Instruction&);
    void Op_Unreachable(const Instruction&);
    void Op_Undef(const Instruction&);
    void Op_VectorTimesMatrix(const Instruction&);
    void Op_ULessThanEqual(const Instruction&);
    void Op_SLessThanEqual(const Instruction&);
    void Op_SGreaterThanEqual(const Instruction&);
    void Op_SGreaterThan(const Instruction&);
    void Op_SDiv(const Instruction&);
    void Op_SNegate(const Instruction&);
    void Op_LogicalEqual(const Instruction& instruction);
    void Op_LogicalNotEqual(const Instruction& instruction);
    void Op_LogicalOr(const Instruction&);
    void Op_LogicalAnd(const Instruction&);
    void Op_MatrixTimesMatrix(const Instruction&);
    void Op_IsNan(const Instruction&);
    void Op_FunctionParameter(const Instruction&);
    void Op_EmitVertex(const Instruction&);
    void Op_EndPrimitive(const Instruction&);
    void Op_FConvert(const Instruction&);
    void Op_Image(const Instruction&);
    void Op_ConvertFToS(const Instruction&);
    void Op_ConvertFToU(const Instruction&);
    void Op_FRem(const Instruction&);
    void Op_FMod(const Instruction&);
    void Op_AtomicOr(const Instruction&);
    void Op_AtomicUMax(const Instruction&);
    void Op_AtomicUMin(const Instruction&);
    void Op_BitReverse(const Instruction&);
    void Op_BitwiseXor(const Instruction&);
    void Op_ControlBarrier(const Instruction&);
    void Op_ShiftRightArithmetic(const Instruction&);
    void Op_GroupNonUniformAll(const Instruction&);
    void Op_GroupNonUniformAny(const Instruction&);
    void Op_GroupNonUniformBallot(const Instruction&);
    void Op_GroupNonUniformBallotBitCount(const Instruction&);
    void Op_GroupNonUniformBroadcastFirst(const Instruction&);
    void Op_GroupNonUniformElect(const Instruction&);
    void Op_GroupNonUniformFMax(const Instruction&);
    void Op_GroupNonUniformFMin(const Instruction&);
    void Op_GroupNonUniformIAdd(const Instruction&);
    void Op_GroupNonUniformShuffle(const Instruction&);
    void Op_GroupNonUniformShuffleXor(const Instruction&);
    void Op_GroupNonUniformUMax(const Instruction&);
    void Op_GroupNonUniformBitwiseAnd(const Instruction&);
    void Op_RayQueryGetIntersectionBarycentricsKHR(const Instruction&);
    void Op_RayQueryGetIntersectionFrontFaceKHR(const Instruction&);
    void Op_RayQueryGetIntersectionGeometryIndexKHR(const Instruction&);
    void Op_RayQueryGetIntersectionInstanceCustomIndexKHR(const Instruction&);
    void Op_RayQueryGetIntersectionInstanceIdKHR(const Instruction&);
    void Op_RayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR(const Instruction&);
    void Op_RayQueryGetIntersectionPrimitiveIndexKHR(const Instruction&);
    void Op_RayQueryGetIntersectionTKHR(const Instruction&);
    void Op_RayQueryGetIntersectionTypeKHR(const Instruction&);
    void Op_RayQueryGetIntersectionWorldToObjectKHR(const Instruction&);
    void Op_RayQueryGetWorldRayDirectionKHR(const Instruction&);
    void Op_RayQueryInitializeKHR(const Instruction&);
    void Op_RayQueryProceedKHR(const Instruction&);
    void Op_DecorateString(const Instruction&);
};

} // namespace SPIRVSimulator

#endif
