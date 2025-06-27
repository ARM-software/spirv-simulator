#pragma once

#ifndef ARM_SPIRV_SIMULATOR_HPP
#define ARM_SPIRV_SIMULATOR_HPP

#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

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

namespace SPIRVSimulator {

// ---------------------------------------------------------------------------
//  Input structure
//
//  This structure defines the shader inputs.
//  This must be populated and passed to the run(...) method to
//  populate the shader input values before and during execution.
//
//  The format of the data must be what the shader expects, eg. if a buffer is bound
//  to a binding with the std430 layout, the data in the byte vectors must obey the rules of
//  that layout

struct InputData{
    // The SpirV ID of the entry point to use
    uint32_t entry_point_id = 0;
    // The OpName label (function name) of the entry point to use, takes priority over entry_point_id if it is set.
    std::string entry_point_op_name = "";

    // Data block pointer -> (byte_offset_to_array, array length)
    std::unordered_map<uint64_t, std::pair<size_t, size_t>> rt_array_lengths;

    // SpecId -> byte offset
    std::unordered_map<uint32_t, size_t> specialization_constant_offsets;
    void* specialization_constants = nullptr;

    // The full binary push_constant block
    void* push_constants = nullptr;

    // DescriptorSet -> Binding -> data
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, void*>> bindings;

    // These can be provided by the user in order to properly initialize PhysicalStorageBuffer storage class values.
    // The keys here are uint64_t values who contain the bits in the physical address pointers
    // The value pair is the size of the buffer (in bytes) followed by the pointer to the host side data
    std::unordered_map<uint64_t, std::pair<size_t, void*>> physical_address_buffers;
};

// ---------------------------------------------------------------------------
// Output structure

enum BitLocation{
    Constant,  // Constant embedded in the shader, offsets will be relative to the start of the spirv binary code
    SpecConstant,  // Spec constant, binding will be SpecId
    StorageClass  // storage_class specifies what block type we are dealing with
};

struct DataSourceBits{
    /*
    Structure describing where a sequence of bits can be found that ended up being used to construct
    (or that was eventually interpreted as) a pointer pointing to data with the PhysicalStorageBuffer storage class
    */

    // Specifies where the data comes from
    BitLocation location;

    // If location is StorageClass, this holds the storage class
    spv::StorageClass storage_class;

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
struct PhysicalAddressData{
    std::vector<DataSourceBits> bit_components;
    uint64_t raw_pointer_value;
};

// ---------------------------------------------------------------------------

struct Instruction{
    spv::Op opcode;

    // word_count is the total number of words this instruction is composed of,
    // including the word holding the opcode + wordcount value.
    // This (along with the opcode above) is a redundant value as the first uint32 in words will also hold it,
    // but it is included/decoded here for ease-of-use and clarity
    uint16_t word_count;
    std::span<const uint32_t> words;
};

struct Type{
    enum class Kind{
        Void,
        BoolT,  // Because x11 headers have a macro called Bool
        Int,
        Float,
        Vector,
        Matrix,
        Array,
        Struct,
        Pointer,
        RuntimeArray,  // TODO: We can/probably should make these maps and use sparse access (eg. add a new map value for these and load during OpAccessChain)
        Image,
        Sampler,
        SampledImage,
        Opaque,
        NamedBarrier
    } kind;

    struct ScalarTypeData{uint32_t width; bool is_signed;};
    struct VectorTypeData{uint32_t elem_type_id; uint32_t elem_count;};
    struct MatrixTypeData{uint32_t col_type_id;  uint32_t col_count;};
    struct ArrayTypeData{uint32_t elem_type_id; uint32_t length_id;};
    struct PointerTypeData{uint32_t storage_class; uint32_t pointee_type_id;};
    struct ImageTypeData{uint32_t sampled_type_id;};
    struct SampledImageTypeData{uint32_t image_type_id;};
    struct OpaqueTypeData{uint32_t name;};

    union{
        ScalarTypeData scalar;
        VectorTypeData vector;
        MatrixTypeData matrix;
        ArrayTypeData array;
        PointerTypeData pointer;
        ImageTypeData image;
        SampledImageTypeData sampled_image;
        OpaqueTypeData opaque;
    };
    Type(): kind(Kind::Void){scalar = {0, false};}
};

struct AggregateV;
struct PointerV;
struct VectorV;
struct MatrixV;

using Value = std::variant<
    std::monostate,
    uint64_t,
    int64_t,
    double,
    std::shared_ptr<VectorV>,
    std::shared_ptr<MatrixV>,
    std::shared_ptr<AggregateV>,
    PointerV>;

struct VectorV{
    std::vector<Value> elems;
};

struct MatrixV{
    std::vector<Value> cols;
};

struct AggregateV{
    std::vector<Value> elems;
};  // array or struct

struct PointerV {
    // Always the index of the value that this pointer points to
    uint32_t obj_id;
    // The TypeID of this pointer
    uint32_t type_id;
    uint32_t storage_class;

    // Optional value, holds the raw pointer value when applicable
    uint64_t raw_pointer;

    // If it points to a value inside a composite, aggregate or array value. This is the indirection path within said value.
    std::vector<uint32_t> idx_path;
    // This is the result_id chain of the objects holding the idx path values
    std::vector<uint32_t> idx_path_ids;
};

struct DecorationInfo{
    spv::Decoration kind;
    std::vector<uint32_t> literals;
};

void DecodeInstruction(std::span<const uint32_t>& program_words, Instruction& instruction);

template<class T>
void extract_bytes(std::vector<std::byte>& output, T input, size_t num_bits){
    if (sizeof(input) != 8){
        throw std::runtime_error("SPIRV simulator: extract_bytes called on type that is not 8 bytes");
    }

    std::array<std::byte, sizeof(T)> arr;

    std::memcpy(arr.data(), &input, sizeof(T));
    if (num_bits > 32){
        output.insert(output.end(), arr.begin(), arr.end());
    } else {
        output.insert(output.end(), arr.begin(), arr.begin() + 4);
    }
}

// Bitcast an be very annoying to import on certain platforms, even if c++20 is supported
// Just do this for now, and we can replace this with the std::bit_cast version in the future
template<class To, class From>
typename std::enable_if_t<sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> && std::is_trivially_copyable_v<To>, To> bit_cast(const From& src) noexcept {
  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}

class SPIRVSimulator{
public:
    explicit SPIRVSimulator(const std::vector<uint32_t>& program_words, const InputData& input_data, bool verbose=false);
    void Run();

    // When called (after a Run()) will write the outputs to the input_data mapped pointers
    void WriteOutputs();

    const std::vector<PhysicalAddressData>& GetPhysicalAddressData() const {return physical_address_pointer_source_data_;}

    const std::unordered_map<uint32_t, Type>& GetTypes() const {return types_;}

    std::set<std::string> unsupported_opcodes;

private:
    // Used to create object id's for entries not created by a spirv instruction
    uint32_t next_external_id_ = 0;

    // Parsing artefacts
    InputData input_data_;
    // Contains entry point ID -> entry point OpName labels (labels may be non-existent/empty)
    std::unordered_map<uint32_t, std::string> entry_points_;
    std::vector<uint32_t> program_words_;
    std::span<const uint32_t> stream_;
    std::vector<Instruction> instructions_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> spec_instr_words_;
    std::unordered_map<uint32_t, Instruction> spec_instructions_;
    std::unordered_map<uint32_t, size_t> result_id_to_inst_index_;
    std::unordered_map<uint32_t, Type> types_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> struct_members_;
    std::unordered_map<uint32_t, uint32_t> forward_type_declarations_;  // Unused, consider removing this
    std::unordered_map<uint32_t, std::vector<DecorationInfo>> decorators_;
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::vector<DecorationInfo>>> struct_decorators_;
    std::unordered_map<uint32_t, std::string> extended_imports_;
    // Any result ID in this set, can be treated as if it has any valid value for the given type
    std::set<uint32_t> arbitrary_values_;
    // This maps the result ID of pointers to the result ID of values stored through them
    std::unordered_map<uint32_t, uint32_t> values_stored_;

    // Debug only
    bool verbose_;

    // These hold information about any pointers that reference physical storage buffers
    std::vector<PointerV> physical_address_pointers_;
    std::vector<std::pair<PointerV, PointerV>> pointers_to_physical_address_pointers_;
    std::vector<PhysicalAddressData> physical_address_pointer_source_data_;
    std::unordered_map<uint32_t, DataSourceBits> data_source_bits_;

    // Control flow
    struct FunctionInfo{
        size_t inst_index;
        size_t first_inst_index;
        std::vector<uint32_t> parameter_ids_;
        std::vector<uint32_t> parameter_type_ids_;
    };
    uint32_t prev_defined_func_id_;
    std::unordered_map<uint32_t, FunctionInfo> funcs_;

    uint32_t prev_block_id_ = 0;
    uint32_t current_block_id_ = 0;

    // Heaps & frames
    struct Frame{
        size_t pc;
        uint32_t result_id;

        // result_id -> Value
        std::unordered_map<uint32_t, Value> locals;

        // obj_id -> Heap Value
        std::unordered_map<uint32_t, Value> func_heap;
    };
    std::vector<Frame> call_stack_;

    // result_id -> Value
    std::unordered_map<uint32_t, Value> globals_;

    // storage_class -> obj_id -> Heap Value
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, Value>> heaps_;

    // Dispatcher
    using DispatcherType = std::function<void(const Instruction&)>;
    std::unordered_map<spv::Op, DispatcherType> opcode_dispatchers_;

    // Handlers used by the OpExtInst handler
    // Implementation of the operations in the GLSL extended set
    void GLSLExtHandler(
        uint32_t type_id,
        uint32_t result_id,
        uint32_t instruction_literal,
        const std::span<const uint32_t>& operand_words);

    // Helpers
    // TODO: Many more of these can be const, fix
    void DecodeHeader();
    void ParseAll();
    void RegisterOpcodeHandlers();
    void CheckOpcodeSupport();
    void Validate();
    void ExecuteInstruction(const Instruction&);
    std::string GetValueString(const Value&);
    std::string GetTypeString(const Type&);
    void PrintInstruction(const Instruction&);
    void HandleUnimplementedOpcode(const Instruction&);
    Value MakeScalar(uint32_t type_id, const uint32_t*& words);
    Value MakeDefault(uint32_t type_id, const uint32_t** initial_data=nullptr);
    Value& Deref(const PointerV &ptr);
    Value& GetValue(uint32_t result_id);
    void SetValue(uint32_t result_id, const Value& value);
    Type GetType(uint32_t result_id) const;
    uint32_t GetTypeID(uint32_t result_id) const;
    void ExtractWords(const std::byte* external_pointer, uint32_t type_id, std::vector<uint32_t>& buffer_data);
    uint64_t GetPointerOffset(const PointerV& pointer_value);
    size_t GetBitizeOfType(uint32_t type_id);
    size_t GetBitizeOfTargetType(const PointerV& pointer);
    void GetBaseTypeIDs(uint32_t type_id, std::vector<uint32_t>& output);
    std::vector<DataSourceBits> FindDataSourcesFromResultID(uint32_t result_id);
    bool HasDecorator(uint32_t result_id, spv::Decoration decorator);
    bool HasDecorator(uint32_t result_id, uint32_t member_id, spv::Decoration decorator);
    uint32_t GetDecoratorLiteral(uint32_t result_id, spv::Decoration decorator, size_t literal_offset=0);
    uint32_t GetDecoratorLiteral(uint32_t result_id, uint32_t member_id, spv::Decoration decorator, size_t literal_offset=0);
    uint32_t GetNextExternalID(){uint32_t new_id = next_external_id_; next_external_id_ += 1; return new_id;}
    bool ValueIsArbitrary(uint32_t result_id) const {return arbitrary_values_.contains(result_id);};
    Value CopyValue(const Value& value) const;
    std::unordered_map<uint32_t,Value>& Heap(uint32_t sc){ return heaps_[sc]; }

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
    void Op_EntryPoint(const Instruction&);
    void Op_ExtInstImport(const Instruction&);
    void Op_Constant(const Instruction&);
    void Op_ConstantComposite(const Instruction&);
    void Op_CompositeConstruct(const Instruction&);
    void Op_Variable(const Instruction&);
    void Op_Load(const Instruction&);
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
    void Op_ExecutionMode(const Instruction&);
    void Op_Source(const Instruction&);
    void Op_SourceExtension(const Instruction&);
    void Op_Name(const Instruction&);
    void Op_MemberName(const Instruction&);
    void Op_Decorate(const Instruction&);
    void Op_MemberDecorate(const Instruction&);
    void Op_SpecConstant(const Instruction&);
    void Op_SpecConstantOp(const Instruction&);
    void Op_SpecConstantComposite(const Instruction&);
    void Op_ArrayLength(const Instruction&);
    void Op_UGreaterThanEqual(const Instruction&);
    void Op_Phi(const Instruction&);
    void Op_ConvertUToF(const Instruction&);
    void Op_ConvertSToF(const Instruction&);
    void Op_FDiv(const Instruction&);
    void Op_FSub(const Instruction&);
    void Op_VectorTimesScalar(const Instruction&);
    void Op_SLessThan(const Instruction&);
    void Op_Dot(const Instruction&);
    void Op_FOrdGreaterThan(const Instruction&);
    void Op_CompositeExtract(const Instruction&);
    void Op_Bitcast(const Instruction&);
    void Op_IMul(const Instruction&);
    void Op_ConvertUToPtr(const Instruction&);
    void Op_UDiv(const Instruction&);
    void Op_UMod(const Instruction&);
    void Op_ULessThan(const Instruction&);
    void Op_ConstantTrue(const Instruction&);
    void Op_ConstantFalse(const Instruction&);
    void Op_ConstantNull(const Instruction&);
    void Op_AtomicIAdd(const Instruction&);
    void Op_AtomicISub(const Instruction&);
    void Op_Select(const Instruction&);
    void Op_IEqual(const Instruction&);
    void Op_ImageTexelPointer(const Instruction&);
    void Op_VectorShuffle(const Instruction&);
    void Op_CompositeInsert(const Instruction&);
    void Op_Transpose(const Instruction&);
    void Op_ImageFetch(const Instruction&);
    void Op_FNegate(const Instruction&);
    void Op_MatrixTimesVector(const Instruction&);
    void Op_UGreaterThan(const Instruction&);
    void Op_FOrdLessThan(const Instruction&);
    void Op_FOrdLessThanEqual(const Instruction&);
    void Op_ShiftRightLogical(const Instruction&);
    void Op_ShiftLeftLogical(const Instruction&);
    void Op_BitwiseOr(const Instruction&);
    void Op_BitwiseAnd(const Instruction&);
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
    void Op_LogicalOr(const Instruction&);
    void Op_LogicalAnd(const Instruction&);
    void Op_MatrixTimesMatrix(const Instruction&);
    void Op_IsNan(const Instruction&);
};

}

#endif
