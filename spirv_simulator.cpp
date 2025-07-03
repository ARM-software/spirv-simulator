#include "spirv_simulator.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <optional>
#include <string_view>
#include <unordered_set>

namespace SPIRVSimulator{

#define assertx(msg) assert((void(msg), false))
#define assertm(exp, msg) assert((void(msg), exp))

constexpr uint32_t kWordCountShift = 16u;
constexpr uint32_t kOpcodeMask = 0xFFFFu;
const std::string execIndent = "                  # ";


void DecodeInstruction(std::span<const uint32_t>& program_words, Instruction& instruction){
    /*
    Decodes an instruction from the given span stream.

    Will update the input stream so it points to the start of the next opcode.
    The results are written to the input instruction.
    */
    uint32_t first = program_words.front();
    instruction.word_count = first >> kWordCountShift;
    instruction.opcode = (spv::Op)(first & kOpcodeMask);

    assertm (instruction.word_count && instruction.word_count <= program_words.size(), "SPIRV simulator: Bad instruction size");

    instruction.words = program_words.first(instruction.word_count);
    program_words = program_words.subspan(instruction.word_count);
}

SPIRVSimulator::SPIRVSimulator(const std::vector<uint32_t>& program_words, const InputData& input_data, bool verbose): program_words_(std::move(program_words)), verbose_(verbose){
    stream_ = program_words_;
    input_data_ = input_data;
    DecodeHeader();
    RegisterOpcodeHandlers();
    CheckOpcodeSupport();
}

void SPIRVSimulator::DecodeHeader(){
    assertm (program_words_.size() >= 5, "SPIRV simulator: SPIRV binary is less than 5 words long, it must at least contain a full valid header.");

    uint32_t magic_number = program_words_[0];
    assertm (magic_number == 0x07230203, "SPIRV simulator: Magic SPIRV header number is invalid, should be: 0x07230203");

    if (verbose_){
        uint32_t version = program_words_[1];
        uint32_t generator = program_words_[2];
        uint32_t bound = program_words_[3];
        uint32_t schema = program_words_[4];

        std::cout << "SPIRV simulator: Shader header parsed as:" << std::endl;
        std::cout << execIndent << "Version: " << version << std::endl;
        std::cout << execIndent << "Generator: " << generator << std::endl;
        std::cout << execIndent << "Bound: " << bound << std::endl;
        std::cout << execIndent << "Schema: " << schema << std::endl << std::endl;
    }

    stream_ = std::span<const uint32_t>(program_words_).subspan(5);
}

void SPIRVSimulator::RegisterOpcodeHandlers(){
    /*
    New opcode implementations must be registered here.
    */
    auto R = [this](spv::Op op, DispatcherType f){
        opcode_dispatchers_[op] = std::move(f);
    };

    R(spv::Op::OpTypeVoid,               [this](const Instruction& i){T_Void(i);});
    R(spv::Op::OpTypeBool,               [this](const Instruction& i){T_Bool(i);});
    R(spv::Op::OpTypeInt,                [this](const Instruction& i){T_Int(i);});
    R(spv::Op::OpTypeFloat,              [this](const Instruction& i){T_Float(i);});
    R(spv::Op::OpTypeVector,             [this](const Instruction& i){T_Vector(i);});
    R(spv::Op::OpTypeMatrix,             [this](const Instruction& i){T_Matrix(i);});
    R(spv::Op::OpTypeArray,              [this](const Instruction& i){T_Array(i);});
    R(spv::Op::OpTypeStruct,             [this](const Instruction& i){T_Struct(i);});
    R(spv::Op::OpTypePointer,            [this](const Instruction& i){T_Pointer(i);});
    R(spv::Op::OpTypeForwardPointer,     [this](const Instruction& i){T_ForwardPointer(i);});
    R(spv::Op::OpTypeRuntimeArray,       [this](const Instruction& i){T_RuntimeArray(i);});
    R(spv::Op::OpTypeFunction,           [this](const Instruction& i){T_Function(i);});
    R(spv::Op::OpTypeImage,              [this](const Instruction& i){T_Image(i);});
    R(spv::Op::OpTypeSampler,            [this](const Instruction& i){T_Sampler(i);});
    R(spv::Op::OpTypeSampledImage,       [this](const Instruction& i){T_SampledImage(i);});
    R(spv::Op::OpTypeOpaque,             [this](const Instruction& i){T_Opaque(i);});
    R(spv::Op::OpTypeNamedBarrier,       [this](const Instruction& i){T_NamedBarrier(i);});
    R(spv::Op::OpEntryPoint,             [this](const Instruction& i){Op_EntryPoint(i);});
    R(spv::Op::OpExtInstImport,          [this](const Instruction& i){Op_ExtInstImport(i);});
    R(spv::Op::OpConstant,               [this](const Instruction& i){Op_Constant(i);});
    R(spv::Op::OpConstantComposite,      [this](const Instruction& i){Op_ConstantComposite(i);});
    R(spv::Op::OpCompositeConstruct,     [this](const Instruction& i){Op_CompositeConstruct(i);});
    R(spv::Op::OpVariable,               [this](const Instruction& i){Op_Variable(i);});
    R(spv::Op::OpLoad,                   [this](const Instruction& i){Op_Load(i);});
    R(spv::Op::OpStore,                  [this](const Instruction& i){Op_Store(i);});
    R(spv::Op::OpAccessChain,            [this](const Instruction& i){Op_AccessChain(i);});
    R(spv::Op::OpInBoundsAccessChain,    [this](const Instruction& i){Op_AccessChain(i);});
    R(spv::Op::OpFunction,               [this](const Instruction& i){Op_Function(i);});
    R(spv::Op::OpFunctionEnd,            [this](const Instruction& i){Op_FunctionEnd(i);});
    R(spv::Op::OpFunctionCall,           [this](const Instruction& i){Op_FunctionCall(i);});
    R(spv::Op::OpLabel,                  [this](const Instruction& i){Op_Label(i);});
    R(spv::Op::OpBranch,                 [this](const Instruction& i){Op_Branch(i);});
    R(spv::Op::OpBranchConditional,      [this](const Instruction& i){Op_BranchConditional(i);});
    R(spv::Op::OpReturn,                 [this](const Instruction& i){Op_Return(i);});
    R(spv::Op::OpReturnValue,            [this](const Instruction& i){Op_ReturnValue(i);});
    R(spv::Op::OpINotEqual,              [this](const Instruction& i){Op_INotEqual(i);});
    R(spv::Op::OpFAdd,                   [this](const Instruction& i){Op_FAdd(i);});
    R(spv::Op::OpExtInst,                [this](const Instruction& i){Op_ExtInst(i);});
    R(spv::Op::OpSelectionMerge,         [this](const Instruction& i){Op_SelectionMerge(i);});
    R(spv::Op::OpFMul,                   [this](const Instruction& i){Op_FMul(i);});
    R(spv::Op::OpLoopMerge,              [this](const Instruction& i){Op_LoopMerge(i);});
    R(spv::Op::OpIAdd,                   [this](const Instruction& i){Op_IAdd(i);});
    R(spv::Op::OpISub,                   [this](const Instruction& i){Op_ISub(i);});
    R(spv::Op::OpLogicalNot,             [this](const Instruction& i){Op_LogicalNot(i);});
    R(spv::Op::OpCapability,             [this](const Instruction& i){Op_Capability(i);});
    R(spv::Op::OpExtension,              [this](const Instruction& i){Op_Extension(i);});
    R(spv::Op::OpMemoryModel,            [this](const Instruction& i){Op_MemoryModel(i);});
    R(spv::Op::OpExecutionMode,          [this](const Instruction& i){Op_ExecutionMode(i);});
    R(spv::Op::OpSource,                 [this](const Instruction& i){Op_Source(i);});
    R(spv::Op::OpSourceExtension,        [this](const Instruction& i){Op_SourceExtension(i);});
    R(spv::Op::OpName,                   [this](const Instruction& i){Op_Name(i);});
    R(spv::Op::OpMemberName,             [this](const Instruction& i){Op_MemberName(i);});
    R(spv::Op::OpDecorate,               [this](const Instruction& i){Op_Decorate(i);});
    R(spv::Op::OpMemberDecorate,         [this](const Instruction& i){Op_MemberDecorate(i);});
    R(spv::Op::OpArrayLength,            [this](const Instruction& i){Op_ArrayLength(i);});
    R(spv::Op::OpSpecConstant,           [this](const Instruction& i){Op_SpecConstant(i);});
    R(spv::Op::OpSpecConstantOp,         [this](const Instruction& i){Op_SpecConstantOp(i);});
    R(spv::Op::OpSpecConstantComposite,  [this](const Instruction& i){Op_SpecConstantComposite(i);});
    R(spv::Op::OpUGreaterThanEqual,      [this](const Instruction& i){Op_UGreaterThanEqual(i);});
    R(spv::Op::OpPhi,                    [this](const Instruction& i){Op_Phi(i);});
    R(spv::Op::OpConvertUToF,            [this](const Instruction& i){Op_ConvertUToF(i);});
    R(spv::Op::OpConvertSToF,            [this](const Instruction& i){Op_ConvertSToF(i);});
    R(spv::Op::OpFDiv,                   [this](const Instruction& i){Op_FDiv(i);});
    R(spv::Op::OpFSub,                   [this](const Instruction& i){Op_FSub(i);});
    R(spv::Op::OpVectorTimesScalar,      [this](const Instruction& i){Op_VectorTimesScalar(i);});
    R(spv::Op::OpSLessThan,              [this](const Instruction& i){Op_SLessThan(i);});
    R(spv::Op::OpDot,                    [this](const Instruction& i){Op_Dot(i);});
    R(spv::Op::OpFOrdGreaterThan,        [this](const Instruction& i){Op_FOrdGreaterThan(i);});
    R(spv::Op::OpCompositeExtract,       [this](const Instruction& i){Op_CompositeExtract(i);});
    R(spv::Op::OpBitcast,                [this](const Instruction& i){Op_Bitcast(i);});
    R(spv::Op::OpIMul,                   [this](const Instruction& i){Op_IMul(i);});
    R(spv::Op::OpConvertUToPtr,          [this](const Instruction& i){Op_ConvertUToPtr(i);});
    R(spv::Op::OpUDiv,                   [this](const Instruction& i){Op_UDiv(i);});
    R(spv::Op::OpUMod,                   [this](const Instruction& i){Op_UMod(i);});
    R(spv::Op::OpULessThan,              [this](const Instruction& i){Op_ULessThan(i);});
    R(spv::Op::OpConstantTrue,           [this](const Instruction& i){Op_ConstantTrue(i);});
    R(spv::Op::OpConstantFalse,          [this](const Instruction& i){Op_ConstantFalse(i);});
    R(spv::Op::OpConstantNull,           [this](const Instruction& i){Op_ConstantNull(i);});
    R(spv::Op::OpAtomicIAdd,             [this](const Instruction& i){Op_AtomicIAdd(i);});
    R(spv::Op::OpAtomicISub,             [this](const Instruction& i){Op_AtomicISub(i);});
    R(spv::Op::OpSelect,                 [this](const Instruction& i){Op_Select(i);});
    R(spv::Op::OpIEqual,                 [this](const Instruction& i){Op_IEqual(i);});
    R(spv::Op::OpImageTexelPointer,      [this](const Instruction& i){Op_ImageTexelPointer(i);});
    R(spv::Op::OpVectorShuffle,          [this](const Instruction& i){Op_VectorShuffle(i);});
    R(spv::Op::OpCompositeInsert,        [this](const Instruction& i){Op_CompositeInsert(i);});
    R(spv::Op::OpTranspose,              [this](const Instruction& i){Op_Transpose(i);});
    R(spv::Op::OpImageFetch,             [this](const Instruction& i){Op_ImageFetch(i);});
    R(spv::Op::OpFNegate,                [this](const Instruction& i){Op_FNegate(i);});
    R(spv::Op::OpMatrixTimesVector,      [this](const Instruction& i){Op_MatrixTimesVector(i);});
    R(spv::Op::OpUGreaterThan,           [this](const Instruction& i){Op_UGreaterThan(i);});
    R(spv::Op::OpFOrdLessThan,           [this](const Instruction& i){Op_FOrdLessThan(i);});
    R(spv::Op::OpFOrdLessThanEqual,      [this](const Instruction& i){Op_FOrdLessThanEqual(i);});
    R(spv::Op::OpShiftRightLogical,      [this](const Instruction& i){Op_ShiftRightLogical(i);});
    R(spv::Op::OpShiftLeftLogical,       [this](const Instruction& i){Op_ShiftLeftLogical(i);});
    R(spv::Op::OpBitwiseOr,              [this](const Instruction& i){Op_BitwiseOr(i);});
    R(spv::Op::OpBitwiseAnd,             [this](const Instruction& i){Op_BitwiseAnd(i);});
    R(spv::Op::OpSwitch,                 [this](const Instruction& i){Op_Switch(i);});
    R(spv::Op::OpAll,                    [this](const Instruction& i){Op_All(i);});
    R(spv::Op::OpAny,                    [this](const Instruction& i){Op_Any(i);});
    R(spv::Op::OpBitCount,               [this](const Instruction& i){Op_BitCount(i);});
    R(spv::Op::OpKill,                   [this](const Instruction& i){Op_Kill(i);});
    R(spv::Op::OpUnreachable,            [this](const Instruction& i){Op_Unreachable(i);});
    R(spv::Op::OpUndef,                  [this](const Instruction& i){Op_Undef(i);});
    R(spv::Op::OpVectorTimesMatrix,      [this](const Instruction& i){Op_VectorTimesMatrix(i);});
    R(spv::Op::OpULessThanEqual,         [this](const Instruction& i){Op_ULessThanEqual(i);});
    R(spv::Op::OpSLessThanEqual,         [this](const Instruction& i){Op_SLessThanEqual(i);});
    R(spv::Op::OpSGreaterThanEqual,      [this](const Instruction& i){Op_SGreaterThanEqual(i);});
    R(spv::Op::OpSGreaterThan,           [this](const Instruction& i){Op_SGreaterThan(i);});
    R(spv::Op::OpSDiv,                   [this](const Instruction& i){Op_SDiv(i);});
    R(spv::Op::OpSNegate,                [this](const Instruction& i){Op_SNegate(i);});
    R(spv::Op::OpLogicalOr,              [this](const Instruction& i){Op_LogicalOr(i);});
    R(spv::Op::OpLogicalAnd,             [this](const Instruction& i){Op_LogicalAnd(i);});
    R(spv::Op::OpMatrixTimesMatrix,      [this](const Instruction& i){Op_MatrixTimesMatrix(i);});
    R(spv::Op::OpIsNan,                  [this](const Instruction& i){Op_IsNan(i);});
    R(spv::Op::OpConvertFToS,            [this](const Instruction& i){OpConvertFToS(i);});
    R(spv::Op::OpConvertFToU,            [this](const Instruction& i){OpConvertFToU(i);});
    R(spv::Op::OpFOrdEqual,              [this](const Instruction& i){OpFOrdEqual(i);});
    R(spv::Op::OpFOrdGreaterThanEqual,   [this](const Instruction& i){OpFOrdGreaterThanEqual(i);});
    R(spv::Op::OpFOrdNotEqual,           [this](const Instruction& i){OpFOrdNotEqual(i);});
    R(spv::Op::OpFRem,                   [this](const Instruction& i){OpFRem(i);});
}

void SPIRVSimulator::CheckOpcodeSupport(){
    /*
    Verifies that all opcodes in the instructions in the input shaders have a registered handled in RegisterOpcodeHandlers.
    */
    // Check that program_words_ has not been messed with
    uint32_t magic_number = program_words_[0];
    assertm (magic_number == 0x07230203, "SPIRV simulator: Magic SPIRV header number wrong, should be: 0x07230203");

    size_t current_word = 5;

    std::set<spv::Op> unimplemented_opcodes;
    while (current_word < program_words_.size()){
        uint32_t header_word = program_words_[current_word];
        uint32_t word_count = header_word >> kWordCountShift;
        spv::Op opcode = (spv::Op)(header_word & kOpcodeMask);

        assertm (word_count > 0, "SPIRV simulator: Word count was 0 (or less) for instruction. Input SPIRV is broken.");

        bool is_implemented = opcode_dispatchers_.find(opcode) != opcode_dispatchers_.end();
        if (!is_implemented){
            unimplemented_opcodes.insert(opcode);
        }

        if (opcode == spv::Op::OpExtInst){
            uint32_t set_id = program_words_[current_word + 3];
            uint32_t instruction_literal = program_words_[current_word + 4];

            if (verbose_){
                std::cout << execIndent << "Found OpExtInst instruction with set ID: " << set_id << ", instruction literal: " << instruction_literal << std::endl;
            }
        }

        current_word += word_count;
    }

    if (!unimplemented_opcodes.empty()){
        if (verbose_){
            std::cout << "SPIRV simulator: Unimplemented OpCodes detected:" << std::endl;
        }
        for (auto it = unimplemented_opcodes.begin(); it != unimplemented_opcodes.end(); ++it){
            if (verbose_){
                std::cout << execIndent << spv::OpToString(*it) << std::endl;
            }
            unsupported_opcodes.insert(spv::OpToString(*it));
        }
    }

    if (verbose_){
        std::cout << std::endl;
    }
}

void SPIRVSimulator::Validate(){
    /*
    Do some early sanity checking and validation.
    */
    // TODO: Expand this (a lot)
    for(auto &[id, t] : types_){

        assertm (!(t.kind == Type::Kind::Array && !types_.contains(t.array.elem_type_id)), "SPIRV simulator: Missing  array elem type");
        assertm (!(t.kind == Type::Kind::Vector && !types_.contains(t.vector.elem_type_id)), "SPIRV simulator: Missing vector elem type");
        assertm (!(t.kind == Type::Kind::RuntimeArray && !types_.contains(t.array.elem_type_id)), "SPIRV simulator: Missing runtie array elem type");
        assertm (!(t.kind == Type::Kind::Matrix && !types_.contains(t.matrix.col_type_id)), "SPIRV simulator: Missing matrix col type");
        assertm (!(t.kind == Type::Kind::Pointer && !types_.contains(t.pointer.pointee_type_id)), "SPIRV simulator: Missing pointee type");

        if(t.kind == Type::Kind::BoolT || t.kind == Type::Kind::Int || t.kind == Type::Kind::Float){
            if (t.scalar.width == 8 || t.scalar.width == 16){
                std::cout << execIndent << "Scalar width is: " << t.scalar.width << ", this is untested but should work (if errors, suspect this and investigate)" << std::endl;
            }

            assertm (t.scalar.width % 8 == 0, "SPIRV simulator: Scalar bit width is not a multiple of eight, we dont support this at present");
            assertm (t.scalar.width == 8 || t.scalar.width == 16 || t.scalar.width == 32 || t.scalar.width == 64, "SPIRV simulator: We only allow 8, 16, 32 and 64 bit scalars at present");
        }
    }

    assertm (sizeof(void*) == 8, "SPIRV simulator: Systems with non 64 bit pointers are not supported");
}

void SPIRVSimulator::ParseAll(){
    size_t instruction_index = 0;

    if (verbose_){
        std::cout << "SPIRV simulator: Parsing instructions:" << std::endl;
    }

    bool in_function = false;

    while (!stream_.empty()){
        Instruction instruction;
        DecodeInstruction(stream_, instruction);
        instructions_.push_back(instruction);

        bool has_result = false;
        bool has_type = false;
        spv::HasResultAndType(instruction.opcode, &has_result, &has_type);

        if (has_result){
            if (has_type){
                result_id_to_inst_index_[instruction.words[2]] = instruction_index;
            } else {
                result_id_to_inst_index_[instruction.words[1]] = instruction_index;
            }
        }

        if (verbose_){
            PrintInstruction(instruction);
        }

        switch (instruction.opcode){
            case spv::Op::OpFunction:{
                in_function = true;
                funcs_[instruction.words[2]] = {instruction_index, instruction_index + 1, {}, {}};
                prev_defined_func_id_ = instruction.words[2];
                break;
            }
            case spv::Op::OpFunctionEnd:
                in_function = false;
                break;
            case spv::Op::OpFunctionParameter:{
                funcs_[prev_defined_func_id_].parameter_ids_.push_back(instruction.words[2]);
                funcs_[prev_defined_func_id_].parameter_type_ids_.push_back(instruction.words[1]);
                break;
            }
            case spv::Op::OpEntryPoint:{
                uint32_t entry_point_id = instruction.words[2];
                entry_points_[entry_point_id] = "";
                break;
            }
            default:{
                if (!in_function){
                    ExecuteInstruction(instruction);
                }
                break;
            }
        }

        ++instruction_index;
    }
}

void SPIRVSimulator::Run(){
    assertm (unsupported_opcodes.size() == 0, "SPIRV simulator: Unhandled opcodes detected, implement them to run!");

    ParseAll();
    Validate();

    std::cout << std::endl;

    if(funcs_.empty()){
        if (verbose_){
            std::cerr << "SPIRV simulator: No functions defined in the shader, cannot start execution" << std::endl;
        }
        return;
    }

    uint32_t entry_point_function_id = 0;

    if (input_data_.entry_point_op_name != ""){
        for (const auto& it : entry_points_){
            if (it.second == input_data_.entry_point_op_name){
                std::cout << "SPIRV simulator: Using entry point with OpName label: " << it.second << std::endl;
                entry_point_function_id = it.first;
                break;
            }
        }

        assertm (entry_point_function_id != 0, "SPIRV simulator: Failed to find an entry point with the given OpName label");
    }

    if (entry_point_function_id == 0){
        if (entry_points_.find(input_data_.entry_point_id) == entry_points_.end()){
            if (verbose_){
                std::cout << "SPIRV simulator: Warning, entry point function with index: " << input_data_.entry_point_id << " not found, using first available" << std::endl;
            }

            entry_point_function_id = entry_points_.begin()->first;
        } else {
            std::cout << "SPIRV simulator: Using entry point with ID: " << input_data_.entry_point_id << std::endl;
            entry_point_function_id = input_data_.entry_point_id;
        }
    }

    if (verbose_){
        std::cout << "SPIRV simulator: Starting execution at entry point with function ID: " << entry_point_function_id << std::endl;
    }

    FunctionInfo& function_info = funcs_[entry_point_function_id];
    // We can set the return value to whatever, ignored if the call stack is empty on return
    call_stack_.push_back({function_info.first_inst_index, 0, {}, {}});

    while (!call_stack_.empty()){
        auto& stack_frame = call_stack_.back();
        const Instruction& instruction = instructions_[stack_frame.pc++];

        if (verbose_){
            PrintInstruction(instruction);
        }

        ExecuteInstruction(instruction);
    }

    if (verbose_){
        std::cout << "SPIRV simulator: Execution complete!\n" << std::endl;
    }

    for (const std::pair<PointerV, PointerV>& pointer_pair : pointers_to_physical_address_pointers_){
        const PointerV& phys_ppointer = pointer_pair.first;
        const PointerV& phys_pointer = pointer_pair.second;

        DataSourceBits source_data;
        source_data.location = BitLocation::StorageClass;
        source_data.storage_class = (spv::StorageClass)phys_ppointer.storage_class;
        source_data.idx = 0;
        source_data.bit_offset = 0;
        source_data.bitcount = 64;
        source_data.val_bit_offset = 0;

        if (phys_ppointer.storage_class == spv::StorageClass::StorageClassFunction){
            // We dont care about these, pointers that are temporary wont exist outside the shader execution context
            // and there will be other references to the actual buffer inputs
            continue;
        }

        if (phys_ppointer.storage_class == spv::StorageClass::StorageClassPushConstant){
            source_data.binding_id = 0;
            source_data.set_id = 0;
        } else if (phys_ppointer.storage_class != spv::StorageClass::StorageClassPhysicalStorageBuffer){
            assertm (HasDecorator(phys_ppointer.obj_id, spv::Decoration::DecorationDescriptorSet), "SPIRV simulator: Missing DecorationDescriptorSet for pointee object");
            assertm (HasDecorator(phys_ppointer.obj_id, spv::Decoration::DecorationBinding), "SPIRV simulator: Missing DecorationBinding for pointee object");

            source_data.binding_id = GetDecoratorLiteral(phys_ppointer.obj_id, spv::Decoration::DecorationBinding);
            source_data.set_id = GetDecoratorLiteral(phys_ppointer.obj_id, spv::Decoration::DecorationDescriptorSet);
        } else {
            source_data.binding_id = 0;
            source_data.set_id = 0;
        }

        source_data.byte_offset = GetPointerOffset(phys_ppointer);

        PhysicalAddressData output_result;
        output_result.raw_pointer_value = phys_pointer.raw_pointer;
        output_result.bit_components.push_back(source_data);
        physical_address_pointer_source_data_.push_back(output_result);
    }
}

void SPIRVSimulator::WriteOutputs(){
    assertx ("SPIRV simulator: Value writeout not implemented yet");
}

void SPIRVSimulator::ExecuteInstruction(const Instruction& instruction){
    auto dispatcher = opcode_dispatchers_.find(instruction.opcode);
    if(dispatcher == opcode_dispatchers_.end()){
        HandleUnimplementedOpcode(instruction);
    }
    else{
        dispatcher->second(instruction);
    }
}

void SPIRVSimulator::HandleUnimplementedOpcode(const Instruction& instruction){
    if (verbose_){
        std::cout << execIndent << "Found unimplemented opcode during execution of instruction: " << std::endl;
        PrintInstruction(instruction);
    }
}

std::string SPIRVSimulator::GetValueString(const Value& value){
    if (std::holds_alternative<double>(value)){
        return "double";
    }
    if (std::holds_alternative<uint64_t>(value)){
        return "uint64_t";
    }
    if (std::holds_alternative<int64_t>(value)){
        return "int64_t";
    }
    if (std::holds_alternative<std::monostate>(value)){
        return "std::monostate";
    }
    if (std::holds_alternative<std::shared_ptr<VectorV>>(value)){
        return "std::shared_ptr<VectorV>";
    }
    if (std::holds_alternative<std::shared_ptr<MatrixV>>(value)){
        return "std::shared_ptr<MatrixV>";
    }
    if (std::holds_alternative<std::shared_ptr<AggregateV>>(value)){
        return "std::shared_ptr<AggregateV>";
    }
    if (std::holds_alternative<PointerV>(value)){
        return "PointerV";
    }

    return "";
}

std::string SPIRVSimulator::GetTypeString(const Type& type){
    if (type.kind == Type::Kind::Void){
        return "void";
    }
    if (type.kind == Type::Kind::BoolT){
        return "bool";
    }
    if (type.kind == Type::Kind::Int){
        return "int";
    }
    if (type.kind == Type::Kind::Float){
        return "float";
    }
    if (type.kind == Type::Kind::Vector){
        return "vector";
    }
    if (type.kind == Type::Kind::Matrix){
        return "matrix";
    }
    if (type.kind == Type::Kind::Array){
        return "array";
    }
    if (type.kind == Type::Kind::RuntimeArray){
        return "runtime_array";
    }
    if (type.kind == Type::Kind::Struct){
        return "struct";
    }
    if (type.kind == Type::Kind::Pointer){
        return "pointer";
    }

    return "";
}

void SPIRVSimulator::PrintInstruction(const Instruction& instruction){
    bool has_result = false;
    bool has_type = false;
    spv::HasResultAndType(instruction.opcode, &has_result, &has_type);

    if (verbose_){
        std::stringstream result_and_type;

        uint32_t result_offset = 0;
        if (has_result){
            if (has_type){
                result_offset = 2;
            } else {
                result_offset = 1;
            }
        }

        if (has_type){
            bool has_type_value = types_.find(instruction.words[1]) != types_.end();
            if (has_type_value){
                result_and_type << GetTypeString(types_.at(instruction.words[1])) << "(" << instruction.words[1] << ") ";
            }
        }

        if (result_offset){
            result_and_type << instruction.words[result_offset] << " ";
        }

        std::cout << std::right << std::setw(18) << result_and_type.str() << spv::OpToString(instruction.opcode) << " ";

        if (instruction.opcode == spv::Op::OpExtInstImport) {
            std::cout << std::string((char*)(&instruction.words[2]), (instruction.word_count - 2) * 4);
        } else if (instruction.opcode == spv::Op::OpName) {
            std::cout << instruction.words[1] << " ";
            std::cout << std::string((char*)(&instruction.words[2]), (instruction.word_count - 2) * 4);
        } else if (instruction.opcode == spv::Op::OpTypePointer) {
            std::cout << spv::StorageClassToString((spv::StorageClass)instruction.words[2]) << " " << GetTypeString(types_.at(instruction.words[3])) << "(" << instruction.words[3] << ") ";
        } else if (instruction.opcode == spv::Op::OpVariable) {
            std::cout << spv::StorageClassToString((spv::StorageClass)instruction.words[3]) << " ";
            for (uint32_t i = 4; i < instruction.word_count; ++i){
                std::cout << instruction.words[i] << " ";
            }
        } else {
            for (uint32_t i = result_offset; i < instruction.word_count; ++i){
                if (i == result_offset){
                    continue;
                }
                if (instruction.opcode == spv::Op::OpDecorate){
                    if (i == 2) {
                        std::cout << spv::DecorationToString((spv::Decoration)instruction.words[i]) << " ";
                    } else {
                        std::cout << instruction.words[i] << " ";
                    }
                }else if(instruction.opcode == spv::Op::OpMemberDecorate){
                    if (i == 3) {
                        std::cout << spv::DecorationToString((spv::Decoration)instruction.words[i]) << " ";
                    } else {
                        std::cout << instruction.words[i] << " ";
                    }
                } else {
                    std::cout << instruction.words[i] << " ";
                }
            }
        }

        std::cout << std::endl;
    }
}

bool SPIRVSimulator::HasDecorator(uint32_t result_id, spv::Decoration decorator){
    /*
    Checks if a result_id has been decorated with the given decoration.
    */
    if (decorators_.find(result_id) != decorators_.end()){
        for (const auto& decorator_data : decorators_.at(result_id)){
            if (decorator == decorator_data.kind){
                return true;
            }
        }
    } else if (struct_decorators_.find(result_id) != struct_decorators_.end()){
        assertx ("SPIRV simulator: Unimplemented branch in HasDecorator");
    }

    return false;
}

bool SPIRVSimulator::HasDecorator(uint32_t result_id, uint32_t member_id, spv::Decoration decorator){
    /*
    Checks if a given member in a result_id has been decorated with the given decoration.
    */
    if (struct_decorators_.find(result_id) != struct_decorators_.end()){
        if (struct_decorators_.at(result_id).find(member_id) != struct_decorators_.at(result_id).end()){
            for (const auto& decorator_data : struct_decorators_.at(result_id).at(member_id)){
                if (decorator == decorator_data.kind){
                    return true;
                }
            }
        } else {
            return false;
        }
    } else if (decorators_.find(result_id) != decorators_.end()){
        assertx ("SPIRV simulator: Unimplemented branch in HasDecorator (member version)");
    }

    return false;
}

uint32_t SPIRVSimulator::GetDecoratorLiteral(uint32_t result_id, spv::Decoration decorator, size_t literal_offset){
    /*
    This will abort if the target id does not have the given decorator
    Check with HasDecorator first
    */
    if (decorators_.find(result_id) != decorators_.end()){
        for (const auto& decorator_data : decorators_.at(result_id)){
            if (decorator_data.kind == decorator){
                if (decorator_data.literals.size() <= literal_offset){
                    assertx ("SPIRV simulator: Literal offset OOB");
                }

                return decorator_data.literals[literal_offset];
            }
        }
    }

    assertx ("SPIRV simulator: No matching decorators for result ID");
}

uint32_t SPIRVSimulator::GetDecoratorLiteral(uint32_t result_id, uint32_t member_id, spv::Decoration decorator, size_t literal_offset){
    /*
    This will abort if the target id does not have the given decorator
    Check with HasDecorator first
    */
    if (struct_decorators_.find(result_id) != struct_decorators_.end()){
        if (struct_decorators_.at(result_id).find(member_id) != struct_decorators_.at(result_id).end()){
            for (const auto& decorator_data : struct_decorators_.at(result_id).at(member_id)){
                if (decorator_data.kind == decorator){
                    assertm (decorator_data.literals.size() > literal_offset, "SPIRV simulator: Literal offset OOB");

                    return decorator_data.literals[literal_offset];
                }
            }
        }
    }

    assertx ("SPIRV simulator: Not decorators for struct member");
}

Type SPIRVSimulator::GetType(uint32_t result_id) const{
    /*
    Returns the type struct mapping to a given result_id.
    result_id must be the result ID of a spirv instruction.
    */
    assertm (result_id_to_inst_index_.find(result_id) != result_id_to_inst_index_.end(), "SPIRV simulator: No instruction found for result_id");

    size_t instruction_index = result_id_to_inst_index_.at(result_id);
    const Instruction& instruction = instructions_[instruction_index];

    bool has_result = false;
    bool has_type = false;
    spv::HasResultAndType(instruction.opcode, &has_result, &has_type);

    if (has_type){
        uint32_t inst_type_id = instruction.words[1];
        assertm (types_.find(inst_type_id) != types_.end(), "SPIRV simulator: No type found for type_id");
        return types_.at(inst_type_id);
    } else {
        Type void_type;
        void_type.kind = Type::Kind::Void;
        void_type.scalar = {
            0,
            false
        };
        return void_type;
    }
}

// ---------------------------------------------------------------------------
//  Value creation and inspect helpers
// ---------------------------------------------------------------------------

size_t SPIRVSimulator::GetBitizeOfType(uint32_t type_id){
    /*
    Returns the full bitsize of the type associated with the given type ID.
    type_id must be the result of a OpType* instruction.
    */
    assertm (types_.find(type_id) != types_.end(), "SPIRV simulator: No valid type for the given ID was found");

    const Type& type = types_.at(type_id);

    assertm (type.kind != Type::Kind::Void, "SPIRV simulator: Attempt to extract size of a void type");

    if (verbose_){
        std::cout << execIndent << "Fetching bitsize of type with ID: " << type_id << std::endl;
    }

    size_t bitcount = 0;
    if (type.kind == Type::Kind::BoolT || type.kind == Type::Kind::Int || type.kind == Type::Kind::Float){
        bitcount += type.scalar.width;
    } else if (type.kind == Type::Kind::Vector){
        uint32_t elem_type_id = type.vector.elem_type_id;
        bitcount += GetBitizeOfType(elem_type_id) * type.vector.elem_count;
    } else if (type.kind == Type::Kind::Matrix){
        uint32_t col_type_id = type.matrix.col_type_id;
        bitcount += GetBitizeOfType(col_type_id) * type.matrix.col_count;
    } else if (type.kind == Type::Kind::Array){
        uint32_t elem_type_id = type.vector.elem_type_id;
        uint64_t array_len = std::get<uint64_t>(GetValue(type.array.length_id));

        bitcount += GetBitizeOfType(elem_type_id) * array_len;
    } else if (type.kind == Type::Kind::RuntimeArray){
        assertx ("SPIRV simulator: Fetching bitsize of RuntimeArray, this is currently not implemented");

        //uint32_t elem_type_id = type.vector.elem_type_id;
        //uint64_t array_len = std::get<uint64_t>(GetValue(type.array.length_id));
        //bitcount += GetBitizeOfType(elem_type_id);
    } else if (type.kind == Type::Kind::Struct){
        assertm (struct_members_.find(type_id) != struct_members_.end(), "SPIRV simulator: Struct has no members");

        for (uint32_t member_type_id : struct_members_.at(type_id)){
            bitcount += GetBitizeOfType(member_type_id);
        }
    } else if (type.kind == Type::Kind::Pointer){
        bitcount += 8 * 8;
    }

    return bitcount;
}

size_t SPIRVSimulator::GetBitizeOfTargetType(const PointerV& pointer){
    /*
    Returns the full bitsize of the type pointed to by the given pointer.
    The pointers type_id field must be the result of a OpType* instruction.
    */
    assertm (types_.find(pointer.type_id) != types_.end(), "SPIRV simulator: No valid type for the given pointer type ID was found");

    const Type* type = &types_.at(pointer.type_id);

    uint32_t type_id = type->pointer.pointee_type_id;
    type = &types_.at(type_id);
    for (uint32_t idx : pointer.idx_path){
        if (type->kind == Type::Kind::Struct){
            assertm (struct_members_.find(type_id) != struct_members_.end(), "SPIRV simulator: Struct has no members");

            type_id = struct_members_.at(type_id)[idx];
            type = &types_.at(type_id);
        } else if ((type->kind == Type::Kind::Array) || (type->kind == Type::Kind::RuntimeArray)) {
            type_id = type->array.elem_type_id;
            type = &types_.at(type_id);
        } else if (type->kind == Type::Kind::Vector) {
            type_id = type->vector.elem_type_id;
            type = &types_.at(type_id);
        } else if (type->kind == Type::Kind::Matrix) {
            type_id = type->matrix.col_type_id;
            type = &types_.at(type_id);
        } else if (type->kind == Type::Kind::Pointer) {
            type_id = type->pointer.pointee_type_id;
            type = &types_.at(type_id);
        } else {
            assertx ("SPIRV simulator: Unhandled type in GetBitizeOfTargetType");
        }
    }

    return GetBitizeOfType(type_id);
}

void SPIRVSimulator::GetBaseTypeIDs(uint32_t type_id, std::vector<uint32_t>& output){
    /*
    Gets all the scalar types in a compond types, laid out as they are in memory.
    */
    const Type& type = types_.at(type_id);

    assertm (type.kind != Type::Kind::Void, "SPIRV simulator: Attempt to extract size of a void type");

    if (type.kind == Type::Kind::BoolT || type.kind == Type::Kind::Int || type.kind == Type::Kind::Float || type.kind == Type::Kind::Pointer){
        output.push_back(type_id);
    } else if (type.kind == Type::Kind::Vector){
        uint32_t elem_type_id = type.vector.elem_type_id;
        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            output.push_back(elem_type_id);
        }
    } else if (type.kind == Type::Kind::Matrix){
        uint32_t col_type_id = type.matrix.col_type_id;
        for (uint32_t i = 0; i < type.matrix.col_count; ++i){
            GetBaseTypeIDs(col_type_id, output);
        }
    } else if (type.kind == Type::Kind::Array || type.kind == Type::Kind::RuntimeArray){
        uint32_t elem_type_id = type.vector.elem_type_id;
        uint64_t array_len = std::get<uint64_t>(GetValue(type.array.length_id));
        for (uint64_t i = 0; i < array_len; ++i){
            GetBaseTypeIDs(elem_type_id, output);
        }
    } else if (type.kind == Type::Kind::Struct){
        for (uint32_t member_type_id : struct_members_.at(type_id)){
            GetBaseTypeIDs(member_type_id, output);
        }
    }
}

void SPIRVSimulator::ExtractWords(const std::byte* external_pointer, uint32_t type_id, std::vector<uint32_t>& buffer_data){
    /*
    Extracts 32 bit word values with type matching type_id from the external_pointer byte buffer
    */
    const Type& type = types_.at(type_id);

    assertm (type.kind != Type::Kind::Void, "SPIRV simulator: Attempt to extract a void type from a buffer");

    if (type.kind == Type::Kind::Struct){
        uint32_t member_offset_id = 0;
        for (uint32_t member_type_id : struct_members_.at(type_id)){
            // They must have offset decorators
            assertm (HasDecorator(type_id, member_offset_id, spv::Decoration::DecorationOffset), "SPIRV simulator: No offset decorator for input struct member");

            const std::byte* member_offset_pointer = external_pointer + GetDecoratorLiteral(type_id, member_offset_id, spv::Decoration::DecorationOffset);
            ExtractWords(member_offset_pointer, member_type_id, buffer_data);
            member_offset_id += 1;
        }
    } else if (type.kind == Type::Kind::Array || type.kind == Type::Kind::RuntimeArray){
        // They must have a stride decorator (TODO: unless they contain blocks, but we can deal with that later)
        assertm (HasDecorator(type_id, spv::Decoration::DecorationArrayStride), "SPIRV simulator: No ArrayStride decorator for input array");

        uint32_t array_stride = GetDecoratorLiteral(type_id, spv::Decoration::DecorationArrayStride);

        if (type.array.length_id == 0){
            // Runtime array, special handling, extract one element
            // TODO: We should probably change this and use sparse loads with maps or something
            ExtractWords(external_pointer, type.array.elem_type_id, buffer_data);
        } else {
            uint64_t array_len = std::get<uint64_t>(GetValue(type.array.length_id));

            for (uint64_t array_index = 0; array_index < array_len; ++array_index){
                const std::byte* member_offset_pointer = external_pointer + array_stride * array_index;
                ExtractWords(member_offset_pointer, type.array.elem_type_id, buffer_data);
            }
        }
    } else if (type.kind == Type::Kind::Matrix){
        assertm (HasDecorator(type_id, spv::Decoration::DecorationMatrixStride), "SPIRV simulator: No MatrixStride decorator for input matrix");
        assertm (HasDecorator(type_id, spv::Decoration::DecorationRowMajor) || HasDecorator(type_id, spv::Decoration::DecorationColMajor), "SPIRV simulator: No RowMajor or ColMajor decorator for input matrix");

        const Type& col_type = types_.at(type.matrix.col_type_id);
        assertm (col_type.kind == Type::Kind::Vector, "SPIRV simulator: Non-vector column type found in matrix");

        // Because row-major matrices may not have a valid col type, we extract the subcomponents directly
        // We basically treat it as an array
        // Always extract to a column major order to simplify stuff later
        uint32_t col_count = type.matrix.col_count;
        uint32_t row_count = col_type.vector.elem_count;

        uint32_t component_stride = GetDecoratorLiteral(type_id, spv::Decoration::DecorationMatrixStride);
        bool row_major = HasDecorator(type_id, spv::Decoration::DecorationRowMajor);

        uint32_t bytes_per_subcomponent = std::ceil((double)(GetBitizeOfType(col_type.vector.elem_type_id) / 8));

        for (uint64_t col_index = 0; col_index < col_count; ++col_index){
            for (uint64_t row_index = 0; row_index < row_count; ++row_index){
                const std::byte* member_offset_pointer;
                if (row_major) {
                    member_offset_pointer = external_pointer + row_index * component_stride + col_index * bytes_per_subcomponent;
                } else {
                    member_offset_pointer = external_pointer + col_index * component_stride + row_index * bytes_per_subcomponent;
                }
                ExtractWords(member_offset_pointer, col_type.vector.elem_type_id, buffer_data);
            }
        }

    } else {
        // Assume everything else is tightly packed
        std::vector<uint32_t> base_type_ids;
        GetBaseTypeIDs(type_id, base_type_ids);
        size_t ext_ptr_offset = 0;
        for (auto base_type_id : base_type_ids){
            const Type& base_type = types_.at(base_type_id);
            size_t bytes_to_extract;

            if (base_type.kind == Type::Kind::Pointer){
                bytes_to_extract = 8;
            } else {
                bytes_to_extract = std::ceil((double)base_type.scalar.width / 8.0);
            }

            size_t output_index = buffer_data.size();
            buffer_data.reserve(output_index + std::ceil((double)bytes_to_extract / 4.0));
            std::memcpy(&(buffer_data[output_index]), external_pointer + ext_ptr_offset, bytes_to_extract);
            ext_ptr_offset += bytes_to_extract;
        }
    }
}

uint64_t SPIRVSimulator::GetPointerOffset(const PointerV& pointer_value){
    /*
    Given a pointer, this will get the correct offset into the memory where its value resides (relative to its base).
    */
    uint64_t offset = 0;
    uint32_t type_id = pointer_value.type_id;
    const Type* type = &types_.at(type_id);
    type_id = type->pointer.pointee_type_id;
    type = &types_.at(type_id);

    assertm (type->kind != Type::Kind::Void, "SPIRV simulator: Attempt to extract a void type offset");

    for (uint32_t indirection_index : pointer_value.idx_path){
        if (type->kind == Type::Kind::Struct){
            // They must have offset decorators
            assertm (HasDecorator(type_id, indirection_index, spv::Decoration::DecorationOffset), "SPIRV simulator: No offset decorator for input struct member");

            offset += GetDecoratorLiteral(type_id, indirection_index, spv::Decoration::DecorationOffset);
            type_id = struct_members_.at(type_id)[indirection_index];
            type = &types_.at(type_id);
        } else if (type->kind == Type::Kind::Array || type->kind == Type::Kind::RuntimeArray){
            // They must have a stride decorator (TODO: unless they contain blocks, but we can deal with that later)
            assertm (HasDecorator(type_id, spv::Decoration::DecorationArrayStride), "SPIRV simulator: No ArrayStride decorator for input array");

            uint32_t array_stride = GetDecoratorLiteral(type_id, spv::Decoration::DecorationArrayStride);
            offset += indirection_index * array_stride;
            type_id = type->array.elem_type_id;
            type = &types_.at(type_id);
        } else if (type->kind == Type::Kind::Matrix){
            assertm (HasDecorator(type_id, spv::Decoration::DecorationColMajor), "SPIRV simulator: Attempt to get pointer offset to row-major matrix, this is illegal and violates contiguity requirements");

            uint32_t matrix_stride = GetDecoratorLiteral(type_id, spv::Decoration::DecorationMatrixStride);
            offset += indirection_index * matrix_stride;
            type_id = type->matrix.col_type_id;
            type = &types_.at(type_id);
        } else if (type->kind == Type::Kind::Vector){
            type_id = type->vector.elem_type_id;
            type = &types_.at(type->vector.elem_type_id);
            offset += indirection_index * std::ceil(type->scalar.width / 8.0);
        } else {
            // Crash, this should never happen
            assertx ("SPIRV simulator: Pointer attempts to index a type that cant be indexed");
        }
    }

    return offset;
}

uint32_t SPIRVSimulator::GetTypeID(uint32_t result_id) const{
    /*
    Given a result ID, return the type ID of the value it maps to.
    */
    assertm (result_id_to_inst_index_.find(result_id) != result_id_to_inst_index_.end(), "SPIRV simulator: No instruction found for result_id");

    size_t instruction_index = result_id_to_inst_index_.at(result_id);
    const Instruction& instruction = instructions_[instruction_index];

    bool has_result = false;
    bool has_type = false;
    spv::HasResultAndType(instruction.opcode, &has_result, &has_type);

    if (has_type){
        return instruction.words[1];
    }

    assertx ("SPIRV simulator: No type found for result_id");
}

Value SPIRVSimulator::MakeScalar(uint32_t type_id, const uint32_t*& words){
    const Type& type = types_.at(type_id);

    switch(type.kind){
        case Type::Kind::Int:{
            assertm (type.scalar.width <= 64, "SPIRV simulator: We do not support types wider than 64 bits");

            if (type.scalar.width > 32){
                if (type.scalar.is_signed){
                    int64_t tmp_value;
                    std::memcpy(&tmp_value, words, 8);
                    words += 2;
                    return tmp_value;
                } else {
                    uint64_t tmp_value = (static_cast<uint64_t>(words[1]) << 32) | words[0];
                    words += 2;
                    return tmp_value;
                }
            } else {
                if (type.scalar.is_signed){
                    int32_t tmp_value;
                    std::memcpy(&tmp_value, &words[0], 4);
                    words += 1;
                    return (int64_t)tmp_value;
                } else {
                    uint64_t tmp_value = (uint64_t)words[0];
                    words += 1;
                    return tmp_value;
                }
            }
        }
        case Type::Kind::BoolT:{
            // Just treat bools as uint64_t types for simplicity
            assertm (type.scalar.width <= 64, "SPIRV simulator: Bool value with more than 64 bits detected, this is not handled at present");
            uint64_t tmp_value = (uint64_t)words[0];
            words += 1;
            return tmp_value;
        }
        case Type::Kind::Float:{
            assertm (type.scalar.width <= 64, "SPIRV simulator: We do not support types wider than 64 bits");
            if (type.scalar.width > 32){
                double tmp_value;
                std::memcpy(&tmp_value, &words[0], 8);
                words += 2;
                return tmp_value;
            } else {
                float tmp_value;
                std::memcpy(&tmp_value, &words[0], 4);
                words += 1;
                return (double)tmp_value;
            }
        }
        default:{
            assertx ("SPIRV simulator: Unsupported scalar type, instructions are possibly corrupt");
        }
    }
}

Value SPIRVSimulator::MakeDefault(uint32_t type_id, const uint32_t** initial_data){
    const Type& type = types_.at(type_id);

    switch(type.kind){
        case Type::Kind::Int:
        case Type::Kind::Float:
        case Type::Kind::BoolT:{
            if (initial_data != nullptr){
                return MakeScalar(type_id, *initial_data);
            } else {
                const uint32_t empty_array[]{0,0};
                const uint32_t* buffer_pointer = empty_array;
                return MakeScalar(type_id, buffer_pointer);
            }
        }
        case Type::Kind::Image:{
            assertm (!initial_data, "SPIRV simulator: Cannot create Image handle with initial_data unless we know the size of the opaque types");
            return (uint64_t)(0);
        }
        case Type::Kind::Vector:{
            auto vec = std::make_shared<VectorV>();
            vec->elems.reserve(type.vector.elem_count);
            for (uint32_t i = 0; i < type.vector.elem_count; ++i){
                vec->elems.push_back(MakeDefault(type.vector.elem_type_id, initial_data));
            }

            return vec;
        }
        case Type::Kind::Matrix:{
            // We dont have to deal with col/row major here since we do that on buffer extraction
            auto matrix = std::make_shared<MatrixV>();
            matrix->cols.reserve(type.matrix.col_count);
            for(uint32_t i = 0; i < type.matrix.col_count; ++i){
                Value mat_val = MakeDefault(type.matrix.col_type_id, initial_data);
                matrix->cols.push_back(mat_val);
            }

            return matrix;
        }
        case Type::Kind::Array:{
            uint64_t len = std::get<uint64_t>(GetValue(type.array.length_id));
            auto aggregate = std::make_shared<AggregateV>();
            aggregate->elems.reserve(len);
            for(uint32_t i = 0; i < len; ++i){
                aggregate->elems.push_back(MakeDefault(type.array.elem_type_id, initial_data));
            }

            return aggregate;
        }
        case Type::Kind::RuntimeArray: {
            uint64_t len = 1;
            if (type.array.length_id != 0){
                len = std::get<uint64_t>(GetValue(type.array.length_id));
            }

            auto aggregate = std::make_shared<AggregateV>();
            aggregate->elems.reserve(len);
            for(uint32_t i = 0; i < len; ++i){
                aggregate->elems.push_back(MakeDefault(type.array.elem_type_id, initial_data));
            }

            return aggregate;
        }
        case Type::Kind::Struct:{
            auto structure = std::make_shared<AggregateV>();
            for(auto member : struct_members_.at(type_id)){
                structure->elems.push_back(MakeDefault(member, initial_data));
            }

            return structure;
        }
        case Type::Kind::Pointer:{
            if (type.pointer.storage_class == spv::StorageClass::StorageClassPhysicalStorageBuffer){
                uint64_t pointer_value = 0;

                if (initial_data){
                    std::memcpy(&pointer_value, reinterpret_cast<const std::byte*>(*initial_data), sizeof(uint64_t));
                } else {
                    if (verbose_){
                        std::cout << execIndent << "A pointer with StorageClassPhysicalStorageBuffer was default initialized without input buffer data available. The actual pointer address will be unknown (null)" << std::endl;
                    }
                }

                if (initial_data){
                    (*initial_data) += 2;
                }

                const std::byte* remapped_pointer = nullptr;

                for (const auto& map_entry : input_data_.physical_address_buffers){
                    uint64_t buffer_address = map_entry.first;
                    size_t buffer_size = map_entry.second.first;

                    const std::byte* buffer_data = static_cast<std::byte*>(map_entry.second.second);

                    if ((pointer_value >= buffer_address) && (pointer_value < (buffer_address + buffer_size))){
                        remapped_pointer = &(buffer_data[buffer_address - pointer_value]);
                        break;
                    }
                }

                Value init;
                if (remapped_pointer){
                    std::vector<uint32_t> buffer_data;
                    ExtractWords(remapped_pointer, type.pointer.pointee_type_id, buffer_data);
                    const uint32_t* buffer_pointer = buffer_data.data();
                    init = MakeDefault(type.pointer.pointee_type_id, &buffer_pointer);
                } else {
                    init = MakeDefault(type.pointer.pointee_type_id);
                }

                uint32_t pointee_obj_id = GetNextExternalID();
                Heap(type.pointer.storage_class)[pointee_obj_id] = init;

                PointerV new_pointer{pointee_obj_id, type_id, type.pointer.storage_class, pointer_value, {}, {}};
                physical_address_pointers_.push_back(new_pointer);
                return new_pointer;
            } else {
                assertx ("SPIRV simulator: Attempting to initialize a raw pointer whose storage class is not PhysicalStorageBuffer");
            }
        }
        default:{
            assertx ("SPIRV simulator: Invalid input type to MakeDefault");
        }
    }
}

std::vector<DataSourceBits> SPIRVSimulator::FindDataSourcesFromResultID(uint32_t result_id){
    std::vector<DataSourceBits> results;

    uint32_t instruction_index = result_id_to_inst_index_[result_id];
    const Instruction& instruction = instructions_[instruction_index];

    bool has_result = false;
    bool has_type = false;
    spv::HasResultAndType(instruction.opcode, &has_result, &has_type);

    uint32_t type_id = 0;
    if (has_type){
        type_id = instruction.words[1];
    }

    if (verbose_){
        std::cout << execIndent << "Tracing value source backwards through: " << spv::OpToString(instruction.opcode) << ": " << result_id << std::endl;
    }

    switch (instruction.opcode){
        case spv::Op::OpSpecConstantComposite:{
            for (uint32_t component_id = 3; component_id < instruction.word_count; ++component_id){
                std::vector<DataSourceBits> component_result = FindDataSourcesFromResultID(instruction.words[component_id]);
                results.insert(results.end(), component_result.begin(), component_result.end());
            }

            DataSourceBits* prev_source = nullptr;
            for (auto& component_data : results){
                if (prev_source){
                    component_data.val_bit_offset += prev_source->val_bit_offset + prev_source->bitcount;
                }

                prev_source = &component_data;
            }
            break;
        }
        case spv::Op::OpSpecConstant:{
            assertm (HasDecorator(result_id, spv::Decoration::DecorationSpecId), "SPIRV simulator: Op_SpecConstant type is not decorated with SpecId");
            uint32_t spec_id = GetDecoratorLiteral(result_id, spv::Decoration::DecorationSpecId);

            DataSourceBits data_source;
            data_source.location = BitLocation::SpecConstant;
            data_source.idx = 0;
            data_source.binding_id = spec_id;
            data_source.set_id = 0;
            data_source.byte_offset = 0;
            data_source.bit_offset = 0;
            data_source.bitcount = GetBitizeOfType(type_id);;
            data_source.val_bit_offset = 0;
            results.push_back(data_source);
            break;
        }
        case spv::Op::OpLoad:{
            uint32_t pointer_id = instruction.words[3];


            if (values_stored_.find(pointer_id) == values_stored_.end()){
                const PointerV& pointer = std::get<PointerV>(GetValue(pointer_id));

                assertm (pointer.storage_class != spv::StorageClass::StorageClassFunction, "SPIRV simulator: A StorageClassFunction is being read from while backtracing operands without having been stored to, this is a symptom of a serious error somewhere");

                DataSourceBits data_source;
                data_source.location = BitLocation::StorageClass;
                data_source.storage_class = (spv::StorageClass)pointer.storage_class;
                data_source.idx = 0;

                if (pointer.storage_class == spv::StorageClass::StorageClassPushConstant){
                    data_source.binding_id = 0;
                    data_source.set_id = 0;
                } else if (pointer.storage_class != spv::StorageClass::StorageClassPhysicalStorageBuffer){
                    assertm (HasDecorator(pointer.obj_id, spv::Decoration::DecorationDescriptorSet), "SPIRV simulator: Missing DecorationDescriptorSet for pointee object");
                    assertm (HasDecorator(pointer.obj_id, spv::Decoration::DecorationBinding), "SPIRV simulator: Missing DecorationBinding for pointee object");

                    data_source.binding_id = GetDecoratorLiteral(pointer.obj_id, spv::Decoration::DecorationBinding);
                    data_source.set_id = GetDecoratorLiteral(pointer.obj_id, spv::Decoration::DecorationDescriptorSet);
                } else {
                    data_source.binding_id = 0;
                    data_source.set_id = 0;
                }

                data_source.byte_offset = GetPointerOffset(pointer);
                data_source.bit_offset = 0;
                // This does not account for padding, but its probably fine here since it makes little sense to load complex constructs here
                data_source.bitcount = GetBitizeOfTargetType(pointer);
                data_source.val_bit_offset = 0;
                results.push_back(data_source);
            } else {
                uint32_t true_source = values_stored_.at(pointer_id);

                std::vector<DataSourceBits> component_result = FindDataSourcesFromResultID(true_source);
                results.insert(results.end(), component_result.begin(), component_result.end());
            }
            break;
        }
        case spv::Op::OpConstant:{
            DataSourceBits data_source;
            data_source.location = BitLocation::Constant;
            data_source.idx = 0;
            data_source.binding_id = 0;
            data_source.set_id = 0;
            uint32_t header_word_count = 5;
            data_source.byte_offset = (instruction_index + header_word_count) * sizeof(uint32_t);
            data_source.bit_offset = 0;
            data_source.bitcount = GetBitizeOfType(type_id);;
            data_source.val_bit_offset = 0;
            results.push_back(data_source);
            break;
        }
        default:{
            assertx ("SPIRV simulator: Unimplemented opcode in FindDataSourcesFromResultID");
        }
    }

    return results;
}

Value SPIRVSimulator::CopyValue(const Value& value) const{
    /*
    Creates a copy of a Value object, will recursively copy all pointers
    and components.
    */

    if (std::holds_alternative<std::shared_ptr<VectorV>>(value)){
        std::shared_ptr<VectorV> new_vector = std::make_shared<VectorV>();

        for (const auto& elem : std::get<std::shared_ptr<VectorV>>(value)->elems){
            new_vector->elems.push_back(CopyValue(elem));
        }

        return new_vector;
    } else if (std::holds_alternative<std::shared_ptr<MatrixV>>(value)){
        std::shared_ptr<MatrixV> new_matrix = std::make_shared<MatrixV>();

        for (const auto& col : std::get<std::shared_ptr<MatrixV>>(value)->cols){
            new_matrix->cols.push_back(CopyValue(col));
        }

        return new_matrix;

    } else if (std::holds_alternative<std::shared_ptr<AggregateV>>(value)){
        std::shared_ptr<AggregateV> new_aggregate = std::make_shared<AggregateV>();

        for (const auto& elem : std::get<std::shared_ptr<AggregateV>>(value)->elems){
            new_aggregate->elems.push_back(CopyValue(elem));
        }

        return new_aggregate;
    }

    return value;
}

// ---------------------------------------------------------------------------
//  Dereference and access helpers
// ---------------------------------------------------------------------------

Value& SPIRVSimulator::Deref(const PointerV &ptr){
    auto& heap = (ptr.storage_class == (uint32_t)spv::StorageClass::StorageClassFunction) ? call_stack_.back().func_heap : Heap(ptr.storage_class);

    Value* value = &heap.at(ptr.obj_id);
    for(size_t depth = 0; depth < ptr.idx_path.size(); ++depth){
        uint32_t indirection_index = ptr.idx_path[depth];

        if(std::holds_alternative<std::shared_ptr<AggregateV>>(*value)){
            auto agg = std::get<std::shared_ptr<AggregateV>>(*value);

            if(indirection_index >= agg->elems.size()){
                // We assume a runtime array here and just return the first entry
                // TODO: We should probably change this to use sparse access with maps or something
                if (verbose_){
                    std::cout << execIndent << "Array index OOB, assuming runtime array and returning first element" << std::endl;
                }
                value = &agg->elems[0];
            } else {
                value = &agg->elems[indirection_index];
            }
        } else if (std::holds_alternative<std::shared_ptr<VectorV>>(*value)){
            auto vec = std::get<std::shared_ptr<VectorV>>(*value);

            assertm (indirection_index < vec->elems.size(), "SPIRV simulator: Vector index OOB");

            value = &vec->elems[indirection_index];
        } else if (std::holds_alternative<std::shared_ptr<MatrixV>>(*value)){
            auto matrix = std::get<std::shared_ptr<MatrixV>>(*value);

            assertm (indirection_index < matrix->cols.size(), "SPIRV simulator: Matrix index OOB");

            value = &matrix->cols[indirection_index];
        }
        else{
            assertx ("SPIRV simulator: Pointer dereference into non-composite object");
        }
    }

    return *value;
}

Value& SPIRVSimulator::GetValue(uint32_t result_id){
    for (auto riter = call_stack_.rbegin(); riter != call_stack_.rend(); ++riter) {
        if (riter->locals.find(result_id) != riter->locals.end()){
            return riter->locals.at(result_id);
        }
    }

    assertm (globals_.find(result_id) != globals_.end(), "SPIRV simulator: Access to undefined variable");

    return globals_.at(result_id);
}

void SPIRVSimulator::SetValue(uint32_t result_id, const Value& value){
    if (call_stack_.size()){
        call_stack_.back().locals[result_id] = value;
    } else {
        globals_[result_id] = value;
    }
}

// ---------------------------------------------------------------------------
//  Ext Import implementations
// ---------------------------------------------------------------------------

void SPIRVSimulator::GLSLExtHandler(
    uint32_t type_id,
    uint32_t result_id,
    uint32_t instruction_literal,
    const std::span<const uint32_t>& operand_words
){
    const Type& type = types_.at(type_id);

    switch(instruction_literal){
        case 14:{  // Cos
            const Value& operand = GetValue(operand_words[0]);

            if (type.kind == Type::Kind::Vector){
                assertm (std::holds_alternative<std::shared_ptr<VectorV>>(operand), "SPIRV simulator: Operands not of vector type in GLSLExtHandler::cos");

                Value result = std::make_shared<VectorV>();
                auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto vec = std::get<std::shared_ptr<VectorV>>(operand);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i){
                    Value elem_result = (double)std::cos(std::get<double>(vec->elems[i]));
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);

            } else if (type.kind == Type::Kind::Float){
                Value result = (double)std::cos(std::get<double>(operand));
                SetValue(result_id, result);
            }
            break;
        }
        default:{
            if (verbose_){
                std::cout << "SPIRV simulator: Unhandled OpExtInst GLSL set operation: " << instruction_literal << std::endl;
                std::cout << "SPIRV simulator: Setting output to default value, this will likely crash" << std::endl;
            }
            SetValue(result_id, MakeDefault(type_id));
        }
    }
}


// ---------------------------------------------------------------------------
//  Type creation handlers
// ---------------------------------------------------------------------------
void SPIRVSimulator::T_Void(const Instruction& instruction){
    assert(instruction.opcode == spv::Op::OpTypeVoid);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind = Type::Kind::Void;
    type.scalar = {
        0,
        false
    };

    types_[result_id] = type;
}

void SPIRVSimulator::T_Bool(const Instruction& instruction){
    assert(instruction.opcode == spv::Op::OpTypeBool);

    // We treat bools as 64 bit unsigned ints for simplicity
    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind = Type::Kind::BoolT;
    type.scalar = {
        64,
        false
    };

    types_[result_id] = type;
}

void SPIRVSimulator::T_Int(const Instruction& instruction){
    assert(instruction.opcode == spv::Op::OpTypeInt);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind = Type::Kind::Int;
    type.scalar = {
        instruction.words[2],
        (bool)instruction.words[3]
    };

    types_[result_id] = type;
}

void SPIRVSimulator::T_Float(const Instruction& instruction){
    assert(instruction.opcode == spv::Op::OpTypeFloat);

    // We dont handle floats encoded in other formats than the default at present
    uint32_t result_id = instruction.words[1];

    assertm (instruction.word_count <= 3, "SPIRV simulator: Simulator only supports IEEE 754 encoded floats at present.");

    Type type;
    type.kind = Type::Kind::Float;
    type.scalar = {
        instruction.words[2],
        false
    };

    types_[result_id] = type;
}

void SPIRVSimulator::T_Vector(const Instruction& instruction){
    assert(instruction.opcode == spv::Op::OpTypeVector);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind = Type::Kind::Vector;
    type.vector = {
        instruction.words[2],
        instruction.words[3]
    };

    types_[result_id] = type;
}

void SPIRVSimulator::T_Matrix(const Instruction& instruction){
    assert(instruction.opcode == spv::Op::OpTypeMatrix);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind = Type::Kind::Matrix;
    type.matrix = {
        instruction.words[2],
        instruction.words[3]
    };

    types_[result_id] = type;
}

void SPIRVSimulator::T_Array(const Instruction& instruction){
    assert(instruction.opcode == spv::Op::OpTypeArray);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind = Type::Kind::Array;
    type.array = {
        instruction.words[2],
        instruction.words[3]
    };

    types_[result_id] = type;
}

void SPIRVSimulator::T_Struct(const Instruction& instruction){
    assert(instruction.opcode == spv::Op::OpTypeStruct);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind = Type::Kind::Struct;

    types_[instruction.words[1]] = type;

    std::vector<uint32_t> members;
    for(auto i = 2; i < instruction.word_count; ++i){
        members.push_back(instruction.words[i]);
    }

    struct_members_[result_id] = std::move(members);
}

void SPIRVSimulator::T_Pointer(const Instruction& instruction){
    assert(instruction.opcode == spv::Op::OpTypePointer);

    uint32_t result_id = instruction.words[1];
    uint32_t storage_class = instruction.words[2];
    uint32_t pointee_type_id = instruction.words[3];

    Type type;
    type.kind = Type::Kind::Pointer;
    type.pointer = {
        storage_class,
        pointee_type_id
    };
    types_[result_id] = type;
}

void SPIRVSimulator::T_ForwardPointer(const Instruction& instruction) {
    assert(instruction.opcode == spv::Op::OpTypeForwardPointer);

    // TODO: May not need this
    uint32_t pointer_type_id = instruction.words[1];
    uint32_t storage_class = instruction.words[2];
    forward_type_declarations_[pointer_type_id] = storage_class;
}

void SPIRVSimulator::T_RuntimeArray(const Instruction& instruction) {
    assert(instruction.opcode == spv::Op::OpTypeRuntimeArray);

    uint32_t result_id = instruction.words[1];
    uint32_t elem_type_id = instruction.words[2];

    Type type;
    type.kind = Type::Kind::RuntimeArray;
    type.array = {
        elem_type_id,
        0
    };
    types_[result_id] = type;

}

void SPIRVSimulator::T_Function(const Instruction& instruction){
    // This info is redundant for us, so treat it as a NOP
    assert(instruction.opcode == spv::Op::OpTypeFunction);
}


void SPIRVSimulator::T_Image(const Instruction& instruction){
    assert(instruction.opcode == spv::Op::OpTypeImage);

    uint32_t result_id = instruction.words[1];
    uint32_t sampled_type_id = instruction.words[2];
    //uint32_t dim = instruction.words[3];
    //uint32_t depth = instruction.words[4];
    //uint32_t arrayed = instruction.words[5];
    //uint32_t multisampled = instruction.words[6];
    //uint32_t sampled = instruction.words[7];
    //uint32_t image_format = instruction.words[8];

    Type type;
    type.kind = Type::Kind::Image;
    type.image = {
        sampled_type_id
    };
    types_[result_id] = type;
}

void SPIRVSimulator::T_Sampler(const Instruction& instruction){
    assert(instruction.opcode == spv::Op::OpTypeSampler);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind = Type::Kind::Sampler;
    types_[result_id] = type;
}

void SPIRVSimulator::T_SampledImage(const Instruction& instruction){
    assert(instruction.opcode == spv::Op::OpTypeSampledImage);

    uint32_t result_id = instruction.words[1];
    uint32_t image_type_id = instruction.words[2];

    Type type;
    type.kind = Type::Kind::SampledImage;
    type.sampled_image = {
        image_type_id
    };
    types_[result_id] = type;
}

void SPIRVSimulator::T_Opaque(const Instruction& instruction){
    assert(instruction.opcode == spv::Op::OpTypeOpaque);

    uint32_t result_id = instruction.words[1];
    uint32_t name_literal = instruction.words[2];

    Type type;
    type.kind = Type::Kind::Opaque;
    type.opaque = {
        name_literal
    };
    types_[result_id] = type;
}

void SPIRVSimulator::T_NamedBarrier(const Instruction& instruction){
    assert(instruction.opcode == spv::Op::OpTypeNamedBarrier);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind = Type::Kind::NamedBarrier;
    types_[result_id] = type;
}

// ---------------------------------------------------------------------------
//  Oparation implementations
// ---------------------------------------------------------------------------

void SPIRVSimulator::Op_EntryPoint(const Instruction& instruction){
    // We handle these during init/parsing
    assert(instruction.opcode == spv::Op::OpEntryPoint);
}

void SPIRVSimulator::Op_ExtInstImport(const Instruction& instruction){
    /*
    OpExtInstImport

    Import an extended set of instructions. It can be later referenced by the Result <id>.

    Name is the extended instruction-set’s name string. Before version 1.6, there must be an external specification defining the semantics
    for this extended instruction set. Starting with version 1.6, if Name starts with "NonSemantic.", including the period that separates
    the namespace "NonSemantic" from the rest of the name, it is encouraged for a specification to exist on the SPIR-V Registry,
    but it is not required.

    Starting with version 1.6, an extended instruction-set name which is prefixed with "NonSemantic." is guaranteed to contain only
    non-semantic instructions, and all OpExtInst instructions referencing this set can be ignored. All instructions within such a set
    must have only <id> operands; no literals. When literals are needed, then the Result <id> from an OpConstant or OpString instruction
    is referenced as appropriate. Result <id>s from these non-semantic instruction-set instructions must be used only in other non-semantic
    instructions.

    See Extended Instruction Sets for more information.
    */
    assert(instruction.opcode == spv::Op::OpExtInstImport);

    uint32_t result_id = instruction.words[1];
    // SPIRV string literals are UTF-8 encoded, so basic c++ string functionality can be used to decode them
    extended_imports_[result_id] = std::string((char*)(&instruction.words[2]), (instruction.word_count - 2) * 4);
}

void SPIRVSimulator::Op_Constant(const Instruction& instruction){
    /*
    OpConstant
    Declare a new integer-type or floating-point-type scalar constant.
    Result Type must be a scalar integer type or floating-point type.
    Value is the bit pattern for the constant. Types 32 bits wide or smaller take one word.
    Larger types take multiple words, with low-order words appearing first.
    */
    assert(instruction.opcode == spv::Op::OpConstant || instruction.opcode == spv::Op::OpSpecConstant);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    const Type& type = types_.at(type_id);

    assertm ((type.kind == Type::Kind::Int) || (type.kind == Type::Kind::Float), "SPIRV simulator: Constant type unsupported");

    if (HasDecorator(result_id, spv::Decoration::DecorationSpecId)){
        uint32_t spec_id = GetDecoratorLiteral(result_id, spv::Decoration::DecorationSpecId);
        if (input_data_.specialization_constant_offsets.find(spec_id) != input_data_.specialization_constant_offsets.end()){
            size_t spec_id_offset = input_data_.specialization_constant_offsets.at(spec_id);
            const std::byte* raw_spec_const_data = static_cast<std::byte*>(input_data_.specialization_constants) + spec_id_offset;
            std::vector<uint32_t> buffer_data;
            ExtractWords(raw_spec_const_data, type_id, buffer_data);

            const uint32_t* buffer_pointer = buffer_data.data();
            SetValue(result_id, MakeScalar(type_id, buffer_pointer));
        } else {
            if (verbose_){
                std::cout << execIndent << "No spec constant data provided for result_id: " << result_id << ", using default" << std::endl;
            }
            const uint32_t* buffer_pointer = instruction.words.subspan(3).data();
            SetValue(result_id, MakeScalar(type_id, buffer_pointer));
        }
    } else {
        const uint32_t* buffer_pointer = instruction.words.subspan(3).data();
        SetValue(result_id, MakeScalar(type_id, buffer_pointer));
    }
}

void SPIRVSimulator::Op_ConstantComposite(const Instruction& instruction){
    /*
    OpConstantComposite

    Declare a new composite constant.

    Result Type must be a composite type, whose top-level members/elements/components/columns have the same type as the
    types of the Constituents. The ordering must be the same between the top-level types in Result Type and the Constituents.

    Constituents become members of a structure, or elements of an array, or components of a vector, or columns of a matrix.
    There must be exactly one Constituent for each top-level member/element/component/column of the result.
    The Constituents must appear in the order needed by the definition of the Result Type.
    The Constituents must all be <id>s of non-specialization constant-instruction declarations or an OpUndef.
    */
    assert(instruction.opcode == spv::Op::OpConstantComposite || instruction.opcode == spv::Op::OpSpecConstantComposite);
    Op_CompositeConstruct(instruction);
}

void SPIRVSimulator::Op_CompositeConstruct(const Instruction& instruction){
    /*
    OpCompositeConstruct

    Construct a new composite object from a set of constituent objects.

    Result Type must be a composite type, whose top-level members/elements/components/columns have the same
    type as the types of the operands, with one exception.

    The exception is that for constructing a vector, the operands may also be vectors with the same component
    type as the Result Type component type.

    If constructing a vector, the total number of components in all the operands must equal
    the number of components in Result Type.

    Constituents become members of a structure, or elements of an array, or components of a vector, or columnsof a matrix.
    There must be exactly one Constituent for each top-level member/element/component/column of the result,with one exception.

    The exception is that for constructing a vector, a contiguous subset of the scalars consumed can be represented by
    a vector operand instead.

    The Constituents must appear in the order needed by the definition of the type of the result.
    If constructing a vector, there must be at least two Constituent operands.

    */
    assert(instruction.opcode == spv::Op::OpCompositeConstruct || instruction.opcode == spv::Op::OpConstantComposite || instruction.opcode == spv::Op::OpSpecConstantComposite);

    // Composite: An aggregate (structure or an array), a matrix, or a vector.
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    const Type& type = types_.at(type_id);

    if(type.kind == Type::Kind::Vector){
        auto vec = std::make_shared<VectorV>();
        for(auto i = 3; i < instruction.word_count; ++i){
            const Value& component_value = GetValue(instruction.words[i]);

            if (std::holds_alternative<std::shared_ptr<VectorV>>(component_value)){
                std::shared_ptr<VectorV> component_vector = std::get<std::shared_ptr<VectorV>>(component_value);

                for (auto& vec_component : component_vector->elems){
                    vec->elems.push_back(vec_component);
                }

            } else {
                vec->elems.push_back(component_value);
            }
        }

        SetValue(result_id, vec);
    }
    else if (type.kind == Type::Kind::Matrix){
        auto matrix = std::make_shared<MatrixV>();
        for(auto i = 3; i < instruction.word_count; ++i){
            matrix->cols.push_back(GetValue(instruction.words[i]));
        }

        SetValue(result_id, matrix);
    }
    else if(type.kind == Type::Kind::Struct || type.kind == Type::Kind::Array || type.kind == Type::Kind::RuntimeArray){
        auto aggregate = std::make_shared<AggregateV>();
        for(auto i = 3; i < instruction.word_count; ++i){
            aggregate->elems.push_back(GetValue(instruction.words[i]));
        }

        SetValue(result_id, aggregate);
    }
    else{
        assertx ("SPIRV simulator: CompositeConstruct not implemented yet for type");
    }
}

void SPIRVSimulator::Op_Variable(const Instruction& instruction){
    /*
    OpVariable

    Allocate an object in memory, resulting in a pointer to it, which can be used with OpLoad and OpStore.

    Result Type must be an OpTypePointer. Its Type operand is the type of object in memory.
    Storage Class is the Storage Class of the memory holding the object. It must not be Generic.
    It must be the same as the Storage Class operand of the Result Type.

    If Storage Class is Function, the memory is allocated on execution of the instruction for the current invocation for
    each dynamic instance of the function. The current invocation’s memory is deallocated when it executes any function
    termination instruction of the dynamic instance of the function it was allocated by.

    Initializer is optional. If Initializer is present, it will be the initial value of the variable’s memory content.
    Initializer must be an <id> from a constant instruction or a global (module scope) OpVariable instruction.
    Initializer must have the same type as the type pointed to by Result Type.
    */
    assert(instruction.opcode == spv::Op::OpVariable);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t storage_class = instruction.words[3];

    const Type& type = types_.at(type_id);

    assertm (type.kind == Type::Kind::Pointer, "SPIRV simulator: Op_Variable must only be used to create pointer types");

    PointerV new_pointer{result_id, type_id, storage_class, 0, {}, {}};

    if (type.pointer.storage_class == spv::StorageClass::StorageClassPushConstant){
        const std::byte* external_pointer = static_cast<std::byte*>(input_data_.push_constants);
        if (!input_data_.push_constants){
            if (verbose_){
                std::cout << execIndent << "No push constant initialization data mapped in the inputs, setting to defaults, this may crash" << std::endl;
            }
            Value init = MakeDefault(type.pointer.pointee_type_id);
            Heap(storage_class)[result_id] = init;
        } else {
            std::vector<uint32_t> buffer_data;
            ExtractWords(external_pointer, type.pointer.pointee_type_id, buffer_data);

            const uint32_t* buffer_pointer = buffer_data.data();
            Value init = MakeDefault(type.pointer.pointee_type_id, &buffer_pointer);
            Heap(storage_class)[result_id] = init;
        }
    } else if (type.pointer.storage_class == spv::StorageClass::StorageClassUniform || type.pointer.storage_class == spv::StorageClass::StorageClassUniformConstant || type.pointer.storage_class == spv::StorageClass::StorageClassStorageBuffer){
        assertm (HasDecorator(result_id, spv::Decoration::DecorationDescriptorSet), "SPIRV simulator: OpVariable called with result_id that lacks the DescriptorSet decoration, but the storage class requires it");
        assertm (HasDecorator(result_id, spv::Decoration::DecorationBinding), "SPIRV simulator: OpVariable called with result_id that lacks the Binding decoration, but the storage class requires it");

        uint32_t descriptor_set = GetDecoratorLiteral(result_id, spv::Decoration::DecorationDescriptorSet);
        uint32_t binding = GetDecoratorLiteral(result_id, spv::Decoration::DecorationBinding);

        const std::byte* external_pointer = nullptr;

        if (input_data_.bindings.find(descriptor_set) != input_data_.bindings.end()){
            if (input_data_.bindings.at(descriptor_set).find(binding) != input_data_.bindings.at(descriptor_set).end()){
                external_pointer = static_cast<std::byte*>(input_data_.bindings.at(descriptor_set).at(binding));
            }
        }

        if (!external_pointer){
            if (verbose_){
                std::cout << execIndent << "No binding initialization data mapped in the inputs for descriptor set: " << descriptor_set << ", binding: " << binding << ", setting to defaults, this may crash" << std::endl;
            }
            Value init = MakeDefault(type.pointer.pointee_type_id);
            Heap(storage_class)[result_id] = init;
        } else {
            std::vector<uint32_t> buffer_data;
            ExtractWords(external_pointer, type.pointer.pointee_type_id, buffer_data);

            const uint32_t* buffer_pointer = buffer_data.data();
            Value init = MakeDefault(type.pointer.pointee_type_id, &buffer_pointer);
            Heap(storage_class)[result_id] = init;
        }
    } else if (type.pointer.storage_class == spv::StorageClass::StorageClassPhysicalStorageBuffer){
        // This is illegal
        assertx ("SPIRV simulator: Op_Variable must not be used to create pointer types with the PhysicalStorageBuffer storage class");
    }
    else {
        Value init;
        if(instruction.word_count >= 5){
            // The instruction has initialization data
            init = GetValue(instruction.words[4]);
        }
        else{
            // No init data, set to default
            init = MakeDefault(type.pointer.pointee_type_id);
        }

        if(storage_class == (uint32_t)spv::StorageClass::StorageClassFunction){
            call_stack_.back().func_heap[result_id] = init;
        }
        else{
            Heap(storage_class)[result_id] = init;
        }
    }

    const Type& pointee_type = types_.at(type.pointer.pointee_type_id);
    if ((pointee_type.kind == Type::Kind::Pointer) && (pointee_type.pointer.storage_class == spv::StorageClass::StorageClassPhysicalStorageBuffer)){
        // This pointer points to a physical storage buffer pointer
        // This is the easy case where we can extract the location of the physical
        // pointer from this pointer's offsets and storage class
        PointerV ppointer = std::get<PointerV>(Deref(new_pointer));
        pointers_to_physical_address_pointers_.push_back(std::pair<PointerV, PointerV>{new_pointer, ppointer});
    }

    SetValue(result_id, new_pointer);
}

void SPIRVSimulator::Op_Load(const Instruction& instruction){
    /*
    OpLoad

    Load through a pointer.

    Result Type is the type of the loaded object. It must be a type with fixed size; i.e., it must not be, nor include,
    any OpTypeRuntimeArray types.

    Pointer is the pointer to load through.
    Its type must be an OpTypePointer whose Type operand is the same as Result Type.

    If present, any Memory Operands must begin with a memory operand literal.
    If not present, it is the same as specifying the memory operand None.
    */
    assert(instruction.opcode == spv::Op::OpLoad);

    //uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t pointer_id = instruction.words[3];

    const PointerV& pointer = std::get<PointerV>(GetValue(pointer_id));

    SetValue(result_id, Deref(pointer));
}

void SPIRVSimulator::Op_Store(const Instruction& instruction){
    /*
    OpStore

    Store through a pointer.

    Pointer is the pointer to store through. Its type must be an OpTypePointer whose Type operand is the same as the type of Object.
    Object is the object to store.

    If present, any Memory Operands must begin with a memory operand literal.
    If not present, it is the same as specifying the memory operand None.
    */
    assert(instruction.opcode == spv::Op::OpStore);

    uint32_t pointer_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    const PointerV& pointer = std::get<PointerV>(GetValue(pointer_id));
    Deref(pointer) = GetValue(result_id);

    values_stored_[pointer_id] = result_id;

}

void SPIRVSimulator::Op_AccessChain(const Instruction& instruction){
    /*
    OpAccessChain

    Create a pointer into a composite object.

    Result Type must be an OpTypePointer. Its Type operand must be the type reached by walking the Base’s type
    hierarchy down to the last provided index in Indexes, and its Storage Class operand must be the same as the
    Storage Class of Base.
    If Result Type is an array-element pointer that is decorated with ArrayStride, its Array Stride must match the
    Array Stride of the array’s type. If the array’s type is not decorated with ArrayStride, Result Type also must not
    be decorated with ArrayStride.

    Base must be a pointer, pointing to the base of a composite object.

    Indexes walk the type hierarchy to the desired depth, potentially down to scalar granularity.
    The first index in Indexes selects the top-level member/element/component/column of the base composite.
    All composite constituents use zero-based numbering, as described by their OpType…​ instruction.
    The second index applies similarly to that result, and so on. Once any non-composite type is reached, there must be
    no remaining (unused) indexes.

    Each index in Indexes
    - must have a scalar integer type
    - is treated as signed
    - if indexing into a structure, must be an OpConstant whose value is in bounds for selecting a member
    - if indexing into a vector, array, or matrix, with the result type being a logical pointer type,
      causes undefined behavior if not in bounds.
    */
    assert(instruction.opcode == spv::Op::OpAccessChain || instruction.opcode == spv::Op::OpInBoundsAccessChain);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t base_id = instruction.words[3];

    const Value& base_value = GetValue(base_id);
    Type base_type = GetType(base_id);

    assertm (std::holds_alternative<PointerV>(base_value), "SPIRV simulator: Attempt to use OpAccessChain on a non-pointer value");

    PointerV new_pointer = std::get<PointerV>(base_value);
    for(auto i = 4; i < instruction.word_count; ++i){
        const Value& index_value = GetValue(instruction.words[i]);

        // Used to calculate arbitrary offsets
        new_pointer.idx_path_ids.push_back(instruction.words[i]);

        if (std::holds_alternative<uint64_t>(index_value)){
            new_pointer.idx_path.push_back((uint32_t)std::get<uint64_t>(index_value));
        } else if (std::holds_alternative<int64_t>(index_value)){
            new_pointer.idx_path.push_back((uint32_t)std::get<int64_t>(index_value));
        } else {
            assertx ("SPIRV simulator: Index not of integer type in Op_AccessChain");
        }
    }

    if (base_type.pointer.storage_class == spv::StorageClass::StorageClassPhysicalStorageBuffer){
        physical_address_pointers_.push_back(new_pointer);
    }

    const Type& result_type = types_.at(type_id);
    const Type& result_pointee_type = types_.at(result_type.pointer.pointee_type_id);
    if ((result_pointee_type.kind == Type::Kind::Pointer) && (result_pointee_type.pointer.storage_class == spv::StorageClass::StorageClassPhysicalStorageBuffer)){
        // This pointer points to a physical storage buffer pointer
        // This is the semi-easy case where we can extract the location of the physical
        // pointer from this pointer's offsets and storage class, but with the caveat that the resulting pointer
        // is itself stored in a physical storage buffer (hence we need the containing buffer to find its actual address)
        PointerV ppointer = std::get<PointerV>(Deref(new_pointer));
        pointers_to_physical_address_pointers_.push_back(std::pair<PointerV, PointerV>{new_pointer, ppointer});
    }

    SetValue(result_id, new_pointer);
}

void SPIRVSimulator::Op_Function(const Instruction& instruction){
    /*
    OpFunction

    Add a function. This instruction must be immediately followed by one OpFunctionParameter instruction per each
    formal parameter of this function. This function’s body or declaration terminates with the next OpFunctionEnd instruction.

    Result Type must be the same as the Return Type declared in Function Type.

    Function Type is the result of an OpTypeFunction, which declares the types of the return value and parameters of the function.
    */
    assert(instruction.opcode == spv::Op::OpFunction);
    // Nothing to do, we handle this when parsing instructions
}

void SPIRVSimulator::Op_FunctionEnd(const Instruction& instruction){
    // This is a NOP in our design
    assert(instruction.opcode == spv::Op::OpFunctionEnd);
}

void SPIRVSimulator::Op_FunctionCall(const Instruction& instruction){
    /*
    OpFunctionCall

    Call a function.

    Result Type is the type of the return value of the function.
    It must be the same as the Return Type operand of the Function Type operand of the Function operand.

    Function is an OpFunction instruction. This could be a forward reference.

    Argument N is the object to copy to parameter N of Function.

    Note: A forward call is possible because there is no missing type information: Result Type must match the Return
    Type of the function, and the calling argument types must match the formal parameter types.
    */
    assert(instruction.opcode == spv::Op::OpFunctionCall);

    uint32_t result_id = instruction.words[2];
    uint32_t function_id = instruction.words[3];

    FunctionInfo& function_info = funcs_[function_id];
    call_stack_.push_back({function_info.first_inst_index, result_id, {}, {}});

    for (auto i = 4; i < instruction.word_count; ++i){
        // Push parameters to the local scope
        call_stack_.back().locals[function_info.parameter_ids_[i]] = GetValue(instruction.words[i]);
    }
}

void SPIRVSimulator::Op_Label(const Instruction& instruction){
    /*
    OpLabel

    The label instruction of a block.

    References to a block are through the Result <id> of its label.
    */
    assert(instruction.opcode == spv::Op::OpLabel);

    uint32_t result_id = instruction.words[1];
    prev_block_id_ = current_block_id_;
    current_block_id_ = result_id;
}

void SPIRVSimulator::Op_Branch(const Instruction& instruction){
    /*
    OpBranch

    Unconditional branch to Target Label.
    Target Label must be the Result <id> of an OpLabel instruction in the current function.
    This instruction must be the last instruction in a block.
    */
    assert(instruction.opcode == spv::Op::OpBranch);

    uint32_t result_id = instruction.words[1];
    call_stack_.back().pc = result_id_to_inst_index_.at(result_id);
}

void SPIRVSimulator::Op_BranchConditional(const Instruction& instruction){
    /*
    OpBranchConditional

    If Condition is true, branch to True Label, otherwise branch to False Label.
    Condition must be a Boolean type scalar.

    True Label must be an OpLabel in the current function.
    False Label must be an OpLabel in the current function.
    Starting with version 1.6, True Label and False Label must not be the same <id>.
    Branch weights are unsigned 32-bit integer literals.
    There must be either no Branch Weights or exactly two branch weights.
    If present, the first is the weight for branching to True Label, and the second is the
    weight for branching to False Label. The implied probability that a branch is taken is
    its weight divided by the sum of the two Branch weights. At least one weight must be non-zero.
    A weight of zero does not imply a branch is dead or permit its removal; branch weights are only hints.
    The sum of the two weights must not overflow a 32-bit unsigned integer.

    This instruction must be the last instruction in a block.
    */
    assert(instruction.opcode == spv::Op::OpBranchConditional);

    uint64_t condition = std::get<uint64_t>(GetValue(instruction.words[1]));
    call_stack_.back().pc = result_id_to_inst_index_.at(condition ? instruction.words[2] : instruction.words[3]);
}

void SPIRVSimulator::Op_Return(const Instruction& instruction){
    /*
    OpReturn

    Return with no value from a function with void return type.
    This instruction must be the last instruction in a block.
    */
    assert(instruction.opcode == spv::Op::OpReturn);

    call_stack_.pop_back();
}

void SPIRVSimulator::Op_ReturnValue(const Instruction& instruction){
    /*
    OpReturnValue

    Return a value from a function.

    Value is the value returned, by copy, and must match the Return Type operand of the OpTypeFunction
    type of the OpFunction body this return instruction is in. Value must not have type OpTypeVoid.

    This instruction must be the last instruction in a block.
    */
    assert(instruction.opcode == spv::Op::OpReturnValue);

    uint32_t value_id = instruction.words[1];
    uint32_t result_id = call_stack_.back().result_id;
    Value return_value = GetValue(value_id);

    call_stack_.pop_back();

    if (call_stack_.size()){
        SetValue(result_id, return_value);
    }
}

void SPIRVSimulator::Op_FAdd(const Instruction& instruction){
    /*
    OpFAdd

    Floating-point addition of Operand 1 and Operand 2.
    Result Type must be a scalar or vector of floating-point type.
    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFAdd);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm ((std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2)), "SPIRV simulator: Operands not of vector type in Op_FAdd");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands not of equal/correct length in Op_FAdd");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            double elem_result = std::get<double>(vec1->elems[i]) + std::get<double>(vec2->elems[i]);
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Float){
        Value result;
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        assertm ((std::holds_alternative<double>(op1) && std::holds_alternative<double>(op2)), "SPIRV simulator: Operands not of float type in Op_FAdd");

        result = std::get<double>(op1) + std::get<double>(op2);

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type int Op_FAdd, must be vector or float");
    }
}

void SPIRVSimulator::Op_ExtInst(const Instruction& instruction){
    /*
    Execute an instruction in an imported set of extended instructions.

    Result Type is defined, per Instruction, in the external specification for Set.
    Set is the result of an OpExtInstImport instruction.
    Instruction is the enumerant of the instruction to execute within Set.
    It is an unsigned 32-bit integer. The semantics of the instruction are defined in the external specification for Set.

    Operand 1, …​ are the operands to the extended instruction.
    */
    assert(instruction.opcode == spv::Op::OpExtInst);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t set_id = instruction.words[3];
    uint32_t instruction_literal = instruction.words[4];

    assertm (extended_imports_.find(set_id) != extended_imports_.end(), "SPIRV simulator: Unsupported set ID (it has not been imported9) for Op_ExtInst");

    std::string set_literal = extended_imports_[set_id];
    const std::span<const uint32_t> operand_words = std::span<const uint32_t>(instruction.words).subspan(5);
    if (!std::strncmp(set_literal.c_str(), "GLSL.std.450", set_literal.length())){
        GLSLExtHandler(type_id, result_id, instruction_literal, operand_words);
    } else {
        if (verbose_){
            std::cout << execIndent << "OpExtInst set with literal: " << set_literal << " (length: " << set_literal.length() << ") " << " does not exist" << std::endl;
        }
        SetValue(result_id, MakeDefault(type_id));
    }
}


void SPIRVSimulator::Op_SelectionMerge(const Instruction& instruction){
    // This is a NOP in our design
    assert(instruction.opcode == spv::Op::OpSelectionMerge);
}

void SPIRVSimulator::Op_FMul(const Instruction& instruction){
    /*
    OpFMul

    Floating-point multiplication of Operand 1 and Operand 2.
    Result Type must be a scalar or vector of floating-point type.
    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFMul);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm ((std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2)), "SPIRV simulator: Operands not of vector type in Op_FMul");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands not of equal/correct length in Op_FMul");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            double elem_result = std::get<double>(vec1->elems[i]) * std::get<double>(vec2->elems[i]);
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Float){
        Value result;
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        assertm ((std::holds_alternative<double>(op1) && std::holds_alternative<double>(op2)), "SPIRV simulator: Operands are not floats/doubles in Op_FMul");

        result = std::get<double>(op1) * std::get<double>(op2);

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type int Op_FMul, must be vector or float");
    }
}

void SPIRVSimulator::Op_LoopMerge(const Instruction& instruction){
    // This is a NOP in our design
    // TODO: Double check this
    assert(instruction.opcode == spv::Op::OpLoopMerge);
}

void SPIRVSimulator::Op_INotEqual(const Instruction& instruction){
    /*
    OpINotEqual

    Integer comparison for inequality.
    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.
    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpINotEqual);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm ((std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2)), "SPIRV simulator: Operands not of vector type in Op_INotEqual");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands not of equal/correct length in Op_INotEqual");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            uint64_t elem_result;

            // This should compare equal if different types but same number, so cant use variant operators here
            // TODO: Refactor this and the similar blocks below
            if(std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])){
                elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) != std::get<uint64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i])){
                elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) != (uint64_t)std::get<int64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i])){
                elem_result = (uint64_t)(std::get<int64_t>(vec1->elems[i]) != std::get<int64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])){
                elem_result = (uint64_t)((uint64_t)std::get<int64_t>(vec1->elems[i]) != std::get<uint64_t>(vec2->elems[i]));
            } else {
                assertx ("SPIRV simulator: Could not find valid parameter type combination for Op_INotEqual vector operand");
            }

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::BoolT){
        Value result;
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        if(std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2)){
            result = (uint64_t)(std::get<uint64_t>(op1) != std::get<uint64_t>(op2));
        } else if(std::holds_alternative<uint64_t>(op1) && std::holds_alternative<int64_t>(op2)){
            result = (uint64_t)(std::get<uint64_t>(op1) != (uint64_t)std::get<int64_t>(op2));
        } else if(std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2)){
            result = (uint64_t)(std::get<int64_t>(op1) != std::get<int64_t>(op2));
        } else if(std::holds_alternative<int64_t>(op1) && std::holds_alternative<uint64_t>(op2)){
            result = (uint64_t)((uint64_t)std::get<int64_t>(op1) != std::get<uint64_t>(op2));
        } else {
            assertx ("SPIRV simulator: Could not find valid parameter type combination for Op_INotEqual");
        }

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type int Op_IAdd, must be vector or bool");
    }
}

void SPIRVSimulator::Op_IAdd(const Instruction& instruction){
    /*
    OpIAdd

    Integer addition of Operand 1 and Operand 2.

    Result Type must be a scalar or vector of integer type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type. They must have the same number of components
    as Result Type. They must have the same component width as Result Type.

    The resulting value equals the low-order N bits of the correct result R, where N is the component
    width and R is computed with enough precision to avoid overflow and underflow.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpIAdd);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands not of vector type in Op_IAdd");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands not of equal/correct length in Op_IAdd");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            Value elem_result;

            if(std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])){
                elem_result = (std::get<uint64_t>(vec1->elems[i]) + std::get<uint64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i])){
                elem_result = (std::get<uint64_t>(vec1->elems[i]) + std::get<int64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i])){
                elem_result = (std::get<int64_t>(vec1->elems[i]) + std::get<int64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])){
                elem_result = (std::get<int64_t>(vec1->elems[i]) + std::get<uint64_t>(vec2->elems[i]));
            } else {
                assertx ("SPIRV simulator: Could not find valid parameter type combination for Op_IAdd vector operand");
            }

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Int){
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        if(std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2)){
            result = (std::get<uint64_t>(op1) + std::get<uint64_t>(op2));
        } else if(std::holds_alternative<uint64_t>(op1) && std::holds_alternative<int64_t>(op2)){
            result = (std::get<uint64_t>(op1) + std::get<int64_t>(op2));
        } else if(std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2)){
            result = (std::get<int64_t>(op1) + std::get<int64_t>(op2));
        } else if(std::holds_alternative<int64_t>(op1) && std::holds_alternative<uint64_t>(op2)){
            result = (std::get<int64_t>(op1) + std::get<uint64_t>(op2));
        } else {
            assertx ("SPIRV simulator: Could not find valid parameter type combination for Op_IAdd");
        }

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type int Op_IAdd, must be vector or int");
    }
}

void SPIRVSimulator::Op_ISub(const Instruction& instruction){
    /*
    OpISub

    Integer subtraction of Operand 2 from Operand 1.
    Result Type must be a scalar or vector of integer type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same number of components as Result Type. They must have the same component width as Result Type.
    The resulting value equals the low-order N bits of the correct result R, where N is the component width
    and R is computed with enough precision to avoid overflow and underflow.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpISub);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands not of vector type in Op_IAdd");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands not of equal/correct length in Op_IAdd");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            Value elem_result;

            if(std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])){
                elem_result = (std::get<uint64_t>(vec1->elems[i]) - std::get<uint64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i])){
                elem_result = (std::get<uint64_t>(vec1->elems[i]) - std::get<int64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i])){
                elem_result = (std::get<int64_t>(vec1->elems[i]) - std::get<int64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])){
                elem_result = (std::get<int64_t>(vec1->elems[i]) - std::get<uint64_t>(vec2->elems[i]));
            } else {
                assertx ("SPIRV simulator: Could not find valid parameter type combination for Op_ISub vector operand");
            }

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Int){
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        if(std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2)){
            result = (std::get<uint64_t>(op1) - std::get<uint64_t>(op2));
        } else if(std::holds_alternative<uint64_t>(op1) && std::holds_alternative<int64_t>(op2)){
            result = (std::get<uint64_t>(op1) - std::get<int64_t>(op2));
        } else if(std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2)){
            result = (std::get<int64_t>(op1) - std::get<int64_t>(op2));
        } else if(std::holds_alternative<int64_t>(op1) && std::holds_alternative<uint64_t>(op2)){
            result = (std::get<int64_t>(op1) - std::get<uint64_t>(op2));
        } else {
            assertx ("SPIRV simulator: Could not find valid parameter type combination for Op_ISub");
        }

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type int Op_ISub, must be vector or int");
    }
}

void SPIRVSimulator::Op_LogicalNot(const Instruction& instruction){
    /*
    OpLogicalNot

    Result is true if Operand is false. Result is false if Operand is true.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand must be the same as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpLogicalNot);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand_id = instruction.words[3];

    const Type& type = types_.at(type_id);
    const Value& operand = GetValue(operand_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(operand), "SPIRV simulator: Invalid value type, must be vector when using vector type");

        auto vec = std::get<std::shared_ptr<VectorV>>(operand);

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            assertm (std::holds_alternative<uint64_t>(vec->elems[i]), "SPIRV simulator: Non-boolean type found in vector operand");
            result_vec->elems.push_back((uint64_t)!(std::get<uint64_t>(vec->elems[i])));
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::BoolT){
        Value result;

        assertm (std::holds_alternative<uint64_t>(operand), "SPIRV simulator: Non-boolean type found in operand");
        result = (uint64_t)!(std::get<uint64_t>(operand));

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type, must be vector or bool");
    }
}

void SPIRVSimulator::Op_Capability(const Instruction& instruction) {
    // This is a NOP in our design
    assert(instruction.opcode == spv::Op::OpCapability);
}

void SPIRVSimulator::Op_Extension(const Instruction& instruction) {
    // This is a NOP in our design (at least for now)
    assert(instruction.opcode == spv::Op::OpExtension);
}

void SPIRVSimulator::Op_MemoryModel(const Instruction& instruction) {
    // This is a NOP in our design
    assert(instruction.opcode == spv::Op::OpMemoryModel);
}

void SPIRVSimulator::Op_ExecutionMode(const Instruction& instruction) {
    // We may need this later
    assert(instruction.opcode == spv::Op::OpExecutionMode);
}

void SPIRVSimulator::Op_Source(const Instruction& instruction) {
    // This is a NOP in our design
    assert(instruction.opcode == spv::Op::OpSource);
}

void SPIRVSimulator::Op_SourceExtension(const Instruction& instruction) {
    // This is a NOP in our design
    assert(instruction.opcode == spv::Op::OpSourceExtension);
}

void SPIRVSimulator::Op_Name(const Instruction& instruction) {
    /*
    OpName

    Assign a name string to another instruction’s Result <id>.
    This has no semantic impact and can safely be removed from a module.

    Target is the Result <id> to assign a name to.
    It can be the Result <id> of any other instruction; a variable, function, type, intermediate result, etc.

    Name is the string to assign.
    */
    assert(instruction.opcode == spv::Op::OpName);

    uint32_t target_id = instruction.words[1];

    std::string label = std::string((char*)(&instruction.words[2]), (instruction.word_count - 2) * 4);
    label.erase(std::find(label.begin(), label.end(), '\0'), label.end());

    if (entry_points_.find(target_id) != entry_points_.end()){
        entry_points_[target_id] = label;
    }
}

void SPIRVSimulator::Op_MemberName(const Instruction& instruction) {
    // We could use this for debug info later, for now we leave it as a NOP
    assert(instruction.opcode == spv::Op::OpMemberName);
}

void SPIRVSimulator::Op_Decorate(const Instruction& instruction) {
    /*
    OpDecorate

    Add a Decoration to another <id>.

    Target is the <id> to decorate. It can potentially be any <id> that is a forward reference.
    A set of decorations can be grouped together by having multiple decoration instructions targeting the same
    OpDecorationGroup instruction.

    This instruction is only valid if the Decoration operand is a decoration that takes no Extra Operands, or takes
    Extra Operands that are not <id> operands.
    */
    assert(instruction.opcode == spv::Op::OpDecorate);

    uint32_t target_id = instruction.words[1];
    spv::Decoration kind = static_cast<spv::Decoration>(instruction.words[2]);

    std::vector<uint32_t> literals;
    for (uint32_t i = 3; i < instruction.word_count; ++i){
        literals.push_back(instruction.words[i]);
    }

    DecorationInfo info{kind, std::move(literals)};
    decorators_[target_id].emplace_back(std::move(info));
}

void SPIRVSimulator::Op_MemberDecorate(const Instruction& instruction) {
    /*
    OpMemberDecorate

    Add a Decoration to a member of a structure type.
    Structure type is the <id> of a type from OpTypeStruct.
    Member is the number of the member to decorate in the type. The first member is member 0, the next is member 1, …​

    Note: See OpDecorate for creating groups of decorations for consumption by OpGroupMemberDecorate
    */
    assert(instruction.opcode == spv::Op::OpMemberDecorate);

    uint32_t structure_type_id = instruction.words[1];
    uint32_t member_literal = instruction.words[2];
    spv::Decoration kind = static_cast<spv::Decoration>(instruction.words[3]);

    std::vector<uint32_t> literals;
    for (uint32_t i = 4; i < instruction.word_count; ++i){
        literals.push_back(instruction.words[i]);
    }

    DecorationInfo info{kind, std::move(literals)};
    struct_decorators_[structure_type_id][member_literal].emplace_back(std::move(info));
}

void SPIRVSimulator::Op_ArrayLength(const Instruction& instruction){
    /*
    OpArrayLength

    Length of a run-time array.
    Result Type must be an OpTypeInt with 32-bit Width and 0 Signedness.

    Structure must be a logical pointer to an OpTypeStruct whose last member is a run-time array.
    Array member is an unsigned 32-bit integer index of the last member of the structure that Structure points to.
    That member’s type must be from OpTypeRuntimeArray.
    */
    assert(instruction.opcode == spv::Op::OpArrayLength);

    //uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    //uint32_t structure_id = instruction.words[3];
    //uint32_t literal_array_member = instruction.words[4];

    // TODO: Must query input data here to find the length
    //       Should be enough to check the binding of the result_id and the size of
    //       the mapped data vector (in number of elements encoded)
    assertx ("SPIRV simulator: Op_ArrayLength is unimplemented! Fix this.");

    SetValue(result_id, 0);
}

void SPIRVSimulator::Op_SpecConstant(const Instruction& instruction) {
    /*
    OpSpecConstant

    Declare a new integer-type or floating-point-type scalar specialization constant.
    Result Type must be a scalar integer type or floating-point type.
    Value is the bit pattern for the default value of the constant. Types 32 bits wide or smaller take one word.
    Larger types take multiple words, with low-order words appearing first.
    This instruction can be specialized to become an OpConstant instruction.

    See Specialization.
    */
    assert(instruction.opcode == spv::Op::OpSpecConstant);

    uint32_t result_id = instruction.words[2];
    assertm (HasDecorator(result_id, spv::Decoration::DecorationSpecId), "SPIRV simulator: Op_SpecConstant type is not decorated with SpecId");

    Op_Constant(instruction);
}

void SPIRVSimulator::Op_SpecConstantOp(const Instruction& instruction) {
    /*
    OpSpecConstantOp

    Declare a new specialization constant that results from doing an operation.
    Result Type must be the type required by the Result Type of Opcode.

    Opcode is an unsigned 32-bit integer. It must equal one of the following opcodes.
    OpSConvert, OpUConvert (missing before version 1.4), OpFConvert
    OpSNegate, OpNot, OpIAdd, OpISub
    OpIMul, OpUDiv, OpSDiv, OpUMod, OpSRem, OpSMod
    OpShiftRightLogical, OpShiftRightArithmetic, OpShiftLeftLogical
    OpBitwiseOr, OpBitwiseXor, OpBitwiseAnd
    OpVectorShuffle, OpCompositeExtract, OpCompositeInsert
    OpLogicalOr, OpLogicalAnd, OpLogicalNot,
    OpLogicalEqual, OpLogicalNotEqual
    OpSelect
    OpIEqual, OpINotEqual
    OpULessThan, OpSLessThan
    OpUGreaterThan, OpSGreaterThan
    OpULessThanEqual, OpSLessThanEqual
    OpUGreaterThanEqual, OpSGreaterThanEqual

    If the Shader capability was declared, OpQuantizeToF16 is also valid.

    If the Kernel capability was declared, the following opcodes are also valid:
    OpConvertFToS, OpConvertSToF
    OpConvertFToU, OpConvertUToF
    OpUConvert, OpConvertPtrToU, OpConvertUToPtr
    OpGenericCastToPtr, OpPtrCastToGeneric, OpBitcast
    OpFNegate, OpFAdd, OpFSub, OpFMul, OpFDiv, OpFRem, OpFMod
    OpAccessChain, OpInBoundsAccessChain
    OpPtrAccessChain, OpInBoundsPtrAccessChain

    Operands are the operands required by opcode, and satisfy the semantics of opcode.
    In addition, all Operands that are <id>s must be either:
    - the <id>s of other constant instructions, or
    - OpUndef, when allowed by opcode, or
    - for the AccessChain named opcodes, their Base is allowed to be a global (module scope) OpVariable instruction.

    See Specialization.
    */
    assert(instruction.opcode == spv::Op::OpSpecConstantOp);

    uint32_t result_id = instruction.words[2];

    // TODO: Double check this after thoroughly reading the spec.
    if (spec_instructions_.find(result_id) == spec_instructions_.end()){
        uint32_t type_id = instruction.words[1];
        uint32_t opcode = instruction.words[3];

        auto& spec_instr_words = spec_instr_words_[result_id];

        Instruction spec_instruction;
        spec_instruction.opcode = (spv::Op)opcode;
        spec_instruction.word_count = instruction.words.size() - 1;

        uint32_t header_word = (spec_instruction.word_count << kWordCountShift) | spec_instruction.opcode;
        spec_instr_words.push_back(header_word);
        spec_instr_words.push_back(type_id);
        spec_instr_words.push_back(result_id);

        for (uint32_t operand_index = 4; operand_index < instruction.word_count; ++operand_index){
            spec_instr_words.push_back(instruction.words[operand_index]);
        }

        spec_instruction.words = std::span<const uint32_t>{spec_instr_words.data(), spec_instr_words.size()};
        spec_instructions_[result_id] = spec_instruction;
    }

    if (verbose_){
        PrintInstruction(spec_instructions_[result_id]);
    }

    ExecuteInstruction(spec_instructions_[result_id]);
}

void SPIRVSimulator::Op_SpecConstantComposite(const Instruction& instruction) {
    /*
    OpSpecConstantComposite

    Declare a new composite specialization constant.
    Result Type must be a composite type, whose top-level members/elements/components/columns have the
    same type as the types of the Constituents. The ordering must be the same between the top-level types in Result Type and the Constituents.
    Constituents become members of a structure, or elements of an array, or components of a vector, or columns of a matrix.
    There must be exactly one Constituent for each top-level member/element/component/column of the result.
    The Constituents must appear in the order needed by the definition of the type of the result.
    The Constituents must be the <id> of other specialization constants, constant declarations, or an OpUndef.
    This instruction will be specialized to an OpConstantComposite instruction.

    See Specialization.
    */
    assert(instruction.opcode == spv::Op::OpSpecConstantComposite);
    Op_ConstantComposite(instruction);
}

void SPIRVSimulator::Op_UGreaterThanEqual(const Instruction& instruction){
    /*
    OpUGreaterThanEqual

    Unsigned-integer comparison if Operand 1 is greater than or equal to Operand 2.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type. They must have the same component width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpUGreaterThanEqual);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    const Type& type = types_.at(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type in Op_UGreaterThanEqual, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length in Op_UGreaterThanEqual");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            assertm (std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec1->elems[i]), "SPIRV simulator: Found non-unsigned integer operand in Op_UGreaterThanEqual vector operands");

            Value elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) >= std::get<uint64_t>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);

    } else if (type.kind == Type::Kind::BoolT){
        Value result = (uint64_t)(std::get<uint64_t>(val_op1) >= std::get<uint64_t>(val_op2));
        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type in Op_UGreaterThanEqual, must be vector or bool");
    }
}

void SPIRVSimulator::Op_Phi(const Instruction& instruction){
    /*

    OpPhi

    The SSA phi function.
    The result is selected based on control flow: If control reached the current block from Parent i, Result Id gets
    the value that Variable i had at the end of Parent i.

    Result Type can be any type except OpTypeVoid.

    Operands are a sequence of pairs: (Variable 1, Parent 1 block), (Variable 2, Parent 2 block), …​
    Each Parent i block is the label of an immediate predecessor in the CFG of the current block.
    There must be exactly one Parent i for each parent block of the current block in the CFG.
    If Parent i is reachable in the CFG and Variable i is defined in a block, that defining block must dominate Parent i.
    All Variables must have a type matching Result Type.

    Within a block, this instruction must appear before all non-OpPhi instructions (except for OpLine and OpNoLine, which can
    be mixed with OpPhi).
    */
    assert(instruction.opcode == spv::Op::OpPhi);

    //uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    for (uint32_t operand_index = 3; operand_index < instruction.word_count; operand_index += 2){
        uint32_t variable_id = instruction.words[operand_index];
        uint64_t block_id = instruction.words[operand_index + 1];

        if (block_id == prev_block_id_){
            SetValue(result_id, GetValue(variable_id));
            return;
        }
    }

    assertx ("SPIRV simulator: Op_Phi faield to find a valid source block ID, something is broken in the control flow handling.");
}

void SPIRVSimulator::Op_ConvertUToF(const Instruction& instruction){
    /*
    OpConvertUToF

    Convert value numerically from unsigned integer to floating point.
    Result Type must be a scalar or vector of floating-point type.
    Unsigned Value must be a scalar or vector of integer type. It must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpConvertUToF);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t value_id = instruction.words[3];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op = GetValue(value_id);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op), "SPIRV simulator: Operand set to be vector type in OpConvertUToF, but it is not, illegal input parameters");

        auto vec = std::get<std::shared_ptr<VectorV>>(val_op);

        assertm (vec->elems.size() == type.vector.elem_count, "SPIRV simulator: Operands are vector type but not of valid length in OpConvertUToF");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            assertm (std::holds_alternative<uint64_t>(vec->elems[i]), "SPIRV simulator: Found non-unsigned integer operand in OpConvertUToF vector operands");

            Value elem_result = (double)std::get<uint64_t>(vec->elems[i]);

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Float){
        const Value& op = GetValue(value_id);

        assertm (std::holds_alternative<uint64_t>(op), "SPIRV simulator: Found non-unsigned integer operand in OpConvertUToF");

        Value result = (double)std::get<uint64_t>(op);
        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid return type in OpConvertUToF, must be vector or float");
    }
}

void SPIRVSimulator::Op_ConvertSToF(const Instruction& instruction){
    /*
    OpConvertSToF

    Convert value numerically from signed integer to floating point.
    Result Type must be a scalar or vector of floating-point type.
    Signed Value must be a scalar or vector of integer type. It must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpConvertSToF);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t value_id = instruction.words[3];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op = GetValue(value_id);
        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op), "SPIRV simulator: Operand set to be vector type in Op_ConvertSToF, but it is not, illegal input parameters");

        auto vec = std::get<std::shared_ptr<VectorV>>(val_op);

        assertm (vec->elems.size() == type.vector.elem_count, "SPIRV simulator: Operands are vector type but not of valid length in Op_ConvertSToF");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            assertm (std::holds_alternative<int64_t>(vec->elems[i]), "SPIRV simulator: Found non-signed integer operand in Op_ConvertSToF vector operands");

            Value elem_result = (double)std::get<int64_t>(vec->elems[i]);

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Float){
        const Value& op = GetValue(value_id);

        assertm (std::holds_alternative<int64_t>(op), "SPIRV simulator: Found non-signed integer operand in Op_ConvertSToF");

        Value result = (double)std::get<int64_t>(op);
        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type in Op_ConvertSToF, must be vector or float");
    }
}


void SPIRVSimulator::Op_FDiv(const Instruction& instruction){
    /*
    OpFDiv

    Floating-point division of Operand 1 divided by Operand 2.

    Result Type must be a scalar or vector of floating-point type.
    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFDiv);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm ((std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2)), "SPIRV simulator: Operands set to be vector type in Op_FDiv, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length in Op_FDiv");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            Value elem_result;

            assertm (std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]), "SPIRV simulator: Found non-floating point operand in Op_FDiv vector operands");

            elem_result = std::get<double>(vec1->elems[i]) / std::get<double>(vec2->elems[i]);

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Float){
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        assertm (std::holds_alternative<double>(op1) && std::holds_alternative<double>(op2), "SPIRV simulator: Found non-floating point operand in Op_FDiv");

        result = std::get<double>(op1) / std::get<double>(op2);

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type int Op_FDiv, must be vector or float");
    }
}

void SPIRVSimulator::Op_FSub(const Instruction& instruction){
    /*
    OpFSub

    Floating-point subtraction of Operand 2 from Operand 1.
    Result Type must be a scalar or vector of floating-point type.
    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFSub);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type in Op_FSub, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length in Op_FSub");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            Value elem_result;

            assertm (std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]), "SPIRV simulator: Found non-floating point operand in Op_FSub vector operands");

            elem_result = std::get<double>(vec1->elems[i]) - std::get<double>(vec2->elems[i]);

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Float){
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        assertm (std::holds_alternative<double>(op1) && std::holds_alternative<double>(op2), "SPIRV simulator: Found non-floating point operand in Op_FSub");

        result = std::get<double>(op1) - std::get<double>(op2);

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type int Op_FSub, must be vector or float");
    }
}

void SPIRVSimulator::Op_VectorTimesScalar(const Instruction& instruction){
    /*
    OpVectorTimesScalar

    Scale a floating-point vector.
    Result Type must be a vector of floating-point type.
    The type of Vector must be the same as Result Type. Each component of Vector is multiplied by Scalar.

    Scalar must have the same type as the Component Type in Result Type.
    */
    assert(instruction.opcode == spv::Op::OpVectorTimesScalar);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t vector_id = instruction.words[3];
    uint32_t scalar_id = instruction.words[4];

    const Type& type = types_.at(type_id);

    Value result = std::make_shared<VectorV>();
    auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

    const Value& vec_operand = GetValue(vector_id);
    assertm (std::holds_alternative<std::shared_ptr<VectorV>>(vec_operand), "SPIRV simulator: Found non-vector operand in Op_VectorTimesScalar");
    auto vec = std::get<std::shared_ptr<VectorV>>(vec_operand);

    const Value& scalar_operand = GetValue(scalar_id);
    assertm (std::holds_alternative<double>(scalar_operand), "SPIRV simulator: Found non-floating point operand in Op_VectorTimesScalar");
    double scalar_value = std::get<double>(scalar_operand);

    for (uint32_t i = 0; i < type.vector.elem_count; ++i){
        Value elem_result;

        assertm (std::holds_alternative<double>(vec->elems[i]), "SPIRV simulator: Found non-floating point operand in Op_VectorTimesScalar vector operands");

        elem_result = std::get<double>(vec->elems[i]) * scalar_value;

        result_vec->elems.push_back(elem_result);
    }

    SetValue(result_id, result);
}

void SPIRVSimulator::Op_SLessThan(const Instruction& instruction){
    /*
    OpSLessThan

    Signed-integer comparison if Operand 1 is less than Operand 2.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpSLessThan);

    // No explicit requirement for ints to be signed? Assume they have to be for now (but detect if they aint)
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type in Op_SLessThan, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length in Op_SLessThan");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            Value elem_result;

            assertm (std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]), "SPIRV simulator: Found non-signed integer operand in Op_SLessThan vector operands");

            elem_result = (uint64_t)(std::get<int64_t>(vec1->elems[i]) < std::get<int64_t>(vec2->elems[i]));

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::BoolT){
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        assertm (std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2), "SPIRV simulator: Found non-signed integer operand in Op_SLessThan");

        result = (uint64_t)(std::get<int64_t>(op1) < std::get<int64_t>(op2));

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type int Op_SLessThan, must be vector or bool");
    }
}

void SPIRVSimulator::Op_Dot(const Instruction& instruction){
    /*
    OpDot

    Dot product of Vector 1 and Vector 2.
    Result Type must be a floating-point type scalar.
    Vector 1 and Vector 2 must be vectors of the same type, and their component type must be Result Type.
    */
    assert(instruction.opcode == spv::Op::OpDot);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Float){
        double result = 0.0;

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands not of vector type in Op_Dot");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm (vec1->elems.size() == vec2->elems.size(), "SPIRV simulator: Operands not of equal/correct length in Op_Dot");

        for (uint32_t i = 0; i < vec1->elems.size(); ++i){
            assertm (std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]), "SPIRV simulator: Found non-floating point operand in Op_Dot vector operands");

            result += std::get<double>(vec1->elems[i]) * std::get<double>(vec2->elems[i]);
        }

        Value val_result = result;
        SetValue(result_id, val_result);
    }  else {
        assertx ("SPIRV simulator: Invalid result type int Op_Dot, must be float");
    }
}

void SPIRVSimulator::Op_FOrdGreaterThan(const Instruction& instruction){
    /*
    OpFOrdGreaterThan

    Floating-point comparison if operands are ordered and Operand 1 is greater than Operand 2.
    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of floating-point type.
    They must have the same type, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFOrdGreaterThan);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    const Type& type = types_.at(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type in Op_UGreaterThanEqual, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length in Op_UGreaterThanEqual");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            assertm (std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]), "SPIRV simulator: Found non-floating point operand in Op_FOrdGreaterThan vector operands");

            Value elem_result = (uint64_t)(std::get<double>(vec1->elems[i]) > std::get<double>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::BoolT){
        Value result = (uint64_t)(std::get<double>(val_op1) > std::get<double>(val_op2));
        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type in Op_FOrdGreaterThan, must be vector or float");
    }
}

void SPIRVSimulator::Op_CompositeExtract(const Instruction& instruction){
    /*
    OpCompositeExtract

    Extract a part of a composite object.
    Result Type must be the type of object selected by the last provided index. The instruction result is the extracted object.
    Composite is the composite to extract from.

    Indexes walk the type hierarchy, potentially down to component granularity, to select the part to extract.
    All indexes must be in bounds. All composite constituents use zero-based numbering, as described by their OpType…​ instruction.
    Each index is an unsigned 32-bit integer.
    */
    assert(instruction.opcode == spv::Op::OpCompositeExtract);

    //uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t composite_id = instruction.words[3];

    Value* current_composite = &(GetValue(composite_id));
    for (uint32_t i = 4; i < instruction.word_count; ++i){
        uint32_t literal_index = instruction.words[i];

        if(std::holds_alternative<std::shared_ptr<AggregateV>>(*current_composite)){
            auto agg = std::get<std::shared_ptr<AggregateV>>(*current_composite);

            assertm (literal_index < agg->elems.size(), "SPIRV simulator: Aggregate index OOB");

            current_composite = &agg->elems[literal_index];
        } else if (std::holds_alternative<std::shared_ptr<VectorV>>(*current_composite)){
            auto vec = std::get<std::shared_ptr<VectorV>>(*current_composite);

            assertm (literal_index < vec->elems.size(), "SPIRV simulator: Vector index OOB");

            current_composite = &vec->elems[literal_index];
        } else if (std::holds_alternative<std::shared_ptr<MatrixV>>(*current_composite)){
            auto matrix = std::get<std::shared_ptr<MatrixV>>(*current_composite);

            assertm (literal_index < matrix->cols.size(), "SPIRV simulator: Matrix index OOB");

            current_composite = &matrix->cols[literal_index];
        }
        else{
            assertx ("SPIRV simulator: Pointer dereference into non-composite object");
        }
    }

    SetValue(result_id, *current_composite);
}

void SPIRVSimulator::Op_Bitcast(const Instruction& instruction){
    /*
    OpBitcast

    Bit pattern-preserving type conversion.

    Result Type must be an OpTypePointer, or a scalar or vector of numerical-type.

    Operand must have a type of OpTypePointer, or a scalar or vector of numerical-type.
    It must be a different type than Result Type.

    Before version 1.5: If either Result Type or Operand is a pointer, the other must be a pointer or an integer scalar.
    Starting with version 1.5: If either Result Type or Operand is a pointer, the other must be a pointer,
    an integer scalar, or an integer vector.

    If both Result Type and the type of Operand are pointers, they both must point into same storage class.

    Behavior is undefined if the storage class of Result Type does not match the one used by the operation that
    produced the value of Operand.

    If Result Type has the same number of components as Operand, they must also have the same component width,
    and results are computed per component.

    If Result Type has a different number of components than Operand, the total number of bits in Result Type must
    equal the total number of
    bits in Operand.

    Let L be the type, either Result Type or Operand’s type, that has the larger number of components. Let S be the other type,
    with the smaller number of components. The number of components in L must be an integer multiple of the number of components in S.
    The first component (that is, the only or lowest-numbered component) of S maps to the first components of L,
    and so on, up to the last component of S mapping to the last components of L.
    Within this mapping, any single component of S (mapping to multiple components of L) maps
    its lower-ordered bits to the lower-numbered components of L.
    */
    assert(instruction.opcode == spv::Op::OpBitcast);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand_id = instruction.words[3];

    const Value& operand = GetValue(operand_id);
    Type operand_type = GetType(operand_id);

    const Type& type = types_.at(type_id);

    //
    // First, we extract all the data from the operands into a vector
    //
    std::vector<std::byte> bytes;
    if (std::holds_alternative<std::shared_ptr<VectorV>>(operand)){
        const Type& elem_type = types_.at(operand_type.vector.elem_type_id);
        std::shared_ptr<VectorV> vec = std::get<std::shared_ptr<VectorV>>(operand);
        for (const Value& element : vec->elems){
            if (std::holds_alternative<double>(element)){
                double value = std::get<double>(element);
                extract_bytes<double>(bytes, value, elem_type.scalar.width);
            } else if (std::holds_alternative<uint64_t>(element)){
                uint64_t value = std::get<uint64_t>(element);
                extract_bytes<uint64_t>(bytes, value, elem_type.scalar.width);
            } else if (std::holds_alternative<int64_t>(element)){
                int64_t value = std::get<int64_t>(element);
                extract_bytes<int64_t>(bytes, value, elem_type.scalar.width);
            } else {
                assertx ("SPIRV simulator: invalid operand element type in Op_Bitcast, must be numeric");
            }
        }
    } else if (std::holds_alternative<double>(operand)){
        double value = std::get<double>(operand);
        extract_bytes<double>(bytes, value, operand_type.scalar.width);
    } else if (std::holds_alternative<uint64_t>(operand)){
        uint64_t value = std::get<uint64_t>(operand);
        extract_bytes<uint64_t>(bytes, value, operand_type.scalar.width);
    } else if (std::holds_alternative<int64_t>(operand)){
        int64_t value = std::get<int64_t>(operand);
        extract_bytes<int64_t>(bytes, value, operand_type.scalar.width);
    } else if (std::holds_alternative<PointerV>(operand)){
        // Take the easy out if its just pointer to pointer conversion
        if (type.kind == Type::Kind::Pointer){
            SetValue(result_id, operand);
            return;
        }
        // We currently dont handle this, we could do it by storing the pointer in a
        // special container and storing a index into that container in the result here
        assertx ("SPIRV simulator: Pointer to non-pointer Op_Bitcast detected, must add support for this!");
    } else {
        assertx ("SPIRV simulator: invalid operand type in Op_Bitcast, must be vector or numeric");
    }

    //
    // Then we map this memory to the result value
    //
    Value result;
    if (type.kind == Type::Kind::Vector){
        const Type& elem_type = types_.at(type.vector.elem_type_id);
        uint32_t elem_size_bytes = elem_type.scalar.width / 8;
        std::shared_ptr<VectorV> vec = std::get<std::shared_ptr<VectorV>>(result);
        uint32_t current_byte = 0;

        for (Value& element : vec->elems){
            if (std::holds_alternative<double>(element)){
                double value;
                std::memcpy(&value, &(bytes[current_byte]), elem_size_bytes);
                element = value;
            } else if (std::holds_alternative<uint64_t>(element)){
                uint64_t value;
                std::memcpy(&value, &(bytes[current_byte]), elem_size_bytes);
                element = value;
            } else if (std::holds_alternative<int64_t>(element)){
                int64_t value;
                std::memcpy(&value, &(bytes[current_byte]), elem_size_bytes);
                element = value;
            } else {
                assertx ("SPIRV simulator: invalid result element type in Op_Bitcast, must be numeric");
            }

            current_byte += elem_size_bytes;
        }
    } else if (type.kind == Type::Kind::Float){
        double value;
        std::memcpy(&value, bytes.data(), type.scalar.width / 8);
        result = value;
    } else if ((type.kind == Type::Kind::Int) && !type.scalar.is_signed){
        uint64_t value;
        std::memcpy(&value, bytes.data(), type.scalar.width / 8);
        result = value;
    } else if ((type.kind == Type::Kind::Int) && type.scalar.is_signed){
        int64_t value;
        std::memcpy(&value, bytes.data(), type.scalar.width / 8);
        result = value;
    } else if (type.kind == Type::Kind::Pointer) {
        // This is one of the main cases we want to detect, a non-pointer type is cast to a pointer
        // If the storage class is PhysicalStorageBuffer we map it to an external address handle
        // In turn, this can be used in combination with inputs to read from the pbuffer

        // This is unhandled (and probably illegal?)
        assertm (type.pointer.storage_class == spv::StorageClass::StorageClassPhysicalStorageBuffer, "SPIRV simulator: Attempt to Op_Bitcast to a non PhysicalStorageBuffer storage class object");

        uint64_t pointer_value;
        std::memcpy(&pointer_value, bytes.data(), sizeof(uint64_t));

        const std::byte* remapped_pointer = nullptr;

        for (const auto& map_entry : input_data_.physical_address_buffers){
            uint64_t buffer_address = map_entry.first;
            size_t buffer_size = map_entry.second.first;

            const std::byte* buffer_data = static_cast<std::byte*>(map_entry.second.second);

            if ((pointer_value >= buffer_address) && (pointer_value < (buffer_address + buffer_size))){
                remapped_pointer = &(buffer_data[buffer_address - pointer_value]);
                break;
            }
        }

        Value init;
        if (remapped_pointer){
            std::vector<uint32_t> buffer_data;
            ExtractWords(remapped_pointer, type.pointer.pointee_type_id, buffer_data);
            const uint32_t* buffer_pointer = buffer_data.data();
            init = MakeDefault(type.pointer.pointee_type_id, &buffer_pointer);
        } else {
            init = MakeDefault(type.pointer.pointee_type_id);
        }

        Heap(type.pointer.storage_class)[result_id] = init;

        PointerV new_pointer{result_id, type_id, type.pointer.storage_class, pointer_value, {}, {}};
        physical_address_pointers_.push_back(new_pointer);
        result = new_pointer;

        // Here we need to find the source of the values that eventually became the pointer above
        // so that any tool using the simulator can extract and deal with them.
        PhysicalAddressData pointer_data;
        pointer_data.bit_components = FindDataSourcesFromResultID(operand_id);
        pointer_data.raw_pointer_value = pointer_value;
        physical_address_pointer_source_data_.push_back(std::move(pointer_data));
    } else {
        assertx ("SPIRV simulator: invalid result type in Op_Bitcast, must be vector, pointer or numeric");
    }

    SetValue(result_id, result);
}

void SPIRVSimulator::Op_IMul(const Instruction& instruction){
    /*
    OpIMul

    Integer multiplication of Operand 1 and Operand 2.
    Result Type must be a scalar or vector of integer type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same number of components as Result Type. They must have the same component width as Result Type.

    The resulting value equals the low-order N bits of the correct result R, where N is the component width and R is computed with enough
    precision to avoid overflow and underflow.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpIMul);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands not of vector type in Op_IMul");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands not of equal/correct length in Op_IMul");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            uint64_t elem_result;

            if(std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])){
                elem_result = (std::get<uint64_t>(vec1->elems[i]) * std::get<uint64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i])){
                elem_result = (std::get<int64_t>(vec1->elems[i]) * std::get<int64_t>(vec2->elems[i]));
            } else {
                assertx ("SPIRV simulator: Could not find valid parameter type combination for Op_IMul vector operand");
            }

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Int){
        Value result;
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        if(std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2)){
            result = (std::get<uint64_t>(op1) * std::get<uint64_t>(op2));
        } else if(std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2)){
            result = (std::get<int64_t>(op1) * std::get<int64_t>(op2));
        } else {
            assertx ("SPIRV simulator: Could not find valid parameter type combination for Op_IMul");
        }

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type int Op_IMul, must be vector or integer type");
    }
}


void SPIRVSimulator::Op_ConvertUToPtr(const Instruction& instruction){
    /*
    OpConvertUToPtr

    Bit pattern-preserving conversion of an unsigned scalar integer to a pointer.

    Result Type must be a physical pointer type.

    Integer Value must be a scalar of integer type, whose Signedness operand is 0. If the bit width of
    Integer Value is smaller than that of Result Type, the conversion zero extends Integer Value.
    If the bit width of Integer Value is larger than that of Result Type, the conversion truncates Integer Value.
    For same-width Integer Value and Result Type, this is the same as OpBitcast.

    Behavior is undefined if the storage class of Result Type does not match the one used by the operation
    that produced the value of Integer Value.
    */
    assert(instruction.opcode == spv::Op::OpConvertUToPtr);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t integer_id = instruction.words[3];

    const Type& type = types_.at(type_id);
    const Value& operand = GetValue(integer_id);

    // This is unhandled (and probably illegal?)
    assertm (type.pointer.storage_class == spv::StorageClass::StorageClassPhysicalStorageBuffer, "SPIRV simulator: Attempt to Op_ConvertUToPtr to a non PhysicalStorageBuffer storage class object");

    uint64_t pointer_value = std::get<uint64_t>(operand);

    const std::byte* remapped_pointer = nullptr;

    for (const auto& map_entry : input_data_.physical_address_buffers){
        uint64_t buffer_address = map_entry.first;
        size_t buffer_size = map_entry.second.first;

        const std::byte* buffer_data = static_cast<std::byte*>(map_entry.second.second);

        if ((pointer_value >= buffer_address) && (pointer_value < (buffer_address + buffer_size))){
            remapped_pointer = &(buffer_data[buffer_address - pointer_value]);
            break;
        }
    }

    Value init;
    if (remapped_pointer){
        std::vector<uint32_t> buffer_data;
        ExtractWords(remapped_pointer, type.pointer.pointee_type_id, buffer_data);
        const uint32_t* buffer_pointer = buffer_data.data();
        init = MakeDefault(type.pointer.pointee_type_id, &buffer_pointer);
    } else {
        init = MakeDefault(type.pointer.pointee_type_id);
    }

    Heap(type.pointer.storage_class)[result_id] = init;

    PointerV new_pointer{result_id, type_id, type.pointer.storage_class, pointer_value, {}, {}};
    physical_address_pointers_.push_back(new_pointer);
    SetValue(result_id, new_pointer);

    // Here we need to find the source of the values that eventually became the pointer above
    // so that any tool using the simulator can extract and deal with them.
    PhysicalAddressData pointer_data;
    pointer_data.bit_components = FindDataSourcesFromResultID(integer_id);
    pointer_data.raw_pointer_value = pointer_value;
    physical_address_pointer_source_data_.push_back(std::move(pointer_data));
}

void SPIRVSimulator::Op_UDiv(const Instruction& instruction){
    /*
    OpUDiv

    Unsigned-integer division of Operand 1 divided by Operand 2.
    Result Type must be a scalar or vector of integer type, whose Signedness operand is 0.
    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component. Behavior is undefined if Operand 2 is 0.
    */
    assert(instruction.opcode == spv::Op::OpUDiv);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);
    const Value& val_op1 = GetValue(instruction.words[3]);
    const Value& val_op2 = GetValue(instruction.words[4]);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            Value elem_result;

            // TODO: Operands dont have to be unsigned, deal with it and remove the asserts
            assertm (std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i]), "SPIRV simulator: Found non-unsigned int operand vector operands");

            elem_result = std::get<uint64_t>(vec1->elems[i]) / std::get<uint64_t>(vec2->elems[i]);

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Int){
        // TODO: Operands dont have to be unsigned, deal with it and remove the asserts
        assertm (std::holds_alternative<uint64_t>(val_op1) && std::holds_alternative<uint64_t>(val_op2), "SPIRV simulator: Found non-unsigned int operand");

        Value result = std::get<uint64_t>(val_op1) / std::get<uint64_t>(val_op2);

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type, must be vector or unsigned-integer");
    }
}

void SPIRVSimulator::Op_UMod(const Instruction& instruction){
    /*
    OpUMod

    Unsigned modulo operation of Operand 1 modulo Operand 2.
    Result Type must be a scalar or vector of integer type, whose Signedness operand is 0.
    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component. Behavior is undefined if Operand 2 is 0.
    */
    assert(instruction.opcode == spv::Op::OpUMod);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            Value elem_result;

            assertm (std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i]), "SPIRV simulator: Found non-unsigned int operand in vector operands");

            elem_result = std::get<uint64_t>(vec1->elems[i]) % std::get<uint64_t>(vec2->elems[i]);

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Int){
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        assertm (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2), "SPIRV simulator: Found non-unsigned int operand");

        result = std::get<uint64_t>(op1) % std::get<uint64_t>(op2);

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type, must be vector or unsigned-integer");
    }
}

void SPIRVSimulator::Op_ULessThan(const Instruction& instruction){
    /*
    OpULessThan

    Unsigned-integer comparison if Operand 1 is less than Operand 2.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type. They must have the same component width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpULessThan);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            Value elem_result;

            assertm (std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i]), "SPIRV simulator: Found non-unsigned integer operand in vector operands");

            elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) < std::get<uint64_t>(vec2->elems[i]));

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::BoolT){
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        assertm (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2), "SPIRV simulator: Found non-unsigned integer operand");

        result = (uint64_t)(std::get<uint64_t>(op1) < std::get<uint64_t>(op2));

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type, must be vector or bool");
    }
}

void SPIRVSimulator::Op_ConstantTrue(const Instruction& instruction){
    /*
    OpConstantTrue
    Declare a true Boolean-type scalar constant.
    Result Type must be the scalar Boolean type.
    */
    assert(instruction.opcode == spv::Op::OpConstantTrue);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    const Type& type = types_.at(type_id);

    assertm (type.kind == Type::Kind::BoolT, "SPIRV simulator: Constant type must be bool");

    Value result = (uint64_t)1;
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_ConstantFalse(const Instruction& instruction){
    /*
    OpConstantFalse
    Declare a false Boolean-type scalar constant.
    Result Type must be the scalar Boolean type.
    */
    assert(instruction.opcode == spv::Op::OpConstantFalse);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    const Type& type = types_.at(type_id);

    assertm (type.kind == Type::Kind::BoolT, "SPIRV simulator: Constant type must be bool");

    Value result = (uint64_t)0;
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_ConstantNull(const Instruction& instruction){
    /*
    OpConstantNull

    Declare a new null constant value.

    The null value is type dependent, defined as follows:
    - Scalar Boolean: false
    - Scalar integer: 0
    - Scalar floating point: +0.0 (all bits 0)
    - All other scalars: Abstract
    - Composites: Members are set recursively to the null constant according to the null value of their constituent types.

    Result Type must be one of the following types:
    - Scalar or vector Boolean type
    - Scalar or vector integer type
    - Scalar or vector floating-point type
    - Pointer type
    - Event type
    - Device side event type
    - Reservation id type
    - Queue type
    - Composite type
    */
    assert(instruction.opcode == spv::Op::OpConstantNull);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    const Type& type = types_.at(type_id);

    // TODO: This will crash for most pointers, we have to handle that case without MakeDefault
    assertm (type.kind != Type::Kind::Pointer, "SPIRV simulator: Op_ConstantNull for pointer types is currently not supported");

    SetValue(result_id, MakeDefault(type_id));
}

void SPIRVSimulator::Op_AtomicIAdd(const Instruction& instruction){
    /*
    OpAtomicIAdd

    Perform the following steps atomically with respect to any other atomic accesses within Memory to the same location:
    1) load through Pointer to get an Original Value,
    2) get a New Value by integer addition of Original Value and Value, and
    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

    The type of Value must be the same as Result Type.
    The type of the value pointed to by Pointer must be the same as Result Type.

    Memory is a memory Scope.
    */
    assert(instruction.opcode == spv::Op::OpAtomicIAdd);

    uint32_t result_id = instruction.words[2];
    uint32_t pointer_id = instruction.words[3];
    uint32_t value_id = instruction.words[6];

    PointerV pointer = std::get<PointerV>(GetValue(pointer_id));
    Value pointee_val = Deref(pointer);
    Value source_value = GetValue(value_id);

    Value result;
    if (std::holds_alternative<uint64_t>(pointee_val) && std::holds_alternative<uint64_t>(source_value)){
        result = std::get<uint64_t>(pointee_val) + std::get<uint64_t>(source_value);
    } else if (std::holds_alternative<int64_t>(pointee_val) && std::holds_alternative<int64_t>(source_value)){
        result = std::get<int64_t>(pointee_val) + std::get<int64_t>(source_value);
    } else {
        assertx ("SPIRV simulator: Invalid type match in Op_AtomicIAdd, must be same type scalar integers");
    }

    Deref(pointer) = result;
    SetValue(result_id, pointee_val);
}

void SPIRVSimulator::Op_AtomicISub(const Instruction& instruction){
    /*
    OpAtomicISub

    Perform the following steps atomically with respect to any other atomic accesses within Memory to the same location:
    1) load through Pointer to get an Original Value,
    2) get a New Value by integer subtraction of Value from Original Value, and
    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

    The type of Value must be the same as Result Type.
    The type of the value pointed to by Pointer must be the same as Result Type.

    Memory is a memory Scope.
    */
    assert(instruction.opcode == spv::Op::OpAtomicISub);

    uint32_t result_id = instruction.words[2];
    uint32_t pointer_id = instruction.words[3];
    uint32_t value_id = instruction.words[6];

    PointerV pointer = std::get<PointerV>(GetValue(pointer_id));
    Value pointee_val = Deref(pointer);
    Value source_value = GetValue(value_id);

    Value result;
    if (std::holds_alternative<uint64_t>(pointee_val) && std::holds_alternative<uint64_t>(source_value)){
        result = std::get<uint64_t>(pointee_val) - std::get<uint64_t>(source_value);
    } else if (std::holds_alternative<int64_t>(pointee_val) && std::holds_alternative<int64_t>(source_value)){
        result = std::get<int64_t>(pointee_val) - std::get<int64_t>(source_value);
    } else {
        assertx ("SPIRV simulator: Invalid type match in Op_AtomicISub, must be same type scalar integers");
    }

    Deref(pointer) = result;
    SetValue(result_id, pointee_val);
}

void SPIRVSimulator::Op_Select(const Instruction& instruction){
    /*
    OpSelect

    Select between two objects. Before version 1.4, results are only computed per component.
    Before version 1.4, Result Type must be a pointer, scalar, or vector.
    Starting with version 1.4, Result Type can additionally be a composite type other than a vector.
    The types of Object 1 and Object 2 must be the same as Result Type.
    Condition must be a scalar or vector of Boolean type.
    If Condition is a scalar and true, the result is Object 1. If Condition is a scalar and false, the result is Object 2.

    If Condition is a vector, Result Type must be a vector with the same number of components as
    Condition and the result is a mix of Object 1 and Object 2: If a component of Condition is true, the corresponding component in the result is taken from Object 1, otherwise it is taken from Object 2.
    */
    assert(instruction.opcode == spv::Op::OpSelect);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t condition_id = instruction.words[3];
    uint32_t obj_1_id = instruction.words[4];
    uint32_t obj_2_id = instruction.words[5];

    const Type& type = types_.at(type_id);
    const Value& condition_val = GetValue(condition_id);
    const Value& val_op1 = GetValue(obj_1_id);
    const Value& val_op2 = GetValue(obj_2_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type in Op_Select, but they are not, illegal input parameters");
        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(condition_val), "SPIRV simulator: Condition operand set to be vector type in Op_Select, but is is not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);
        auto cond_vec = std::get<std::shared_ptr<VectorV>>(condition_val);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == cond_vec->elems.size()) && (cond_vec->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length in Op_Select");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            Value elem_result;

            uint64_t cond_val = std::get<uint64_t>(cond_vec->elems[i]);

            if (cond_val){
                result_vec->elems.push_back(vec1->elems[i]);
            } else {
                result_vec->elems.push_back(vec2->elems[i]);
            }
        }

        SetValue(result_id, result);
    } else {
        assertm (std::holds_alternative<uint64_t>(condition_val), "SPIRV simulator: Op_Select condition must be a bool or a vector of bools");
        uint64_t condition_int = std::get<uint64_t>(condition_val);

        if (condition_int){
            SetValue(result_id, val_op1);
        } else {
            SetValue(result_id, val_op2);
        }
    }
}

void SPIRVSimulator::Op_IEqual(const Instruction& instruction){
    /*
    OpIEqual

    Integer comparison for equality.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.
    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpIEqual);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands not of vector type in Op_IEqual");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands not of equal/correct length in Op_IEqual");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            Value elem_result;

            if(std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])){
                elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) == std::get<uint64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i])){
                elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) == (uint64_t)std::get<int64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i])){
                elem_result = (uint64_t)((uint64_t)std::get<int64_t>(vec1->elems[i]) == (uint64_t)std::get<int64_t>(vec2->elems[i]));
            } else if(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])){
                elem_result = (uint64_t)((uint64_t)std::get<int64_t>(vec1->elems[i]) == std::get<uint64_t>(vec2->elems[i]));
            } else {
                assertx ("SPIRV simulator: Could not find valid parameter type combination for Op_IEqual vector operand");
            }

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Int){
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        if(std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2)){
            result = (uint64_t)(std::get<uint64_t>(op1) == std::get<uint64_t>(op2));
        } else if(std::holds_alternative<uint64_t>(op1) && std::holds_alternative<int64_t>(op2)){
            result = (uint64_t)(std::get<uint64_t>(op1) == (uint64_t)std::get<int64_t>(op2));
        } else if(std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2)){
            result = (uint64_t)((uint64_t)std::get<int64_t>(op1) == (uint64_t)std::get<int64_t>(op2));
        } else if(std::holds_alternative<int64_t>(op1) && std::holds_alternative<uint64_t>(op2)){
            result = (uint64_t)((uint64_t)std::get<int64_t>(op1) == std::get<uint64_t>(op2));
        } else {
            assertx ("SPIRV simulator: Could not find valid parameter type combination for Op_IEqual");
        }

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type int Op_IEqual, must be vector or int");
    }
}

void SPIRVSimulator::Op_CompositeInsert(const Instruction& instruction){
    /*
    OpCompositeInsert

    Make a copy of a composite object, while modifying one part of it.
    Result Type must be the same type as Composite.

    Object is the object to use as the modified part.
    Composite is the composite to copy all but the modified part from.
    Indexes walk the type hierarchy of Composite to the desired depth, potentially down to component granularity,
    to select the part to modify. All indexes must be in bounds. All composite constituents use zero-based numbering,
    as described by their OpType…​ instruction.
    The type of the part selected to modify must match the type of Object. Each index is an unsigned 32-bit integer.
    */
    assert(instruction.opcode == spv::Op::OpCompositeInsert);

    uint32_t result_id = instruction.words[2];
    uint32_t obj_id = instruction.words[3];
    uint32_t composite_id = instruction.words[4];

    const Value& source_composite = GetValue(composite_id);
    Value composite_copy = CopyValue(source_composite);

    Value* current_composite = &composite_copy;
    for (uint32_t i = 5; i < instruction.word_count; ++i){
        uint32_t literal_index = instruction.words[i];

        if(std::holds_alternative<std::shared_ptr<AggregateV>>(*current_composite)){
            auto agg = std::get<std::shared_ptr<AggregateV>>(*current_composite);

            assertm (literal_index < agg->elems.size(), "SPIRV simulator: Aggregate index OOB");

            current_composite = &agg->elems[literal_index];
        } else if (std::holds_alternative<std::shared_ptr<VectorV>>(*current_composite)){
            auto vec = std::get<std::shared_ptr<VectorV>>(*current_composite);

            assertm (literal_index < vec->elems.size(), "SPIRV simulator: Vector index OOB");

            current_composite = &vec->elems[literal_index];
        } else if (std::holds_alternative<std::shared_ptr<MatrixV>>(*current_composite)){
            auto matrix = std::get<std::shared_ptr<MatrixV>>(*current_composite);

            assertm (literal_index < matrix->cols.size(), "SPIRV simulator: Matrix index OOB");

            current_composite = &matrix->cols[literal_index];
        }
        else{
            assertx ("SPIRV simulator: Pointer dereference into non-composite object");
        }
    }

    const Value& source_object = GetValue(obj_id);
    *current_composite = CopyValue(source_object);
    SetValue(result_id, composite_copy);
}

void SPIRVSimulator::Op_Transpose(const Instruction& instruction){
    /*

    OpTranspose

    Transpose a matrix.
    Result Type must be an OpTypeMatrix.
    Matrix must be an object of type OpTypeMatrix. The number of columns and the column size of Matrix must be the
    reverse of those in Result Type. The types of the scalar components in Matrix and Result Type must be the same.

    Matrix must have of type of OpTypeMatrix.
    */
    assert(instruction.opcode == spv::Op::OpTranspose);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t matrix_id = instruction.words[3];
    const Type& type = types_.at(type_id);

    assertm (type.kind == Type::Kind::Matrix, "SPIRV simulator: Non-matrix type given to Op_Transpose");
    assertm (type.matrix.col_count > 0, "SPIRV simulator: Matrix type with no columns encountered");

    const Type& col_type = types_.at(type.matrix.col_type_id);
    assertm (col_type.kind == Type::Kind::Vector, "SPIRV simulator: Non-vector column type in matrix type given to Op_Transpose");
    assertm (col_type.vector.elem_count > 0, "SPIRV simulator: Vector type with no elements encountered");

    Value source_matrix_value = GetValue(matrix_id);
    assertm (std::holds_alternative<std::shared_ptr<MatrixV>>(source_matrix_value), "SPIRV simulator: Simulator value does not hold a MatrixV shared pointer in Op_Transpose");

    std::shared_ptr<MatrixV> new_matrix = std::make_shared<MatrixV>();
    std::shared_ptr<MatrixV> source_matrix = std::get<std::shared_ptr<MatrixV>>(source_matrix_value);
    assertm (source_matrix->cols.size() == col_type.vector.elem_count, "SPIRV simulator: Column vs row mismatch in Op_Transpose");

    for (uint64_t target_column = 0; target_column < type.matrix.col_count; ++target_column){
        std::shared_ptr<VectorV> new_column = std::make_shared<VectorV>();

        for (uint64_t source_row = 0; source_row < type.matrix.col_count; ++source_row){
            for (uint64_t source_column = 0; source_column < col_type.vector.elem_count; ++source_column){
                new_column->elems.push_back(CopyValue(std::get<std::shared_ptr<VectorV>>(source_matrix->cols[source_column])->elems[source_row]));
            }
        }

        new_matrix->cols.push_back(new_column);
    }

    SetValue(result_id, new_matrix);
}

void SPIRVSimulator::Op_FNegate(const Instruction& instruction){
    /*
    OpFNegate

    Inverts the sign bit of Operand. (Note, however, that OpFNegate is still considered a floating-point instruction,
    and so is subject to the general floating-point rules regarding, for example, subnormals and NaN propagation).

    Result Type must be a scalar or vector of floating-point type.
    The type of Operand must be the same as Result Type.
    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFNegate);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand_id = instruction.words[3];

    const Type& type = types_.at(type_id);
    const Value& val_op = GetValue(operand_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op), "SPIRV simulator: Operand not of vector type");

        auto vec = std::get<std::shared_ptr<VectorV>>(val_op);

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            double elem_result = -1.0 * std::get<double>(vec->elems[i]);
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Float){
        assertm (std::holds_alternative<double>(val_op), "SPIRV simulator: Operands not of float type");

        Value result = -1.0 * std::get<double>(val_op);

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type int, must be vector or float");
    }
}

void SPIRVSimulator::Op_UGreaterThan(const Instruction& instruction){
    /*
    OpUGreaterThan

    Unsigned-integer comparison if Operand 1 is greater than Operand 2.

    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpUGreaterThan);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    const Type& type = types_.at(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            assertm (std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec1->elems[i]), "SPIRV simulator: Found non-unsigned integer operand in vector operands");

            Value elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) > std::get<uint64_t>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);

    } else if (type.kind == Type::Kind::BoolT){
        Value result = (uint64_t)(std::get<uint64_t>(val_op1) > std::get<uint64_t>(val_op2));
        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type, must be vector or bool");
    }
}

void SPIRVSimulator::Op_FOrdLessThan(const Instruction& instruction){
    /*

    OpFOrdLessThan

    Floating-point comparison if operands are ordered and Operand 1 is less than Operand 2.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of floating-point type.
    They must have the same type, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFOrdLessThan);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    const Type& type = types_.at(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            assertm (std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]), "SPIRV simulator: Found non-floating point operand in vector operands");

            Value elem_result = (uint64_t)(std::get<double>(vec1->elems[i]) < std::get<double>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::BoolT){
        Value result = (uint64_t)(std::get<double>(val_op1) < std::get<double>(val_op2));
        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type in, must be vector or float");
    }
}

void SPIRVSimulator::Op_FOrdLessThanEqual(const Instruction& instruction){
    /*
    OpFOrdLessThanEqual

    Floating-point comparison if operands are ordered and Operand 1 is less than or equal to Operand 2.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of floating-point type.
    They must have the same type, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFOrdLessThanEqual);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    const Type& type = types_.at(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            assertm (std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]), "SPIRV simulator: Found non-floating point operand in vector operands");

            Value elem_result = (uint64_t)(std::get<double>(vec1->elems[i]) <= std::get<double>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::BoolT){
        Value result = (uint64_t)(std::get<double>(val_op1) <= std::get<double>(val_op2));
        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type, must be vector or float");
    }
}

void SPIRVSimulator::Op_Switch(const Instruction& instruction){
    /*
    OpSwitch

    Multi-way branch to one of the operand label <id>.
    Selector must have a type of OpTypeInt. Selector is compared for equality to the Target literals.
    Default must be the <id> of a label. If Selector does not equal any of the Target literals,
    control flow branches to the Default label <id>.

    Target must be alternating scalar integer literals and the <id> of a label.
    If Selector equals a literal, control flow branches to the following label <id>.
    It is invalid for any two literal to be equal to each other.
    If Selector does not equal any literal, control flow branches to the Default label <id>.
    Each literal is interpreted with the type of Selector: The bit width of Selector’s type is the width
    of each literal’s type. If this width is not a multiple of 32-bits and the OpTypeInt Signedness is set to 1,
    the literal values are interpreted as being sign extended.

    This instruction must be the last instruction in a block.
    */
    assert(instruction.opcode == spv::Op::OpSwitch);

    uint32_t selector_id = instruction.words[1];
    uint32_t default_id = instruction.words[2];

    const Value& selector_value = GetValue(selector_id);
    uint64_t selector;
    if (std::holds_alternative<uint64_t>(selector_value)){
        selector = std::get<uint64_t>(selector_value);
    } else if (std::holds_alternative<int64_t>(selector_value)){
        selector = (uint64_t)std::get<int64_t>(selector_value);
    } else {
        assertx ("SPIRV simulator: Selector value is not an integer");
    }

    const Type& type = GetType(selector_id);
    assertm (type.scalar.width <= 32, "SPIRV simulator: Selector ID uses more than 32 bits, this is not handled at present and should be implemented");

    uint32_t label_id = default_id;
    for (uint32_t i = 3; i < instruction.word_count; i += 2){
        uint32_t literal = instruction.words[i];

        if (selector == literal){
            label_id = instruction.words[i + 1];
            break;
        }
    }

    call_stack_.back().pc = result_id_to_inst_index_.at(label_id);
}

void SPIRVSimulator::Op_MatrixTimesVector(const Instruction& instruction){
    /*
    OpMatrixTimesVector

    Linear-algebraic Matrix X Vector.

    Result Type must be a vector of floating-point type.
    Matrix must be an OpTypeMatrix whose Column Type is Result Type.
    Vector must be a vector with the same Component Type as the Component Type in Result Type.

    Its number of components must equal the number of columns in Matrix.
    */
    assert(instruction.opcode == spv::Op::OpMatrixTimesVector);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t matrix_id = instruction.words[3];
    uint32_t vector_id = instruction.words[4];

    const Type& type = GetType(type_id);
    assertm (type.kind == Type::Kind::Vector, "SPIRV simulator: Result operand is not a vector");
    assertm (GetType(matrix_id).kind == Type::Kind::Matrix, "SPIRV simulator: First operand is not a matrix");
    assertm (GetType(vector_id).kind == Type::Kind::Vector, "SPIRV simulator: Second operand is not a vector");

    const std::shared_ptr<VectorV>& vector = std::get<std::shared_ptr<VectorV>>(GetValue(vector_id));
    const std::shared_ptr<MatrixV>& matrix = std::get<std::shared_ptr<MatrixV>>(GetValue(matrix_id));

    std::vector<double> tmp_result;
    tmp_result.resize(type.vector.elem_count);

    for (uint32_t col_index = 0; col_index < matrix->cols.size(); ++col_index){
        for (uint32_t row_index = 0; row_index < type.vector.elem_count; ++row_index){
            assertm (std::holds_alternative<std::shared_ptr<VectorV>>(matrix->cols[col_index]), "SPIRV simulator: Non-vector column value found in matrix operand");
            assertm (std::holds_alternative<double>(vector->elems[row_index]), "SPIRV simulator: Non-floating point value found in vector operand");

            const std::shared_ptr<VectorV>& col_vector = std::get<std::shared_ptr<VectorV>>(matrix->cols[col_index]);
            assertm (std::holds_alternative<double>(col_vector->elems[row_index]), "SPIRV simulator: Non-floating point value found in column vector operand");

            tmp_result[row_index] += std::get<double>(col_vector->elems[row_index]) * std::get<double>(vector->elems[col_index]);
        }

    }

    std::shared_ptr<VectorV> result = std::make_shared<VectorV>();
    for (double result_val : tmp_result){
        result->elems.push_back(result_val);
    }

    SetValue(result_id, result);
}

void SPIRVSimulator::Op_VectorShuffle(const Instruction& instruction){
    /*
    OpVectorShuffle

    Select arbitrary components from two vectors to make a new vector.

    Result Type must be an OpTypeVector.
    The number of components in Result Type must be the same as the number of Component operands.

    Vector 1 and Vector 2 must both have vector types, with the same Component Type as Result Type.
    They do not have to have the same number of components as Result Type or with each other. They are logically concatenated, forming a single vector with Vector 1’s components appearing before Vector 2’s. The components of this logical vector are logically numbered with a single consecutive set of numbers from 0 to N - 1, where N is the total number of components.

    Components are these logical numbers (see above), selecting which of the logically numbered components form the result.
    Each component is an unsigned 32-bit integer. They can select the components in any order and can repeat components. The first component of the result is selected by the first Component operand, the second component of the result is selected by the second Component operand, etc. A Component literal may also be FFFFFFFF, which means the corresponding result component has no source and is undefined. All Component literals must either be FFFFFFFF or in [0, N - 1] (inclusive).

    Note: A vector “swizzle” can be done by using the vector for both Vector operands, or
    using an OpUndef for one of the Vector operands.
    */
    assert(instruction.opcode == spv::Op::OpVectorShuffle);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t vec1_id = instruction.words[3];
    uint32_t vec2_id = instruction.words[4];

    assertm (types_.at(type_id).kind == Type::Kind::Vector, "SPIRV simulator: Non-vector result type");

    const Value& vector1_val = GetValue(vec1_id);
    const Value& vector2_val = GetValue(vec2_id);

    assertm (std::holds_alternative<std::shared_ptr<VectorV>>(vector1_val), "SPIRV simulator: Non-vector value in vector operand 1");
    assertm (std::holds_alternative<std::shared_ptr<VectorV>>(vector2_val), "SPIRV simulator: Non-vector value in vector operand 2");

    const std::shared_ptr<VectorV>& vector1 = std::get<std::shared_ptr<VectorV>>(vector1_val);
    const std::shared_ptr<VectorV>& vector2 = std::get<std::shared_ptr<VectorV>>(vector2_val);

    std::vector<Value> values;
    values.insert(values.end(), vector1->elems.begin(), vector1->elems.end());
    values.insert(values.end(), vector2->elems.begin(), vector2->elems.end());

    std::shared_ptr<VectorV> result = std::make_shared<VectorV>();
    for (uint32_t literal_index = 5; literal_index < instruction.word_count; ++literal_index){
        assertm (literal_index < values.size(), "SPIRV simulator: Literal index OOB");

        if (literal_index == 0xFFFFFFFF){
            Value undef_val = (uint64_t)0xFFFFFFFF;
            result->elems.push_back(undef_val);
        } else {
            result->elems.push_back(values[literal_index]);
        }
    }

    SetValue(result_id, result);
}

void SPIRVSimulator::Op_ShiftRightLogical(const Instruction& instruction){
    /*
    OpShiftRightLogical

    Shift the bits in Base right by the number of bits specified in Shift. The most-significant bits are zero filled.
    Result Type must be a scalar or vector of integer type.
    The type of each Base and Shift must be a scalar or vector of integer type. Base and Shift must have the same number of components. The number of components and bit width of the type of Base must be the same as in Result Type.
    Shift is consumed as an unsigned integer. The resulting value is undefined if Shift is greater than or equal to the bit width of the components of Base.
    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpShiftRightLogical);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t op1_id = instruction.words[3];
    uint32_t op2_id = instruction.words[4];

    const Type& type = types_.at(type_id);
    const Value& op1 = GetValue(op1_id);
    const Value& op2 = GetValue(op2_id);

    if (type.kind == Type::Kind::Vector) {
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        auto vec1 = std::get<std::shared_ptr<VectorV>>(op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(op2);

        assertm(vec1->elems.size() == vec2->elems.size() && vec1->elems.size() == type.vector.elem_count, "SPIRV simulator: Vector size mismatch in Op_ShiftRightLogical");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i) {
            if (std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])) {
                result_vec->elems.push_back(std::get<uint64_t>(vec1->elems[i]) >> std::get<uint64_t>(vec2->elems[i]));
            } else if (std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])) {
                result_vec->elems.push_back((uint64_t)std::get<int64_t>(vec1->elems[i]) >> std::get<uint64_t>(vec2->elems[i]));
            } else {
                assertx("SPIRV simulator: Invalid operand types in Op_ShiftRightLogical vector");
            }
        }
        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Int) {
        Value result;
        if (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2)) {
            result = std::get<uint64_t>(op1) >> std::get<uint64_t>(op2);
        } else if (std::holds_alternative<int64_t>(op1) && std::holds_alternative<uint64_t>(op2)) {
            result = (uint64_t)std::get<int64_t>(op1) >> std::get<uint64_t>(op2);
        } else {
            assertx("SPIRV simulator: Invalid operand types in Op_ShiftRightLogical");
        }
        SetValue(result_id, result);
    } else {
        assertx("SPIRV simulator: Invalid result type in Op_ShiftRightLogical, must be vector or int");
    }
}

void SPIRVSimulator::Op_ShiftLeftLogical(const Instruction& instruction){
   /*
    OpShiftLeftLogical

    Shift the bits in Base left by the number of bits specified in Shift. The most-significant bits are zero filled.
    Result Type must be a scalar or vector of integer type.
    The type of each Base and Shift must be a scalar or vector of integer type. Base and Shift must have the same number of components. The number of components and bit width of the type of Base must be the same as in Result Type.
    Shift is consumed as an unsigned integer. The resulting value is undefined if Shift is greater than or equal to the bit width of the components of Base.
    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpShiftLeftLogical);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t op1_id = instruction.words[3];
    uint32_t op2_id = instruction.words[4];

    const Type& type = types_.at(type_id);
    const Value& op1 = GetValue(op1_id);
    const Value& op2 = GetValue(op2_id);

    if (type.kind == Type::Kind::Vector) {
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        auto vec1 = std::get<std::shared_ptr<VectorV>>(op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(op2);

        assertm(vec1->elems.size() == vec2->elems.size() && vec1->elems.size() == type.vector.elem_count, "SPIRV simulator: Vector size mismatch in Op_ShiftLeftLogical");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i) {
            if (std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])) {
                result_vec->elems.push_back(std::get<uint64_t>(vec1->elems[i]) << std::get<uint64_t>(vec2->elems[i]));
            } else if (std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i])) {
                result_vec->elems.push_back((uint64_t)std::get<int64_t>(vec1->elems[i]) << std::get<uint64_t>(vec2->elems[i]));
            } else {
                assertx("SPIRV simulator: Invalid operand types in Op_ShiftLeftLogical vector");
            }
        }
        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Int) {
        Value result;
        if (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2)) {
            result = std::get<uint64_t>(op1) << std::get<uint64_t>(op2);
        } else if (std::holds_alternative<int64_t>(op1) && std::holds_alternative<uint64_t>(op2)) {
            result = (uint64_t)std::get<int64_t>(op1) << std::get<uint64_t>(op2);
        } else {
            assertx("SPIRV simulator: Invalid operand types in Op_ShiftLeftLogical");
        }
        SetValue(result_id, result);
    } else {
        assertx("SPIRV simulator: Invalid result type in Op_ShiftLeftLogical, must be vector or int");
    }
}

void SPIRVSimulator::Op_BitwiseOr(const Instruction& instruction){
    /*
    OpBitwiseOr

    Result is 1 if either Operand 1 or Operand 2 is 1. Result is 0 if both Operand 1 and Operand 2 are 0.

    Results are computed per component, and within each component, per bit.

    Result Type must be a scalar or vector of integer type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same number of components as Result Type.
    They must have the same component width as Result Type.
    */
    assert(instruction.opcode == spv::Op::OpBitwiseOr);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t op1_id = instruction.words[3];
    uint32_t op2_id = instruction.words[4];

    const Type& type = GetType(type_id);
    const Value& val_op1 = GetValue(op1_id);
    const Value& val_op2 = GetValue(op2_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length");
        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            uint64_t val1;
            if (std::holds_alternative<int64_t>(vec1->elems[i])){
                val1 =  bit_cast<uint64_t>(std::get<int64_t>(vec1->elems[i]));
            } else {
                val1 = std::get<uint64_t>(vec1->elems[i]);
            }

            uint64_t val2;
            if (std::holds_alternative<int64_t>(vec2->elems[i])){
                val2 = bit_cast<uint64_t>(std::get<int64_t>(vec2->elems[i]));
            } else {
                val2 = std::get<uint64_t>(vec2->elems[i]);
            }

            Value elem_result = (uint64_t)(val1 | val2);
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Int){
        uint64_t val1;
        if (std::holds_alternative<int64_t>(val_op1)){
            val1 =  bit_cast<uint64_t>(std::get<int64_t>(val_op1));
        } else {
            val1 = std::get<uint64_t>(val_op1);
        }

        uint64_t val2;
        if (std::holds_alternative<int64_t>(val_op2)){
            val2 = bit_cast<uint64_t>(std::get<int64_t>(val_op2));
        } else {
            val2 = std::get<uint64_t>(val_op2);
        }
        Value result = (uint64_t)(val1 | val2);
        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type, must be vector or int");
    }
}

void SPIRVSimulator::Op_BitwiseAnd(const Instruction& instruction){
    /*
    OpBitwiseAnd

    Result is 1 if both Operand 1 and Operand 2 are 1. Result is 0 if either Operand 1 or Operand 2 are 0.

    Results are computed per component, and within each component, per bit.

    Result Type must be a scalar or vector of integer type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same number of components as Result Type. They must have the same component width as Result Type.
    */
    assert(instruction.opcode == spv::Op::OpBitwiseAnd);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t op1_id = instruction.words[3];
    uint32_t op2_id = instruction.words[4];

    const Type& type = GetType(type_id);
    const Value& val_op1 = GetValue(op1_id);
    const Value& val_op2 = GetValue(op2_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length");
        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            uint64_t val1;
            if (std::holds_alternative<int64_t>(vec1->elems[i])){
                val1 =  bit_cast<uint64_t>(std::get<int64_t>(vec1->elems[i]));
            } else {
                val1 = std::get<uint64_t>(vec1->elems[i]);
            }

            uint64_t val2;
            if (std::holds_alternative<int64_t>(vec2->elems[i])){
                val2 = bit_cast<uint64_t>(std::get<int64_t>(vec2->elems[i]));
            } else {
                val2 = std::get<uint64_t>(vec2->elems[i]);
            }

            Value elem_result = (uint64_t)(val1 & val2);
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Int){
        uint64_t val1;
        if (std::holds_alternative<int64_t>(val_op1)){
            val1 =  bit_cast<uint64_t>(std::get<int64_t>(val_op1));
        } else {
            val1 = std::get<uint64_t>(val_op1);
        }

        uint64_t val2;
        if (std::holds_alternative<int64_t>(val_op2)){
            val2 = bit_cast<uint64_t>(std::get<int64_t>(val_op2));
        } else {
            val2 = std::get<uint64_t>(val_op2);
        }
        Value result = (uint64_t)(val1 & val2);
        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type, must be vector or int");
    }
}

void SPIRVSimulator::Op_All(const Instruction& instruction){
    /*
    OpAll

    Result is true if all components of Vector are true, otherwise result is false.

    Result Type must be a Boolean type scalar.

    Vector must be a vector of Boolean type.
    */
    assert(instruction.opcode == spv::Op::OpAll);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t vector_id = instruction.words[3];

    const Type& type = GetType(type_id);
    const Value& vector_val = GetValue(vector_id);

    assertm (type.kind == Type::Kind::Vector, "SPIRV simulator: Operand is not of vector type");
    assertm (std::holds_alternative<std::shared_ptr<VectorV>>(vector_val), "SPIRV simulator: Operand is of vector type but does not hold a vector");

    const std::shared_ptr<VectorV>& vec = std::get<std::shared_ptr<VectorV>>(vector_val);

    bool result_bool = true;
    for (const auto& bool_val : vec->elems){
        result_bool = result_bool && (bool)std::get<uint64_t>(bool_val);
    }

    Value result = (uint64_t)result_bool;
    SetValue(result_id, result);
}


void SPIRVSimulator::Op_Any(const Instruction& instruction){
    /*
    OpAny

    Result is true if any component of Vector is true, otherwise result is false.

    Result Type must be a Boolean type scalar.

    Vector must be a vector of Boolean type.
    */
    assert(instruction.opcode == spv::Op::OpAny);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t vector_id = instruction.words[3];

    const Type& type = GetType(type_id);
    const Value& vector_val = GetValue(vector_id);

    assertm (type.kind == Type::Kind::Vector, "SPIRV simulator: Operand is not of vector type");
    assertm (std::holds_alternative<std::shared_ptr<VectorV>>(vector_val), "SPIRV simulator: Operand is of vector type but does not hold a vector");

    const std::shared_ptr<VectorV>& vec = std::get<std::shared_ptr<VectorV>>(vector_val);

    bool result_bool = false;
    for (const auto& bool_val : vec->elems){
        result_bool = result_bool || (bool)std::get<uint64_t>(bool_val);
    }

    Value result = (uint64_t)result_bool;
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_BitCount(const Instruction& instruction){
    /*
    OpBitCount

    Count the number of set bits in an object.

    Results are computed per component.

    Result Type must be a scalar or vector of integer type.
    The components must be wide enough to hold the unsigned Width of Base as an unsigned value.
    That is, no sign bit is needed or counted when checking for a wide enough result width.

    Base must be a scalar or vector of integer type. It must have the same number of components as Result Type.

    The result is the unsigned value that is the number of bits in Base that are 1.
    */
    assert(instruction.opcode == spv::Op::OpBitCount);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t base_id = instruction.words[3];

    const Type& type = GetType(type_id);
    const Value& base_val = GetValue(base_id);

    uint32_t base_type_id = GetTypeID(base_id);

    if (type.kind == Type::Kind::Vector){
        const Type& base_type = GetType(base_type_id);
        const std::shared_ptr<VectorV> vec = std::get<std::shared_ptr<VectorV>>(base_val);
        std::shared_ptr<VectorV> result_vec = std::make_shared<VectorV>();

        for (const Value& val : vec->elems){
            (void)val;
            result_vec->elems.push_back(GetBitizeOfType(base_type.vector.elem_type_id));
        }

        SetValue(result_id, result_vec);
    } else if (type.kind == Type::Kind::Int){
        SetValue(result_id, (uint64_t)GetBitizeOfType(base_type_id));
    } else {
        assertx ("SPIRV simulator: Invalid result value, must be vector or int");
    }
}

void SPIRVSimulator::Op_Kill(const Instruction& instruction){
    /*
    OpKill

    Deprecated (use OpTerminateInvocation or OpDemoteToHelperInvocation).

    Fragment-shader discard.
    Ceases all further processing in any invocation that executes it: Only instructions these invocations
    executed before OpKill have observable side effects.
    If this instruction is executed in non-uniform control flow, all subsequent control flow is non-uniform
    (for invocations that continue to execute).

    This instruction must be the last instruction in a block.

    This instruction is only valid in the Fragment Execution Model.
    */
    assert(instruction.opcode == spv::Op::OpKill);

    if (verbose_){
        std::cout << execIndent << "Thread killed by OpKill, ceasing all further processing" << std::endl;
    }

    call_stack_.clear();
}

void SPIRVSimulator::Op_Unreachable(const Instruction& instruction){
    /*
    OpUnreachable

    Behavior is undefined if this instruction is executed.

    This instruction must be the last instruction in a block.
    */
    assert(instruction.opcode == spv::Op::OpUnreachable);

    assertx ("SPIRV simulator: OpUnreachable executed, this is undefined behaviour");
}

void SPIRVSimulator::Op_Undef(const Instruction& instruction){
    /*
    OpUndef

    Make an intermediate object whose value is undefined.

    Result Type is the type of object to make. Result Type can be any type except OpTypeVoid.

    Each consumption of Result <id> yields an arbitrary, possibly different bit pattern or abstract value
    resulting in possibly different concrete, abstract, or opaque values.
    */
    assert(instruction.opcode == spv::Op::OpUndef);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    SetValue(result_id, MakeDefault(type_id));
}

void SPIRVSimulator::Op_VectorTimesMatrix(const Instruction& instruction){
    /*
    OpVectorTimesMatrix

    Linear-algebraic Vector X Matrix.

    Result Type must be a vector of floating-point type.

    Vector must be a vector with the same Component Type as the Component Type in Result Type.
    Its number of components must equal the number of components in each column in Matrix.

    Matrix must be a matrix with the same Component Type as the Component Type in Result Type.

    Its number of columns must equal the number of components in Result Type.
    */
    assert(instruction.opcode == spv::Op::OpVectorTimesMatrix);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t vector_id = instruction.words[3];
    uint32_t matrix_id = instruction.words[4];

    const Type& type = GetType(type_id);
    assertm (type.kind == Type::Kind::Vector, "SPIRV simulator: Result operand is not a vector");
    assertm (GetType(matrix_id).kind == Type::Kind::Matrix, "SPIRV simulator: Second operand is not a matrix");
    assertm (GetType(vector_id).kind == Type::Kind::Vector, "SPIRV simulator: First operand is not a vector");

    const std::shared_ptr<VectorV>& vector = std::get<std::shared_ptr<VectorV>>(GetValue(vector_id));
    const std::shared_ptr<MatrixV>& matrix = std::get<std::shared_ptr<MatrixV>>(GetValue(matrix_id));

    std::vector<double> tmp_result;
    tmp_result.resize(type.vector.elem_count);

    for (uint32_t col_index = 0; col_index < matrix->cols.size(); ++col_index){
        for (uint32_t row_index = 0; row_index < type.vector.elem_count; ++row_index){
            assertm (std::holds_alternative<std::shared_ptr<VectorV>>(matrix->cols[col_index]), "SPIRV simulator: Non-vector column value found in matrix operand");
            assertm (std::holds_alternative<double>(vector->elems[row_index]), "SPIRV simulator: Non-floating point value found in vector operand");

            const std::shared_ptr<VectorV>& col_vector = std::get<std::shared_ptr<VectorV>>(matrix->cols[col_index]);
            assertm (std::holds_alternative<double>(col_vector->elems[row_index]), "SPIRV simulator: Non-floating point value found in column vector operand");

            tmp_result[col_index] += std::get<double>(col_vector->elems[row_index]) * std::get<double>(vector->elems[row_index]);
        }
    }

    std::shared_ptr<VectorV> result = std::make_shared<VectorV>();
    for (double result_val : tmp_result){
        result->elems.push_back(result_val);
    }

    SetValue(result_id, result);
}

void SPIRVSimulator::Op_ULessThanEqual(const Instruction& instruction){
    /*
    OpULessThanEqual

    Unsigned-integer comparison if Operand 1 is less than or equal to Operand 2.

    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpULessThanEqual);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    const Type& type = types_.at(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            assertm (std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i]), "SPIRV simulator: Found non-unsigned int operand in vector operands");

            Value elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) <= std::get<uint64_t>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::BoolT){
        Value result = (uint64_t)(std::get<uint64_t>(val_op1) <= std::get<uint64_t>(val_op2));
        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type, must be vector or bool");
    }
}

void SPIRVSimulator::Op_SLessThanEqual(const Instruction& instruction){
    /*
    OpSLessThanEqual

    Signed-integer comparison if Operand 1 is less than or equal to Operand 2.

    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpSLessThanEqual);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    const Type& type = types_.at(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            assertm (std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]), "SPIRV simulator: Found non-signed int operand in vector operands");

            Value elem_result = (uint64_t)(std::get<int64_t>(vec1->elems[i]) <= std::get<int64_t>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::BoolT){
        Value result = (uint64_t)(std::get<int64_t>(val_op1) <= std::get<int64_t>(val_op2));
        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type, must be vector or bool");
    }
}

void SPIRVSimulator::Op_SGreaterThanEqual(const Instruction& instruction){
    /*
    OpSGreaterThanEqual

    Signed-integer comparison if Operand 1 is greater than or equal to Operand 2.

    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpSGreaterThanEqual);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    const Type& type = types_.at(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            assertm (std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]), "SPIRV simulator: Found non-signed int operand in vector operands");

            Value elem_result = (uint64_t)(std::get<int64_t>(vec1->elems[i]) >= std::get<int64_t>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::BoolT){
        Value result = (uint64_t)(std::get<int64_t>(val_op1) >= std::get<int64_t>(val_op2));
        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type, must be vector or bool");
    }
}

void SPIRVSimulator::Op_SGreaterThan(const Instruction& instruction){
    /*
    OpSGreaterThan

    Signed-integer comparison if Operand 1 is greater than Operand 2.

    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpSGreaterThan);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    const Type& type = types_.at(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            assertm (std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]), "SPIRV simulator: Found non-signed int operand in vector operands");

            Value elem_result = (uint64_t)(std::get<int64_t>(vec1->elems[i]) > std::get<int64_t>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::BoolT){
        Value result = (uint64_t)(std::get<int64_t>(val_op1) > std::get<int64_t>(val_op2));
        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type, must be vector or bool");
    }
}

void SPIRVSimulator::Op_SDiv(const Instruction& instruction){
    /*
    OpSDiv

    Signed-integer division of Operand 1 divided by Operand 2.

    Result Type must be a scalar or vector of integer type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same number of components as Result Type. They must have the same component width as Result Type.

    Results are computed per component. Behavior is undefined if Operand 2 is 0.
    Behavior is undefined if Operand 2 is -1 and Operand 1 is the minimum representable value for the operands' type,
    causing signed overflow.
    */
    assert(instruction.opcode == spv::Op::OpSDiv);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = types_.at(type_id);
    const Value& val_op1 = GetValue(instruction.words[3]);
    const Value& val_op2 = GetValue(instruction.words[4]);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) && std::holds_alternative<std::shared_ptr<VectorV>>(val_op2), "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm ((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count), "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            Value elem_result;

            // TODO: Operands dont have to be signed, deal with it and remove the asserts
            assertm (std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]), "SPIRV simulator: Found non-signed int operand vector operands");

            elem_result = std::get<int64_t>(vec1->elems[i]) / std::get<int64_t>(vec2->elems[i]);

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Int){
        // TODO: Operands dont have to be signed, deal with it and remove the asserts
        assertm (std::holds_alternative<int64_t>(val_op1) && std::holds_alternative<int64_t>(val_op2), "SPIRV simulator: Found non-signed int operand");

        Value result = std::get<int64_t>(val_op1) / std::get<int64_t>(val_op2);

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type, must be vector or signed int");
    }
}

void SPIRVSimulator::Op_SNegate(const Instruction& instruction){
    /*

    OpSNegate

    Signed-integer subtract of Operand from zero.

    Result Type must be a scalar or vector of integer type.

    Operand’s type must be a scalar or vector of integer type.
    It must have the same number of components as Result Type.
    The component width must equal the component width in Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpSNegate);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand_id = instruction.words[3];

    const Type& type = types_.at(type_id);
    const Value& val_op = GetValue(operand_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        // TODO: Operands dont have to be signed? If so, fix it
        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(val_op), "SPIRV simulator: Operand not of vector type");

        auto vec = std::get<std::shared_ptr<VectorV>>(val_op);

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            int64_t elem_result = 0 - std::get<int64_t>(vec->elems[i]);
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::Float){
        // TODO: Operands dont have to be signed? If so, fix it
        assertm (std::holds_alternative<int64_t>(val_op), "SPIRV simulator: Operands not of int type");

        Value result = 0 - std::get<int64_t>(val_op);

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type int, must be vector or integer");
    }
}

void SPIRVSimulator::Op_LogicalOr(const Instruction& instruction){
    /*
    OpLogicalOr

    Result is true if either Operand 1 or Operand 2 is true. Result is false if both Operand 1 and Operand 2 are false.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 must be the same as Result Type.
    The type of Operand 2 must be the same as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpLogicalOr);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    const Type& type = types_.at(type_id);
    const Value& operand1 = GetValue(operand1_id);
    const Value& operand2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(operand1), "SPIRV simulator: Invalid value type for operand 1, must be vector when using vector type");
        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(operand2), "SPIRV simulator: Invalid value type for operand 2, must be vector when using vector type");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(operand1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(operand2);

        assertm (std::holds_alternative<uint64_t>(vec1->elems[0]), "SPIRV simulator: Invalid vector component value for operand 1, must be bool");
        assertm (std::holds_alternative<uint64_t>(vec2->elems[0]), "SPIRV simulator: Invalid vector component value for operand 2, must be bool");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            result_vec->elems.push_back((uint64_t)(std::get<uint64_t>(vec1->elems[i]) || std::get<uint64_t>(vec2->elems[i])));
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::BoolT){
        assertm (std::holds_alternative<uint64_t>(operand1), "SPIRV simulator: Invalid value for operand 1, must be bool");
        assertm (std::holds_alternative<uint64_t>(operand2), "SPIRV simulator: Invalid value for operand 2, must be bool");
        Value result = (uint64_t)(std::get<uint64_t>(operand1) || std::get<uint64_t>(operand2));

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type, must be vector or bool");
    }
}

void SPIRVSimulator::Op_LogicalAnd(const Instruction& instruction){
    /*
    OpLogicalAnd

    Result is true if both Operand 1 and Operand 2 are true. Result is false if either Operand 1 or Operand 2 are false.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 must be the same as Result Type.
    The type of Operand 2 must be the same as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpLogicalAnd);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    const Type& type = types_.at(type_id);
    const Value& operand1 = GetValue(operand1_id);
    const Value& operand2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(operand1), "SPIRV simulator: Invalid value type for operand 1, must be vector when using vector type");
        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(operand2), "SPIRV simulator: Invalid value type for operand 2, must be vector when using vector type");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(operand1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(operand2);

        assertm (std::holds_alternative<uint64_t>(vec1->elems[0]), "SPIRV simulator: Invalid vector component value for operand 1, must be bool");
        assertm (std::holds_alternative<uint64_t>(vec2->elems[0]), "SPIRV simulator: Invalid vector component value for operand 2, must be bool");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            result_vec->elems.push_back((uint64_t)(std::get<uint64_t>(vec1->elems[i]) && std::get<uint64_t>(vec2->elems[i])));
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::BoolT){
        assertm (std::holds_alternative<uint64_t>(operand1), "SPIRV simulator: Invalid value for operand 1, must be bool");
        assertm (std::holds_alternative<uint64_t>(operand2), "SPIRV simulator: Invalid value for operand 2, must be bool");
        Value result = (uint64_t)(std::get<uint64_t>(operand1) && std::get<uint64_t>(operand2));

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type, must be vector or bool");
    }
}

void SPIRVSimulator::Op_MatrixTimesMatrix(const Instruction& instruction){
    /*
    OpMatrixTimesMatrix

    Linear-algebraic multiply of LeftMatrix X RightMatrix.
    Result Type must be an OpTypeMatrix whose Column Type is a vector of floating-point type.
    LeftMatrix must be a matrix whose Column Type is the same as the Column Type in Result Type.
    RightMatrix must be a matrix with the same Component Type as the Component Type in Result Type.
    Its number of columns must equal the number of columns in Result Type.
    Its columns must have the same number of components as the number of columns in LeftMatrix.
    */
    assert(instruction.opcode == spv::Op::OpMatrixTimesMatrix);

    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t matrix_left_id = instruction.words[3];
    uint32_t matrix_right_id = instruction.words[4];

    const Type& type = types_.at(type_id);
    const Type& left_type = GetType(matrix_left_id);
    const Type& right_type = GetType(matrix_right_id);

    assertm (type.kind == Type::Kind::Matrix, "SPIRV simulator: Result operand is not a matrix");
    assertm (left_type.kind == Type::Kind::Matrix, "SPIRV simulator: First operand is not a matrix");
    assertm (right_type.kind == Type::Kind::Matrix, "SPIRV simulator: Second operand is not a matrix");

    const Type& left_col_type = GetType(left_type.matrix.col_type_id);
    assertm (left_col_type.kind == Type::Kind::Vector, "SPIRV simulator: Left matrix col type is not vector");

    const std::shared_ptr<MatrixV>& matrix_left = std::get<std::shared_ptr<MatrixV>>(GetValue(matrix_left_id));
    const std::shared_ptr<MatrixV>& matrix_right = std::get<std::shared_ptr<MatrixV>>(GetValue(matrix_right_id));

    assertm (type.matrix.col_count == matrix_right->cols.size(), "SPIRV simulator: Second operand matrix number of columns dont match the result type");

    std::shared_ptr<MatrixV> result_matrix = std::make_shared<MatrixV>();

    for (uint32_t right_col_index = 0; right_col_index < matrix_right->cols.size(); ++right_col_index){
        const auto& right_col_vec = std::get<std::shared_ptr<VectorV>>(matrix_right->cols[right_col_index]);

        std::vector<double> tmp_result;
        tmp_result.resize(left_col_type.vector.elem_count);

        for (uint32_t left_col_index = 0; left_col_index < matrix_left->cols.size(); ++left_col_index){
            assertm (std::holds_alternative<std::shared_ptr<VectorV>>(matrix_left->cols[left_col_index]), "SPIRV simulator: Non-vector column value found in matrix operand");
            assertm (std::holds_alternative<double>(right_col_vec->elems[left_col_index]), "SPIRV simulator: Non-floating point value found in vector operand");

            const std::shared_ptr<VectorV>& left_col_vector = std::get<std::shared_ptr<VectorV>>(matrix_left->cols[left_col_index]);

            for (uint32_t row_index = 0; row_index < left_col_vector->elems.size(); ++row_index){
                assertm (std::holds_alternative<double>(left_col_vector->elems[row_index]), "SPIRV simulator: Non-floating point value found in column vector operand");
                tmp_result[row_index] += std::get<double>(left_col_vector->elems[row_index]) * std::get<double>(right_col_vec->elems[left_col_index]);
            }
        }

        std::shared_ptr<VectorV> new_column = std::make_shared<VectorV>();
        for (auto col_val : tmp_result){
            new_column->elems.push_back(col_val);
        }
        result_matrix->cols.push_back(new_column);
    }

    SetValue(result_id, result_matrix);
}

void SPIRVSimulator::Op_IsNan(const Instruction& instruction){
    /*
    OpIsNan

    Result is true if x is a NaN for the floating-point encoding used by the type of x, otherwise result is false.

    Result Type must be a scalar or vector of Boolean type.

    x must be a scalar or vector of floating-point type. It must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpIsNan);

    /*
    Floats are guaranteed to be nan under the following conditions GPU side:

        Case                   Guaranteed NaN on GPU?        Notes
           0.0 / 0.0              ✅                            Division by zero with zero numerator
           sqrt(x < 0)            ✅                            May be clamped under fast math (only in strict mode)
           log(x <= 0)            ✅                            log(0) = -Inf; log(x < 0) = NaN
           asin(x > 1 or < -1)    ✅                            Domain violation
           acos(x > 1 or < -1)    ✅                            Domain violation
           Arithmetic on NaN      ✅                            Follows IEEE NaN propagation

    For other cases, the results are not guaranteed and we can do whatever we want.

    Still this stuff is chaotic due to the large amount of slack compilers have here,
    results can be undefined or take multiple values in practice for many operations so always print a
    warning when we encounter this instruction.

    Also, C++ math functions can do a lot of wild stuff here, but apart from the operands above there are no guarantees GPU
    side either so we should be good.

    NOTE: We should investigate this carefully and in depth if we ever see broken behaviour in applications that use
    OpIsNan.
    */
    uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t x_id = instruction.words[3];

    std::cout << execIndent << "WARNING: OpIsNan executed, keep this in mind if you see broken behaviour here" << std::endl;

    const Type& type = types_.at(type_id);
    const Value& x_val = GetValue(x_id);

    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(x_val), "SPIRV simulator: Invalid value type for operand 1, must be vector when using vector type");

        auto x_vec = std::get<std::shared_ptr<VectorV>>(x_val);

        assertm (std::holds_alternative<double>(x_vec->elems[0]), "SPIRV simulator: Invalid vector component value for operand 1, must be bool");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            result_vec->elems.push_back((uint64_t)(std::isnan(std::get<double>(x_vec->elems[i]))));
        }

        SetValue(result_id, result);
    } else if (type.kind == Type::Kind::BoolT){
        assertm (std::holds_alternative<double>(x_val), "SPIRV simulator: Invalid value for operand 1, must be bool");
        Value result = (uint64_t)(std::isnan(std::get<double>(x_val)));

        SetValue(result_id, result);
    } else {
        assertx ("SPIRV simulator: Invalid result type, must be vector or bool");
    }
}

void SPIRVSimulator::Op_ImageFetch(const Instruction& instruction){
    /*
    OpImageFetch

    Fetch a single texel from an image whose Sampled operand is 1.

    Result Type must be a vector of four components of floating-point type or integer type.
    Its components must be the same as Sampled Type of the underlying OpTypeImage
    (unless that underlying Sampled Type is OpTypeVoid).

    Image must be an object whose type is OpTypeImage. Its Dim operand must not be Cube, and its Sampled operand must be 1.

    Coordinate must be a scalar or vector of integer type. It contains (u[, v] …​ [, array layer]) as needed
    by the definition of Sampled Image.

    Image Operands encodes what operands follow, as per Image Operands.
    */
    assert(instruction.opcode == spv::Op::OpImageFetch);
    assertx ("SPIRV simulator: Op_ImageFetch is currently unimplemented");
}

void SPIRVSimulator::Op_ImageTexelPointer(const Instruction& instruction){
    /*
    OpImageTexelPointer

    Form a pointer to a texel of an image. Use of such a pointer is limited to atomic operations.
    Result Type must be an OpTypePointer whose Storage Class operand is Image.
    Its Type operand must be a scalar numerical type or OpTypeVoid.

    Image must have a type of OpTypePointer with Type OpTypeImage.
    The Sampled Type of the type of Image must be the same as the Type pointed to by Result Type. The Dim operand of Type must not be SubpassData.

    Coordinate and Sample specify which texel and sample within the image to form a pointer to.

    Coordinate must be a scalar or vector of integer type. It must have the number of components specified below,
    given the following Arrayed and Dim operands of the type of the OpTypeImage.

    If Arrayed is 0:
    1D: scalar
    2D: 2 components
    3D: 3 components
    Cube: 3 components
    Rect: 2 components
    Buffer: scalar

    If Arrayed is 1:
    1D: 2 components
    2D: 3 components
    Cube: 3 components; the face and layer combine into the 3rd component, layer_face,
    such that face is layer_face % 6 and layer is floor(layer_face / 6)

    Sample must be an integer type scalar. It specifies which sample to select at the given coordinate.
    Behavior is undefined unless it is a valid <id> for the value 0 when the OpTypeImage has MS of 0.
    */
    assert(instruction.opcode == spv::Op::OpImageTexelPointer);
    assertx ("SPIRV simulator: Op_ImageTexelPointer is currently unimplemented");
}

void SPIRVSimulator::OpConvertFToS(const Instruction& instruction){
    /*
    Convert value numerically from floating point to signed integer, with round toward 0.0.

    Result Type must be a scalar or vector of integer type. Behavior is undefined if Result Type is not wide enough to hold the converted value.

    Float Value must be a scalar or vector of floating-point type. It must have the same number of components as Result Type.

    Results are computed per component.
    */
   assert(instruction.opcode == spv::Op::OpConvertFToS);
    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t val_id    = instruction.words[3];

    const Type& type = types_.at(type_id);
    const Value& x_val     = GetValue(val_id);
    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(x_val), "SPIRV simulator: Invalid value type for operand 1, must be vector when using vector type");

        auto x_vec = std::get<std::shared_ptr<VectorV>>(x_val);

        assertm (std::holds_alternative<double>(x_vec->elems[0]), "SPIRV simulator: Invalid vector component value for operand 1, must be bool");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            result_vec->elems.push_back((int64_t)(truncf(std::get<double>(x_vec->elems[i]))));
        }
        SetValue(result_id, result);
    }
    else if(type.kind == Type::Kind::Int)
    {
        Value result = (int64_t)(truncf(std::get<double>(x_val)));

        SetValue(result_id, result);
    }
}
void SPIRVSimulator::OpConvertFToU(const Instruction& instruction){
    /*
    Convert value numerically from floating point to unsigned integer, with round toward 0.0.

    Result Type must be a scalar or vector of integer type, whose Signedness operand is 0. Behavior is undefined if Result Type is not wide enough to hold the converted value.

    Float Value must be a scalar or vector of floating-point type. It must have the same number of components as Result Type.

    Results are computed per component.
    */
   assert(instruction.opcode == spv::Op::OpConvertFToU);
   uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t val_id    = instruction.words[3];

    const Type& type = types_.at(type_id);
    const Value& x_val     = GetValue(val_id);
    if (type.kind == Type::Kind::Vector){
        Value result = std::make_shared<VectorV>();
        auto result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm (std::holds_alternative<std::shared_ptr<VectorV>>(x_val), "SPIRV simulator: Invalid value type for operand 1, must be vector when using vector type");

        auto x_vec = std::get<std::shared_ptr<VectorV>>(x_val);

        assertm (std::holds_alternative<double>(x_vec->elems[0]), "SPIRV simulator: Invalid vector component value for operand 1, must be float");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i){
            result_vec->elems.push_back((uint64_t)(truncf(std::get<double>(x_vec->elems[i]))));
        }
        SetValue(result_id, result);
    }
    else if(type.kind == Type::Kind::Int)
    {
        Value result = (uint64_t)(truncf(std::get<float>(x_val)));

        SetValue(result_id, result);
    }
}
void SPIRVSimulator::OpFOrdEqual(const Instruction& instruction){
    /*
    Floating-point comparison for being ordered and equal.

    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of floating-point type. They must have the same type, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFOrdEqual);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t op1_id    = instruction.words[3];
    uint32_t op2_id    = instruction.words[4];

    const Type& type     = types_.at(type_id);
    const Value& op1_val = GetValue(op1_id);
    const Value& op2_val = GetValue(op2_id);

    if(type.kind == Type::Kind::Vector){
        assertm (std::holds_alternative<VectorVPtr>(op1_val), "SPIRV simulator: Invalid value type for operand 1, must be vector when using vector type");
        assertm (std::holds_alternative<VectorVPtr>(op2_val), "SPIRV simulator: Invalid value type for operand 2, must be vector when using vector type");
        Value res = std::make_shared<VectorV>();
        auto result_vec = std::get<VectorVPtr>(res);
        auto op1_vec = std::get<VectorVPtr>(op1_val);
        auto op2_vec = std::get<VectorVPtr>(op2_val);
        for(uint32_t i = 0; i < type.vector.elem_count; i++){
            bool is_ordered = !std::isnan(std::get<double>(op1_vec->elems[i])) && !std::isnan(std::get<double>(op2_vec->elems[i]));
            bool is_equal   = std::get<double>(op1_vec->elems[i]) == std::get<double>(op2_vec->elems[i]);
            result_vec->elems.push_back((uint64_t)(is_ordered && is_equal));
        }
        SetValue(result_id, res);
    }
    else if(type.kind == Type::Kind::BoolT){
        bool is_ordered = !std::isnan(std::get<double>(op1_val)) && !std::isnan(std::get<double>(op2_val));
        bool is_equal   = std::get<double>(op1_val) == std::get<double>(op2_val);
        Value res = (uint64_t)(is_ordered && is_equal);
        SetValue(result_id, res);
    }
}
void SPIRVSimulator::OpFOrdGreaterThanEqual(const Instruction& instruction){
    /*
    Floating-point comparison if operands are ordered and Operand 1 is greater than or equal to Operand 2.

    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of floating-point type. They must have the same type, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFOrdGreaterThanEqual);
    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t op1_id    = instruction.words[3];
    uint32_t op2_id    = instruction.words[4];

    const Type& type     = types_.at(type_id);
    const Value& op1_val = GetValue(op1_id);
    const Value& op2_val = GetValue(op2_id);

    if(type.kind == Type::Kind::Vector){
        assertm (std::holds_alternative<VectorVPtr>(op1_val), "SPIRV simulator: Invalid value type for operand 1, must be vector when using vector type");
        assertm (std::holds_alternative<VectorVPtr>(op2_val), "SPIRV simulator: Invalid value type for operand 2, must be vector when using vector type");
        Value res = std::make_shared<VectorV>();
        auto result_vec = std::get<VectorVPtr>(res);
        auto op1_vec = std::get<VectorVPtr>(op1_val);
        auto op2_vec = std::get<VectorVPtr>(op2_val);
        for(uint32_t i = 0; i < type.vector.elem_count; i++){
            bool is_ordered = !std::isnan(std::get<double>(op1_vec->elems[i])) && !std::isnan(std::get<double>(op2_vec->elems[i]));
            bool is_gr_or_eq   = std::get<double>(op1_vec->elems[i]) >= std::get<double>(op2_vec->elems[i]);
            result_vec->elems.push_back((uint64_t)(is_ordered && is_gr_or_eq));
        }
        SetValue(result_id, res);
    }
    else if(type.kind == Type::Kind::BoolT){
        bool is_ordered = !std::isnan(std::get<double>(op1_val)) && !std::isnan(std::get<double>(op2_val));
        bool is_gr_or_eq   = std::get<double>(op1_val) >= std::get<double>(op2_val);
        Value res = (uint64_t)(is_ordered && is_gr_or_eq);
        SetValue(result_id, res);
    }
}
void SPIRVSimulator::OpFOrdNotEqual(const Instruction& instruction){
    /*
    Floating-point comparison for being ordered and not equal.

    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of floating-point type. They must have the same type, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFOrdNotEqual);
    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t op1_id    = instruction.words[3];
    uint32_t op2_id    = instruction.words[4];

    const Type& type     = types_.at(type_id);
    const Value& op1_val = GetValue(op1_id);
    const Value& op2_val = GetValue(op2_id);

    if(type.kind == Type::Kind::Vector){
        assertm (std::holds_alternative<VectorVPtr>(op1_val), "SPIRV simulator: Invalid value type for operand 1, must be vector when using vector type");
        assertm (std::holds_alternative<VectorVPtr>(op2_val), "SPIRV simulator: Invalid value type for operand 2, must be vector when using vector type");
        Value res = std::make_shared<VectorV>();
        auto result_vec = std::get<VectorVPtr>(res);
        auto op1_vec = std::get<VectorVPtr>(op1_val);
        auto op2_vec = std::get<VectorVPtr>(op2_val);
        for(uint32_t i = 0; i < type.vector.elem_count; i++){
            bool is_ordered = !std::isnan(std::get<double>(op1_vec->elems[i])) && !std::isnan(std::get<double>(op2_vec->elems[i]));
            bool is_equal   = std::get<double>(op1_vec->elems[i]) == std::get<double>(op2_vec->elems[i]);
            result_vec->elems.push_back((uint64_t)(is_ordered && !is_equal));
        }
        SetValue(result_id, res);
    }
    else if(type.kind == Type::Kind::BoolT){
        bool is_ordered = !std::isnan(std::get<double>(op1_val)) && !std::isnan(std::get<double>(op2_val));
        bool is_equal   = std::get<double>(op1_val) == std::get<double>(op2_val);
        Value res = (uint64_t)(is_ordered && !is_equal);
        SetValue(result_id, res);
    }
}
void SPIRVSimulator::OpFRem(const Instruction& instruction){
    /*
    The floating-point remainder whose sign matches the sign of Operand 1.

    Result Type must be a scalar or vector of floating-point type.

    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component. The resulting value is undefined if Operand 2 is 0. Otherwise, the result is the remainder r of Operand 1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the sign of Operand 1.
    */
    assert(instruction.opcode == spv::Op::OpFRem);
    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t op1_id    = instruction.words[3];
    uint32_t op2_id    = instruction.words[4];

    const Type& type     = types_.at(type_id);
    const Value& op1_val = GetValue(op1_id);
    const Value& op2_val = GetValue(op2_id);
    if(type.kind == Type::Kind::Vector){
        assertm (std::holds_alternative<VectorVPtr>(op1_val), "SPIRV simulator: Invalid value type for operand 1, must be vector when using vector type");
        assertm (std::holds_alternative<VectorVPtr>(op2_val), "SPIRV simulator: Invalid value type for operand 2, must be vector when using vector type");
        Value res = std::make_shared<VectorV>();
        auto result_vec = std::get<VectorVPtr>(res);
        auto op1_vec = std::get<VectorVPtr>(op1_val);
        auto op2_vec = std::get<VectorVPtr>(op2_val);
        for(uint32_t i = 0; i < type.vector.elem_count; i++){
            bool is_positive = std::get<double>(op1_vec->elems[i]) >= 0.0;
            double rem = remainder(std::get<double>(op1_vec->elems[i]), std::get<double>(op2_vec->elems[i]));
            if((!is_positive && rem > 0.0) || (is_positive && rem < 0.0))
            {
                rem = -rem;
            }
            result_vec->elems.push_back(rem);
        }
        SetValue(result_id, res);
    }
    else if(type.kind == Type::Kind::Float){
        bool is_positive = std::get<double>(op1_val) >= 0.0;
        double rem = remainder(std::get<double>(op1_val), std::get<double>(op2_val));
        if((!is_positive && rem > 0.0) || (is_positive && rem < 0.0))
        {
            rem = -rem;
        }
        Value res = res;
        SetValue(result_id, res);
    }
}

#undef assertx
#undef assertm

}
