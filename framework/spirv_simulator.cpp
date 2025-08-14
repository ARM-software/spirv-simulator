#include "spirv_simulator.hpp"
#include "util.hpp"

#include <iostream>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <variant>

#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-variable"

namespace SPIRVSimulator
{

constexpr uint32_t kWordCountShift = 16u;
constexpr uint32_t kOpcodeMask     = 0xFFFFu;
const std::string  execIndent      = "                      # ";

void DecodeInstruction(std::span<const uint32_t>& program_words, Instruction& instruction)
{
    /*
    Decodes an instruction from the given span stream.

    Will update the input stream so it points to the start of the next opcode.
    The results are written to the input instruction.
    */
    uint32_t first         = program_words.front();
    instruction.word_count = first >> kWordCountShift;
    instruction.opcode     = (spv::Op)(first & kOpcodeMask);

    assertm(instruction.word_count && instruction.word_count <= program_words.size(),
            "SPIRV simulator: Bad instruction size");

    instruction.words = program_words.first(instruction.word_count);
    program_words     = program_words.subspan(instruction.word_count);
}

SPIRVSimulator::SPIRVSimulator(const std::vector<uint32_t>& program_words, const InputData& input_data, bool verbose) :
    program_words_(std::move(program_words)), verbose_(verbose)
{
    if (input_data.shader_id && (input_data_.persistent_data.uninteresting_shaders.contains(input_data.shader_id)))
    {
        done_ = true;
        return;
    }

    stream_     = program_words_;
    input_data_ = input_data;

    void_type_.kind   = Type::Kind::Void;
    void_type_.scalar = { 0, false };

    DecodeHeader();

    assertm(unsupported_opcodes.size() == 0, "SPIRV simulator: Unhandled opcodes detected, implement them to run!");

    ParseAll();
    Validate();
}

void SPIRVSimulator::DecodeHeader()
{
    assertm(program_words_.size() >= 5,
            "SPIRV simulator: SPIRV binary is less than 5 words long, it must at least contain a full valid header.");

    uint32_t magic_number = program_words_[0];
    assertm(magic_number == 0x07230203, "SPIRV simulator: Magic SPIRV header number is invalid, should be: 0x07230203");

    if (verbose_)
    {
        uint32_t version   = program_words_[1];
        uint32_t generator = program_words_[2];
        uint32_t bound     = program_words_[3];
        uint32_t schema    = program_words_[4];

        std::cout << "SPIRV simulator: Shader header parsed as:" << std::endl;
        std::cout << execIndent << "Version: " << version << std::endl;
        std::cout << execIndent << "Generator: " << generator << std::endl;
        std::cout << execIndent << "Bound: " << bound << std::endl;
        std::cout << execIndent << "Schema: " << schema << std::endl << std::endl;
    }

    stream_ = std::span<const uint32_t>(program_words_).subspan(5);
}

void SPIRVSimulator::Validate()
{
    /*
    Do some early sanity checking and validation.
    */
    // TODO: Expand this (a lot)
    for (auto& [id, t] : types_)
    {
        assertm(!(t.kind == Type::Kind::Array && !types_.contains(t.array.elem_type_id)),
                "SPIRV simulator: Missing  array elem type");
        assertm(!(t.kind == Type::Kind::Vector && !types_.contains(t.vector.elem_type_id)),
                "SPIRV simulator: Missing vector elem type");
        assertm(!(t.kind == Type::Kind::RuntimeArray && !types_.contains(t.array.elem_type_id)),
                "SPIRV simulator: Missing runtie array elem type");
        assertm(!(t.kind == Type::Kind::Matrix && !types_.contains(t.matrix.col_type_id)),
                "SPIRV simulator: Missing matrix col type");
        assertm(!(t.kind == Type::Kind::Pointer && !types_.contains(t.pointer.pointee_type_id)),
                "SPIRV simulator: Missing pointee type");

        if (t.kind == Type::Kind::BoolT || t.kind == Type::Kind::Int || t.kind == Type::Kind::Float)
        {
            if (t.scalar.width == 8 || t.scalar.width == 16)
            {
                std::cout << execIndent << "Scalar width is: " << t.scalar.width
                          << ", this is untested but should work (if errors, suspect this and investigate)"
                          << std::endl;
            }

            assertm(t.scalar.width % 8 == 0,
                    "SPIRV simulator: Scalar bit width is not a multiple of eight, we dont support this at present");
            assertm(t.scalar.width == 8 || t.scalar.width == 16 || t.scalar.width == 32 || t.scalar.width == 64,
                    "SPIRV simulator: We only allow 8, 16, 32 and 64 bit scalars at present");
        }
    }

    assertm(sizeof(void*) == 8, "SPIRV simulator: Systems with non 64 bit pointers are not supported");
}

void SPIRVSimulator::ParseAll()
{
    size_t instruction_index = 0;

    if (verbose_)
    {
        std::cout << "SPIRV simulator: Parsing instructions:" << std::endl;
    }

    bool              in_function = false;
    std::set<spv::Op> unimplemented_opcodes;

    while (!stream_.empty())
    {
        Instruction instruction;
        DecodeInstruction(stream_, instruction);
        instructions_.push_back(instruction);

        bool is_implemented = ExecuteInstruction(instruction, true);
        if (!is_implemented)
        {
            unimplemented_opcodes.insert(instruction.opcode);
        }

        if ((spv::Op)instruction.opcode == spv::Op::OpExtInst)
        {
            uint32_t set_id              = instruction.words[3];
            uint32_t instruction_literal = instruction.words[4];

            if (verbose_)
            {
                std::cout << execIndent << "Found OpExtInst instruction with set ID: " << set_id
                          << ", instruction literal: " << instruction_literal << std::endl;
            }
        }

        bool has_result = false;
        bool has_type   = false;
        spv::HasResultAndType(instruction.opcode, &has_result, &has_type);

        if (has_result)
        {
            if (has_type)
            {
                result_id_to_inst_index_[instruction.words[2]] = instruction_index;
                num_result_ids_                                = std::max(num_result_ids_, instruction.words[2]) + 1;
            }
            else
            {
                result_id_to_inst_index_[instruction.words[1]] = instruction_index;
                num_result_ids_                                = std::max(num_result_ids_, instruction.words[1]) + 1;
            }
        }

        instruction_index += 1;
    }

    if (!unimplemented_opcodes.empty())
    {
        std::cout << "SPIRV simulator: Unimplemented OpCodes detected:" << std::endl;
        for (auto it = unimplemented_opcodes.begin(); it != unimplemented_opcodes.end(); ++it)
        {
            std::cout << execIndent << spv::OpToString(*it) << std::endl;
            unsupported_opcodes.insert(spv::OpToString(*it));
        }
    }

    // Preinitialize to max result ID
    values_.resize(num_result_ids_, std::monostate{});

    instruction_index = 0;
    for (const auto& instruction : instructions_)
    {

        if (verbose_)
        {
            PrintInstruction(instruction);
        }

        switch (instruction.opcode)
        {
            case spv::Op::OpFunction:
            {
                in_function                  = true;
                funcs_[instruction.words[2]] = { instruction_index, instruction_index + 1, {}, {} };
                prev_defined_func_id_        = instruction.words[2];
                break;
            }
            case spv::Op::OpFunctionEnd:
                in_function = false;
                break;
            case spv::Op::OpFunctionParameter:
            {
                funcs_[prev_defined_func_id_].parameter_ids_.push_back(instruction.words[2]);
                funcs_[prev_defined_func_id_].parameter_type_ids_.push_back(instruction.words[1]);
                break;
            }
            case spv::Op::OpEntryPoint:
            {
                uint32_t entry_point_id       = instruction.words[2];
                entry_points_[entry_point_id] = "";
                break;
            }
            default:
            {
                if (!in_function)
                {
                    ExecuteInstruction(instruction);
                }
                break;
            }
        }

        ++instruction_index;
    }
}

bool SPIRVSimulator::Run()
{
    if (done_)
    {
        return false;
    }

    if (funcs_.empty())
    {
        std::cerr << "SPIRV simulator: No functions defined in the shader, cannot start execution" << std::endl;
        return false;
    }

    uint32_t entry_point_function_id = 0;

    if (input_data_.entry_point_op_name != "")
    {
        for (const auto& it : entry_points_)
        {
            if (it.second == input_data_.entry_point_op_name)
            {
                if (verbose_)
                    std::cout << "SPIRV simulator: Using entry point with OpName label: " << it.second << std::endl;
                entry_point_function_id = it.first;
                break;
            }
        }

        assertm(entry_point_function_id != 0,
                "SPIRV simulator: Failed to find an entry point with the given OpName label");
    }

    if (entry_point_function_id == 0)
    {
        if (entry_points_.find(input_data_.entry_point_id) == entry_points_.end())
        {
            if (verbose_)
                std::cout << "SPIRV simulator: Warning, entry point function with index: " << input_data_.entry_point_id
                          << " not found, using first available" << std::endl;
            entry_point_function_id = entry_points_.begin()->first;
        }
        else
        {
            if (verbose_)
                std::cout << "SPIRV simulator: Using entry point with ID: " << input_data_.entry_point_id << std::endl;
            entry_point_function_id = input_data_.entry_point_id;
        }
    }

    if (verbose_)
    {
        std::cout << "SPIRV simulator: Starting execution at entry point with function ID: " << entry_point_function_id
                  << std::endl;
    }

    FunctionInfo& function_info = funcs_[entry_point_function_id];

    // We can set the return value to whatever, ignored if the call stack is empty on return
    call_stack_.push_back({ function_info.first_inst_index, 0, current_heap_index_ });
    ExecuteInstructions();

    if ((!has_buffer_writes_) && (physical_address_pointer_source_data_.size() == 0) && input_data_.shader_id)
    {
        input_data_.persistent_data.uninteresting_shaders.insert(input_data_.shader_id);
    }

    return false;
}

void SPIRVSimulator::ExecuteInstructions()
{
    while (!call_stack_.empty())
    {
        auto&              stack_frame = call_stack_.back();
        const Instruction& instruction = instructions_[stack_frame.pc++];

        if (verbose_)
        {
            PrintInstruction(instruction);
        }

        if (!ExecuteInstruction(instruction))
        {
            HandleUnimplementedOpcode(instruction);
        }
    }

    if (verbose_)
    {
        std::cout << "SPIRV simulator: Execution complete!\n" << std::endl;
    }

    for (const std::pair<PointerV, PointerV>& pointer_pair : pointers_to_physical_address_pointers_)
    {
        const PointerV& phys_ppointer = pointer_pair.first;
        const PointerV& phys_pointer  = pointer_pair.second;

        DataSourceBits source_data;
        source_data.location       = BitLocation::StorageClass;
        source_data.storage_class  = (spv::StorageClass)phys_ppointer.storage_class;
        source_data.idx            = 0;
        source_data.bit_offset     = 0;
        source_data.bitcount       = 64;
        source_data.val_bit_offset = 0;

        if (phys_ppointer.storage_class == spv::StorageClass::StorageClassFunction)
        {
            // We dont care about these, pointers that are temporary wont exist outside the shader execution context
            // and there will be other references to the actual buffer inputs
            continue;
        }

        if (phys_ppointer.storage_class == spv::StorageClass::StorageClassPushConstant)
        {
            source_data.binding_id = 0;
            source_data.set_id     = 0;
        }
        else if (phys_ppointer.storage_class != spv::StorageClass::StorageClassPhysicalStorageBuffer)
        {
            assertm(HasDecorator(phys_ppointer.base_result_id, spv::Decoration::DecorationDescriptorSet),
                    "SPIRV simulator: Missing DecorationDescriptorSet for pointee object");
            assertm(HasDecorator(phys_ppointer.base_result_id, spv::Decoration::DecorationBinding),
                    "SPIRV simulator: Missing DecorationBinding for pointee object");

            source_data.binding_id =
                GetDecoratorLiteral(phys_ppointer.base_result_id, spv::Decoration::DecorationBinding);
            source_data.set_id =
                GetDecoratorLiteral(phys_ppointer.base_result_id, spv::Decoration::DecorationDescriptorSet);
        }
        else
        {
            source_data.binding_id = 0;
            source_data.set_id     = 0;
        }

        source_data.byte_offset = GetPointerOffset(phys_ppointer);

        PhysicalAddressData output_result;
        output_result.raw_pointer_value = RemapHostToClientPointer(phys_pointer.pointer_handle);
        output_result.bit_components.push_back(source_data);
        physical_address_pointer_source_data_.push_back(output_result);
    }
}

void SPIRVSimulator::WriteOutputs()
{
    assertx("SPIRV simulator: Value writeout not implemented yet");
}

bool SPIRVSimulator::ExecuteInstruction(const Instruction& instruction, bool dummy_exec)
{
#define R(OPF)                \
    {                         \
        if (!dummy_exec)      \
        {                     \
            OPF(instruction); \
        }                     \
        return true;          \
    }

    switch (instruction.opcode)
    {
        case spv::Op::OpTypeVoid:
            R(T_Void)
        case spv::Op::OpTypeBool:
            R(T_Bool)
        case spv::Op::OpTypeInt:
            R(T_Int)
        case spv::Op::OpTypeFloat:
            R(T_Float)
        case spv::Op::OpTypeVector:
            R(T_Vector)
        case spv::Op::OpTypeMatrix:
            R(T_Matrix)
        case spv::Op::OpTypeArray:
            R(T_Array)
        case spv::Op::OpTypeStruct:
            R(T_Struct)
        case spv::Op::OpTypePointer:
            R(T_Pointer)
        case spv::Op::OpTypeForwardPointer:
            R(T_ForwardPointer)
        case spv::Op::OpTypeRuntimeArray:
            R(T_RuntimeArray)
        case spv::Op::OpTypeFunction:
            R(T_Function)
        case spv::Op::OpTypeImage:
            R(T_Image)
        case spv::Op::OpTypeSampler:
            R(T_Sampler)
        case spv::Op::OpTypeSampledImage:
            R(T_SampledImage)
        case spv::Op::OpTypeOpaque:
            R(T_Opaque)
        case spv::Op::OpTypeNamedBarrier:
            R(T_NamedBarrier)
        case spv::Op::OpTypeAccelerationStructureKHR:
            R(T_AccelerationStructureKHR)
        case spv::Op::OpTypeRayQueryKHR:
            R(T_RayQueryKHR)
        case spv::Op::OpEntryPoint:
            R(Op_EntryPoint)
        case spv::Op::OpExtInstImport:
            R(Op_ExtInstImport)
        case spv::Op::OpConstant:
            R(Op_Constant)
        case spv::Op::OpConstantComposite:
            R(Op_ConstantComposite)
        case spv::Op::OpCompositeConstruct:
            R(Op_CompositeConstruct)
        case spv::Op::OpVariable:
            R(Op_Variable)
        case spv::Op::OpImageTexelPointer:
            R(Op_ImageTexelPointer)
        case spv::Op::OpLoad:
            R(Op_Load)
        case spv::Op::OpStore:
            R(Op_Store)
        case spv::Op::OpAccessChain:
            R(Op_AccessChain)
        case spv::Op::OpInBoundsAccessChain:
            R(Op_AccessChain)
        case spv::Op::OpFunction:
            R(Op_Function)
        case spv::Op::OpFunctionEnd:
            R(Op_FunctionEnd)
        case spv::Op::OpFunctionCall:
            R(Op_FunctionCall)
        case spv::Op::OpLabel:
            R(Op_Label)
        case spv::Op::OpBranch:
            R(Op_Branch)
        case spv::Op::OpBranchConditional:
            R(Op_BranchConditional)
        case spv::Op::OpReturn:
            R(Op_Return)
        case spv::Op::OpReturnValue:
            R(Op_ReturnValue)
        case spv::Op::OpINotEqual:
            R(Op_INotEqual)
        case spv::Op::OpFAdd:
            R(Op_FAdd)
        case spv::Op::OpExtInst:
            R(Op_ExtInst)
        case spv::Op::OpSelectionMerge:
            R(Op_SelectionMerge)
        case spv::Op::OpFMul:
            R(Op_FMul)
        case spv::Op::OpLoopMerge:
            R(Op_LoopMerge)
        case spv::Op::OpIAdd:
            R(Op_IAdd)
        case spv::Op::OpISub:
            R(Op_ISub)
        case spv::Op::OpLogicalNot:
            R(Op_LogicalNot)
        case spv::Op::OpCapability:
            R(Op_Capability)
        case spv::Op::OpExtension:
            R(Op_Extension)
        case spv::Op::OpMemoryModel:
            R(Op_MemoryModel)
        case spv::Op::OpExecutionMode:
            R(Op_ExecutionMode)
        case spv::Op::OpSource:
            R(Op_Source)
        case spv::Op::OpSourceExtension:
            R(Op_SourceExtension)
        case spv::Op::OpName:
            R(Op_Name)
        case spv::Op::OpMemberName:
            R(Op_MemberName)
        case spv::Op::OpDecorate:
            R(Op_Decorate)
        case spv::Op::OpMemberDecorate:
            R(Op_MemberDecorate)
        case spv::Op::OpArrayLength:
            R(Op_ArrayLength)
        case spv::Op::OpSpecConstant:
            R(Op_SpecConstant)
        case spv::Op::OpSpecConstantOp:
            R(Op_SpecConstantOp)
        case spv::Op::OpSpecConstantComposite:
            R(Op_SpecConstantComposite)
        case spv::Op::OpSpecConstantFalse:
            R(Op_SpecConstantFalse)
        case spv::Op::OpSpecConstantTrue:
            R(Op_SpecConstantTrue)
        case spv::Op::OpUGreaterThanEqual:
            R(Op_UGreaterThanEqual)
        case spv::Op::OpPhi:
            R(Op_Phi)
        case spv::Op::OpConvertUToF:
            R(Op_ConvertUToF)
        case spv::Op::OpConvertSToF:
            R(Op_ConvertSToF)
        case spv::Op::OpFDiv:
            R(Op_FDiv)
        case spv::Op::OpFSub:
            R(Op_FSub)
        case spv::Op::OpVectorTimesScalar:
            R(Op_VectorTimesScalar)
        case spv::Op::OpSLessThan:
            R(Op_SLessThan)
        case spv::Op::OpDot:
            R(Op_Dot)
        case spv::Op::OpFOrdGreaterThan:
            R(Op_FOrdGreaterThan)
        case spv::Op::OpFOrdGreaterThanEqual:
            R(Op_FOrdGreaterThanEqual)
        case spv::Op::OpFOrdEqual:
            R(Op_FOrdEqual)
        case spv::Op::OpFOrdNotEqual:
            R(Op_FOrdNotEqual)
        case spv::Op::OpCompositeExtract:
            R(Op_CompositeExtract)
        case spv::Op::OpBitcast:
            R(Op_Bitcast)
        case spv::Op::OpIMul:
            R(Op_IMul)
        case spv::Op::OpConvertUToPtr:
            R(Op_ConvertUToPtr)
        case spv::Op::OpUDiv:
            R(Op_UDiv)
        case spv::Op::OpUMod:
            R(Op_UMod)
        case spv::Op::OpULessThan:
            R(Op_ULessThan)
        case spv::Op::OpConstantTrue:
            R(Op_ConstantTrue)
        case spv::Op::OpConstantFalse:
            R(Op_ConstantFalse)
        case spv::Op::OpConstantNull:
            R(Op_ConstantNull)
        case spv::Op::OpAtomicIAdd:
            R(Op_AtomicIAdd)
        case spv::Op::OpAtomicISub:
            R(Op_AtomicISub)
        case spv::Op::OpSelect:
            R(Op_Select)
        case spv::Op::OpIEqual:
            R(Op_IEqual)
        case spv::Op::OpVectorShuffle:
            R(Op_VectorShuffle)
        case spv::Op::OpCompositeInsert:
            R(Op_CompositeInsert)
        case spv::Op::OpTranspose:
            R(Op_Transpose)
        case spv::Op::OpSampledImage:
            R(Op_SampledImage)
        case spv::Op::OpImageSampleImplicitLod:
            R(Op_ImageSampleImplicitLod)
        case spv::Op::OpImageSampleExplicitLod:
            R(Op_ImageSampleExplicitLod)
        case spv::Op::OpImageFetch:
            R(Op_ImageFetch)
        case spv::Op::OpImageGather:
            R(Op_ImageGather)
        case spv::Op::OpImageRead:
            R(Op_ImageRead)
        case spv::Op::OpImageWrite:
            R(Op_ImageWrite)
        case spv::Op::OpImageQuerySize:
            R(Op_ImageQuerySize)
        case spv::Op::OpImageQuerySizeLod:
            R(Op_ImageQuerySizeLod)
        case spv::Op::OpFNegate:
            R(Op_FNegate)
        case spv::Op::OpMatrixTimesVector:
            R(Op_MatrixTimesVector)
        case spv::Op::OpUGreaterThan:
            R(Op_UGreaterThan)
        case spv::Op::OpFOrdLessThan:
            R(Op_FOrdLessThan)
        case spv::Op::OpFOrdLessThanEqual:
            R(Op_FOrdLessThanEqual)
        case spv::Op::OpShiftRightLogical:
            R(Op_ShiftRightLogical)
        case spv::Op::OpShiftLeftLogical:
            R(Op_ShiftLeftLogical)
        case spv::Op::OpBitwiseOr:
            R(Op_BitwiseOr)
        case spv::Op::OpBitwiseAnd:
            R(Op_BitwiseAnd)
        case spv::Op::OpSwitch:
            R(Op_Switch)
        case spv::Op::OpAll:
            R(Op_All)
        case spv::Op::OpAny:
            R(Op_Any)
        case spv::Op::OpBitCount:
            R(Op_BitCount)
        case spv::Op::OpKill:
            R(Op_Kill)
        case spv::Op::OpUnreachable:
            R(Op_Unreachable)
        case spv::Op::OpUndef:
            R(Op_Undef)
        case spv::Op::OpVectorTimesMatrix:
            R(Op_VectorTimesMatrix)
        case spv::Op::OpULessThanEqual:
            R(Op_ULessThanEqual)
        case spv::Op::OpSLessThanEqual:
            R(Op_SLessThanEqual)
        case spv::Op::OpSGreaterThanEqual:
            R(Op_SGreaterThanEqual)
        case spv::Op::OpSGreaterThan:
            R(Op_SGreaterThan)
        case spv::Op::OpSDiv:
            R(Op_SDiv)
        case spv::Op::OpSNegate:
            R(Op_SNegate)
        case spv::Op::OpLogicalOr:
            R(Op_LogicalOr)
        case spv::Op::OpLogicalAnd:
            R(Op_LogicalAnd)
        case spv::Op::OpMatrixTimesMatrix:
            R(Op_MatrixTimesMatrix)
        case spv::Op::OpIsNan:
            R(Op_IsNan)
        case spv::Op::OpFunctionParameter:
            R(Op_FunctionParameter)
        case spv::Op::OpEmitVertex:
            R(Op_EmitVertex)
        case spv::Op::OpEndPrimitive:
            R(Op_EndPrimitive)
        case spv::Op::OpFConvert:
            R(Op_FConvert)
        case spv::Op::OpImage:
            R(Op_Image)
        case spv::Op::OpConvertFToS:
            R(Op_ConvertFToS)
        case spv::Op::OpConvertFToU:
            R(Op_ConvertFToU)
        case spv::Op::OpFRem:
            R(Op_FRem)
        case spv::Op::OpFMod:
            R(Op_FMod)
        case spv::Op::OpAtomicOr:
            R(Op_AtomicOr)
        case spv::Op::OpAtomicUMax:
            R(Op_AtomicUMax)
        case spv::Op::OpAtomicUMin:
            R(Op_AtomicUMin)
        case spv::Op::OpBitReverse:
            R(Op_BitReverse)
        case spv::Op::OpBitwiseXor:
            R(Op_BitwiseXor)
        case spv::Op::OpControlBarrier:
            R(Op_ControlBarrier)
        case spv::Op::OpShiftRightArithmetic:
            R(Op_ShiftRightArithmetic)
        case spv::Op::OpGroupNonUniformAll:
            R(Op_GroupNonUniformAll)
        case spv::Op::OpGroupNonUniformAny:
            R(Op_GroupNonUniformAny)
        case spv::Op::OpGroupNonUniformBallot:
            R(Op_GroupNonUniformBallot)
        case spv::Op::OpGroupNonUniformBallotBitCount:
            R(Op_GroupNonUniformBallotBitCount)
        case spv::Op::OpGroupNonUniformBroadcastFirst:
            R(Op_GroupNonUniformBroadcastFirst)
        case spv::Op::OpGroupNonUniformElect:
            R(Op_GroupNonUniformElect)
        case spv::Op::OpGroupNonUniformFMax:
            R(Op_GroupNonUniformFMax)
        case spv::Op::OpGroupNonUniformFMin:
            R(Op_GroupNonUniformFMin)
        case spv::Op::OpGroupNonUniformIAdd:
            R(Op_GroupNonUniformIAdd)
        case spv::Op::OpGroupNonUniformShuffle:
            R(Op_GroupNonUniformShuffle)
        case spv::Op::OpGroupNonUniformUMax:
            R(Op_GroupNonUniformUMax)
        case spv::Op::OpRayQueryGetIntersectionBarycentricsKHR:
            R(Op_RayQueryGetIntersectionBarycentricsKHR)
        case spv::Op::OpRayQueryGetIntersectionFrontFaceKHR:
            R(Op_RayQueryGetIntersectionFrontFaceKHR)
        case spv::Op::OpRayQueryGetIntersectionGeometryIndexKHR:
            R(Op_RayQueryGetIntersectionGeometryIndexKHR)
        case spv::Op::OpRayQueryGetIntersectionInstanceCustomIndexKHR:
            R(Op_RayQueryGetIntersectionInstanceCustomIndexKHR)
        case spv::Op::OpRayQueryGetIntersectionInstanceIdKHR:
            R(Op_RayQueryGetIntersectionInstanceIdKHR)
        case spv::Op::OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR:
            R(Op_RayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR)
        case spv::Op::OpRayQueryGetIntersectionPrimitiveIndexKHR:
            R(Op_RayQueryGetIntersectionPrimitiveIndexKHR)
        case spv::Op::OpRayQueryGetIntersectionTKHR:
            R(Op_RayQueryGetIntersectionTKHR)
        case spv::Op::OpRayQueryGetIntersectionTypeKHR:
            R(Op_RayQueryGetIntersectionTypeKHR)
        case spv::Op::OpRayQueryGetIntersectionWorldToObjectKHR:
            R(Op_RayQueryGetIntersectionWorldToObjectKHR)
        case spv::Op::OpRayQueryGetWorldRayDirectionKHR:
            R(Op_RayQueryGetWorldRayDirectionKHR)
        case spv::Op::OpRayQueryInitializeKHR:
            R(Op_RayQueryInitializeKHR)
        case spv::Op::OpRayQueryProceedKHR:
            R(Op_RayQueryProceedKHR)
        case spv::Op::OpDecorateString:
            R(Op_DecorateString)
        default:
        {
            return false;
        }
    }

#undef R
}

void SPIRVSimulator::CreateExecutionFork(const SPIRVSimulator& source,
                                         uint32_t              branching_value_id,
                                         uint32_t              target_block_id)
{
    // Do a shallow copy
    *this = source;

    // Then duplicate the values
    for (auto& value : values_)
    {
        value = CopyValue(value);
    }

    for (auto& value : function_heap_)
    {
        value = CopyValue(value);
    }

    for (auto& heap_pair : heaps_)
    {
        for (auto& value : heap_pair.second)
        {
            value = CopyValue(value);
        }
    }

    is_execution_fork = true;
    current_fork_index_ += 1;

    fork_abort_trigger_id_ = branching_value_id;
    fork_abort_target_id_  = target_block_id;

    auto& stack_frame = call_stack_.back();
    stack_frame.pc -= 1;

    // For now, just invert the value, this allows us to continue execution in release builds for some more testing
    // TODO: If it ever becomes necessary, we should backtrack from the candidate branching boolean and change the
    // operands in the
    //       instructions resulting in its current value such that the result of its source instruction
    //       becomes the inverse of its current value

    const Value& branch_val  = GetValue(branching_value_id);
    uint64_t     branch_bool = std::get<uint64_t>(branch_val);

    if (branch_bool)
    {
        SetValue(branching_value_id, (uint64_t)(0));
    }
    else
    {
        SetValue(branching_value_id, (uint64_t)(1));
    }

    ClearIsArbitrary(branching_value_id);

    ExecuteInstructions();
}

void SPIRVSimulator::HandleUnimplementedOpcode(const Instruction& instruction)
{
    if (verbose_)
    {
        std::cout << execIndent << "Found unimplemented opcode during execution of instruction: " << std::endl;
        PrintInstruction(instruction);
    }
}

std::string SPIRVSimulator::GetValueString(const Value& value)
{
    if (std::holds_alternative<double>(value))
    {
        return "double";
    }
    if (std::holds_alternative<uint64_t>(value))
    {
        return "uint64_t";
    }
    if (std::holds_alternative<int64_t>(value))
    {
        return "int64_t";
    }
    if (std::holds_alternative<std::monostate>(value))
    {
        return "std::monostate";
    }
    if (std::holds_alternative<std::shared_ptr<VectorV>>(value))
    {
        return "std::shared_ptr<VectorV>";
    }
    if (std::holds_alternative<std::shared_ptr<MatrixV>>(value))
    {
        return "std::shared_ptr<MatrixV>";
    }
    if (std::holds_alternative<std::shared_ptr<AggregateV>>(value))
    {
        return "std::shared_ptr<AggregateV>";
    }
    if (std::holds_alternative<PointerV>(value))
    {
        return "PointerV";
    }
    if (std::holds_alternative<SampledImageV>(value))
    {
        return "SampledImageV";
    }

    return "";
}

std::string SPIRVSimulator::GetTypeString(const Type& type)
{
    if (type.kind == Type::Kind::Void)
    {
        return "void";
    }
    if (type.kind == Type::Kind::BoolT)
    {
        return "bool";
    }
    if (type.kind == Type::Kind::Int)
    {
        return "int";
    }
    if (type.kind == Type::Kind::Float)
    {
        return "float";
    }
    if (type.kind == Type::Kind::Vector)
    {
        return "vector";
    }
    if (type.kind == Type::Kind::Matrix)
    {
        return "matrix";
    }
    if (type.kind == Type::Kind::Array)
    {
        return "array";
    }
    if (type.kind == Type::Kind::RuntimeArray)
    {
        return "runtime_array";
    }
    if (type.kind == Type::Kind::Struct)
    {
        return "struct";
    }
    if (type.kind == Type::Kind::Pointer)
    {
        return "pointer";
    }

    if (type.kind == Type::Kind::Image)
    {
        return "Image";
    }
    if (type.kind == Type::Kind::Sampler)
    {
        return "Sampler";
    }
    if (type.kind == Type::Kind::SampledImage)
    {
        return "SampledImage";
    }
    if (type.kind == Type::Kind::Opaque)
    {
        return "Opaque";
    }
    if (type.kind == Type::Kind::NamedBarrier)
    {
        return "NamedBarrier";
    }
    if (type.kind == Type::Kind::AccelerationStructureKHR)
    {
        return "AccelerationStructureKHR";
    }
    if (type.kind == Type::Kind::RayQueryKHR)
    {
        return "RayQueryKHR";
    }

    return "";
}

void SPIRVSimulator::PrintInstruction(const Instruction& instruction)
{
    bool has_result = false;
    bool has_type   = false;
    spv::HasResultAndType(instruction.opcode, &has_result, &has_type);

    std::stringstream result_and_type;

    uint32_t result_offset = 0;
    if (has_result)
    {
        if (has_type)
        {
            result_offset = 2;
        }
        else
        {
            result_offset = 1;
        }
    }

    if (has_type)
    {
        bool has_type_value = types_.find(instruction.words[1]) != types_.end();
        if (has_type_value)
        {
            result_and_type << GetTypeString(GetTypeByTypeId(instruction.words[1])) << "(" << instruction.words[1]
                            << ") ";
        }
    }

    if (result_offset)
    {
        result_and_type << instruction.words[result_offset] << " ";
    }

    std::cout << std::right << std::setw(22) << result_and_type.str() << spv::OpToString(instruction.opcode) << " ";

    if (instruction.opcode == spv::Op::OpExtInstImport)
    {
        std::cout << std::string((char*)(&instruction.words[2]), (instruction.word_count - 2) * 4);
    }
    else if (instruction.opcode == spv::Op::OpName)
    {
        std::cout << instruction.words[1] << " ";
        std::cout << std::string((char*)(&instruction.words[2]), (instruction.word_count - 2) * 4);
    }
    else if (instruction.opcode == spv::Op::OpMemberName)
    {
        std::cout << instruction.words[1] << " " << instruction.words[2] << " ";
        std::cout << std::string((char*)(&instruction.words[3]), (instruction.word_count - 3) * 4);
    }
    else if (instruction.opcode == spv::Op::OpExtension)
    {
        std::cout << instruction.words[1] << " ";
        std::cout << std::string((char*)(&instruction.words[2]), (instruction.word_count - 2) * 4);
    }
    else if (instruction.opcode == spv::Op::OpEntryPoint)
    {
        std::cout << instruction.words[1] << " " << instruction.words[2] << " ";
        std::cout << std::string((char*)(&instruction.words[3]), (instruction.word_count - 3) * 4);
    }
    else if (instruction.opcode == spv::Op::OpDecorateString)
    {
        std::cout << instruction.words[1] << " " << instruction.words[2] << " " << "<...>";
    }
    else if (instruction.opcode == spv::Op::OpTypePointer)
    {
        std::cout << spv::StorageClassToString((spv::StorageClass)instruction.words[2]) << " "
                  << GetTypeString(GetTypeByTypeId(instruction.words[3])) << "(" << instruction.words[3] << ") ";
    }
    else if (instruction.opcode == spv::Op::OpVariable)
    {
        std::cout << spv::StorageClassToString((spv::StorageClass)instruction.words[3]) << " ";
        for (uint32_t i = 4; i < instruction.word_count; ++i)
        {
            std::cout << instruction.words[i] << " ";
        }
    }
    else
    {
        for (uint32_t i = result_offset; i < instruction.word_count; ++i)
        {
            if (i == result_offset)
            {
                continue;
            }
            if (instruction.opcode == spv::Op::OpDecorate)
            {
                if (i == 2)
                {
                    std::cout << spv::DecorationToString((spv::Decoration)instruction.words[i]) << " ";
                }
                else
                {
                    std::cout << instruction.words[i] << " ";
                }
            }
            else if (instruction.opcode == spv::Op::OpMemberDecorate)
            {
                if (i == 3)
                {
                    std::cout << spv::DecorationToString((spv::Decoration)instruction.words[i]) << " ";
                }
                else
                {
                    std::cout << instruction.words[i] << " ";
                }
            }
            else
            {
                std::cout << instruction.words[i] << " ";
            }
        }
    }

    std::cout << std::endl;
}

bool SPIRVSimulator::HasDecorator(uint32_t result_id, spv::Decoration decorator)
{
    /*
    Checks if a result_id has been decorated with the given decoration.
    */
    if (decorators_.find(result_id) != decorators_.end())
    {
        for (const auto& decorator_data : decorators_.at(result_id))
        {
            if (decorator == decorator_data.kind)
            {
                return true;
            }
        }
    }
    else if (struct_decorators_.find(result_id) != struct_decorators_.end())
    {
        assertx("SPIRV simulator: Unimplemented branch in HasDecorator");
    }

    return false;
}

bool SPIRVSimulator::HasDecorator(uint32_t result_id, uint32_t member_id, spv::Decoration decorator)
{
    /*
    Checks if a given member in a result_id has been decorated with the given decoration.
    */
    if (struct_decorators_.find(result_id) != struct_decorators_.end())
    {
        if (struct_decorators_.at(result_id).find(member_id) != struct_decorators_.at(result_id).end())
        {
            for (const auto& decorator_data : struct_decorators_.at(result_id).at(member_id))
            {
                if (decorator == decorator_data.kind)
                {
                    return true;
                }
            }
        }
        else
        {
            return false;
        }
    }
    else if (decorators_.find(result_id) != decorators_.end())
    {
        assertx("SPIRV simulator: Unimplemented branch in HasDecorator (member version)");
    }

    return false;
}

uint32_t SPIRVSimulator::GetDecoratorLiteral(uint32_t result_id, spv::Decoration decorator, size_t literal_offset)
{
    /*
    This will abort if the target id does not have the given decorator
    Check with HasDecorator first
    */
    if (decorators_.find(result_id) != decorators_.end())
    {
        for (const auto& decorator_data : decorators_.at(result_id))
        {
            if (decorator_data.kind == decorator)
            {
                if (decorator_data.literals.size() <= literal_offset)
                {
                    assertx("SPIRV simulator: Literal offset OOB");
                }

                return decorator_data.literals[literal_offset];
            }
        }
    }

    // Should never happen
    assertx("SPIRV simulator: No matching decorators for result ID");
    return 0;
}

uint32_t SPIRVSimulator::GetDecoratorLiteral(uint32_t        result_id,
                                             uint32_t        member_id,
                                             spv::Decoration decorator,
                                             size_t          literal_offset)
{
    /*
    This will abort if the target id does not have the given decorator
    Check with HasDecorator first
    */
    if (struct_decorators_.find(result_id) != struct_decorators_.end())
    {
        if (struct_decorators_.at(result_id).find(member_id) != struct_decorators_.at(result_id).end())
        {
            for (const auto& decorator_data : struct_decorators_.at(result_id).at(member_id))
            {
                if (decorator_data.kind == decorator)
                {
                    assertm(decorator_data.literals.size() > literal_offset, "SPIRV simulator: Literal offset OOB");

                    return decorator_data.literals[literal_offset];
                }
            }
        }
    }

    // Should never happen
    assertx("SPIRV simulator: No matching decorators for result ID");
    return 0;
}

const Type& SPIRVSimulator::GetTypeByResultId(uint32_t result_id) const
{
    /*
    Returns the type struct mapping to a given result_id.
    result_id must be the result ID of a spirv instruction.
    */
    assertm(result_id_to_inst_index_.find(result_id) != result_id_to_inst_index_.end(),
            "SPIRV simulator: No instruction found for result_id");

    size_t             instruction_index = result_id_to_inst_index_.at(result_id);
    const Instruction& instruction       = instructions_[instruction_index];

    bool has_result = false;
    bool has_type   = false;
    spv::HasResultAndType(instruction.opcode, &has_result, &has_type);

    if (has_type)
    {
        uint32_t inst_type_id = instruction.words[1];
        assertm(types_.find(inst_type_id) != types_.end(), "SPIRV simulator: No type found for type_id");
        return types_.at(inst_type_id);
    }
    else
    {
        return void_type_;
    }
}

const Type& SPIRVSimulator::GetTypeByTypeId(uint32_t type_id) const
{
    /*
    Returns the type struct mapping to a given type_id.
    */
    assertm(types_.find(type_id) != types_.end(), "SPIRV simulator: Type does not exist");
    return types_.at(type_id);
}

// ---------------------------------------------------------------------------
//  Value creation and inspect helpers
// ---------------------------------------------------------------------------

size_t CountBitsUInt(uint64_t value, size_t max_bits)
{
    size_t count = 0;

    while (max_bits)
    {
        count += value & 1;
        value >>= 1;
        max_bits -= 1;
    }
    return count;
}

size_t SPIRVSimulator::CountSetBits(const Value& value, uint32_t type_id, bool* is_arbitrary)
{
    assertm(types_.find(type_id) != types_.end(), "SPIRV simulator: No valid type for the given ID was found");

    const Type& type = GetTypeByTypeId(type_id);

    assertm(type.kind != Type::Kind::Void, "SPIRV simulator: Attempt to extract set bits of a void type value");

    *is_arbitrary   = false;
    size_t bitcount = 0;
    if (type.kind == Type::Kind::BoolT)
    {
        bitcount += CountBitsUInt(std::get<uint64_t>(value), type.scalar.width);
    }
    else if (type.kind == Type::Kind::Int)
    {
        if (!type.scalar.is_signed)
        {
            bitcount += CountBitsUInt(std::get<uint64_t>(value), type.scalar.width);
        }
        else
        {
            bitcount += CountBitsUInt(bit_cast<uint64_t>(std::get<int64_t>(value)), type.scalar.width);
        }
    }
    else if (type.kind == Type::Kind::Float)
    {
        bitcount += CountBitsUInt(bit_cast<uint64_t>(std::get<double>(value)), type.scalar.width);
    }
    else if (type.kind == Type::Kind::Vector)
    {
        uint32_t                        elem_type_id = type.vector.elem_type_id;
        const std::shared_ptr<VectorV>& vec          = std::get<std::shared_ptr<VectorV>>(value);

        for (size_t i = 0; i < type.vector.elem_count; ++i)
        {
            bitcount += CountSetBits(vec->elems[i], elem_type_id, is_arbitrary);
        }
    }
    else if (type.kind == Type::Kind::Matrix)
    {
        uint32_t                        col_type_id = type.matrix.col_type_id;
        const std::shared_ptr<MatrixV>& mat         = std::get<std::shared_ptr<MatrixV>>(value);

        for (size_t i = 0; i < type.matrix.col_count; ++i)
        {
            bitcount += CountSetBits(mat->cols[i], col_type_id, is_arbitrary);
        }
    }
    else if (type.kind == Type::Kind::Array)
    {
        uint32_t                           elem_type_id = type.vector.elem_type_id;
        uint64_t                           array_len    = std::get<uint64_t>(GetValue(type.array.length_id));
        const std::shared_ptr<AggregateV>& agg          = std::get<std::shared_ptr<AggregateV>>(value);

        for (size_t i = 0; i < array_len; ++i)
        {
            bitcount += CountSetBits(agg->elems[i], elem_type_id, is_arbitrary);
        }
    }
    else if (type.kind == Type::Kind::RuntimeArray)
    {
        assertx("SPIRV simulator: Fetching bitsize of RuntimeArray, this is currently not implemented");

        // uint32_t elem_type_id = type.vector.elem_type_id;
        // uint64_t array_len = std::get<uint64_t>(GetValue(type.array.length_id));
        // bitcount += GetBitizeOfType(elem_type_id);
    }
    else if (type.kind == Type::Kind::Struct)
    {
        assertm(struct_members_.find(type_id) != struct_members_.end(), "SPIRV simulator: Struct has no members");
        const std::shared_ptr<AggregateV>& agg = std::get<std::shared_ptr<AggregateV>>(value);

        uint32_t member_index = 0;
        for (uint32_t member_type_id : struct_members_.at(type_id))
        {
            bitcount += CountSetBits(agg->elems[member_index], member_type_id, is_arbitrary);
            member_index += 1;
        }
    }
    else if (type.kind == Type::Kind::Pointer)
    {
        // This makes the result arbitrary
        *is_arbitrary = true;
        bitcount += 8 * 8;
    }

    return bitcount;
}

size_t SPIRVSimulator::GetBitizeOfType(uint32_t type_id)
{
    /*
    Returns the full bitsize of the type associated with the given type ID.
    type_id must be the result of a OpType* instruction.
    */
    assertm(types_.find(type_id) != types_.end(), "SPIRV simulator: No valid type for the given ID was found");

    const Type& type = GetTypeByTypeId(type_id);

    assertm(type.kind != Type::Kind::Void, "SPIRV simulator: Attempt to extract size of a void type");

    if (verbose_)
    {
        std::cout << execIndent << "Fetching bitsize of type with ID: " << type_id << std::endl;
    }

    size_t bitcount = 0;
    if (type.kind == Type::Kind::BoolT || type.kind == Type::Kind::Int || type.kind == Type::Kind::Float)
    {
        bitcount += type.scalar.width;
    }
    else if (type.kind == Type::Kind::Vector)
    {
        uint32_t elem_type_id = type.vector.elem_type_id;
        bitcount += GetBitizeOfType(elem_type_id) * type.vector.elem_count;
    }
    else if (type.kind == Type::Kind::Matrix)
    {
        uint32_t col_type_id = type.matrix.col_type_id;
        bitcount += GetBitizeOfType(col_type_id) * type.matrix.col_count;
    }
    else if (type.kind == Type::Kind::Array)
    {
        uint32_t elem_type_id = type.vector.elem_type_id;
        uint64_t array_len    = std::get<uint64_t>(GetValue(type.array.length_id));

        bitcount += GetBitizeOfType(elem_type_id) * array_len;
    }
    else if (type.kind == Type::Kind::RuntimeArray)
    {
        assertx("SPIRV simulator: Fetching bitsize of RuntimeArray, this is currently not implemented");

        // uint32_t elem_type_id = type.vector.elem_type_id;
        // uint64_t array_len = std::get<uint64_t>(GetValue(type.array.length_id));
        // bitcount += GetBitizeOfType(elem_type_id);
    }
    else if (type.kind == Type::Kind::Struct)
    {
        assertm(struct_members_.find(type_id) != struct_members_.end(), "SPIRV simulator: Struct has no members");

        for (uint32_t member_type_id : struct_members_.at(type_id))
        {
            bitcount += GetBitizeOfType(member_type_id);
        }
    }
    else if (type.kind == Type::Kind::Pointer)
    {
        bitcount += 8 * 8;
    }

    return bitcount;
}

uint32_t SPIRVSimulator::GetTargetPointerType(const PointerV& pointer)
{
    assertm(types_.find(pointer.base_type_id) != types_.end(),
            "SPIRV simulator: No valid type for the given pointer type ID was found");

    const Type* type = &GetTypeByTypeId(pointer.base_type_id);

    uint32_t type_id = type->pointer.pointee_type_id;
    type             = &GetTypeByTypeId(type_id);
    for (uint32_t idx : pointer.idx_path)
    {
        if (type->kind == Type::Kind::Struct)
        {
            assertm(struct_members_.find(type_id) != struct_members_.end(), "SPIRV simulator: Struct has no members");

            type_id = struct_members_.at(type_id)[idx];
            type    = &GetTypeByTypeId(type_id);
        }
        else if ((type->kind == Type::Kind::Array) || (type->kind == Type::Kind::RuntimeArray))
        {
            type_id = type->array.elem_type_id;
            type    = &GetTypeByTypeId(type_id);
        }
        else if (type->kind == Type::Kind::Vector)
        {
            type_id = type->vector.elem_type_id;
            type    = &GetTypeByTypeId(type_id);
        }
        else if (type->kind == Type::Kind::Matrix)
        {
            type_id = type->matrix.col_type_id;
            type    = &GetTypeByTypeId(type_id);
        }
        else if (type->kind == Type::Kind::Pointer)
        {
            type_id = type->pointer.pointee_type_id;
            type    = &GetTypeByTypeId(type_id);
        }
        else
        {
            assertx("SPIRV simulator: Unhandled type in GetBitizeOfTargetType");
        }
    }

    return type_id;
}

size_t SPIRVSimulator::GetBitizeOfTargetType(const PointerV& pointer)
{
    /*
    Returns the full bitsize of the type pointed to by the given pointer.
    The pointers type_id field must be the result of a OpType* instruction.
    */
    assertm(types_.find(pointer.base_type_id) != types_.end(),
            "SPIRV simulator: No valid type for the given pointer type ID was found");

    uint32_t type_id = GetTargetPointerType(pointer);

    return GetBitizeOfType(type_id);
}

void SPIRVSimulator::GetBaseTypeIDs(uint32_t type_id, std::vector<uint32_t>& output)
{
    /*
    Gets all the scalar types in a compond types, laid out as they are in memory.
    */
    const Type& type = GetTypeByTypeId(type_id);

    assertm(type.kind != Type::Kind::Void, "SPIRV simulator: Attempt to extract size of a void type");

    if (type.kind == Type::Kind::BoolT || type.kind == Type::Kind::Int || type.kind == Type::Kind::Float ||
        type.kind == Type::Kind::Pointer)
    {
        output.push_back(type_id);
    }
    else if (type.kind == Type::Kind::Vector)
    {
        uint32_t elem_type_id = type.vector.elem_type_id;
        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            output.push_back(elem_type_id);
        }
    }
    else if (type.kind == Type::Kind::Matrix)
    {
        uint32_t col_type_id = type.matrix.col_type_id;
        for (uint32_t i = 0; i < type.matrix.col_count; ++i)
        {
            GetBaseTypeIDs(col_type_id, output);
        }
    }
    else if (type.kind == Type::Kind::Array || type.kind == Type::Kind::RuntimeArray)
    {
        uint32_t elem_type_id = type.vector.elem_type_id;
        uint64_t array_len    = std::get<uint64_t>(GetValue(type.array.length_id));
        for (uint64_t i = 0; i < array_len; ++i)
        {
            GetBaseTypeIDs(elem_type_id, output);
        }
    }
    else if (type.kind == Type::Kind::Struct)
    {
        for (uint32_t member_type_id : struct_members_.at(type_id))
        {
            GetBaseTypeIDs(member_type_id, output);
        }
    }
}

void SPIRVSimulator::ReadWords(const std::byte* external_pointer, uint32_t type_id, std::vector<uint32_t>& buffer_data)
{
    /*
    Extracts 32 bit word values with type matching type_id from the external_pointer byte buffer
    */
    const Type& type = GetTypeByTypeId(type_id);

    assertm(type.kind != Type::Kind::Void, "SPIRV simulator: Attempt to extract a void type from a buffer");

    if (type.kind == Type::Kind::Struct)
    {
        uint32_t member_offset_id = 0;
        for (uint32_t member_type_id : struct_members_.at(type_id))
        {
            // They must have offset decorators
            assertm(HasDecorator(type_id, member_offset_id, spv::Decoration::DecorationOffset),
                    "SPIRV simulator: No offset decorator for input struct member");

            const std::byte* member_offset_pointer =
                external_pointer + GetDecoratorLiteral(type_id, member_offset_id, spv::Decoration::DecorationOffset);
            ReadWords(member_offset_pointer, member_type_id, buffer_data);
            member_offset_id += 1;
        }
    }
    else if (type.kind == Type::Kind::Array || type.kind == Type::Kind::RuntimeArray)
    {
        // They must have a stride decorator (TODO: unless they contain blocks, but we can deal with that later)
        assertm(HasDecorator(type_id, spv::Decoration::DecorationArrayStride),
                "SPIRV simulator: No ArrayStride decorator for input array, check if this is a block array and add "
                "support for it if so");

        uint32_t array_stride = GetDecoratorLiteral(type_id, spv::Decoration::DecorationArrayStride);

        if (type.array.length_id == 0)
        {
            // Runtime array, special handling, extract one element
            // TODO: We should probably change this and use sparse loads with maps or something
            ReadWords(external_pointer, type.array.elem_type_id, buffer_data);
        }
        else
        {
            uint64_t array_len = std::get<uint64_t>(GetValue(type.array.length_id));

            for (uint64_t array_index = 0; array_index < array_len; ++array_index)
            {
                const std::byte* member_offset_pointer = external_pointer + array_stride * array_index;
                ReadWords(member_offset_pointer, type.array.elem_type_id, buffer_data);
            }
        }
    }
    else if (type.kind == Type::Kind::Matrix)
    {
        assertm(HasDecorator(type_id, spv::Decoration::DecorationMatrixStride),
                "SPIRV simulator: No MatrixStride decorator for input matrix");
        assertm(HasDecorator(type_id, spv::Decoration::DecorationRowMajor) ||
                    HasDecorator(type_id, spv::Decoration::DecorationColMajor),
                "SPIRV simulator: No RowMajor or ColMajor decorator for input matrix");

        const Type& col_type = GetTypeByTypeId(type.matrix.col_type_id);
        assertm(col_type.kind == Type::Kind::Vector, "SPIRV simulator: Non-vector column type found in matrix");

        // Because row-major matrices may not have a valid col type, we extract the subcomponents directly
        // We basically treat it as an array
        // Always extract to a column major order to simplify stuff later
        uint32_t col_count = type.matrix.col_count;
        uint32_t row_count = col_type.vector.elem_count;

        uint32_t component_stride = GetDecoratorLiteral(type_id, spv::Decoration::DecorationMatrixStride);
        bool     row_major        = HasDecorator(type_id, spv::Decoration::DecorationRowMajor);

        uint32_t bytes_per_subcomponent = std::ceil((double)(GetBitizeOfType(col_type.vector.elem_type_id) / 8));

        for (uint64_t col_index = 0; col_index < col_count; ++col_index)
        {
            for (uint64_t row_index = 0; row_index < row_count; ++row_index)
            {
                const std::byte* member_offset_pointer;
                if (row_major)
                {
                    member_offset_pointer =
                        external_pointer + row_index * component_stride + col_index * bytes_per_subcomponent;
                }
                else
                {
                    member_offset_pointer =
                        external_pointer + col_index * component_stride + row_index * bytes_per_subcomponent;
                }
                ReadWords(member_offset_pointer, col_type.vector.elem_type_id, buffer_data);
            }
        }
    }
    else
    {
        // Assume everything else is tightly packed
        std::vector<uint32_t> base_type_ids;
        GetBaseTypeIDs(type_id, base_type_ids);
        size_t ext_ptr_offset = 0;
        for (auto base_type_id : base_type_ids)
        {
            const Type& base_type = GetTypeByTypeId(base_type_id);
            size_t      bytes_to_extract;

            if (base_type.kind == Type::Kind::Pointer)
            {
                bytes_to_extract = 8;
            }
            else
            {
                bytes_to_extract = std::ceil((double)base_type.scalar.width / 8.0);
            }

            size_t output_index = buffer_data.size();
            buffer_data.resize(output_index + std::ceil((double)bytes_to_extract / 4.0));
            std::memcpy(&(buffer_data[output_index]), external_pointer + ext_ptr_offset, bytes_to_extract);
            ext_ptr_offset += bytes_to_extract;
        }
    }
}

void SPIRVSimulator::WriteValue(std::byte* external_pointer, uint32_t type_id, const Value& value)
{
    /*
    Writes the value stored in result_id to the external pointer
    */

    const Type& type = GetTypeByTypeId(type_id);

    assertm(type.kind != Type::Kind::Void, "SPIRV simulator: Attempt to write a void type to a buffer");

    if (type.kind == Type::Kind::Struct)
    {
        const std::shared_ptr<AggregateV>& agg_ptr = std::get<std::shared_ptr<AggregateV>>(value);

        uint32_t member_offset_index = 0;
        for (uint32_t member_type_id : struct_members_.at(type_id))
        {
            // They must have offset decorators
            assertm(HasDecorator(type_id, member_offset_index, spv::Decoration::DecorationOffset),
                    "SPIRV simulator: No offset decorator for input struct member");

            std::byte* member_offset_pointer =
                external_pointer + GetDecoratorLiteral(type_id, member_offset_index, spv::Decoration::DecorationOffset);

            WriteValue(member_offset_pointer, member_type_id, agg_ptr->elems[member_offset_index]);
            member_offset_index += 1;
        }
    }
    else if (type.kind == Type::Kind::Array || type.kind == Type::Kind::RuntimeArray)
    {
        // They must have a stride decorator (TODO: unless they contain blocks, but we can deal with that later)
        assertm(HasDecorator(type_id, spv::Decoration::DecorationArrayStride),
                "SPIRV simulator: No ArrayStride decorator for input array, check if this is a block array and add "
                "support for it if so");

        uint32_t array_stride = GetDecoratorLiteral(type_id, spv::Decoration::DecorationArrayStride);

        assertm(type.array.length_id != 0,
                "SPIRV simulator: Attempt to write out a runtime array, this should never happen");

        const std::shared_ptr<AggregateV>& agg_ptr = std::get<std::shared_ptr<AggregateV>>(value);

        uint64_t array_len = std::get<uint64_t>(GetValue(type.array.length_id));

        for (uint64_t array_index = 0; array_index < array_len; ++array_index)
        {
            std::byte* member_offset_pointer = external_pointer + array_stride * array_index;
            WriteValue(member_offset_pointer, type.array.elem_type_id, agg_ptr->elems[array_index]);
        }
    }
    else if (type.kind == Type::Kind::Matrix)
    {
        assertm(HasDecorator(type_id, spv::Decoration::DecorationMatrixStride),
                "SPIRV simulator: No MatrixStride decorator for input matrix");
        assertm(HasDecorator(type_id, spv::Decoration::DecorationRowMajor) ||
                    HasDecorator(type_id, spv::Decoration::DecorationColMajor),
                "SPIRV simulator: No RowMajor or ColMajor decorator for input matrix");

        const Type& col_type = GetTypeByTypeId(type.matrix.col_type_id);
        assertm(col_type.kind == Type::Kind::Vector, "SPIRV simulator: Non-vector column type found in matrix");

        uint32_t col_count = type.matrix.col_count;
        uint32_t row_count = col_type.vector.elem_count;

        uint32_t component_stride = GetDecoratorLiteral(type_id, spv::Decoration::DecorationMatrixStride);
        bool     row_major        = HasDecorator(type_id, spv::Decoration::DecorationRowMajor);

        uint32_t bytes_per_subcomponent = std::ceil((double)(GetBitizeOfType(col_type.vector.elem_type_id) / 8));

        const std::shared_ptr<MatrixV>& matrix_ptr = std::get<std::shared_ptr<MatrixV>>(value);

        for (uint64_t col_index = 0; col_index < col_count; ++col_index)
        {
            const std::shared_ptr<VectorV>& column_val =
                std::get<std::shared_ptr<VectorV>>(matrix_ptr->cols[col_index]);

            for (uint64_t row_index = 0; row_index < row_count; ++row_index)
            {
                std::byte* member_offset_pointer;
                if (row_major)
                {
                    member_offset_pointer =
                        external_pointer + row_index * component_stride + col_index * bytes_per_subcomponent;
                }
                else
                {
                    member_offset_pointer =
                        external_pointer + col_index * component_stride + row_index * bytes_per_subcomponent;
                }
                WriteValue(member_offset_pointer, col_type.vector.elem_type_id, column_val->elems[row_index]);
            }
        }
    }
    else if (type.kind == Type::Kind::Vector)
    {
        // TODO: More checks
        const std::shared_ptr<VectorV>& vector_ptr = std::get<std::shared_ptr<VectorV>>(value);

        const Type& elem_type = GetTypeByTypeId(type.vector.elem_type_id);

        uint32_t scalar_width = elem_type.scalar.width / 8;

        for (uint64_t elem_index = 0; elem_index < type.vector.elem_count; ++elem_index)
        {
            std::byte* member_offset_pointer = external_pointer + scalar_width * elem_index;
            WriteValue(member_offset_pointer, type.array.elem_type_id, vector_ptr->elems[elem_index]);
        }
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        uint32_t scalar_width = type.scalar.width / 8;
        uint64_t raw_value    = std::get<uint64_t>(value);

        std::memcpy(external_pointer, (const void*)(&raw_value), scalar_width);
    }
    else if (type.kind == Type::Kind::Int)
    {
        uint32_t scalar_width = type.scalar.width / 8;

        uint64_t raw_value;
        if (type.scalar.is_signed)
        {
            raw_value = bit_cast<uint64_t>(std::get<int64_t>(value));
        }
        else
        {
            raw_value = std::get<uint64_t>(value);
        }

        std::memcpy(external_pointer, (const void*)(&raw_value), scalar_width);
    }
    else if (type.kind == Type::Kind::Float)
    {
        uint32_t scalar_width = type.scalar.width / 8;
        uint64_t raw_value    = bit_cast<uint64_t>(std::get<double>(value));

        std::memcpy(external_pointer, (const void*)(&raw_value), scalar_width);
    }
    else
    {
        assertx("SPIRV simulator: Unhandled type in output writer");
    }
}

uint64_t SPIRVSimulator::GetPointerOffset(const PointerV& pointer_value)
{
    /*
    Given a pointer, this will get the correct offset into the memory where its value resides (relative to its base).
    */
    uint64_t offset  = 0;
    uint32_t type_id = pointer_value.base_type_id;

    const Type& pointer_type = GetTypeByTypeId(type_id);
    type_id                  = pointer_type.pointer.pointee_type_id;
    const Type* type         = &GetTypeByTypeId(type_id);

    assertm(type->kind != Type::Kind::Void, "SPIRV simulator: Attempt to extract a void type offset");

    for (uint32_t indirection_index : pointer_value.idx_path)
    {
        if (type->kind == Type::Kind::Struct)
        {
            // They must have offset decorators
            assertm(HasDecorator(type_id, indirection_index, spv::Decoration::DecorationOffset),
                    "SPIRV simulator: No offset decorator for input struct member");

            offset += GetDecoratorLiteral(type_id, indirection_index, spv::Decoration::DecorationOffset);
            type_id = struct_members_.at(type_id)[indirection_index];
            type    = &GetTypeByTypeId(type_id);
        }
        else if (type->kind == Type::Kind::Array || type->kind == Type::Kind::RuntimeArray)
        {
            // They must have a stride decorator (TODO: unless they contain blocks, but we can deal with that later)
            assertm(HasDecorator(type_id, spv::Decoration::DecorationArrayStride),
                    "SPIRV simulator: No ArrayStride decorator for input array");

            uint32_t array_stride = GetDecoratorLiteral(type_id, spv::Decoration::DecorationArrayStride);
            offset += indirection_index * array_stride;
            type_id = type->array.elem_type_id;
            type    = &GetTypeByTypeId(type_id);
        }
        else if (type->kind == Type::Kind::Matrix)
        {
            assertm(HasDecorator(type_id, spv::Decoration::DecorationColMajor),
                    "SPIRV simulator: Attempt to get pointer offset to row-major matrix, this is illegal and violates "
                    "contiguity requirements");

            uint32_t matrix_stride = GetDecoratorLiteral(type_id, spv::Decoration::DecorationMatrixStride);
            offset += indirection_index * matrix_stride;
            type_id = type->matrix.col_type_id;
            type    = &GetTypeByTypeId(type_id);
        }
        else if (type->kind == Type::Kind::Vector)
        {
            type_id = type->vector.elem_type_id;
            type    = &GetTypeByTypeId(type->vector.elem_type_id);
            offset += indirection_index * std::ceil(type->scalar.width / 8.0);
        }
        else
        {
            // Crash, this should never happen
            assertx("SPIRV simulator: Pointer attempts to index a type that cant be indexed");
        }
    }

    return offset;
}

uint32_t SPIRVSimulator::GetTypeID(uint32_t result_id) const
{
    /*
    Given a result ID, return the type ID of the value it maps to.
    */
    assertm(result_id_to_inst_index_.find(result_id) != result_id_to_inst_index_.end(),
            "SPIRV simulator: No instruction found for result_id");

    size_t             instruction_index = result_id_to_inst_index_.at(result_id);
    const Instruction& instruction       = instructions_[instruction_index];

    bool has_result = false;
    bool has_type   = false;
    spv::HasResultAndType(instruction.opcode, &has_result, &has_type);

    if (has_type)
    {
        return instruction.words[1];
    }

    assertx("SPIRV simulator: No type found for result_id");
    return 0;
}

Value SPIRVSimulator::MakeScalar(uint32_t type_id, const uint32_t*& words)
{
    const Type& type = GetTypeByTypeId(type_id);

    switch (type.kind)
    {
        case Type::Kind::Int:
        {
            assertm(type.scalar.width <= 64, "SPIRV simulator: We do not support types wider than 64 bits");

            if (type.scalar.width > 32)
            {
                if (type.scalar.is_signed)
                {
                    int64_t tmp_value;
                    std::memcpy(&tmp_value, words, 8);
                    words += 2;
                    return tmp_value;
                }
                else
                {
                    uint64_t tmp_value = (static_cast<uint64_t>(words[1]) << 32) | words[0];
                    words += 2;
                    return tmp_value;
                }
            }
            else
            {
                if (type.scalar.is_signed)
                {
                    int32_t tmp_value;
                    std::memcpy(&tmp_value, &words[0], 4);
                    words += 1;
                    return (int64_t)tmp_value;
                }
                else
                {
                    uint64_t tmp_value = (uint64_t)words[0];
                    words += 1;
                    return tmp_value;
                }
            }
        }
        case Type::Kind::BoolT:
        {
            // Just treat bools as uint64_t types for simplicity
            assertm(type.scalar.width <= 64,
                    "SPIRV simulator: Bool value with more than 64 bits detected, this is not handled at present");
            uint64_t tmp_value = (uint64_t)words[0];
            words += 1;
            return tmp_value;
        }
        case Type::Kind::Float:
        {
            assertm(type.scalar.width <= 64, "SPIRV simulator: We do not support types wider than 64 bits");
            if (type.scalar.width > 32)
            {
                double tmp_value;
                std::memcpy(&tmp_value, &words[0], 8);
                words += 2;
                return tmp_value;
            }
            else
            {
                float tmp_value;
                std::memcpy(&tmp_value, &words[0], 4);
                words += 1;
                return (double)tmp_value;
            }
        }
        default:
        {
            assertx("SPIRV simulator: Unsupported scalar type, instructions are possibly corrupt");
            return 0;
        }
    }
}

Value SPIRVSimulator::MakeDefault(uint32_t type_id, const uint32_t** initial_data)
{
    const Type& type = GetTypeByTypeId(type_id);

    switch (type.kind)
    {
        case Type::Kind::Int:
        case Type::Kind::Float:
        case Type::Kind::BoolT:
        {
            if (initial_data != nullptr)
            {
                return MakeScalar(type_id, *initial_data);
            }
            else
            {
                const uint32_t  empty_array[]{ 0, 0 };
                const uint32_t* buffer_pointer = empty_array;
                return MakeScalar(type_id, buffer_pointer);
            }
        }
        case Type::Kind::Image:
        {
            assertm(!initial_data,
                    "SPIRV simulator: Cannot create Image handle with initial_data unless we know the size of the "
                    "opaque types");
            return (uint64_t)(0);
        }
        case Type::Kind::Sampler:
        {
            assertm(!initial_data,
                    "SPIRV simulator: Cannot create Sampler with initial_data unless we know the size of the "
                    "opaque types");
            return (uint64_t)0;
        }
        case Type::Kind::SampledImage:
        {
            assertm(!initial_data,
                    "SPIRV simulator: Cannot create SampledImage with initial_data unless we know the size of the "
                    "opaque types");
            SampledImageV new_sampled_image{ 0, 0 };
            return new_sampled_image;
        }
        case Type::Kind::Opaque:
        {
            assertm(!initial_data,
                    "SPIRV simulator: Cannot create Opaque value with initial_data unless we know the size of the "
                    "opaque types");
            return (uint64_t)0;
        }
        case Type::Kind::NamedBarrier:
        {
            assertx("SPIRV simulator: NamedBarrier is not supported by MakeDefault, implement it to continue.");
        }
        case Type::Kind::Vector:
        {
            auto vec = std::make_shared<VectorV>();
            vec->elems.reserve(type.vector.elem_count);
            for (uint32_t i = 0; i < type.vector.elem_count; ++i)
            {
                vec->elems.push_back(MakeDefault(type.vector.elem_type_id, initial_data));
            }

            return vec;
        }
        case Type::Kind::Matrix:
        {
            // We dont have to deal with col/row major here since we do that on buffer extraction
            auto matrix = std::make_shared<MatrixV>();
            matrix->cols.reserve(type.matrix.col_count);
            for (uint32_t i = 0; i < type.matrix.col_count; ++i)
            {
                Value mat_val = MakeDefault(type.matrix.col_type_id, initial_data);
                matrix->cols.push_back(mat_val);
            }

            return matrix;
        }
        case Type::Kind::Array:
        {
            uint64_t len       = std::get<uint64_t>(GetValue(type.array.length_id));
            auto     aggregate = std::make_shared<AggregateV>();
            aggregate->elems.reserve(len);
            for (uint32_t i = 0; i < len; ++i)
            {
                aggregate->elems.push_back(MakeDefault(type.array.elem_type_id, initial_data));
            }

            return aggregate;
        }
        case Type::Kind::RuntimeArray:
        {
            uint64_t len = 1;
            if (type.array.length_id != 0)
            {
                len = std::get<uint64_t>(GetValue(type.array.length_id));
            }

            auto aggregate = std::make_shared<AggregateV>();
            aggregate->elems.reserve(len);
            for (uint32_t i = 0; i < len; ++i)
            {
                aggregate->elems.push_back(MakeDefault(type.array.elem_type_id, initial_data));
            }

            return aggregate;
        }
        case Type::Kind::Struct:
        {
            auto structure = std::make_shared<AggregateV>();
            for (auto member : struct_members_.at(type_id))
            {
                structure->elems.push_back(MakeDefault(member, initial_data));
            }

            return structure;
        }
        case Type::Kind::Pointer:
        {
            if (type.pointer.storage_class == spv::StorageClass::StorageClassPhysicalStorageBuffer)
            {
                uint64_t pointer_value = 0;

                if (initial_data)
                {
                    std::memcpy(&pointer_value, reinterpret_cast<const std::byte*>(*initial_data), sizeof(uint64_t));
                }
                else
                {
                    if (verbose_)
                    {
                        std::cout << execIndent
                                  << "A pointer with StorageClassPhysicalStorageBuffer was default initialized without "
                                     "input buffer data available. The actual pointer address will be unknown (null)"
                                  << std::endl;
                    }
                }

                if (initial_data)
                {
                    (*initial_data) += 2;
                }

                const std::byte* remapped_pointer = nullptr;

                for (const auto& map_entry : input_data_.physical_address_buffers)
                {
                    uint64_t buffer_address = map_entry.first;
                    size_t   buffer_size    = map_entry.second.first;

                    const std::byte* buffer_data = static_cast<std::byte*>(map_entry.second.second);

                    if ((pointer_value >= buffer_address) && (pointer_value < (buffer_address + buffer_size)))
                    {
                        remapped_pointer = &(buffer_data[buffer_address - pointer_value]);
                        break;
                    }
                }

                PointerV new_pointer{
                    bit_cast<uint64_t>(remapped_pointer), type_id, 0, type.pointer.storage_class, {}
                };
                physical_address_pointers_.push_back(new_pointer);
                return new_pointer;
            }
            else
            {
                assertx("SPIRV simulator: Attempting to initialize a raw pointer whose storage class is not "
                        "PhysicalStorageBuffer");
                return 0;
            }
        }
        default:
        {
            std::cout << (uint32_t)type.kind << std::endl;
            assertx("SPIRV simulator: Invalid input type to MakeDefault");
            return 0;
        }
    }
}

std::vector<DataSourceBits> SPIRVSimulator::FindDataSourcesFromResultID(uint32_t result_id)
{
    std::vector<DataSourceBits> results;

    uint32_t           instruction_index = result_id_to_inst_index_[result_id];
    const Instruction& instruction       = instructions_[instruction_index];

    bool has_result = false;
    bool has_type   = false;
    spv::HasResultAndType(instruction.opcode, &has_result, &has_type);

    uint32_t type_id = 0;
    if (has_type)
    {
        type_id = instruction.words[1];
    }

    if (verbose_)
    {
        std::cout << execIndent << "Tracing value source backwards through: " << spv::OpToString(instruction.opcode)
                  << ": " << result_id << std::endl;
    }

    switch (instruction.opcode)
    {
        case spv::Op::OpSpecConstantComposite:
        {
            for (uint32_t component_id = 3; component_id < instruction.word_count; ++component_id)
            {
                std::vector<DataSourceBits> component_result =
                    FindDataSourcesFromResultID(instruction.words[component_id]);
                results.insert(results.end(), component_result.begin(), component_result.end());
            }

            DataSourceBits* prev_source = nullptr;
            for (auto& component_data : results)
            {
                if (prev_source)
                {
                    component_data.val_bit_offset += prev_source->val_bit_offset + prev_source->bitcount;
                }

                prev_source = &component_data;
            }
            break;
        }
        case spv::Op::OpSpecConstant:
        {
            assertm(HasDecorator(result_id, spv::Decoration::DecorationSpecId),
                    "SPIRV simulator: Op_SpecConstant type is not decorated with SpecId");
            uint32_t spec_id = GetDecoratorLiteral(result_id, spv::Decoration::DecorationSpecId);

            DataSourceBits data_source;
            data_source.location    = BitLocation::SpecConstant;
            data_source.idx         = 0;
            data_source.binding_id  = spec_id;
            data_source.set_id      = 0;
            data_source.byte_offset = 0;
            data_source.bit_offset  = 0;
            data_source.bitcount    = GetBitizeOfType(type_id);
            ;
            data_source.val_bit_offset = 0;
            results.push_back(data_source);
            break;
        }
        case spv::Op::OpLoad:
        {
            uint32_t pointer_id = instruction.words[3];

            if (values_stored_.find(pointer_id) == values_stored_.end())
            {
                const PointerV& pointer = std::get<PointerV>(GetValue(pointer_id));

                assertm(pointer.storage_class != spv::StorageClass::StorageClassFunction,
                        "SPIRV simulator: A StorageClassFunction is being read from while backtracing operands without "
                        "having been stored to, this is a symptom of a serious error somewhere");

                DataSourceBits data_source;
                data_source.location      = BitLocation::StorageClass;
                data_source.storage_class = (spv::StorageClass)pointer.storage_class;
                data_source.idx           = 0;

                if (pointer.storage_class == spv::StorageClass::StorageClassPushConstant)
                {
                    data_source.binding_id = 0;
                    data_source.set_id     = 0;
                }
                else if (pointer.storage_class != spv::StorageClass::StorageClassPhysicalStorageBuffer)
                {
                    assertm(HasDecorator(pointer.base_result_id, spv::Decoration::DecorationDescriptorSet),
                            "SPIRV simulator: Missing DecorationDescriptorSet for pointee object");
                    assertm(HasDecorator(pointer.base_result_id, spv::Decoration::DecorationBinding),
                            "SPIRV simulator: Missing DecorationBinding for pointee object");

                    data_source.binding_id =
                        GetDecoratorLiteral(pointer.base_result_id, spv::Decoration::DecorationBinding);
                    data_source.set_id =
                        GetDecoratorLiteral(pointer.base_result_id, spv::Decoration::DecorationDescriptorSet);
                }
                else
                {
                    data_source.binding_id = 0;
                    data_source.set_id     = 0;
                }

                data_source.byte_offset = GetPointerOffset(pointer);
                data_source.bit_offset  = 0;
                // This does not account for padding, but its probably fine here since it makes little sense to load
                // complex constructs here
                data_source.bitcount       = GetBitizeOfTargetType(pointer);
                data_source.val_bit_offset = 0;
                results.push_back(data_source);
            }
            else
            {
                uint32_t true_source = values_stored_.at(pointer_id);

                std::vector<DataSourceBits> component_result = FindDataSourcesFromResultID(true_source);
                results.insert(results.end(), component_result.begin(), component_result.end());
            }
            break;
        }
        case spv::Op::OpConstant:
        {
            DataSourceBits data_source;
            data_source.location       = BitLocation::Constant;
            data_source.idx            = 0;
            data_source.binding_id     = 0;
            data_source.set_id         = 0;
            uint32_t header_word_count = 5;
            data_source.byte_offset    = (instruction_index + header_word_count) * sizeof(uint32_t);
            data_source.bit_offset     = 0;
            data_source.bitcount       = GetBitizeOfType(type_id);
            ;
            data_source.val_bit_offset = 0;
            results.push_back(data_source);
            break;
        }
        default:
        {
            assertx("SPIRV simulator: Unimplemented opcode in FindDataSourcesFromResultID");
        }
    }

    return results;
}

Value SPIRVSimulator::CopyValue(const Value& value) const
{
    /*
    Creates a copy of a Value object, will recursively copy all pointers
    and components.
    */

    if (std::holds_alternative<std::shared_ptr<VectorV>>(value))
    {
        std::shared_ptr<VectorV> new_vector = std::make_shared<VectorV>();

        for (const auto& elem : std::get<std::shared_ptr<VectorV>>(value)->elems)
        {
            new_vector->elems.push_back(CopyValue(elem));
        }

        return new_vector;
    }
    else if (std::holds_alternative<std::shared_ptr<MatrixV>>(value))
    {
        std::shared_ptr<MatrixV> new_matrix = std::make_shared<MatrixV>();

        for (const auto& col : std::get<std::shared_ptr<MatrixV>>(value)->cols)
        {
            new_matrix->cols.push_back(CopyValue(col));
        }

        return new_matrix;
    }
    else if (std::holds_alternative<std::shared_ptr<AggregateV>>(value))
    {
        std::shared_ptr<AggregateV> new_aggregate = std::make_shared<AggregateV>();

        for (const auto& elem : std::get<std::shared_ptr<AggregateV>>(value)->elems)
        {
            new_aggregate->elems.push_back(CopyValue(elem));
        }

        return new_aggregate;
    }

    return value;
}

// ---------------------------------------------------------------------------
//  Dereference and access helpers
// ---------------------------------------------------------------------------

uint64_t SPIRVSimulator::RemapHostToClientPointer(uint64_t host_pointer) const
{
    return 0;
}

void SPIRVSimulator::WritePointer(const PointerV& ptr, const Value& out_value)
{
    const Type& type = GetTypeByTypeId(ptr.base_type_id);

    // To make inputs optional
    if (ptr.pointer_handle == 0)
    {
        return;
    }

    // These are not backed by buffers (yet, input/output is debatable and we will likely need to handle them), write to
    // the internal heaps
    if (type.pointer.storage_class == spv::StorageClass::StorageClassFunction ||
        type.pointer.storage_class == spv::StorageClass::StorageClassWorkgroup ||
        type.pointer.storage_class == spv::StorageClass::StorageClassPrivate ||
        type.pointer.storage_class == spv::StorageClass::StorageClassInput ||
        type.pointer.storage_class == spv::StorageClass::StorageClassOutput ||
        type.pointer.storage_class == spv::StorageClass::StorageClassImage)
    {
        Value* value = &Heap(type.pointer.storage_class)[ptr.pointer_handle];
        for (size_t depth = 0; depth < ptr.idx_path.size(); ++depth)
        {
            uint32_t indirection_index = ptr.idx_path[depth];

            if (std::holds_alternative<std::shared_ptr<AggregateV>>(*value))
            {
                const auto agg = std::get<std::shared_ptr<AggregateV>>(*value);

                assertm(indirection_index < agg->elems.size(), "SPIRV simulator: Arrau index OOB");

                value = &agg->elems[indirection_index];
            }
            else if (std::holds_alternative<std::shared_ptr<VectorV>>(*value))
            {
                auto vec = std::get<std::shared_ptr<VectorV>>(*value);

                assertm(indirection_index < vec->elems.size(), "SPIRV simulator: Vector index OOB");

                value = &vec->elems[indirection_index];
            }
            else if (std::holds_alternative<std::shared_ptr<MatrixV>>(*value))
            {
                auto matrix = std::get<std::shared_ptr<MatrixV>>(*value);

                assertm(indirection_index < matrix->cols.size(), "SPIRV simulator: Matrix index OOB");

                value = &matrix->cols[indirection_index];
            }
            else
            {
                assertx("SPIRV simulator: Pointer dereference into non-composite object");
            }
        }

        *value = out_value;
    }
    // Write back to the input buffers here
    else if (type.pointer.storage_class == spv::StorageClass::StorageClassStorageBuffer ||
             type.pointer.storage_class == spv::StorageClass::StorageClassPhysicalStorageBuffer)
    {
        auto offset = GetPointerOffset(ptr);

        std::byte* external_pointer = bit_cast<std::byte*>(ptr.pointer_handle) + offset;

        WriteValue(external_pointer, type.pointer.pointee_type_id, out_value);
    }
    else if (type.pointer.storage_class == spv::StorageClass::StorageClassPushConstant ||
             type.pointer.storage_class == spv::StorageClass::StorageClassUniform ||
             type.pointer.storage_class == spv::StorageClass::StorageClassUniformConstant)
    {
        assertx("SPIRV simulator: Write to invalid/constant storage class");
    }
    else
    {
        assertx("SPIRV simulator: Unhandled storage class in WritePointer, add support to continue");
    }
}

Value SPIRVSimulator::ReadPointer(const PointerV& ptr)
{
    const Type& type = GetTypeByTypeId(ptr.base_type_id);

    // To make inputs optional
    if (ptr.pointer_handle == 0)
    {
        uint32_t target_type_id = GetTargetPointerType(ptr);
        return MakeDefault(target_type_id);
    }

    // These are stored on the internal heaps
    if (type.pointer.storage_class == spv::StorageClass::StorageClassFunction ||
        type.pointer.storage_class == spv::StorageClass::StorageClassWorkgroup ||
        type.pointer.storage_class == spv::StorageClass::StorageClassPrivate ||
        type.pointer.storage_class == spv::StorageClass::StorageClassInput ||
        type.pointer.storage_class == spv::StorageClass::StorageClassOutput ||
        type.pointer.storage_class == spv::StorageClass::StorageClassImage)
    {
        Value* value = &Heap(type.pointer.storage_class)[ptr.pointer_handle];
        for (size_t depth = 0; depth < ptr.idx_path.size(); ++depth)
        {
            uint32_t indirection_index = ptr.idx_path[depth];

            if (std::holds_alternative<std::shared_ptr<AggregateV>>(*value))
            {
                const auto agg = std::get<std::shared_ptr<AggregateV>>(*value);

                assertm(indirection_index < agg->elems.size(), "SPIRV simulator: Arrau index OOB");

                value = &agg->elems[indirection_index];
            }
            else if (std::holds_alternative<std::shared_ptr<VectorV>>(*value))
            {
                auto vec = std::get<std::shared_ptr<VectorV>>(*value);

                assertm(indirection_index < vec->elems.size(), "SPIRV simulator: Vector index OOB");

                value = &vec->elems[indirection_index];
            }
            else if (std::holds_alternative<std::shared_ptr<MatrixV>>(*value))
            {
                auto matrix = std::get<std::shared_ptr<MatrixV>>(*value);

                assertm(indirection_index < matrix->cols.size(), "SPIRV simulator: Matrix index OOB");

                value = &matrix->cols[indirection_index];
            }
            else
            {
                assertx("SPIRV simulator: Pointer dereference into non-composite object");
            }
        }

        return *value;
    }
    // These can/should have input pointers
    else if (type.pointer.storage_class == spv::StorageClass::StorageClassPushConstant ||
             type.pointer.storage_class == spv::StorageClass::StorageClassUniform ||
             type.pointer.storage_class == spv::StorageClass::StorageClassUniformConstant ||
             type.pointer.storage_class == spv::StorageClass::StorageClassStorageBuffer ||
             type.pointer.storage_class == spv::StorageClass::StorageClassPhysicalStorageBuffer)
    {
        auto             offset           = GetPointerOffset(ptr);
        const std::byte* external_pointer = bit_cast<const std::byte*>(ptr.pointer_handle) + offset;

        std::vector<uint32_t> buffer_data;
        ReadWords(external_pointer, type.pointer.pointee_type_id, buffer_data);

        const uint32_t* buffer_pointer = buffer_data.data();
        return MakeDefault(type.pointer.pointee_type_id, &(buffer_pointer));
    }
    else
    {
        assertx("SPIRV simulator: Unhandled storage class in ReadPointer, add support to continue");
    }

    // TODO: Remove this when we replace the asserts
    Value value;
    return value;
}

const Value& SPIRVSimulator::GetValue(uint32_t result_id)
{
    assertm(!std::holds_alternative<std::monostate>(values_[result_id]),
            "SPIRV simulator: Access to undefined variable");

    return values_[result_id];
}

void SPIRVSimulator::SetValue(uint32_t result_id, const Value& value)
{
    values_[result_id] = value;
}

// ---------------------------------------------------------------------------
//  Ext Import implementations
// ---------------------------------------------------------------------------

void SPIRVSimulator::GLSLExtHandler(uint32_t                         type_id,
                                    uint32_t                         result_id,
                                    uint32_t                         instruction_literal,
                                    const std::span<const uint32_t>& operand_words)
{
    const Type& type = GetTypeByTypeId(type_id);

    switch (instruction_literal)
    {
        case 8:
        { // Floor
            const Value& operand = GetValue(operand_words[0]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::floor");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto vec = std::get<std::shared_ptr<VectorV>>(operand);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    Value elem_result = (double)std::floor(std::get<double>(vec->elems[i]));
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Float)
            {
                Value result = (double)std::floor(std::get<double>(operand));
                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 9:
        { // Ceil
            const Value& operand = GetValue(operand_words[0]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::ceil");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto vec = std::get<std::shared_ptr<VectorV>>(operand);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    Value elem_result = (double)std::ceil(std::get<double>(vec->elems[i]));
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Float)
            {
                Value result = (double)std::ceil(std::get<double>(operand));
                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 10:
        { // Fract
            const Value& operand = GetValue(operand_words[0]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::fract");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto vec = std::get<std::shared_ptr<VectorV>>(operand);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    double tval        = std::get<double>(vec->elems[i]);
                    Value  elem_result = tval - std::floor(tval);
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Float)
            {
                double tval   = std::get<double>(operand);
                Value  result = tval - std::floor(tval);
                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 13:
        { // Sin
            const Value& operand = GetValue(operand_words[0]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::sin");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto vec = std::get<std::shared_ptr<VectorV>>(operand);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    Value elem_result = (double)std::sin(std::get<double>(vec->elems[i]));
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Float)
            {
                Value result = (double)std::sin(std::get<double>(operand));
                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 14:
        { // Cos
            const Value& operand = GetValue(operand_words[0]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::cos");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto vec = std::get<std::shared_ptr<VectorV>>(operand);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    Value elem_result = (double)std::cos(std::get<double>(vec->elems[i]));
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Float)
            {
                Value result = (double)std::cos(std::get<double>(operand));
                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 15:
        { // Tan
            const Value& operand = GetValue(operand_words[0]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::tan");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto vec = std::get<std::shared_ptr<VectorV>>(operand);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    Value elem_result = (double)std::tan(std::get<double>(vec->elems[i]));
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Float)
            {
                Value result = (double)std::tan(std::get<double>(operand));
                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 26:
        { // Pow
            const Value& base     = GetValue(operand_words[0]);
            const Value& exponent = GetValue(operand_words[1]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(base) &&
                            std::holds_alternative<std::shared_ptr<VectorV>>(exponent),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::pow");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto basevec = std::get<std::shared_ptr<VectorV>>(base);
                auto expvec  = std::get<std::shared_ptr<VectorV>>(exponent);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    Value elem_result =
                        (double)std::pow(std::get<double>(basevec->elems[i]), std::get<double>(expvec->elems[i]));
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Float)
            {
                Value result = (double)std::pow(std::get<double>(base), std::get<double>(exponent));
                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 31:
        { // Sqrt
            const Value& operand = GetValue(operand_words[0]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::sqrt");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto vec = std::get<std::shared_ptr<VectorV>>(operand);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    Value elem_result = (double)std::sqrt(std::get<double>(vec->elems[i]));
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Float)
            {
                Value result = (double)std::sqrt(std::get<double>(operand));
                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 38:
        { // UMin
            const Value& operand_1 = GetValue(operand_words[0]);
            const Value& operand_2 = GetValue(operand_words[1]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand_1) &&
                            std::holds_alternative<std::shared_ptr<VectorV>>(operand_2),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::umin");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto operand_1_val = std::get<std::shared_ptr<VectorV>>(operand_1);
                auto operand_2_val = std::get<std::shared_ptr<VectorV>>(operand_2);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    uint64_t elem_result;
                    if (std::holds_alternative<uint64_t>(operand_1_val->elems[i]) &&
                        std::holds_alternative<uint64_t>(operand_2_val->elems[i]))
                    {
                        elem_result = std::min(std::get<uint64_t>(operand_1_val->elems[i]),
                                               std::get<uint64_t>(operand_2_val->elems[i]));
                    }
                    else if (std::holds_alternative<uint64_t>(operand_1_val->elems[i]) &&
                             std::holds_alternative<int64_t>(operand_2_val->elems[i]))
                    {
                        elem_result = std::min(std::get<uint64_t>(operand_1_val->elems[i]),
                                               bit_cast<uint64_t>(std::get<int64_t>(operand_2_val->elems[i])));
                    }
                    else if (std::holds_alternative<int64_t>(operand_1_val->elems[i]) &&
                             std::holds_alternative<uint64_t>(operand_2_val->elems[i]))
                    {
                        elem_result = std::min(bit_cast<uint64_t>(std::get<int64_t>(operand_1_val->elems[i])),
                                               std::get<uint64_t>(operand_2_val->elems[i]));
                    }
                    else if (std::holds_alternative<int64_t>(operand_1_val->elems[i]) &&
                             std::holds_alternative<int64_t>(operand_2_val->elems[i]))
                    {
                        elem_result = std::min(bit_cast<uint64_t>(std::get<int64_t>(operand_1_val->elems[i])),
                                               bit_cast<uint64_t>(std::get<int64_t>(operand_2_val->elems[i])));
                    }
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Int)
            {
                Value result;
                if (std::holds_alternative<uint64_t>(operand_1) && std::holds_alternative<uint64_t>(operand_2))
                {
                    result = std::min(std::get<uint64_t>(operand_1), std::get<uint64_t>(operand_2));
                }
                else if (std::holds_alternative<uint64_t>(operand_1) && std::holds_alternative<int64_t>(operand_2))
                {
                    result = std::min(std::get<uint64_t>(operand_1), bit_cast<uint64_t>(std::get<int64_t>(operand_2)));
                }
                else if (std::holds_alternative<int64_t>(operand_1) && std::holds_alternative<uint64_t>(operand_2))
                {
                    result = std::min(bit_cast<uint64_t>(std::get<int64_t>(operand_1)), std::get<uint64_t>(operand_2));
                }
                else if (std::holds_alternative<int64_t>(operand_1) && std::holds_alternative<int64_t>(operand_2))
                {
                    result = std::min(bit_cast<uint64_t>(std::get<int64_t>(operand_1)),
                                      bit_cast<uint64_t>(std::get<int64_t>(operand_2)));
                }

                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 39:
        { // SMin
            const Value& operand_1 = GetValue(operand_words[0]);
            const Value& operand_2 = GetValue(operand_words[1]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand_1) &&
                            std::holds_alternative<std::shared_ptr<VectorV>>(operand_2),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::smin");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto operand_1_val = std::get<std::shared_ptr<VectorV>>(operand_1);
                auto operand_2_val = std::get<std::shared_ptr<VectorV>>(operand_2);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    int64_t elem_result;
                    if (std::holds_alternative<uint64_t>(operand_1_val->elems[i]) &&
                        std::holds_alternative<uint64_t>(operand_2_val->elems[i]))
                    {
                        elem_result = std::min(bit_cast<int64_t>(std::get<uint64_t>(operand_1_val->elems[i])),
                                               bit_cast<int64_t>(std::get<uint64_t>(operand_2_val->elems[i])));
                    }
                    else if (std::holds_alternative<uint64_t>(operand_1_val->elems[i]) &&
                             std::holds_alternative<int64_t>(operand_2_val->elems[i]))
                    {
                        elem_result = std::min(bit_cast<int64_t>(std::get<uint64_t>(operand_1_val->elems[i])),
                                               std::get<int64_t>(operand_2_val->elems[i]));
                    }
                    else if (std::holds_alternative<int64_t>(operand_1_val->elems[i]) &&
                             std::holds_alternative<uint64_t>(operand_2_val->elems[i]))
                    {
                        elem_result = std::min(std::get<int64_t>(operand_1_val->elems[i]),
                                               bit_cast<int64_t>(std::get<uint64_t>(operand_2_val->elems[i])));
                    }
                    else if (std::holds_alternative<int64_t>(operand_1_val->elems[i]) &&
                             std::holds_alternative<int64_t>(operand_2_val->elems[i]))
                    {
                        elem_result = std::min(std::get<int64_t>(operand_1_val->elems[i]),
                                               std::get<int64_t>(operand_2_val->elems[i]));
                    }
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Int)
            {
                Value result;
                if (std::holds_alternative<uint64_t>(operand_1) && std::holds_alternative<uint64_t>(operand_2))
                {
                    result = std::min(bit_cast<int64_t>(std::get<uint64_t>(operand_1)),
                                      bit_cast<int64_t>(std::get<uint64_t>(operand_2)));
                }
                else if (std::holds_alternative<uint64_t>(operand_1) && std::holds_alternative<int64_t>(operand_2))
                {
                    result = std::min(bit_cast<int64_t>(std::get<uint64_t>(operand_1)), std::get<int64_t>(operand_2));
                }
                else if (std::holds_alternative<int64_t>(operand_1) && std::holds_alternative<uint64_t>(operand_2))
                {
                    result = std::min(std::get<int64_t>(operand_1), bit_cast<int64_t>(std::get<uint64_t>(operand_2)));
                }
                else if (std::holds_alternative<int64_t>(operand_1) && std::holds_alternative<int64_t>(operand_2))
                {
                    result = std::min(std::get<int64_t>(operand_1), std::get<int64_t>(operand_2));
                }

                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 41:
        { // UMax
            const Value& operand_1 = GetValue(operand_words[0]);
            const Value& operand_2 = GetValue(operand_words[1]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand_1) &&
                            std::holds_alternative<std::shared_ptr<VectorV>>(operand_2),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::umax");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto operand_1_val = std::get<std::shared_ptr<VectorV>>(operand_1);
                auto operand_2_val = std::get<std::shared_ptr<VectorV>>(operand_2);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    uint64_t elem_result;
                    if (std::holds_alternative<uint64_t>(operand_1_val->elems[i]) &&
                        std::holds_alternative<uint64_t>(operand_2_val->elems[i]))
                    {
                        elem_result = std::max(std::get<uint64_t>(operand_1_val->elems[i]),
                                               std::get<uint64_t>(operand_2_val->elems[i]));
                    }
                    else if (std::holds_alternative<uint64_t>(operand_1_val->elems[i]) &&
                             std::holds_alternative<int64_t>(operand_2_val->elems[i]))
                    {
                        elem_result = std::max(std::get<uint64_t>(operand_1_val->elems[i]),
                                               bit_cast<uint64_t>(std::get<int64_t>(operand_2_val->elems[i])));
                    }
                    else if (std::holds_alternative<int64_t>(operand_1_val->elems[i]) &&
                             std::holds_alternative<uint64_t>(operand_2_val->elems[i]))
                    {
                        elem_result = std::max(bit_cast<uint64_t>(std::get<int64_t>(operand_1_val->elems[i])),
                                               std::get<uint64_t>(operand_2_val->elems[i]));
                    }
                    else if (std::holds_alternative<int64_t>(operand_1_val->elems[i]) &&
                             std::holds_alternative<int64_t>(operand_2_val->elems[i]))
                    {
                        elem_result = std::max(bit_cast<uint64_t>(std::get<int64_t>(operand_1_val->elems[i])),
                                               bit_cast<uint64_t>(std::get<int64_t>(operand_2_val->elems[i])));
                    }
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Int)
            {
                Value result;
                if (std::holds_alternative<uint64_t>(operand_1) && std::holds_alternative<uint64_t>(operand_2))
                {
                    result = std::max(std::get<uint64_t>(operand_1), std::get<uint64_t>(operand_2));
                }
                else if (std::holds_alternative<uint64_t>(operand_1) && std::holds_alternative<int64_t>(operand_2))
                {
                    result = std::max(std::get<uint64_t>(operand_1), bit_cast<uint64_t>(std::get<int64_t>(operand_2)));
                }
                else if (std::holds_alternative<int64_t>(operand_1) && std::holds_alternative<uint64_t>(operand_2))
                {
                    result = std::max(bit_cast<uint64_t>(std::get<int64_t>(operand_1)), std::get<uint64_t>(operand_2));
                }
                else if (std::holds_alternative<int64_t>(operand_1) && std::holds_alternative<int64_t>(operand_2))
                {
                    result = std::max(bit_cast<uint64_t>(std::get<int64_t>(operand_1)),
                                      bit_cast<uint64_t>(std::get<int64_t>(operand_2)));
                }

                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 43:
        { // FClamp
            const Value& operand = GetValue(operand_words[0]);
            const Value& min_val = GetValue(operand_words[1]);
            const Value& max_val = GetValue(operand_words[2]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand) &&
                            std::holds_alternative<std::shared_ptr<VectorV>>(min_val) &&
                            std::holds_alternative<std::shared_ptr<VectorV>>(max_val),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::fclamp");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto vec     = std::get<std::shared_ptr<VectorV>>(operand);
                auto min_vec = std::get<std::shared_ptr<VectorV>>(min_val);
                auto max_vec = std::get<std::shared_ptr<VectorV>>(max_val);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    Value elem_result = (double)std::clamp(std::get<double>(vec->elems[i]),
                                                           std::get<double>(min_vec->elems[i]),
                                                           std::get<double>(max_vec->elems[i]));
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Float)
            {
                Value result =
                    (double)std::clamp(std::get<double>(operand), std::get<double>(min_val), std::get<double>(max_val));
                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 44:
        { // UClamp
            const Value& operand = GetValue(operand_words[0]);
            const Value& min_val = GetValue(operand_words[1]);
            const Value& max_val = GetValue(operand_words[2]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand) &&
                            std::holds_alternative<std::shared_ptr<VectorV>>(min_val) &&
                            std::holds_alternative<std::shared_ptr<VectorV>>(max_val),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::uclamp");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto vec     = std::get<std::shared_ptr<VectorV>>(operand);
                auto min_vec = std::get<std::shared_ptr<VectorV>>(min_val);
                auto max_vec = std::get<std::shared_ptr<VectorV>>(max_val);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    Value elem_result = (uint64_t)std::clamp(std::get<uint64_t>(vec->elems[i]),
                                                             std::get<uint64_t>(min_vec->elems[i]),
                                                             std::get<uint64_t>(max_vec->elems[i]));
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Float)
            {
                Value result = (uint64_t)std::clamp(
                    std::get<uint64_t>(operand), std::get<uint64_t>(min_val), std::get<uint64_t>(max_val));
                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 45:
        { // SClamp
            const Value& operand = GetValue(operand_words[0]);
            const Value& min_val = GetValue(operand_words[1]);
            const Value& max_val = GetValue(operand_words[2]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand) &&
                            std::holds_alternative<std::shared_ptr<VectorV>>(min_val) &&
                            std::holds_alternative<std::shared_ptr<VectorV>>(max_val),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::sclamp");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto vec     = std::get<std::shared_ptr<VectorV>>(operand);
                auto min_vec = std::get<std::shared_ptr<VectorV>>(min_val);
                auto max_vec = std::get<std::shared_ptr<VectorV>>(max_val);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    Value elem_result = (int64_t)std::clamp(std::get<int64_t>(vec->elems[i]),
                                                            std::get<int64_t>(min_vec->elems[i]),
                                                            std::get<int64_t>(max_vec->elems[i]));
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Float)
            {
                Value result = (int64_t)std::clamp(
                    std::get<int64_t>(operand), std::get<int64_t>(min_val), std::get<int64_t>(max_val));
                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 46:
        { // FMix
            const Value& x = GetValue(operand_words[0]);
            const Value& y = GetValue(operand_words[1]);
            const Value& a = GetValue(operand_words[2]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(x) &&
                            std::holds_alternative<std::shared_ptr<VectorV>>(y) &&
                            std::holds_alternative<std::shared_ptr<VectorV>>(a),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::fmix");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto xvec = std::get<std::shared_ptr<VectorV>>(x);
                auto yvec = std::get<std::shared_ptr<VectorV>>(y);
                auto avec = std::get<std::shared_ptr<VectorV>>(a);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    double x_d         = std::get<double>(xvec->elems[i]);
                    double y_d         = std::get<double>(yvec->elems[i]);
                    double a_d         = std::get<double>(avec->elems[i]);
                    Value  elem_result = (double)(x_d * (1 - a_d) + y_d * a_d);
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Float)
            {
                double x_d    = std::get<double>(x);
                double y_d    = std::get<double>(y);
                double a_d    = std::get<double>(a);
                Value  result = (double)(x_d * (1 - a_d) + y_d * a_d);
                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 50:
        { // Fma
            const Value& a_val = GetValue(operand_words[0]);
            const Value& b_val = GetValue(operand_words[1]);
            const Value& c_val = GetValue(operand_words[2]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(a_val) &&
                            std::holds_alternative<std::shared_ptr<VectorV>>(b_val) &&
                            std::holds_alternative<std::shared_ptr<VectorV>>(c_val),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::fma");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto a_vec = std::get<std::shared_ptr<VectorV>>(a_val);
                auto b_vec = std::get<std::shared_ptr<VectorV>>(b_val);
                auto c_vec = std::get<std::shared_ptr<VectorV>>(c_val);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    Value elem_result = (double)(std::get<double>(a_vec->elems[i]) * std::get<double>(b_vec->elems[i]) +
                                                 std::get<double>(c_vec->elems[i]));
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Float)
            {
                Value result = (double)(std::get<double>(a_val) * std::get<double>(b_val) + std::get<double>(c_val));
                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 66:
        { // Length
            const Value& operand      = GetValue(operand_words[0]);
            const Type&  operand_type = GetTypeByResultId(operand_words[0]);

            if (operand_type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::length");

                const Type& elem_type = GetTypeByTypeId(operand_type.vector.elem_type_id);

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto vec = std::get<std::shared_ptr<VectorV>>(operand);

                double len_sum = 0.0;
                for (uint32_t i = 0; i < operand_type.vector.elem_count; ++i)
                {
                    if (elem_type.kind == Type::Kind::Float)
                    {
                        len_sum += std::get<double>(vec->elems[i]) * std::get<double>(vec->elems[i]);
                    }
                    else if (elem_type.kind == Type::Kind::Int)
                    {
                        if (elem_type.scalar.is_signed)
                        {
                            len_sum += std::get<int64_t>(vec->elems[i]) * std::get<int64_t>(vec->elems[i]);
                        }
                        else
                        {
                            len_sum += std::get<uint64_t>(vec->elems[i]) * std::get<uint64_t>(vec->elems[i]);
                        }
                    }
                    else
                    {
                        assertx("SPIRV simulator: Unhandled type in vector operand for GLSL length");
                    }
                }

                len_sum = std::sqrt(len_sum);
                SetValue(result_id, len_sum);
            }
            else if (operand_type.kind == Type::Kind::Float)
            {
                SetValue(result_id, operand);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 69:
        { // Normalize
            const Value& operand = GetValue(operand_words[0]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::normalize");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto vec = std::get<std::shared_ptr<VectorV>>(operand);

                double len_sum = 0.0;
                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    len_sum += std::get<double>(vec->elems[i]) * std::get<double>(vec->elems[i]);
                }

                len_sum = std::sqrt(len_sum);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    Value elem_result = std::get<double>(vec->elems[i]) / len_sum;
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Float)
            {
                Value result = (double)1.0;
                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        case 79:
        { // NMin
            const Value& x_val = GetValue(operand_words[0]);
            const Value& y_val = GetValue(operand_words[1]);

            if (type.kind == Type::Kind::Vector)
            {
                assertm(std::holds_alternative<std::shared_ptr<VectorV>>(x_val) &&
                            std::holds_alternative<std::shared_ptr<VectorV>>(y_val),
                        "SPIRV simulator: Operands not of vector type in GLSLExtHandler::NMin");

                Value result     = std::make_shared<VectorV>();
                auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

                auto x = std::get<std::shared_ptr<VectorV>>(x_val);
                auto y = std::get<std::shared_ptr<VectorV>>(y_val);

                for (uint32_t i = 0; i < type.vector.elem_count; ++i)
                {
                    double x_d = std::get<double>(x->elems[i]);
                    double y_d = std::get<double>(y->elems[i]);

                    Value elem_result;
                    if (std::isnan(x_d))
                    {
                        elem_result = y_d;
                    }
                    else if (std::isnan(y_d))
                    {
                        elem_result = x_d;
                    }
                    else
                    {
                        elem_result = std::min(x_d, y_d);
                    }
                    result_vec->elems.push_back(elem_result);
                }

                SetValue(result_id, result_vec);
            }
            else if (type.kind == Type::Kind::Float)
            {
                double x_d = std::get<double>(x_val);
                double y_d = std::get<double>(y_val);

                Value result;
                if (std::isnan(x_d))
                {
                    result = y_d;
                }
                else if (std::isnan(y_d))
                {
                    result = x_d;
                }
                else
                {
                    result = std::min(x_d, y_d);
                }
                SetValue(result_id, result);
            }
            else
            {
                assertx("SPIRV simulator: Invalid type encountered in GLSLExtHandler");
            }
            break;
        }
        default:
        {
            if (verbose_)
            {
                std::cout << "SPIRV simulator: Unhandled OpExtInst GLSL set operation: " << instruction_literal
                          << std::endl;
                std::cout << "SPIRV simulator: Setting output to default value, this will likely crash" << std::endl;
            }
            SetValue(result_id, MakeDefault(type_id));
        }
    }
}

// ---------------------------------------------------------------------------
//  Type creation handlers
// ---------------------------------------------------------------------------
void SPIRVSimulator::T_Void(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeVoid);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind   = Type::Kind::Void;
    type.scalar = { 0, false };

    types_[result_id] = type;
}

void SPIRVSimulator::T_Bool(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeBool);

    // We treat bools as 64 bit unsigned ints for simplicity
    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind   = Type::Kind::BoolT;
    type.scalar = { 64, false };

    types_[result_id] = type;
}

void SPIRVSimulator::T_Int(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeInt);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind   = Type::Kind::Int;
    type.scalar = { instruction.words[2], (bool)instruction.words[3] };

    types_[result_id] = type;
}

void SPIRVSimulator::T_Float(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeFloat);

    // We dont handle floats encoded in other formats than the default at present
    uint32_t result_id = instruction.words[1];

    assertm(instruction.word_count <= 3,
            "SPIRV simulator: Simulator only supports IEEE 754 encoded floats at present.");

    Type type;
    type.kind   = Type::Kind::Float;
    type.scalar = { instruction.words[2], false };

    types_[result_id] = type;
}

void SPIRVSimulator::T_Vector(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeVector);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind   = Type::Kind::Vector;
    type.vector = { instruction.words[2], instruction.words[3] };

    types_[result_id] = type;
}

void SPIRVSimulator::T_Matrix(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeMatrix);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind   = Type::Kind::Matrix;
    type.matrix = { instruction.words[2], instruction.words[3] };

    types_[result_id] = type;
}

void SPIRVSimulator::T_Array(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeArray);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind  = Type::Kind::Array;
    type.array = { instruction.words[2], instruction.words[3] };

    types_[result_id] = type;
}

void SPIRVSimulator::T_Struct(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeStruct);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind       = Type::Kind::Struct;
    type.structure.id = instruction.words[1];

    types_[instruction.words[1]] = type;

    std::vector<uint32_t> members;
    for (auto i = 2; i < instruction.word_count; ++i)
    {
        members.push_back(instruction.words[i]);
    }

    struct_members_[result_id] = std::move(members);
}

void SPIRVSimulator::T_Pointer(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypePointer);

    uint32_t result_id       = instruction.words[1];
    uint32_t storage_class   = instruction.words[2];
    uint32_t pointee_type_id = instruction.words[3];

    Type type;
    type.kind         = Type::Kind::Pointer;
    type.pointer      = { storage_class, pointee_type_id };
    types_[result_id] = type;
}

void SPIRVSimulator::T_ForwardPointer(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeForwardPointer);

    // TODO: May not need this
    uint32_t pointer_type_id                    = instruction.words[1];
    uint32_t storage_class                      = instruction.words[2];
    forward_type_declarations_[pointer_type_id] = storage_class;
}

void SPIRVSimulator::T_RuntimeArray(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeRuntimeArray);

    uint32_t result_id    = instruction.words[1];
    uint32_t elem_type_id = instruction.words[2];

    Type type;
    type.kind         = Type::Kind::RuntimeArray;
    type.array        = { elem_type_id, 0 };
    types_[result_id] = type;
}

void SPIRVSimulator::T_Function(const Instruction& instruction)
{
    // This info is redundant for us, so treat it as a NOP
    assert(instruction.opcode == spv::Op::OpTypeFunction);
}

void SPIRVSimulator::T_Image(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeImage);

    uint32_t result_id       = instruction.words[1];
    uint32_t sampled_type_id = instruction.words[2];
    uint32_t dim             = instruction.words[3];
    uint32_t depth           = instruction.words[4];
    uint32_t arrayed         = instruction.words[5];
    uint32_t multisampled    = instruction.words[6];
    uint32_t sampled         = instruction.words[7];
    uint32_t image_format    = instruction.words[8];

    // uint32_t access_qualifier = spv::AccessQualifier::AccessQualifierMax;
    // if (instruction.word_count == 10)
    // {
    //     access_qualifier = instruction.words[9];
    // }

    Type type;
    type.kind         = Type::Kind::Image;
    type.image        = { sampled_type_id, dim, depth, arrayed, multisampled, sampled, image_format };
    types_[result_id] = type;
}

void SPIRVSimulator::T_Sampler(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeSampler);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind         = Type::Kind::Sampler;
    types_[result_id] = type;
}

void SPIRVSimulator::T_SampledImage(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeSampledImage);

    uint32_t result_id     = instruction.words[1];
    uint32_t image_type_id = instruction.words[2];

    Type type;
    type.kind          = Type::Kind::SampledImage;
    type.sampled_image = { image_type_id };
    types_[result_id]  = type;
}

void SPIRVSimulator::T_Opaque(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeOpaque);

    uint32_t result_id    = instruction.words[1];
    uint32_t name_literal = instruction.words[2];

    Type type;
    type.kind         = Type::Kind::Opaque;
    type.opaque       = { name_literal };
    types_[result_id] = type;
}

void SPIRVSimulator::T_NamedBarrier(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeNamedBarrier);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind         = Type::Kind::NamedBarrier;
    types_[result_id] = type;
}

void SPIRVSimulator::T_AccelerationStructureKHR(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeAccelerationStructureKHR);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind         = Type::Kind::AccelerationStructureKHR;
    types_[result_id] = type;
}

void SPIRVSimulator::T_RayQueryKHR(const Instruction& instruction)
{
    assert(instruction.opcode == spv::Op::OpTypeRayQueryKHR);

    uint32_t result_id = instruction.words[1];

    Type type;
    type.kind         = Type::Kind::RayQueryKHR;
    types_[result_id] = type;
}

// ---------------------------------------------------------------------------
//  Oparation implementations
// ---------------------------------------------------------------------------

void SPIRVSimulator::Op_EntryPoint(const Instruction& instruction)
{
    // We handle these during init/parsing
    assert(instruction.opcode == spv::Op::OpEntryPoint);
}

void SPIRVSimulator::Op_ExtInstImport(const Instruction& instruction)
{
    /*
    OpExtInstImport

    Import an extended set of instructions. It can be later referenced by the Result <id>.

    Name is the extended instruction-set’s name string. Before version 1.6, there must be an external specification
    defining the semantics for this extended instruction set. Starting with version 1.6, if Name starts with
    "NonSemantic.", including the period that separates the namespace "NonSemantic" from the rest of the name, it is
    encouraged for a specification to exist on the SPIR-V Registry, but it is not required.

    Starting with version 1.6, an extended instruction-set name which is prefixed with "NonSemantic." is guaranteed to
    contain only non-semantic instructions, and all OpExtInst instructions referencing this set can be ignored. All
    instructions within such a set must have only <id> operands; no literals. When literals are needed, then the Result
    <id> from an OpConstant or OpString instruction is referenced as appropriate. Result <id>s from these non-semantic
    instruction-set instructions must be used only in other non-semantic instructions.

    See Extended Instruction Sets for more information.
    */
    assert(instruction.opcode == spv::Op::OpExtInstImport);

    uint32_t result_id = instruction.words[1];
    // SPIRV string literals are UTF-8 encoded, so basic c++ string functionality can be used to decode them
    extended_imports_[result_id] = std::string((char*)(&instruction.words[2]), (instruction.word_count - 2) * 4);
}

void SPIRVSimulator::Op_Constant(const Instruction& instruction)
{
    /*
    OpConstant

    Declare a new integer-type or floating-point-type scalar constant.

    Result Type must be a scalar integer type or floating-point type.

    Value is the bit pattern for the constant. Types 32 bits wide or smaller take one word.

    Larger types take multiple words, with low-order words appearing first.
    */
    assert(instruction.opcode == spv::Op::OpConstant || instruction.opcode == spv::Op::OpSpecConstant);

    uint32_t    type_id   = instruction.words[1];
    uint32_t    result_id = instruction.words[2];
    const Type& type      = GetTypeByTypeId(type_id);

    assertm((type.kind == Type::Kind::Int) || (type.kind == Type::Kind::Float),
            "SPIRV simulator: Constant type unsupported");

    if (HasDecorator(result_id, spv::Decoration::DecorationSpecId))
    {
        uint32_t spec_id = GetDecoratorLiteral(result_id, spv::Decoration::DecorationSpecId);
        if (input_data_.specialization_constant_offsets.find(spec_id) !=
            input_data_.specialization_constant_offsets.end())
        {
            size_t           spec_id_offset = input_data_.specialization_constant_offsets.at(spec_id);
            const std::byte* raw_spec_const_data =
                static_cast<const std::byte*>(input_data_.specialization_constants) + spec_id_offset;
            std::vector<uint32_t> buffer_data;
            ReadWords(raw_spec_const_data, type_id, buffer_data);

            const uint32_t* buffer_pointer = buffer_data.data();
            SetValue(result_id, MakeScalar(type_id, buffer_pointer));
        }
        else
        {
            if (verbose_)
            {
                std::cout << execIndent << "No spec constant data provided for result_id: " << result_id
                          << ", using default" << std::endl;
            }
            const uint32_t* buffer_pointer = instruction.words.subspan(3).data();
            SetValue(result_id, MakeScalar(type_id, buffer_pointer));
        }
    }
    else
    {
        const uint32_t* buffer_pointer = instruction.words.subspan(3).data();
        SetValue(result_id, MakeScalar(type_id, buffer_pointer));
    }
}

void SPIRVSimulator::Op_ConstantComposite(const Instruction& instruction)
{
    /*
    OpConstantComposite

    Declare a new composite constant.

    Result Type must be a composite type, whose top-level members/elements/components/columns have the same type as the
    types of the Constituents. The ordering must be the same between the top-level types in Result Type and the
    Constituents.

    Constituents become members of a structure, or elements of an array, or components of a vector, or columns of a
    matrix. There must be exactly one Constituent for each top-level member/element/component/column of the result. The
    Constituents must appear in the order needed by the definition of the Result Type. The Constituents must all be
    <id>s of non-specialization constant-instruction declarations or an OpUndef.
    */
    assert(instruction.opcode == spv::Op::OpConstantComposite ||
           instruction.opcode == spv::Op::OpSpecConstantComposite);
    Op_CompositeConstruct(instruction);
}

void SPIRVSimulator::Op_CompositeConstruct(const Instruction& instruction)
{
    /*
    OpCompositeConstruct

    Construct a new composite object from a set of constituent objects.

    Result Type must be a composite type, whose top-level members/elements/components/columns have the same
    type as the types of the operands, with one exception.

    The exception is that for constructing a vector, the operands may also be vectors with the same component
    type as the Result Type component type.

    If constructing a vector, the total number of components in all the operands must equal
    the number of components in Result Type.

    Constituents become members of a structure, or elements of an array, or components of a vector, or columnsof a
    matrix. There must be exactly one Constituent for each top-level member/element/component/column of the result,with
    one exception.

    The exception is that for constructing a vector, a contiguous subset of the scalars consumed can be represented by
    a vector operand instead.

    The Constituents must appear in the order needed by the definition of the type of the result.
    If constructing a vector, there must be at least two Constituent operands.

    */
    assert(instruction.opcode == spv::Op::OpCompositeConstruct || instruction.opcode == spv::Op::OpConstantComposite ||
           instruction.opcode == spv::Op::OpSpecConstantComposite);

    // Composite: An aggregate (structure or an array), a matrix, or a vector.
    uint32_t    type_id   = instruction.words[1];
    uint32_t    result_id = instruction.words[2];
    const Type& type      = GetTypeByTypeId(type_id);

    bool is_arbitrary = false;

    if (type.kind == Type::Kind::Vector)
    {
        auto vec = std::make_shared<VectorV>();
        for (auto i = 3; i < instruction.word_count; ++i)
        {
            const Value& component_value = GetValue(instruction.words[i]);
            is_arbitrary |= ValueIsArbitrary(instruction.words[i]);

            if (std::holds_alternative<std::shared_ptr<VectorV>>(component_value))
            {
                std::shared_ptr<VectorV> component_vector = std::get<std::shared_ptr<VectorV>>(component_value);

                for (auto& vec_component : component_vector->elems)
                {
                    vec->elems.push_back(vec_component);
                }
            }
            else
            {
                vec->elems.push_back(component_value);
            }
        }

        SetValue(result_id, vec);
    }
    else if (type.kind == Type::Kind::Matrix)
    {
        auto matrix = std::make_shared<MatrixV>();
        for (auto i = 3; i < instruction.word_count; ++i)
        {
            is_arbitrary |= ValueIsArbitrary(instruction.words[i]);
            matrix->cols.push_back(GetValue(instruction.words[i]));
        }

        SetValue(result_id, matrix);
    }
    else if (type.kind == Type::Kind::Struct || type.kind == Type::Kind::Array || type.kind == Type::Kind::RuntimeArray)
    {
        auto aggregate = std::make_shared<AggregateV>();
        for (auto i = 3; i < instruction.word_count; ++i)
        {
            is_arbitrary |= ValueIsArbitrary(instruction.words[i]);
            aggregate->elems.push_back(GetValue(instruction.words[i]));
        }

        SetValue(result_id, aggregate);
    }
    else
    {
        assertx("SPIRV simulator: CompositeConstruct not implemented yet for type");
    }

    if (is_arbitrary)
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_Variable(const Instruction& instruction)
{
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

    uint32_t type_id       = instruction.words[1];
    uint32_t result_id     = instruction.words[2];
    uint32_t storage_class = instruction.words[3];

    const Type& type = GetTypeByTypeId(type_id);

    assertm(type.kind == Type::Kind::Pointer, "SPIRV simulator: Op_Variable must only be used to create pointer types");

    PointerV new_pointer{ 0, type_id, result_id, storage_class, {} };

    if (type.pointer.storage_class == spv::StorageClass::StorageClassPushConstant)
    {
        const std::byte* external_pointer = static_cast<const std::byte*>(input_data_.push_constants);
        new_pointer.pointer_handle        = bit_cast<uint64_t>(input_data_.push_constants);
    }
    else if (type.pointer.storage_class == spv::StorageClass::StorageClassUniform ||
             type.pointer.storage_class == spv::StorageClass::StorageClassUniformConstant ||
             type.pointer.storage_class == spv::StorageClass::StorageClassStorageBuffer)
    {
        assertm(HasDecorator(result_id, spv::Decoration::DecorationDescriptorSet),
                "SPIRV simulator: OpVariable called with result_id that lacks the DescriptorSet decoration, but the "
                "storage class requires it");
        assertm(HasDecorator(result_id, spv::Decoration::DecorationBinding),
                "SPIRV simulator: OpVariable called with result_id that lacks the Binding decoration, but the storage "
                "class requires it");

        uint32_t descriptor_set = GetDecoratorLiteral(result_id, spv::Decoration::DecorationDescriptorSet);
        uint32_t binding        = GetDecoratorLiteral(result_id, spv::Decoration::DecorationBinding);

        const std::byte* external_pointer = nullptr;

        if (input_data_.bindings.find(descriptor_set) != input_data_.bindings.end())
        {
            if (input_data_.bindings.at(descriptor_set).find(binding) != input_data_.bindings.at(descriptor_set).end())
            {
                external_pointer = static_cast<std::byte*>(input_data_.bindings.at(descriptor_set).at(binding));
            }
        }

        new_pointer.pointer_handle = external_pointer ? bit_cast<uint64_t>(external_pointer) : 0;
    }
    else if (type.pointer.storage_class == spv::StorageClass::StorageClassPhysicalStorageBuffer)
    {
        // This is illegal
        assertx("SPIRV simulator: Op_Variable must not be used to create pointer types with the PhysicalStorageBuffer "
                "storage class");
    }
    else if (type.pointer.storage_class == spv::StorageClass::StorageClassFunction ||
             type.pointer.storage_class == spv::StorageClass::StorageClassWorkgroup ||
             type.pointer.storage_class == spv::StorageClass::StorageClassPrivate ||
             type.pointer.storage_class == spv::StorageClass::StorageClassInput ||
             type.pointer.storage_class == spv::StorageClass::StorageClassOutput)
    {
        if (instruction.word_count >= 5)
        {
            new_pointer.pointer_handle = HeapAllocate(type.pointer.storage_class, GetValue(instruction.words[4]));
        }
        else
        {
            new_pointer.pointer_handle =
                HeapAllocate(type.pointer.storage_class, MakeDefault(type.pointer.pointee_type_id));
        }
    }
    else
    {
        assertx("SPIRV simulator: Unhandled Op_Variable storage class, add support to continue");
    }

    const Type& pointee_type = GetTypeByTypeId(type.pointer.pointee_type_id);
    if ((pointee_type.kind == Type::Kind::Pointer) &&
        (pointee_type.pointer.storage_class == spv::StorageClass::StorageClassPhysicalStorageBuffer))
    {
        // This pointer points to a physical storage buffer pointer
        // This is the easy case where we can extract the location of the physical
        // pointer from this pointer's offsets and storage class
        PointerV ppointer = std::get<PointerV>(ReadPointer(new_pointer));
        pointers_to_physical_address_pointers_.push_back(std::pair<PointerV, PointerV>{ new_pointer, ppointer });
    }

    // TODO: Compare pointer with candidates here and track

    SetValue(result_id, new_pointer);
}

void SPIRVSimulator::Op_ImageTexelPointer(const Instruction& instruction)
{
    /*
    OpImageTexelPointer

    Form a pointer to a texel of an image. Use of such a pointer is limited to atomic operations.
    Result Type must be an OpTypePointer whose Storage Class operand is Image.
    Its Type operand must be a scalar numerical type or OpTypeVoid.

    Image must have a type of OpTypePointer with Type OpTypeImage.
    The Sampled Type of the type of Image must be the same as the Type pointed to by Result Type. The Dim operand of
    Type must not be SubpassData.

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

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t image_id  = instruction.words[3];
    uint32_t coord_id  = instruction.words[4];
    uint32_t sample_id = instruction.words[5];

    const Type& type = GetTypeByTypeId(type_id);

    assertm(type.kind == Type::Kind::Pointer,
            "SPIRV simulator: Op_ImageTexelPointer must only be used to create pointer types");
    assertm(type.pointer.storage_class == spv::StorageClass::StorageClassImage,
            "SPIRV simulator: Op_ImageTexelPointer must only be used to create pointer types");

    Value    init = MakeDefault(type.pointer.pointee_type_id);
    PointerV new_pointer{ HeapAllocate(spv::StorageClass::StorageClassImage, init),
                          type_id,
                          result_id,
                          spv::StorageClass::StorageClassImage,
                          {} };

    SetValue(result_id, new_pointer);
    SetIsArbitrary(result_id);
}

void SPIRVSimulator::Op_Load(const Instruction& instruction)
{
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

    uint32_t type_id    = instruction.words[1];
    uint32_t result_id  = instruction.words[2];
    uint32_t pointer_id = instruction.words[3];

    const PointerV& pointer = std::get<PointerV>(GetValue(pointer_id));

    // TODO: Compare pointer with candidates here and track
    SetValue(result_id, ReadPointer(pointer));

    if (pointer.storage_class == spv::StorageClass::StorageClassInput ||
        pointer.storage_class == spv::StorageClass::StorageClassOutput)
    {
        SetIsArbitrary(result_id);
    }

    if (values_stored_.find(pointer_id) != values_stored_.end())
    {
        if (ValueIsArbitrary(values_stored_[pointer_id]))
        {
            SetIsArbitrary(result_id);
        }
    }
}

void SPIRVSimulator::Op_Store(const Instruction& instruction)
{
    /*
    OpStore

    Store through a pointer.

    Pointer is the pointer to store through. Its type must be an OpTypePointer whose Type operand is the same as the
    type of Object. Object is the object to store.

    If present, any Memory Operands must begin with a memory operand literal.
    If not present, it is the same as specifying the memory operand None.
    */
    assert(instruction.opcode == spv::Op::OpStore);

    uint32_t        pointer_id = instruction.words[1];
    uint32_t        result_id  = instruction.words[2];
    const PointerV& pointer    = std::get<PointerV>(GetValue(pointer_id));

    WritePointer(pointer, GetValue(result_id));

    // TODO: Compare pointer with candidates here and track

    // If this is a non-interpolated output value, the shader may be important for pbuffer pointer detection
    if (pointer.storage_class == spv::StorageClass::StorageClassOutput)
    {
        // TODO: Double check types, if this is a value that cant be interpolated, it may be flat even if not decorated
        // as such
        if (HasDecorator(pointer.base_result_id, spv::Decoration::DecorationFlat))
        {
            has_buffer_writes_ = true;
        }
    }
    else if ((pointer.storage_class != spv::StorageClass::StorageClassFunction) &&
             (pointer.storage_class != spv::StorageClass::StorageClassImage))
    {
        // If we are writing to any storage class that is not function or image, the shader may be important for pbuffer
        // pointer detection
        has_buffer_writes_ = true;
    }

    values_stored_[pointer_id] = result_id;
}

void SPIRVSimulator::Op_AccessChain(const Instruction& instruction)
{
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

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t base_id   = instruction.words[3];

    const Value& base_value = GetValue(base_id);
    const Type&  base_type  = GetTypeByResultId(base_id);

    assertm(std::holds_alternative<PointerV>(base_value),
            "SPIRV simulator: Attempt to use OpAccessChain on a non-pointer value");

    PointerV new_pointer = std::get<PointerV>(base_value);

    for (auto i = 4; i < instruction.word_count; ++i)
    {
        const Value& index_value = GetValue(instruction.words[i]);

        if (std::holds_alternative<uint64_t>(index_value))
        {
            new_pointer.idx_path.push_back((uint32_t)std::get<uint64_t>(index_value));
        }
        else if (std::holds_alternative<int64_t>(index_value))
        {
            new_pointer.idx_path.push_back((uint32_t)std::get<int64_t>(index_value));
        }
        else
        {
            assertx("SPIRV simulator: Index not of integer type in Op_AccessChain");
        }
    }

    if (base_type.pointer.storage_class == spv::StorageClass::StorageClassPhysicalStorageBuffer)
    {
        physical_address_pointers_.push_back(new_pointer);
    }

    const Type& result_type         = GetTypeByTypeId(type_id);
    const Type& result_pointee_type = GetTypeByTypeId(result_type.pointer.pointee_type_id);
    if ((result_pointee_type.kind == Type::Kind::Pointer) &&
        (result_pointee_type.pointer.storage_class == spv::StorageClass::StorageClassPhysicalStorageBuffer))
    {
        // This pointer points to a physical storage buffer pointer
        // This is the semi-easy case where we can extract the location of the physical
        // pointer from this pointer's offsets and storage class, but with the caveat that the resulting pointer
        // is itself stored in a physical storage buffer (hence we need the containing buffer to find its actual
        // address)
        PointerV ppointer = std::get<PointerV>(ReadPointer(new_pointer));
        pointers_to_physical_address_pointers_.push_back(std::pair<PointerV, PointerV>{ new_pointer, ppointer });
    }

    // TODO: Compare pointer with candidates here and track

    if (ValueIsArbitrary(base_id))
    {
        SetIsArbitrary(result_id);
    }

    SetValue(result_id, new_pointer);
}

void SPIRVSimulator::Op_Function(const Instruction& instruction)
{
    /*
    OpFunction

    Add a function. This instruction must be immediately followed by one OpFunctionParameter instruction per each
    formal parameter of this function. This function’s body or declaration terminates with the next OpFunctionEnd
    instruction.

    Result Type must be the same as the Return Type declared in Function Type.

    Function Type is the result of an OpTypeFunction, which declares the types of the return value and parameters of the
    function.
    */
    assert(instruction.opcode == spv::Op::OpFunction);
    // Nothing to do, we handle this when parsing instructions
}

void SPIRVSimulator::Op_FunctionEnd(const Instruction& instruction)
{
    /*
    OpFunctionEnd

    Last instruction of a function.
    */
    // This is a NOP in our design
    assert(instruction.opcode == spv::Op::OpFunctionEnd);
}

void SPIRVSimulator::Op_FunctionCall(const Instruction& instruction)
{
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

    uint32_t result_id   = instruction.words[2];
    uint32_t function_id = instruction.words[3];

    FunctionInfo& function_info = funcs_[function_id];
    call_stack_.push_back({ function_info.first_inst_index, result_id, current_heap_index_ });

    uint32_t parameter_index = 0;
    for (auto i = 4; i < instruction.word_count; ++i)
    {
        // Push parameters to the local scope
        values_[function_info.parameter_ids_[parameter_index]] = GetValue(instruction.words[i]);

        parameter_index += 1;
    }
}

void SPIRVSimulator::Op_Label(const Instruction& instruction)
{
    /*
    OpLabel

    The label instruction of a block.

    References to a block are through the Result <id> of its label.
    */
    assert(instruction.opcode == spv::Op::OpLabel);

    uint32_t result_id = instruction.words[1];
    prev_block_id_     = current_block_id_;
    current_block_id_  = result_id;

    if ((current_block_id_ == current_merge_block_id_) && is_execution_fork)
    {
        // We are done, merge back and communicate fork info to the callee
        call_stack_.clear();
    }
}

void SPIRVSimulator::Op_Branch(const Instruction& instruction)
{
    /*
    OpBranch

    Unconditional branch to Target Label.
    Target Label must be the Result <id> of an OpLabel instruction in the current function.
    This instruction must be the last instruction in a block.
    */
    assert(instruction.opcode == spv::Op::OpBranch);

    uint32_t result_id    = instruction.words[1];
    call_stack_.back().pc = result_id_to_inst_index_.at(result_id);
}

void SPIRVSimulator::Op_BranchConditional(const Instruction& instruction)
{
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

    uint32_t condition_id = instruction.words[1];
    uint32_t label_1_id   = instruction.words[2];
    uint32_t label_2_id   = instruction.words[3];

    uint64_t condition    = std::get<uint64_t>(GetValue(condition_id));
    uint32_t target_label = condition ? label_1_id : label_2_id;

    // We may need to diverge and execute both branches here.
    // Only do it if the conditional is arbitrary, and if we are looping, only do so if we are skipping the loop
    // (eg. target id is not the continue)
    if (ValueIsArbitrary(condition_id) && (target_label != current_continue_block_id_))
    {

        if ((fork_abort_trigger_id_ == condition_id) && (fork_abort_target_id_ == target_label))
        {
            // Do not fork again, we are entering an infite loop by creating a fork equal to the one that started this
            // one If this ever happens, we are done so just return
            call_stack_.clear();
            return;
        }

        SPIRVSimulator fork;
        fork.CreateExecutionFork(*this, condition_id, target_label);

        const auto& fork_results = fork.GetPhysicalAddressData();

        if (fork_results.size())
        {
            if (verbose_)
            {
                std::cout << "SPIRV simulator: Execution fork complete, got: " << fork_results.size()
                          << " fork results at execution level: " << current_fork_index_ << std::endl;
                std::cout
                    << "                 Note that advanced variable adaptation to the arbitrary branch investigation "
                       "is not implemented, there is a chance that the pbuffer pointer metadata is incomplete."
                    << std::endl;
            }

            physical_address_pointer_source_data_.insert(
                physical_address_pointer_source_data_.end(), fork_results.begin(), fork_results.end());
        }
    }

    call_stack_.back().pc = result_id_to_inst_index_.at(target_label);
}

void SPIRVSimulator::Op_Return(const Instruction& instruction)
{
    /*
    OpReturn

    Return with no value from a function with void return type.
    This instruction must be the last instruction in a block.
    */
    assert(instruction.opcode == spv::Op::OpReturn);

#ifdef DEBUG_BUILD
    // Clear the heap for better error checking
    uint32_t stack_heap_index = call_stack_.back().func_heap_index;
    for (auto heap_index = stack_heap_index; heap_index < current_heap_index_; ++heap_index)
    {
        function_heap_[heap_index] = std::monostate{};
    }
    // TODO: Maybe clear locals as well
#endif

    current_heap_index_ = call_stack_.back().func_heap_index;

    call_stack_.pop_back();
}

void SPIRVSimulator::Op_ReturnValue(const Instruction& instruction)
{
    /*
    OpReturnValue

    Return a value from a function.

    Value is the value returned, by copy, and must match the Return Type operand of the OpTypeFunction
    type of the OpFunction body this return instruction is in. Value must not have type OpTypeVoid.

    This instruction must be the last instruction in a block.
    */
    assert(instruction.opcode == spv::Op::OpReturnValue);

    uint32_t value_id     = instruction.words[1];
    uint32_t result_id    = call_stack_.back().result_id;
    Value    return_value = GetValue(value_id);

#ifdef DEBUG_BUILD
    // Clear the heap for better error checking
    uint32_t stack_heap_index = call_stack_.back().func_heap_index;
    for (auto heap_index = stack_heap_index; heap_index < current_heap_index_; ++heap_index)
    {
        function_heap_[heap_index] = std::monostate{};
    }
    // TODO: Maybe clear locals as well
#endif

    current_heap_index_ = call_stack_.back().func_heap_index;

    call_stack_.pop_back();

    if (call_stack_.size())
    {
        SetValue(result_id, return_value);
        if (ValueIsArbitrary(value_id))
        {
            SetIsArbitrary(result_id);
        }
    }
}

void SPIRVSimulator::Op_FAdd(const Instruction& instruction)
{
    /*
    OpFAdd

    Floating-point addition of Operand 1 and Operand 2.
    Result Type must be a scalar or vector of floating-point type.
    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFAdd);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm((std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                 std::holds_alternative<std::shared_ptr<VectorV>>(val_op2)),
                "SPIRV simulator: Operands not of vector type in Op_FAdd");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands not of equal/correct length in Op_FAdd");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm((std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i])),
                    "SPIRV simulator: vector contains non-doubles in Op_FAdd");
            double elem_result = std::get<double>(vec1->elems[i]) + std::get<double>(vec2->elems[i]);
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Float)
    {
        Value        result;
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        assertm((std::holds_alternative<double>(op1) && std::holds_alternative<double>(op2)),
                "SPIRV simulator: Operands not of float type in Op_FAdd");

        result = std::get<double>(op1) + std::get<double>(op2);

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type for Op_FAdd, must be vector or float");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_ExtInst(const Instruction& instruction)
{
    /*
    Execute an instruction in an imported set of extended instructions.

    Result Type is defined, per Instruction, in the external specification for Set.
    Set is the result of an OpExtInstImport instruction.
    Instruction is the enumerant of the instruction to execute within Set.
    It is an unsigned 32-bit integer. The semantics of the instruction are defined in the external specification for
    Set.

    Operand 1, …​ are the operands to the extended instruction.
    */
    assert(instruction.opcode == spv::Op::OpExtInst);

    uint32_t type_id             = instruction.words[1];
    uint32_t result_id           = instruction.words[2];
    uint32_t set_id              = instruction.words[3];
    uint32_t instruction_literal = instruction.words[4];

    assertm(extended_imports_.find(set_id) != extended_imports_.end(),
            "SPIRV simulator: Unsupported set ID (it has not been imported9) for Op_ExtInst");

    std::string                     set_literal   = extended_imports_[set_id];
    const std::span<const uint32_t> operand_words = std::span<const uint32_t>(instruction.words).subspan(5);
    if (!std::strncmp(set_literal.c_str(), "GLSL.std.450", set_literal.length()))
    {
        GLSLExtHandler(type_id, result_id, instruction_literal, operand_words);
    }
    else
    {
        if (verbose_)
        {
            std::cout << execIndent << "OpExtInst set with literal: " << set_literal
                      << " (length: " << set_literal.length() << ") " << " does not exist" << std::endl;
        }
        SetValue(result_id, MakeDefault(type_id));
    }
}

void SPIRVSimulator::Op_SelectionMerge(const Instruction& instruction)
{
    /*
    OpSelectionMerge

    Declare a structured selection.

    This instruction must immediately precede either an OpBranchConditional or OpSwitch instruction.
    That is, it must be the second-to-last instruction in its block.

    Merge Block is the label of the merge block for this structured selection.

    See Structured Control Flow for more detail.
    */
    assert(instruction.opcode == spv::Op::OpSelectionMerge);

    uint32_t merge_block_id = instruction.words[1];

    current_merge_block_id_ = merge_block_id;
}

void SPIRVSimulator::Op_LoopMerge(const Instruction& instruction)
{
    /*
    OpLoopMerge

    Declare a structured loop.

    This instruction must immediately precede either an OpBranch or OpBranchConditional instruction.
    That is, it must be the second-to-last instruction in its block.

    Merge Block is the label of the merge block for this structured loop.

    Continue Target is the label of a block targeted for processing a loop "continue".

    Loop Control Parameters appear in Loop Control-table order for any Loop Control setting that requires such a
    parameter.

    See Structured Control Flow for more detail.
    */
    assert(instruction.opcode == spv::Op::OpLoopMerge);

    uint32_t merge_block_id     = instruction.words[1];
    uint32_t continue_target_id = instruction.words[2];

    current_merge_block_id_    = merge_block_id;
    current_continue_block_id_ = continue_target_id;
}

void SPIRVSimulator::Op_FMul(const Instruction& instruction)
{
    /*
    OpFMul

    Floating-point multiplication of Operand 1 and Operand 2.
    Result Type must be a scalar or vector of floating-point type.
    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFMul);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm((std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                 std::holds_alternative<std::shared_ptr<VectorV>>(val_op2)),
                "SPIRV simulator: Operands not of vector type in Op_FMul");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands not of equal/correct length in Op_FMul");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm((std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i])),
                    "SPIRV simulator: vector contains non-doubles in Op_FMul");
            double elem_result = std::get<double>(vec1->elems[i]) * std::get<double>(vec2->elems[i]);
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Float)
    {
        Value        result;
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        assertm((std::holds_alternative<double>(op1) && std::holds_alternative<double>(op2)),
                "SPIRV simulator: Operands are not floats/doubles in Op_FMul");

        result = std::get<double>(op1) * std::get<double>(op2);

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type for Op_FMul, must be vector or float");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_INotEqual(const Instruction& instruction)
{
    /*
    OpINotEqual

    Integer comparison for inequality.
    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.
    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpINotEqual);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm((std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                 std::holds_alternative<std::shared_ptr<VectorV>>(val_op2)),
                "SPIRV simulator: Operands not of vector type in Op_INotEqual");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands not of equal/correct length in Op_INotEqual");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            uint64_t elem_result;

            // This should compare equal if different types but same number, so cant use variant operators here
            // TODO: Refactor this and the similar blocks below
            if (std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i]))
            {
                elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) != std::get<uint64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<uint64_t>(vec1->elems[i]) &&
                     std::holds_alternative<int64_t>(vec2->elems[i]))
            {
                elem_result =
                    (uint64_t)(std::get<uint64_t>(vec1->elems[i]) != (uint64_t)std::get<int64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]))
            {
                elem_result = (uint64_t)(std::get<int64_t>(vec1->elems[i]) != std::get<int64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<int64_t>(vec1->elems[i]) &&
                     std::holds_alternative<uint64_t>(vec2->elems[i]))
            {
                elem_result =
                    (uint64_t)((uint64_t)std::get<int64_t>(vec1->elems[i]) != std::get<uint64_t>(vec2->elems[i]));
            }
            else
            {
                assertx(
                    "SPIRV simulator: Could not find valid parameter type combination for Op_INotEqual vector operand");
            }

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        Value        result;
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        if (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2))
        {
            result = (uint64_t)(std::get<uint64_t>(op1) != std::get<uint64_t>(op2));
        }
        else if (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<int64_t>(op2))
        {
            result = (uint64_t)(std::get<uint64_t>(op1) != (uint64_t)std::get<int64_t>(op2));
        }
        else if (std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2))
        {
            result = (uint64_t)(std::get<int64_t>(op1) != std::get<int64_t>(op2));
        }
        else if (std::holds_alternative<int64_t>(op1) && std::holds_alternative<uint64_t>(op2))
        {
            result = (uint64_t)((uint64_t)std::get<int64_t>(op1) != std::get<uint64_t>(op2));
        }
        else
        {
            assertx("SPIRV simulator: Could not find valid parameter type combination for Op_INotEqual");
        }

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type for Op_IAdd, must be vector or bool");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_IAdd(const Instruction& instruction)
{
    /*
    OpIAdd

    Integer addition of Operand 1 and Operand 2.

    Result Type must be a scalar or vector of integer type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type. They must have the same number of
    components as Result Type. They must have the same component width as Result Type.

    The resulting value equals the low-order N bits of the correct result R, where N is the component
    width and R is computed with enough precision to avoid overflow and underflow.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpIAdd);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type type = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands not of vector type in Op_IAdd");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands not of equal/correct length in Op_IAdd");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            Value elem_result;

            if (std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i]))
            {
                elem_result = (std::get<uint64_t>(vec1->elems[i]) + std::get<uint64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<uint64_t>(vec1->elems[i]) &&
                     std::holds_alternative<int64_t>(vec2->elems[i]))
            {
                elem_result = (std::get<uint64_t>(vec1->elems[i]) + std::get<int64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]))
            {
                elem_result = (std::get<int64_t>(vec1->elems[i]) + std::get<int64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<int64_t>(vec1->elems[i]) &&
                     std::holds_alternative<uint64_t>(vec2->elems[i]))
            {
                elem_result = (std::get<int64_t>(vec1->elems[i]) + std::get<uint64_t>(vec2->elems[i]));
            }
            else
            {
                assertx("SPIRV simulator: Could not find valid parameter type combination for Op_IAdd vector operand");
            }

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Int)
    {
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        if (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2))
        {
            result = (std::get<uint64_t>(op1) + std::get<uint64_t>(op2));
        }
        else if (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<int64_t>(op2))
        {
            result = (std::get<uint64_t>(op1) + std::get<int64_t>(op2));
        }
        else if (std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2))
        {
            result = (std::get<int64_t>(op1) + std::get<int64_t>(op2));
        }
        else if (std::holds_alternative<int64_t>(op1) && std::holds_alternative<uint64_t>(op2))
        {
            result = (std::get<int64_t>(op1) + std::get<uint64_t>(op2));
        }
        else
        {
            assertx("SPIRV simulator: Could not find valid parameter type combination for Op_IAdd");
        }

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type for Op_IAdd, must be vector or int");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_ISub(const Instruction& instruction)
{
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

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    Type type = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands not of vector type in Op_ISub");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands not of equal/correct length in Op_ISub");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            Value elem_result;

            if (std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i]))
            {
                elem_result = (std::get<uint64_t>(vec1->elems[i]) - std::get<uint64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<uint64_t>(vec1->elems[i]) &&
                     std::holds_alternative<int64_t>(vec2->elems[i]))
            {
                elem_result = (std::get<uint64_t>(vec1->elems[i]) - std::get<int64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]))
            {
                elem_result = (std::get<int64_t>(vec1->elems[i]) - std::get<int64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<int64_t>(vec1->elems[i]) &&
                     std::holds_alternative<uint64_t>(vec2->elems[i]))
            {
                elem_result = (std::get<int64_t>(vec1->elems[i]) - std::get<uint64_t>(vec2->elems[i]));
            }
            else
            {
                assertx("SPIRV simulator: Could not find valid parameter type combination for Op_ISub vector operand");
            }

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Int)
    {
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        if (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2))
        {
            result = (std::get<uint64_t>(op1) - std::get<uint64_t>(op2));
        }
        else if (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<int64_t>(op2))
        {
            result = (std::get<uint64_t>(op1) - std::get<int64_t>(op2));
        }
        else if (std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2))
        {
            result = (std::get<int64_t>(op1) - std::get<int64_t>(op2));
        }
        else if (std::holds_alternative<int64_t>(op1) && std::holds_alternative<uint64_t>(op2))
        {
            result = (std::get<int64_t>(op1) - std::get<uint64_t>(op2));
        }
        else
        {
            assertx("SPIRV simulator: Could not find valid parameter type combination for Op_ISub");
        }

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type for Op_ISub, must be vector or int");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_LogicalNot(const Instruction& instruction)
{
    /*
    OpLogicalNot

    Result is true if Operand is false. Result is false if Operand is true.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand must be the same as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpLogicalNot);

    uint32_t type_id    = instruction.words[1];
    uint32_t result_id  = instruction.words[2];
    uint32_t operand_id = instruction.words[3];

    Type         type    = GetTypeByTypeId(type_id);
    const Value& operand = GetValue(operand_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand),
                "SPIRV simulator: Invalid value type, must be vector when using vector type");

        auto vec = std::get<std::shared_ptr<VectorV>>(operand);

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<uint64_t>(vec->elems[i]),
                    "SPIRV simulator: Non-boolean type found in vector operand");
            result_vec->elems.push_back((uint64_t)!(std::get<uint64_t>(vec->elems[i])));
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        Value result;

        assertm(std::holds_alternative<uint64_t>(operand), "SPIRV simulator: Non-boolean type found in operand");
        result = (uint64_t)!(std::get<uint64_t>(operand));

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or bool");
    }

    if (ValueIsArbitrary(operand_id))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_Capability(const Instruction& instruction)
{
    // This is a NOP in our design
    assert(instruction.opcode == spv::Op::OpCapability);
}

void SPIRVSimulator::Op_Extension(const Instruction& instruction)
{
    // This is a NOP in our design (at least for now)
    assert(instruction.opcode == spv::Op::OpExtension);
}

void SPIRVSimulator::Op_MemoryModel(const Instruction& instruction)
{
    // This is a NOP in our design
    assert(instruction.opcode == spv::Op::OpMemoryModel);
}

void SPIRVSimulator::Op_ExecutionMode(const Instruction& instruction)
{
    // We may need this later
    assert(instruction.opcode == spv::Op::OpExecutionMode);
}

void SPIRVSimulator::Op_Source(const Instruction& instruction)
{
    // This is a NOP in our design
    assert(instruction.opcode == spv::Op::OpSource);
}

void SPIRVSimulator::Op_SourceExtension(const Instruction& instruction)
{
    // This is a NOP in our design
    assert(instruction.opcode == spv::Op::OpSourceExtension);
}

void SPIRVSimulator::Op_Name(const Instruction& instruction)
{
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

    if (entry_points_.find(target_id) != entry_points_.end())
    {
        entry_points_[target_id] = label;
    }
}

void SPIRVSimulator::Op_MemberName(const Instruction& instruction)
{
    /*
    OpMemberName

    Assign a name string to a member of a structure type. This has no semantic impact and can safely be removed from a
    module.

    Type is the <id> from an OpTypeStruct instruction.

    Member is the number of the member to assign in the structure.
    The first member is member 0, the next is member 1, …​ Member is an unsigned 32-bit integer.

    Name is the string to assign to the member.
    */
    assert(instruction.opcode == spv::Op::OpMemberName);
    // This is a nop for now, can be used for debugging later but will slow things down
}

void SPIRVSimulator::Op_Decorate(const Instruction& instruction)
{
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

    uint32_t        target_id = instruction.words[1];
    spv::Decoration kind      = static_cast<spv::Decoration>(instruction.words[2]);

    std::vector<uint32_t> literals;
    for (uint32_t i = 3; i < instruction.word_count; ++i)
    {
        literals.push_back(instruction.words[i]);
    }

    DecorationInfo info{ kind, std::move(literals) };
    decorators_[target_id].emplace_back(std::move(info));
}

void SPIRVSimulator::Op_MemberDecorate(const Instruction& instruction)
{
    /*
    OpMemberDecorate

    Add a Decoration to a member of a structure type.
    Structure type is the <id> of a type from OpTypeStruct.
    Member is the number of the member to decorate in the type. The first member is member 0, the next is member 1,
    …​

    Note: See OpDecorate for creating groups of decorations for consumption by OpGroupMemberDecorate
    */
    assert(instruction.opcode == spv::Op::OpMemberDecorate);

    uint32_t        structure_type_id = instruction.words[1];
    uint32_t        member_literal    = instruction.words[2];
    spv::Decoration kind              = static_cast<spv::Decoration>(instruction.words[3]);

    std::vector<uint32_t> literals;
    for (uint32_t i = 4; i < instruction.word_count; ++i)
    {
        literals.push_back(instruction.words[i]);
    }

    DecorationInfo info{ kind, std::move(literals) };
    struct_decorators_[structure_type_id][member_literal].emplace_back(std::move(info));
}

void SPIRVSimulator::Op_ArrayLength(const Instruction& instruction)
{
    /*
    OpArrayLength

    Length of a run-time array.
    Result Type must be an OpTypeInt with 32-bit Width and 0 Signedness.

    Structure must be a logical pointer to an OpTypeStruct whose last member is a run-time array.
    Array member is an unsigned 32-bit integer index of the last member of the structure that Structure points to.
    That member’s type must be from OpTypeRuntimeArray.
    */
    assert(instruction.opcode == spv::Op::OpArrayLength);

    uint32_t type_id              = instruction.words[1];
    uint32_t result_id            = instruction.words[2];
    uint32_t structure_pointer_id = instruction.words[3];
    uint32_t literal_array_member = instruction.words[4];

    const Value& structure_pointer_val = GetValue(structure_pointer_id);
    assertm(std::holds_alternative<PointerV>(structure_pointer_val),
            "SPIRV simulator: OpArrayLength called on non-pointer type");

    PointerV pointer = std::get<PointerV>(structure_pointer_val);

    // Add the array index to the indirection path
    pointer.idx_path.push_back(literal_array_member);

    // Must (and should) be present for any pointer to buffers containing runtime arrays
    uint64_t array_pointer = pointer.pointer_handle;
    size_t   array_offset  = GetPointerOffset(pointer);

    if (array_pointer)
    {
        if (input_data_.rt_array_lengths.find(array_pointer) != input_data_.rt_array_lengths.end())
        {
            if (input_data_.rt_array_lengths[array_pointer].find(array_offset) !=
                input_data_.rt_array_lengths[array_pointer].end())
            {
                SetValue(result_id, (uint64_t)input_data_.rt_array_lengths[array_pointer][array_offset]);
            }
            else
            {
                if (verbose_)
                {
                    std::cout << "SPIRV simulator: WARNING: Op_ArrayLength called on pointer with no input size set, "
                                 "the user must provide this for correct behaviour"
                              << std::endl;
                    std::cout << "SPIRV simulator: Pointer:" << array_pointer << ", offset: " << array_offset
                              << std::endl;
                }

                SetValue(result_id, (uint64_t)1);
            }
        }
        else
        {
            if (verbose_)
            {
                std::cout << "SPIRV simulator: WARNING: Op_ArrayLength called on pointer with no input size set for "
                             "the given offset, the user must provide this for correct behaviour"
                          << std::endl;
                std::cout << "SPIRV simulator: Pointer:" << array_pointer << ", offset: " << array_offset << std::endl;
            }

            SetValue(result_id, (uint64_t)1);
        }
    }
    else
    {
        if (verbose_)
        {
            std::cout << "SPIRV simulator: WARNING: Op_ArrayLength called on pointer with no raw_pointer value set"
                      << std::endl;
            std::cout << "SPIRV simulator: Pointer:" << array_pointer << ", offset: " << array_offset << std::endl;
        }

        SetValue(result_id, (uint64_t)1);
    }
}

void SPIRVSimulator::Op_SpecConstant(const Instruction& instruction)
{
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
    assertm(HasDecorator(result_id, spv::Decoration::DecorationSpecId),
            "SPIRV simulator: Op_SpecConstant type is not decorated with SpecId");

    Op_Constant(instruction);
}

void SPIRVSimulator::Op_SpecConstantFalse(const Instruction& instruction)
{
    /*
    OpSpecConstantFalse

    Declare a Boolean-type scalar specialization constant with a default value of false.

    This instruction can be specialized to become either an OpConstantTrue or OpConstantFalse instruction.

    Result Type must be the scalar Boolean type.

    See Specialization.
    */
    assert(instruction.opcode == spv::Op::OpSpecConstantFalse);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    assertm(HasDecorator(result_id, spv::Decoration::DecorationSpecId),
            "SPIRV simulator: Op_SpecConstantFalse type is not decorated with SpecId");

    uint32_t spec_id = GetDecoratorLiteral(result_id, spv::Decoration::DecorationSpecId);
    if (input_data_.specialization_constant_offsets.find(spec_id) != input_data_.specialization_constant_offsets.end())
    {
        assertx("SPIRV simulator: Specialized Op_SpecConstantFalse branch not implemented yet, extract the instruction "
                "and execute");
    }
    else
    {
        SetValue(result_id, (uint64_t)0);
    }
}

void SPIRVSimulator::Op_SpecConstantTrue(const Instruction& instruction)
{
    /*
    OpSpecConstantTrue

    Declare a Boolean-type scalar specialization constant with a default value of true.

    This instruction can be specialized to become either an OpConstantTrue or OpConstantFalse instruction.

    Result Type must be the scalar Boolean type.

    See Specialization.
    */
    assert(instruction.opcode == spv::Op::OpSpecConstantTrue);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    assertm(HasDecorator(result_id, spv::Decoration::DecorationSpecId),
            "SPIRV simulator: Op_SpecConstantFalse type is not decorated with SpecId");

    uint32_t spec_id = GetDecoratorLiteral(result_id, spv::Decoration::DecorationSpecId);
    if (input_data_.specialization_constant_offsets.find(spec_id) != input_data_.specialization_constant_offsets.end())
    {
        assertx("SPIRV simulator: Specialized Op_SpecConstantTrue branch not implemented yet, extract the instruction "
                "and execute");
    }
    else
    {
        SetValue(result_id, (uint64_t)1);
    }
}

void SPIRVSimulator::Op_SpecConstantOp(const Instruction& instruction)
{
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
    if (spec_instructions_.find(result_id) == spec_instructions_.end())
    {
        uint32_t type_id = instruction.words[1];
        uint32_t opcode  = instruction.words[3];

        auto& spec_instr_words = spec_instr_words_[result_id];

        Instruction spec_instruction;
        spec_instruction.opcode     = (spv::Op)opcode;
        spec_instruction.word_count = instruction.words.size() - 1;

        uint32_t header_word = (spec_instruction.word_count << kWordCountShift) | spec_instruction.opcode;
        spec_instr_words.push_back(header_word);
        spec_instr_words.push_back(type_id);
        spec_instr_words.push_back(result_id);

        for (uint32_t operand_index = 4; operand_index < instruction.word_count; ++operand_index)
        {
            spec_instr_words.push_back(instruction.words[operand_index]);
        }

        spec_instruction.words        = std::span<const uint32_t>{ spec_instr_words.data(), spec_instr_words.size() };
        spec_instructions_[result_id] = spec_instruction;
    }

    if (verbose_)
    {
        PrintInstruction(spec_instructions_[result_id]);
    }

    ExecuteInstruction(spec_instructions_[result_id]);
}

void SPIRVSimulator::Op_SpecConstantComposite(const Instruction& instruction)
{
    /*
    OpSpecConstantComposite

    Declare a new composite specialization constant.
    Result Type must be a composite type, whose top-level members/elements/components/columns have the
    same type as the types of the Constituents. The ordering must be the same between the top-level types in Result Type
    and the Constituents. Constituents become members of a structure, or elements of an array, or components of a
    vector, or columns of a matrix. There must be exactly one Constituent for each top-level
    member/element/component/column of the result. The Constituents must appear in the order needed by the definition of
    the type of the result. The Constituents must be the <id> of other specialization constants, constant declarations,
    or an OpUndef. This instruction will be specialized to an OpConstantComposite instruction.

    See Specialization.
    */
    assert(instruction.opcode == spv::Op::OpSpecConstantComposite);
    Op_ConstantComposite(instruction);
}

void SPIRVSimulator::Op_UGreaterThanEqual(const Instruction& instruction)
{
    /*
    OpUGreaterThanEqual

    Unsigned-integer comparison if Operand 1 is greater than or equal to Operand 2.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type. They must have the same component
    width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpUGreaterThanEqual);

    uint32_t type_id     = instruction.words[1];
    uint32_t result_id   = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    Type         type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type in Op_UGreaterThanEqual, but they are not, illegal "
                "input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length in Op_UGreaterThanEqual");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<uint64_t>(vec1->elems[i]) &&
                        std::holds_alternative<uint64_t>(vec1->elems[i]),
                    "SPIRV simulator: Found non-unsigned integer operand in Op_UGreaterThanEqual vector operands");

            Value elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) >= std::get<uint64_t>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        Value result = (uint64_t)(std::get<uint64_t>(val_op1) >= std::get<uint64_t>(val_op2));
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type in Op_UGreaterThanEqual, must be vector or bool");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_Phi(const Instruction& instruction)
{
    /*

    OpPhi

    The SSA phi function.
    The result is selected based on control flow: If control reached the current block from Parent i, Result Id gets
    the value that Variable i had at the end of Parent i.

    Result Type can be any type except OpTypeVoid.

    Operands are a sequence of pairs: (Variable 1, Parent 1 block), (Variable 2, Parent 2 block), …​
    Each Parent i block is the label of an immediate predecessor in the CFG of the current block.
    There must be exactly one Parent i for each parent block of the current block in the CFG.
    If Parent i is reachable in the CFG and Variable i is defined in a block, that defining block must dominate Parent
    i. All Variables must have a type matching Result Type.

    Within a block, this instruction must appear before all non-OpPhi instructions (except for OpLine and OpNoLine,
    which can be mixed with OpPhi).
    */
    assert(instruction.opcode == spv::Op::OpPhi);

    // uint32_t type_id = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    for (uint32_t operand_index = 3; operand_index < instruction.word_count; operand_index += 2)
    {
        uint32_t variable_id = instruction.words[operand_index];
        uint64_t block_id    = instruction.words[operand_index + 1];

        if (block_id == prev_block_id_)
        {
            SetValue(result_id, GetValue(variable_id));

            if (ValueIsArbitrary(variable_id))
            {
                SetIsArbitrary(result_id);
            }
            return;
        }
    }

    assertx("SPIRV simulator: Op_Phi faield to find a valid source block ID, something is broken in the control flow "
            "handling.");
}

void SPIRVSimulator::Op_ConvertUToF(const Instruction& instruction)
{
    /*
    OpConvertUToF

    Convert value numerically from unsigned integer to floating point.
    Result Type must be a scalar or vector of floating-point type.
    Unsigned Value must be a scalar or vector of integer type. It must have the same number of components as Result
    Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpConvertUToF);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t value_id  = instruction.words[3];

    const Type& type = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op = GetValue(value_id);

        assertm(
            std::holds_alternative<std::shared_ptr<VectorV>>(val_op),
            "SPIRV simulator: Operand set to be vector type in OpConvertUToF, but it is not, illegal input parameters");

        auto vec = std::get<std::shared_ptr<VectorV>>(val_op);

        assertm(vec->elems.size() == type.vector.elem_count,
                "SPIRV simulator: Operands are vector type but not of valid length in OpConvertUToF");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<uint64_t>(vec->elems[i]),
                    "SPIRV simulator: Found non-unsigned integer operand in OpConvertUToF vector operands");

            Value elem_result = (double)std::get<uint64_t>(vec->elems[i]);

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Float)
    {
        const Value& op = GetValue(value_id);

        assertm(std::holds_alternative<uint64_t>(op),
                "SPIRV simulator: Found non-unsigned integer operand in OpConvertUToF");

        Value result = (double)std::get<uint64_t>(op);
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid return type in OpConvertUToF, must be vector or float");
    }

    if (ValueIsArbitrary(instruction.words[3]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_ConvertSToF(const Instruction& instruction)
{
    /*
    OpConvertSToF

    Convert value numerically from signed integer to floating point.
    Result Type must be a scalar or vector of floating-point type.
    Signed Value must be a scalar or vector of integer type.
    It must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpConvertSToF);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t value_id  = instruction.words[3];

    const Type& type = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op = GetValue(value_id);
        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op),
                "SPIRV simulator: Operand set to be vector type in Op_ConvertSToF, but it is not, illegal input "
                "parameters");

        auto vec = std::get<std::shared_ptr<VectorV>>(val_op);

        assertm(vec->elems.size() == type.vector.elem_count,
                "SPIRV simulator: Operands are vector type but not of valid length in Op_ConvertSToF");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<int64_t>(vec->elems[i]),
                    "SPIRV simulator: Found non-signed integer operand in Op_ConvertSToF vector operands");

            Value elem_result = (double)std::get<int64_t>(vec->elems[i]);

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Float)
    {
        const Value& op = GetValue(value_id);

        assertm(std::holds_alternative<int64_t>(op),
                "SPIRV simulator: Found non-signed integer operand in Op_ConvertSToF");

        Value result = (double)std::get<int64_t>(op);
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type in Op_ConvertSToF, must be vector or float");
    }

    if (ValueIsArbitrary(instruction.words[3]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_FDiv(const Instruction& instruction)
{
    /*
    OpFDiv

    Floating-point division of Operand 1 divided by Operand 2.

    Result Type must be a scalar or vector of floating-point type.
    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFDiv);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm(
            (std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
             std::holds_alternative<std::shared_ptr<VectorV>>(val_op2)),
            "SPIRV simulator: Operands set to be vector type in Op_FDiv, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length in Op_FDiv");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            Value elem_result;

            assertm(std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]),
                    "SPIRV simulator: Found non-floating point operand in Op_FDiv vector operands");

            elem_result = std::get<double>(vec1->elems[i]) / std::get<double>(vec2->elems[i]);

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Float)
    {
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        assertm(std::holds_alternative<double>(op1) && std::holds_alternative<double>(op2),
                "SPIRV simulator: Found non-floating point operand in Op_FDiv");

        result = std::get<double>(op1) / std::get<double>(op2);

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type for Op_FDiv, must be vector or float");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_FSub(const Instruction& instruction)
{
    /*
    OpFSub

    Floating-point subtraction of Operand 2 from Operand 1.
    Result Type must be a scalar or vector of floating-point type.
    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFSub);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm(
            std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
            "SPIRV simulator: Operands set to be vector type in Op_FSub, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length in Op_FSub");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            Value elem_result;

            assertm(std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]),
                    "SPIRV simulator: Found non-floating point operand in Op_FSub vector operands");

            elem_result = std::get<double>(vec1->elems[i]) - std::get<double>(vec2->elems[i]);

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Float)
    {
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        assertm(std::holds_alternative<double>(op1) && std::holds_alternative<double>(op2),
                "SPIRV simulator: Found non-floating point operand in Op_FSub");

        result = std::get<double>(op1) - std::get<double>(op2);

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type for Op_FSub, must be vector or float");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_VectorTimesScalar(const Instruction& instruction)
{
    /*
    OpVectorTimesScalar

    Scale a floating-point vector.
    Result Type must be a vector of floating-point type.
    The type of Vector must be the same as Result Type. Each component of Vector is multiplied by Scalar.

    Scalar must have the same type as the Component Type in Result Type.
    */
    assert(instruction.opcode == spv::Op::OpVectorTimesScalar);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t vector_id = instruction.words[3];
    uint32_t scalar_id = instruction.words[4];

    const Type& type = GetTypeByTypeId(type_id);

    Value result     = std::make_shared<VectorV>();
    auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

    const Value& vec_operand = GetValue(vector_id);
    assertm(std::holds_alternative<std::shared_ptr<VectorV>>(vec_operand),
            "SPIRV simulator: Found non-vector operand in Op_VectorTimesScalar");
    auto vec = std::get<std::shared_ptr<VectorV>>(vec_operand);

    const Value& scalar_operand = GetValue(scalar_id);
    assertm(std::holds_alternative<double>(scalar_operand),
            "SPIRV simulator: Found non-floating point operand in Op_VectorTimesScalar");
    double scalar_value = std::get<double>(scalar_operand);

    for (uint32_t i = 0; i < type.vector.elem_count; ++i)
    {
        Value elem_result;

        assertm(std::holds_alternative<double>(vec->elems[i]),
                "SPIRV simulator: Found non-floating point operand in Op_VectorTimesScalar vector operands");

        elem_result = std::get<double>(vec->elems[i]) * scalar_value;

        result_vec->elems.push_back(elem_result);
    }

    SetValue(result_id, result);

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_SLessThan(const Instruction& instruction)
{
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
    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type in Op_SLessThan, but they are not, illegal input "
                "parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length in Op_SLessThan");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            Value elem_result;

            assertm(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]),
                    "SPIRV simulator: Found non-signed integer operand in Op_SLessThan vector operands");

            elem_result = (uint64_t)(std::get<int64_t>(vec1->elems[i]) < std::get<int64_t>(vec2->elems[i]));

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        assertm(std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2),
                "SPIRV simulator: Found non-signed integer operand in Op_SLessThan");

        result = (uint64_t)(std::get<int64_t>(op1) < std::get<int64_t>(op2));

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type for Op_SLessThan, must be vector or bool");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_Dot(const Instruction& instruction)
{
    /*
    OpDot

    Dot product of Vector 1 and Vector 2.
    Result Type must be a floating-point type scalar.
    Vector 1 and Vector 2 must be vectors of the same type, and their component type must be Result Type.
    */
    assert(instruction.opcode == spv::Op::OpDot);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Float)
    {
        double result = 0.0;

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands not of vector type in Op_Dot");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm(vec1->elems.size() == vec2->elems.size(),
                "SPIRV simulator: Operands not of equal/correct length in Op_Dot");

        for (uint32_t i = 0; i < vec1->elems.size(); ++i)
        {
            assertm(std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]),
                    "SPIRV simulator: Found non-floating point operand in Op_Dot vector operands");

            result += std::get<double>(vec1->elems[i]) * std::get<double>(vec2->elems[i]);
        }

        Value val_result = result;
        SetValue(result_id, val_result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type for Op_Dot, must be float");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_FOrdGreaterThan(const Instruction& instruction)
{
    /*
    OpFOrdGreaterThan

    Floating-point comparison if operands are ordered and Operand 1 is greater than Operand 2.
    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of floating-point type.
    They must have the same type, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFOrdGreaterThan);

    uint32_t type_id     = instruction.words[1];
    uint32_t result_id   = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    Type         type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type in Op_FOrdGreaterThan, but they are not, illegal "
                "input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length in Op_UGreaterThanEqual");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]),
                    "SPIRV simulator: Found non-floating point operand in Op_FOrdGreaterThan vector operands");

            Value elem_result = (uint64_t)(std::get<double>(vec1->elems[i]) > std::get<double>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        Value result = (uint64_t)(std::get<double>(val_op1) > std::get<double>(val_op2));
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type in Op_FOrdGreaterThan, must be vector or float");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_FOrdGreaterThanEqual(const Instruction& instruction)
{
    /*
    OpFOrdGreaterThanEqual

    Floating-point comparison if operands are ordered and Operand 1 is greater than or equal to Operand 2.

    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of floating-point type.
    They must have the same type, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFOrdGreaterThanEqual);

    uint32_t type_id     = instruction.words[1];
    uint32_t result_id   = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    Type         type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type in Op_FOrdGreaterThanEqual, but they are not, illegal "
                "input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length in Op_UGreaterThanEqual");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]),
                    "SPIRV simulator: Found non-floating point operand in Op_FOrdGreaterThanEqual vector operands");

            Value elem_result = (uint64_t)(std::get<double>(vec1->elems[i]) >= std::get<double>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        Value result = (uint64_t)(std::get<double>(val_op1) >= std::get<double>(val_op2));
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type in Op_FOrdGreaterThanEqual, must be vector or float");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_FOrdEqual(const Instruction& instruction)
{
    /*
    OpFOrdEqual

    Floating-point comparison for being ordered and equal.

    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of floating-point type.
    They must have the same type, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFOrdEqual);

    uint32_t type_id     = instruction.words[1];
    uint32_t result_id   = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    Type         type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type in Op_FOrdEqual, but they are not, illegal "
                "input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length in Op_UGreaterThanEqual");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]),
                    "SPIRV simulator: Found non-floating point operand in Op_FOrdEqual vector operands");

            Value elem_result = (uint64_t)(std::get<double>(vec1->elems[i]) == std::get<double>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        Value result = (uint64_t)(std::get<double>(val_op1) == std::get<double>(val_op2));
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type in Op_FOrdEqual, must be vector or float");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_FOrdNotEqual(const Instruction& instruction)
{
    /*
    OpFOrdNotEqual

    Floating-point comparison for being ordered and not equal.

    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of floating-point type.
    They must have the same type, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFOrdNotEqual);

    uint32_t type_id     = instruction.words[1];
    uint32_t result_id   = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    Type         type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type in Op_FOrdNotEqual, but they are not, illegal "
                "input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length in Op_UGreaterThanEqual");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]),
                    "SPIRV simulator: Found non-floating point operand in Op_FOrdNotEqual vector operands");

            Value elem_result = (uint64_t)(std::get<double>(vec1->elems[i]) != std::get<double>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        Value result = (uint64_t)(std::get<double>(val_op1) != std::get<double>(val_op2));
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type in Op_FOrdNotEqual, must be vector or float");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_CompositeExtract(const Instruction& instruction)
{
    /*
    OpCompositeExtract

    Extract a part of a composite object.
    Result Type must be the type of object selected by the last provided index. The instruction result is the extracted
    object. Composite is the composite to extract from.

    Indexes walk the type hierarchy, potentially down to component granularity, to select the part to extract.
    All indexes must be in bounds. All composite constituents use zero-based numbering, as described by their
    OpType…​ instruction. Each index is an unsigned 32-bit integer.
    */
    assert(instruction.opcode == spv::Op::OpCompositeExtract);

    // uint32_t type_id = instruction.words[1];
    uint32_t result_id    = instruction.words[2];
    uint32_t composite_id = instruction.words[3];

    const Value* current_composite = &(GetValue(composite_id));
    for (uint32_t i = 4; i < instruction.word_count; ++i)
    {
        uint32_t literal_index = instruction.words[i];

        if (std::holds_alternative<std::shared_ptr<AggregateV>>(*current_composite))
        {
            auto agg = std::get<std::shared_ptr<AggregateV>>(*current_composite);

            assertm(literal_index < agg->elems.size(), "SPIRV simulator: Aggregate index OOB");

            current_composite = &agg->elems[literal_index];
        }
        else if (std::holds_alternative<std::shared_ptr<VectorV>>(*current_composite))
        {
            auto vec = std::get<std::shared_ptr<VectorV>>(*current_composite);

            assertm(literal_index < vec->elems.size(), "SPIRV simulator: Vector index OOB");

            current_composite = &vec->elems[literal_index];
        }
        else if (std::holds_alternative<std::shared_ptr<MatrixV>>(*current_composite))
        {
            auto matrix = std::get<std::shared_ptr<MatrixV>>(*current_composite);

            assertm(literal_index < matrix->cols.size(), "SPIRV simulator: Matrix index OOB");

            current_composite = &matrix->cols[literal_index];
        }
        else
        {
            assertx("SPIRV simulator: Pointer dereference into non-composite object");
        }
    }

    SetValue(result_id, *current_composite);

    if (ValueIsArbitrary(instruction.words[3]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_Bitcast(const Instruction& instruction)
{
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

    Let L be the type, either Result Type or Operand’s type, that has the larger number of components. Let S be the
    other type, with the smaller number of components. The number of components in L must be an integer multiple of the
    number of components in S. The first component (that is, the only or lowest-numbered component) of S maps to the
    first components of L, and so on, up to the last component of S mapping to the last components of L. Within this
    mapping, any single component of S (mapping to multiple components of L) maps its lower-ordered bits to the
    lower-numbered components of L.
    */
    assert(instruction.opcode == spv::Op::OpBitcast);

    uint32_t type_id    = instruction.words[1];
    uint32_t result_id  = instruction.words[2];
    uint32_t operand_id = instruction.words[3];

    const Value& operand      = GetValue(operand_id);
    Type         operand_type = GetTypeByResultId(operand_id);

    const Type& type = GetTypeByTypeId(type_id);

    //
    // First, we extract all the data from the operands into a vector
    //
    std::vector<std::byte> bytes;
    if (std::holds_alternative<std::shared_ptr<VectorV>>(operand))
    {
        const Type&              elem_type = GetTypeByTypeId(operand_type.vector.elem_type_id);
        std::shared_ptr<VectorV> vec       = std::get<std::shared_ptr<VectorV>>(operand);
        for (const Value& element : vec->elems)
        {
            if (std::holds_alternative<double>(element))
            {
                double value = std::get<double>(element);
                extract_bytes<double>(bytes, value, elem_type.scalar.width);
            }
            else if (std::holds_alternative<uint64_t>(element))
            {
                uint64_t value = std::get<uint64_t>(element);
                extract_bytes<uint64_t>(bytes, value, elem_type.scalar.width);
            }
            else if (std::holds_alternative<int64_t>(element))
            {
                int64_t value = std::get<int64_t>(element);
                extract_bytes<int64_t>(bytes, value, elem_type.scalar.width);
            }
            else
            {
                assertx("SPIRV simulator: invalid operand element type in Op_Bitcast, must be numeric");
            }
        }
    }
    else if (std::holds_alternative<double>(operand))
    {
        double value = std::get<double>(operand);
        extract_bytes<double>(bytes, value, operand_type.scalar.width);
    }
    else if (std::holds_alternative<uint64_t>(operand))
    {
        uint64_t value = std::get<uint64_t>(operand);
        extract_bytes<uint64_t>(bytes, value, operand_type.scalar.width);
    }
    else if (std::holds_alternative<int64_t>(operand))
    {
        int64_t value = std::get<int64_t>(operand);
        extract_bytes<int64_t>(bytes, value, operand_type.scalar.width);
    }
    else if (std::holds_alternative<PointerV>(operand))
    {
        // Take the easy out if its just pointer to pointer conversion
        if (type.kind == Type::Kind::Pointer)
        {
            SetValue(result_id, operand);

            // TODO: Compare pointer with candidates here and track

            return;
        }
        // We currently dont handle this, we could do it by storing the pointer in a
        // special container and storing a index into that container in the result here
        assertx("SPIRV simulator: Pointer to non-pointer Op_Bitcast detected, must add support for this!");
    }
    else
    {
        assertx("SPIRV simulator: invalid operand type in Op_Bitcast, must be vector or numeric");
    }

    //
    // Then we map this memory to the result value
    //
    Value result;
    if (type.kind == Type::Kind::Vector)
    {
        const Type&              elem_type       = GetTypeByTypeId(type.vector.elem_type_id);
        uint32_t                 elem_size_bytes = elem_type.scalar.width / 8;
        std::shared_ptr<VectorV> vec             = std::make_shared<VectorV>();
        uint32_t                 current_byte    = 0;

        for (unsigned i = 0; i < type.vector.elem_count; ++i)
        {
            if (elem_type.kind == Type::Kind::Float)
            {
                double value = 0.0;
                std::memcpy(&value, &(bytes[current_byte]), elem_size_bytes);
                vec->elems.push_back(value);
            }
            else if ((elem_type.kind == Type::Kind::Int) && !elem_type.scalar.is_signed)
            {
                uint64_t value = 0;
                std::memcpy(&value, &(bytes[current_byte]), elem_size_bytes);
                vec->elems.push_back(value);
            }
            else if ((elem_type.kind == Type::Kind::Int) && elem_type.scalar.is_signed)
            {
                int64_t value = 0;
                std::memcpy(&value, &(bytes[current_byte]), elem_size_bytes);
                vec->elems.push_back(value);
            }
            else
            {
                assertx("SPIRV simulator: invalid result element type in Op_Bitcast, must be numeric");
            }

            current_byte += elem_size_bytes;
        }

        result = vec;
    }
    else if (type.kind == Type::Kind::Float)
    {
        double value = 0.0;
        std::memcpy(&value, bytes.data(), type.scalar.width / 8);

        result = value;
    }
    else if ((type.kind == Type::Kind::Int) && !type.scalar.is_signed)
    {
        uint64_t value = 0;
        std::memcpy(&value, bytes.data(), type.scalar.width / 8);

        result = value;
    }
    else if ((type.kind == Type::Kind::Int) && type.scalar.is_signed)
    {
        int64_t value = 0;
        std::memcpy(reinterpret_cast<std::byte*>(&value), bytes.data(), type.scalar.width / 8);

        result = value;
    }
    else if (type.kind == Type::Kind::Pointer)
    {
        // This is one of the main cases we want to detect, a non-pointer type is cast to a pointer
        // If the storage class is PhysicalStorageBuffer we map it to an external address handle
        // In turn, this can be used in combination with inputs to read from the pbuffer

        // This is unhandled (and probably illegal?)
        assertm(type.pointer.storage_class == spv::StorageClass::StorageClassPhysicalStorageBuffer,
                "SPIRV simulator: Attempt to Op_Bitcast to a non PhysicalStorageBuffer storage class object");

        uint64_t pointer_value = 0;
        std::memcpy(&pointer_value, bytes.data(), sizeof(uint64_t));

        const std::byte* remapped_pointer = nullptr;

        for (const auto& map_entry : input_data_.physical_address_buffers)
        {
            uint64_t buffer_address = map_entry.first;
            size_t   buffer_size    = map_entry.second.first;

            const std::byte* buffer_data = static_cast<std::byte*>(map_entry.second.second);

            if ((pointer_value >= buffer_address) && (pointer_value < (buffer_address + buffer_size)))
            {
                remapped_pointer = &(buffer_data[buffer_address - pointer_value]);
                break;
            }
        }

        PointerV new_pointer{
            bit_cast<uint64_t>(remapped_pointer), type_id, result_id, type.pointer.storage_class, {}
        };
        physical_address_pointers_.push_back(new_pointer);
        result = new_pointer;

        // Here we need to find the source of the values that eventually became the pointer above
        // so that any tool using the simulator can extract and deal with them.
        PhysicalAddressData pointer_data;
        pointer_data.bit_components    = FindDataSourcesFromResultID(operand_id);
        pointer_data.raw_pointer_value = pointer_value;
        physical_address_pointer_source_data_.push_back(std::move(pointer_data));
    }
    else
    {
        assertx("SPIRV simulator: invalid result type in Op_Bitcast, must be vector, pointer or numeric");
    }

    SetValue(result_id, result);

    if (ValueIsArbitrary(instruction.words[3]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_IMul(const Instruction& instruction)
{
    /*
    OpIMul

    Integer multiplication of Operand 1 and Operand 2.
    Result Type must be a scalar or vector of integer type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same number of components as Result Type. They must have the same component width as Result Type.

    The resulting value equals the low-order N bits of the correct result R, where N is the component width and R is
    computed with enough precision to avoid overflow and underflow.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpIMul);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands not of vector type in Op_IMul");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands not of equal/correct length in Op_IMul");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            Value elem_result;

            if (std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i]))
            {
                elem_result = (std::get<uint64_t>(vec1->elems[i]) * std::get<uint64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]))
            {
                elem_result = (std::get<int64_t>(vec1->elems[i]) * std::get<int64_t>(vec2->elems[i]));
            }
            else
            {
                assertx("SPIRV simulator: Could not find valid parameter type combination for Op_IMul vector operand");
            }

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Int)
    {
        Value        result;
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        if (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2))
        {
            result = (std::get<uint64_t>(op1) * std::get<uint64_t>(op2));
        }
        else if (std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2))
        {
            result = (std::get<int64_t>(op1) * std::get<int64_t>(op2));
        }
        else
        {
            assertx("SPIRV simulator: Could not find valid parameter type combination for Op_IMul");
        }

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type for Op_IMul, must be vector or integer type");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_ConvertUToPtr(const Instruction& instruction)
{
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

    uint32_t type_id    = instruction.words[1];
    uint32_t result_id  = instruction.words[2];
    uint32_t integer_id = instruction.words[3];

    Type         type    = GetTypeByTypeId(type_id);
    const Value& operand = GetValue(integer_id);

    // This is unhandled (and probably illegal?)
    assertm(type.pointer.storage_class == spv::StorageClass::StorageClassPhysicalStorageBuffer,
            "SPIRV simulator: Attempt to Op_ConvertUToPtr to a non PhysicalStorageBuffer storage class object");

    uint64_t pointer_value = std::get<uint64_t>(operand);

    const std::byte* remapped_pointer = nullptr;

    for (const auto& map_entry : input_data_.physical_address_buffers)
    {
        uint64_t buffer_address = map_entry.first;
        size_t   buffer_size    = map_entry.second.first;

        const std::byte* buffer_data = static_cast<std::byte*>(map_entry.second.second);

        if ((pointer_value >= buffer_address) && (pointer_value < (buffer_address + buffer_size)))
        {
            remapped_pointer = &(buffer_data[buffer_address - pointer_value]);
            break;
        }
    }

    PointerV new_pointer{ pointer_value, type_id, result_id, type.pointer.storage_class, {} };
    physical_address_pointers_.push_back(new_pointer);
    SetValue(result_id, new_pointer);

    // TODO: Compare pointer with candidates here and track

    // Here we need to find the source of the values that eventually became the pointer above
    // so that any tool using the simulator can extract and deal with them.
    PhysicalAddressData pointer_data;
    pointer_data.bit_components    = FindDataSourcesFromResultID(integer_id);
    pointer_data.raw_pointer_value = pointer_value;
    physical_address_pointer_source_data_.push_back(std::move(pointer_data));

    if (ValueIsArbitrary(instruction.words[3]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_UDiv(const Instruction& instruction)
{
    /*
    OpUDiv

    Unsigned-integer division of Operand 1 divided by Operand 2.
    Result Type must be a scalar or vector of integer type, whose Signedness operand is 0.
    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component. Behavior is undefined if Operand 2 is 0.
    */
    assert(instruction.opcode == spv::Op::OpUDiv);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    Type         type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(instruction.words[3]);
    const Value& val_op2 = GetValue(instruction.words[4]);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            Value elem_result;

            // TODO: Operands dont have to be unsigned, deal with it and remove the asserts
            assertm(std::holds_alternative<uint64_t>(vec1->elems[i]) &&
                        std::holds_alternative<uint64_t>(vec2->elems[i]),
                    "SPIRV simulator: Found non-unsigned int operand vector operands");

            uint64_t op2 = std::get<uint64_t>(vec2->elems[i]);
            if (op2 == 0)
            {
                if (verbose_)
                {
                    std::cout << "SPIRV simulator: Divisor in OpUDiv is 0, this is undefined behaviour, setting to 1"
                              << std::endl;
                }

                op2 = 1;
            }

            elem_result = std::get<uint64_t>(vec1->elems[i]) / op2;

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Int)
    {
        // TODO: Operands dont have to be unsigned, deal with it and remove the asserts
        assertm(std::holds_alternative<uint64_t>(val_op1) && std::holds_alternative<uint64_t>(val_op2),
                "SPIRV simulator: Found non-unsigned int operand");

        uint64_t op2 = std::get<uint64_t>(val_op2);
        if (op2 == 0)
        {
            if (verbose_)
            {
                std::cout << "SPIRV simulator: Divisor in OpUDiv is 0, this is undefined behaviour, setting to 1"
                          << std::endl;
            }

            op2 = 1;
        }

        Value result = std::get<uint64_t>(val_op1) / op2;

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or unsigned-integer");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_UMod(const Instruction& instruction)
{
    /*
    OpUMod

    Unsigned modulo operation of Operand 1 modulo Operand 2.
    Result Type must be a scalar or vector of integer type, whose Signedness operand is 0.
    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component. Behavior is undefined if Operand 2 is 0.
    */
    assert(instruction.opcode == spv::Op::OpUMod);

    uint32_t type_id     = instruction.words[1];
    uint32_t result_id   = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    const Type& type = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(operand1_id);
        const Value& val_op2 = GetValue(operand2_id);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            Value elem_result;

            assertm(std::holds_alternative<uint64_t>(vec1->elems[i]) &&
                        std::holds_alternative<uint64_t>(vec2->elems[i]),
                    "SPIRV simulator: Found non-unsigned int operand in vector operands");

            uint64_t op2_val = std::get<uint64_t>(vec2->elems[i]);

            if (op2_val == 0)
            {
                op2_val = 1;

                if (!ValueIsArbitrary(operand2_id))
                {
                    std::cout
                        << "SPIRV simulator: WARNING: Second operand is 0 in Op_UMod, shader has undefined behaviour"
                        << std::endl;
                }
            }

            elem_result = std::get<uint64_t>(vec1->elems[i]) % op2_val;

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Int)
    {
        const Value& op1 = GetValue(operand1_id);
        const Value& op2 = GetValue(operand2_id);

        Value result;
        assertm(std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2),
                "SPIRV simulator: Found non-unsigned int operand");

        uint64_t op2_val = std::get<uint64_t>(op2);

        if (op2_val == 0)
        {
            op2_val = 1;

            if (!ValueIsArbitrary(operand2_id))
            {
                std::cout << "SPIRV simulator: WARNING: Second operand is 0 in Op_UMod, shader has undefined behaviour"
                          << std::endl;
            }
        }

        result = std::get<uint64_t>(op1) % op2_val;

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or unsigned-integer");
    }

    if (ValueIsArbitrary(operand1_id) || ValueIsArbitrary(operand2_id))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_ULessThan(const Instruction& instruction)
{
    /*
    OpULessThan

    Unsigned-integer comparison if Operand 1 is less than Operand 2.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type. They must have the same component
    width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpULessThan);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            Value elem_result;

            assertm(std::holds_alternative<uint64_t>(vec1->elems[i]) &&
                        std::holds_alternative<uint64_t>(vec2->elems[i]),
                    "SPIRV simulator: Found non-unsigned integer operand in vector operands");

            elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) < std::get<uint64_t>(vec2->elems[i]));

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        assertm(std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2),
                "SPIRV simulator: Found non-unsigned integer operand");

        result = (uint64_t)(std::get<uint64_t>(op1) < std::get<uint64_t>(op2));

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or bool");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_ConstantTrue(const Instruction& instruction)
{
    /*
    OpConstantTrue
    Declare a true Boolean-type scalar constant.
    Result Type must be the scalar Boolean type.
    */
    assert(instruction.opcode == spv::Op::OpConstantTrue);

    uint32_t    type_id   = instruction.words[1];
    uint32_t    result_id = instruction.words[2];
    const Type& type      = GetTypeByTypeId(type_id);

    assertm(type.kind == Type::Kind::BoolT, "SPIRV simulator: Constant type must be bool");

    Value result = (uint64_t)1;
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_ConstantFalse(const Instruction& instruction)
{
    /*
    OpConstantFalse
    Declare a false Boolean-type scalar constant.
    Result Type must be the scalar Boolean type.
    */
    assert(instruction.opcode == spv::Op::OpConstantFalse);

    uint32_t    type_id   = instruction.words[1];
    uint32_t    result_id = instruction.words[2];
    const Type& type      = GetTypeByTypeId(type_id);

    assertm(type.kind == Type::Kind::BoolT, "SPIRV simulator: Constant type must be bool");

    Value result = (uint64_t)0;
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_ConstantNull(const Instruction& instruction)
{
    /*
    OpConstantNull

    Declare a new null constant value.

    The null value is type dependent, defined as follows:
    - Scalar Boolean: false
    - Scalar integer: 0
    - Scalar floating point: +0.0 (all bits 0)
    - All other scalars: Abstract
    - Composites: Members are set recursively to the null constant according to the null value of their constituent
    types.

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

    uint32_t    type_id   = instruction.words[1];
    uint32_t    result_id = instruction.words[2];
    const Type& type      = GetTypeByTypeId(type_id);

    // TODO: This will crash for most pointers, we have to handle that case without MakeDefault
    assertm(type.kind != Type::Kind::Pointer,
            "SPIRV simulator: Op_ConstantNull for pointer types is currently not supported");

    SetValue(result_id, MakeDefault(type_id));
}

void SPIRVSimulator::Op_AtomicIAdd(const Instruction& instruction)
{
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

    uint32_t result_id  = instruction.words[2];
    uint32_t pointer_id = instruction.words[3];
    uint32_t value_id   = instruction.words[6];

    PointerV     pointer      = std::get<PointerV>(GetValue(pointer_id));
    const Value& pointee_val  = ReadPointer(pointer);
    const Value& source_value = GetValue(value_id);

    Value result;
    if (std::holds_alternative<uint64_t>(pointee_val) && std::holds_alternative<uint64_t>(source_value))
    {
        result = std::get<uint64_t>(pointee_val) + std::get<uint64_t>(source_value);
    }
    else if (std::holds_alternative<int64_t>(pointee_val) && std::holds_alternative<int64_t>(source_value))
    {
        result = std::get<int64_t>(pointee_val) + std::get<int64_t>(source_value);
    }
    else
    {
        assertx("SPIRV simulator: Invalid type match in Op_AtomicIAdd, must be same type scalar integers");
    }

    WritePointer(pointer, result);
    SetValue(result_id, pointee_val);
}

void SPIRVSimulator::Op_AtomicISub(const Instruction& instruction)
{
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

    uint32_t result_id  = instruction.words[2];
    uint32_t pointer_id = instruction.words[3];
    uint32_t value_id   = instruction.words[6];

    PointerV     pointer      = std::get<PointerV>(GetValue(pointer_id));
    const Value& pointee_val  = ReadPointer(pointer);
    const Value& source_value = GetValue(value_id);

    Value result;
    if (std::holds_alternative<uint64_t>(pointee_val) && std::holds_alternative<uint64_t>(source_value))
    {
        result = std::get<uint64_t>(pointee_val) - std::get<uint64_t>(source_value);
    }
    else if (std::holds_alternative<int64_t>(pointee_val) && std::holds_alternative<int64_t>(source_value))
    {
        result = std::get<int64_t>(pointee_val) - std::get<int64_t>(source_value);
    }
    else
    {
        assertx("SPIRV simulator: Invalid type match in Op_AtomicISub, must be same type scalar integers");
    }

    WritePointer(pointer, result);
    SetValue(result_id, pointee_val);
}

void SPIRVSimulator::Op_Select(const Instruction& instruction)
{
    /*
    OpSelect

    Select between two objects. Before version 1.4, results are only computed per component.
    Before version 1.4, Result Type must be a pointer, scalar, or vector.
    Starting with version 1.4, Result Type can additionally be a composite type other than a vector.
    The types of Object 1 and Object 2 must be the same as Result Type.
    Condition must be a scalar or vector of Boolean type.
    If Condition is a scalar and true, the result is Object 1. If Condition is a scalar and false, the result is
    Object 2.

    If Condition is a vector, Result Type must be a vector with the same number of components as
    Condition and the result is a mix of Object 1 and Object 2: If a component of Condition is true, the corresponding
    component in the result is taken from Object 1, otherwise it is taken from Object 2.
    */
    assert(instruction.opcode == spv::Op::OpSelect);

    uint32_t type_id      = instruction.words[1];
    uint32_t result_id    = instruction.words[2];
    uint32_t condition_id = instruction.words[3];
    uint32_t obj_1_id     = instruction.words[4];
    uint32_t obj_2_id     = instruction.words[5];

    const Type&  type          = GetTypeByTypeId(type_id);
    const Value& condition_val = GetValue(condition_id);
    const Value& val_op1       = GetValue(obj_1_id);
    const Value& val_op2       = GetValue(obj_2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(
            std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
            "SPIRV simulator: Operands set to be vector type in Op_Select, but they are not, illegal input parameters");
        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(condition_val),
                "SPIRV simulator: Condition operand set to be vector type in Op_Select, but is is not, illegal input "
                "parameters");

        auto vec1     = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2     = std::get<std::shared_ptr<VectorV>>(val_op2);
        auto cond_vec = std::get<std::shared_ptr<VectorV>>(condition_val);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == cond_vec->elems.size()) &&
                    (cond_vec->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length in Op_Select");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            Value elem_result;

            uint64_t cond_val = std::get<uint64_t>(cond_vec->elems[i]);

            if (cond_val)
            {
                result_vec->elems.push_back(vec1->elems[i]);

                if (ValueIsArbitrary(instruction.words[4]))
                {
                    SetIsArbitrary(result_id);
                }
            }
            else
            {
                result_vec->elems.push_back(vec2->elems[i]);

                if (ValueIsArbitrary(instruction.words[5]))
                {
                    SetIsArbitrary(result_id);
                }
            }
        }

        SetValue(result_id, result);
    }
    else
    {
        assertm(std::holds_alternative<uint64_t>(condition_val),
                "SPIRV simulator: Op_Select condition must be a bool or a vector of bools");
        uint64_t condition_int = std::get<uint64_t>(condition_val);

        if (condition_int)
        {
            SetValue(result_id, val_op1);

            if (ValueIsArbitrary(instruction.words[4]))
            {
                SetIsArbitrary(result_id);
            }
        }
        else
        {
            SetValue(result_id, val_op2);

            if (ValueIsArbitrary(instruction.words[5]))
            {
                SetIsArbitrary(result_id);
            }
        }
    }

    if (ValueIsArbitrary(instruction.words[3]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_IEqual(const Instruction& instruction)
{
    /*
    OpIEqual

    Integer comparison for equality.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.
    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpIEqual);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    const Type& type = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const Value& val_op1 = GetValue(instruction.words[3]);
        const Value& val_op2 = GetValue(instruction.words[4]);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands not of vector type in Op_IEqual");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands not of equal/correct length in Op_IEqual");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            Value elem_result;

            if (std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i]))
            {
                elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) == std::get<uint64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<uint64_t>(vec1->elems[i]) &&
                     std::holds_alternative<int64_t>(vec2->elems[i]))
            {
                elem_result =
                    (uint64_t)(std::get<uint64_t>(vec1->elems[i]) == (uint64_t)std::get<int64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]))
            {
                elem_result = (uint64_t)((uint64_t)std::get<int64_t>(vec1->elems[i]) ==
                                         (uint64_t)std::get<int64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<int64_t>(vec1->elems[i]) &&
                     std::holds_alternative<uint64_t>(vec2->elems[i]))
            {
                elem_result =
                    (uint64_t)((uint64_t)std::get<int64_t>(vec1->elems[i]) == std::get<uint64_t>(vec2->elems[i]));
            }
            else
            {
                assertx(
                    "SPIRV simulator: Could not find valid parameter type combination for Op_IEqual vector operand");
            }

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Int || type.kind == Type::Kind::BoolT)
    {
        const Value& op1 = GetValue(instruction.words[3]);
        const Value& op2 = GetValue(instruction.words[4]);

        Value result;
        if (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2))
        {
            result = (uint64_t)(std::get<uint64_t>(op1) == std::get<uint64_t>(op2));
        }
        else if (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<int64_t>(op2))
        {
            result = (uint64_t)(std::get<uint64_t>(op1) == (uint64_t)std::get<int64_t>(op2));
        }
        else if (std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2))
        {
            result = (uint64_t)((uint64_t)std::get<int64_t>(op1) == (uint64_t)std::get<int64_t>(op2));
        }
        else if (std::holds_alternative<int64_t>(op1) && std::holds_alternative<uint64_t>(op2))
        {
            result = (uint64_t)((uint64_t)std::get<int64_t>(op1) == std::get<uint64_t>(op2));
        }
        else
        {
            assertx("SPIRV simulator: Could not find valid parameter type combination for Op_IEqual");
        }

        SetValue(result_id, result);
    }
    else
    {
        std::cout << (uint32_t)type.kind << std::endl;
        assertx("SPIRV simulator: Invalid result type for Op_IEqual, must be vector or int");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_CompositeInsert(const Instruction& instruction)
{
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

    uint32_t result_id    = instruction.words[2];
    uint32_t obj_id       = instruction.words[3];
    uint32_t composite_id = instruction.words[4];

    const Value& source_composite = GetValue(composite_id);
    Value        composite_copy   = CopyValue(source_composite);

    Value* current_composite = &composite_copy;
    for (uint32_t i = 5; i < instruction.word_count; ++i)
    {
        uint32_t literal_index = instruction.words[i];

        if (std::holds_alternative<std::shared_ptr<AggregateV>>(*current_composite))
        {
            auto agg = std::get<std::shared_ptr<AggregateV>>(*current_composite);

            assertm(literal_index < agg->elems.size(), "SPIRV simulator: Aggregate index OOB");

            current_composite = &agg->elems[literal_index];
        }
        else if (std::holds_alternative<std::shared_ptr<VectorV>>(*current_composite))
        {
            auto vec = std::get<std::shared_ptr<VectorV>>(*current_composite);

            assertm(literal_index < vec->elems.size(), "SPIRV simulator: Vector index OOB");

            current_composite = &vec->elems[literal_index];
        }
        else if (std::holds_alternative<std::shared_ptr<MatrixV>>(*current_composite))
        {
            auto matrix = std::get<std::shared_ptr<MatrixV>>(*current_composite);

            assertm(literal_index < matrix->cols.size(), "SPIRV simulator: Matrix index OOB");

            current_composite = &matrix->cols[literal_index];
        }
        else
        {
            assertx("SPIRV simulator: Pointer dereference into non-composite object");
        }
    }

    const Value& source_object = GetValue(obj_id);
    *current_composite         = CopyValue(source_object);
    SetValue(result_id, composite_copy);

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_Transpose(const Instruction& instruction)
{
    /*

    OpTranspose

    Transpose a matrix.
    Result Type must be an OpTypeMatrix.
    Matrix must be an object of type OpTypeMatrix. The number of columns and the column size of Matrix must be the
    reverse of those in Result Type. The types of the scalar components in Matrix and Result Type must be the same.

    Matrix must have of type of OpTypeMatrix.
    */
    assert(instruction.opcode == spv::Op::OpTranspose);

    uint32_t    type_id   = instruction.words[1];
    uint32_t    result_id = instruction.words[2];
    uint32_t    matrix_id = instruction.words[3];
    const Type& type      = GetTypeByTypeId(type_id);

    assertm(type.kind == Type::Kind::Matrix, "SPIRV simulator: Non-matrix type given to Op_Transpose");
    assertm(type.matrix.col_count > 0, "SPIRV simulator: Matrix type with no columns encountered");

    const Type& col_type = GetTypeByTypeId(type.matrix.col_type_id);
    assertm(col_type.kind == Type::Kind::Vector,
            "SPIRV simulator: Non-vector column type in matrix type given to Op_Transpose");
    assertm(col_type.vector.elem_count > 0, "SPIRV simulator: Vector type with no elements encountered");

    Value source_matrix_value = GetValue(matrix_id);
    assertm(std::holds_alternative<std::shared_ptr<MatrixV>>(source_matrix_value),
            "SPIRV simulator: Simulator value does not hold a MatrixV shared pointer in Op_Transpose");

    std::shared_ptr<MatrixV> new_matrix    = std::make_shared<MatrixV>();
    std::shared_ptr<MatrixV> source_matrix = std::get<std::shared_ptr<MatrixV>>(source_matrix_value);
    assertm(source_matrix->cols.size() == col_type.vector.elem_count,
            "SPIRV simulator: Column vs row mismatch in Op_Transpose");

    for (uint64_t target_column = 0; target_column < type.matrix.col_count; ++target_column)
    {
        std::shared_ptr<VectorV> new_column = std::make_shared<VectorV>();

        for (uint64_t source_row = 0; source_row < type.matrix.col_count; ++source_row)
        {
            for (uint64_t source_column = 0; source_column < col_type.vector.elem_count; ++source_column)
            {
                new_column->elems.push_back(CopyValue(
                    std::get<std::shared_ptr<VectorV>>(source_matrix->cols[source_column])->elems[source_row]));
            }
        }

        new_matrix->cols.push_back(new_column);
    }

    SetValue(result_id, new_matrix);

    if (ValueIsArbitrary(instruction.words[3]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_FNegate(const Instruction& instruction)
{
    /*
    OpFNegate

    Inverts the sign bit of Operand. (Note, however, that OpFNegate is still considered a floating-point instruction,
    and so is subject to the general floating-point rules regarding, for example, subnormals and NaN propagation).

    Result Type must be a scalar or vector of floating-point type.
    The type of Operand must be the same as Result Type.
    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFNegate);

    uint32_t type_id    = instruction.words[1];
    uint32_t result_id  = instruction.words[2];
    uint32_t operand_id = instruction.words[3];

    const Type&  type   = GetTypeByTypeId(type_id);
    const Value& val_op = GetValue(operand_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op),
                "SPIRV simulator: Operand not of vector type");

        auto vec = std::get<std::shared_ptr<VectorV>>(val_op);

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            double elem_result = -1.0 * std::get<double>(vec->elems[i]);
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Float)
    {
        assertm(std::holds_alternative<double>(val_op), "SPIRV simulator: Operands not of float type");

        Value result = -1.0 * std::get<double>(val_op);

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or float");
    }

    if (ValueIsArbitrary(instruction.words[3]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_UGreaterThan(const Instruction& instruction)
{
    /*
    OpUGreaterThan

    Unsigned-integer comparison if Operand 1 is greater than Operand 2.

    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpUGreaterThan);

    uint32_t type_id     = instruction.words[1];
    uint32_t result_id   = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    Type         type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<uint64_t>(vec1->elems[i]) &&
                        std::holds_alternative<uint64_t>(vec1->elems[i]),
                    "SPIRV simulator: Found non-unsigned integer operand in vector operands");

            Value elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) > std::get<uint64_t>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        Value result = (uint64_t)(std::get<uint64_t>(val_op1) > std::get<uint64_t>(val_op2));
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or bool");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_FOrdLessThan(const Instruction& instruction)
{
    /*

    OpFOrdLessThan

    Floating-point comparison if operands are ordered and Operand 1 is less than Operand 2.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of floating-point type.
    They must have the same type, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFOrdLessThan);

    uint32_t type_id     = instruction.words[1];
    uint32_t result_id   = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    Type         type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]),
                    "SPIRV simulator: Found non-floating point operand in vector operands");

            Value elem_result = (uint64_t)(std::get<double>(vec1->elems[i]) < std::get<double>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        Value result = (uint64_t)(std::get<double>(val_op1) < std::get<double>(val_op2));
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type in, must be vector or float");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_FOrdLessThanEqual(const Instruction& instruction)
{
    /*
    OpFOrdLessThanEqual

    Floating-point comparison if operands are ordered and Operand 1 is less than or equal to Operand 2.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of floating-point type.
    They must have the same type, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFOrdLessThanEqual);

    uint32_t type_id     = instruction.words[1];
    uint32_t result_id   = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    Type         type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<double>(vec1->elems[i]) && std::holds_alternative<double>(vec2->elems[i]),
                    "SPIRV simulator: Found non-floating point operand in vector operands");

            Value elem_result = (uint64_t)(std::get<double>(vec1->elems[i]) <= std::get<double>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        Value result = (uint64_t)(std::get<double>(val_op1) <= std::get<double>(val_op2));
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or float");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_Switch(const Instruction& instruction)
{
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
    uint32_t default_id  = instruction.words[2];

    const Value& selector_value = GetValue(selector_id);
    uint64_t     selector;
    if (std::holds_alternative<uint64_t>(selector_value))
    {
        selector = std::get<uint64_t>(selector_value);
    }
    else if (std::holds_alternative<int64_t>(selector_value))
    {
        selector = (uint64_t)std::get<int64_t>(selector_value);
    }
    else
    {
        assertx("SPIRV simulator: Selector value is not an integer");
        return;
    }

    const Type& type = GetTypeByResultId(selector_id);
    assertm(type.scalar.width <= 32,
            "SPIRV simulator: Selector ID uses more than 32 bits, this is not handled at present and should be "
            "implemented");

    uint32_t label_id = default_id;
    for (uint32_t i = 3; i < instruction.word_count; i += 2)
    {
        uint32_t literal = instruction.words[i];

        if (selector == literal)
        {
            label_id = instruction.words[i + 1];
            break;
        }
    }

    call_stack_.back().pc = result_id_to_inst_index_.at(label_id);
}

void SPIRVSimulator::Op_MatrixTimesVector(const Instruction& instruction)
{
    /*
    OpMatrixTimesVector

    Linear-algebraic Matrix X Vector.

    Result Type must be a vector of floating-point type.
    Matrix must be an OpTypeMatrix whose Column Type is Result Type.
    Vector must be a vector with the same Component Type as the Component Type in Result Type.

    Its number of components must equal the number of columns in Matrix.
    */
    assert(instruction.opcode == spv::Op::OpMatrixTimesVector);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t matrix_id = instruction.words[3];
    uint32_t vector_id = instruction.words[4];

    const Type& type = GetTypeByTypeId(type_id);
    assertm(type.kind == Type::Kind::Vector, "SPIRV simulator: Result operand is not a vector");
    assertm(GetTypeByResultId(matrix_id).kind == Type::Kind::Matrix, "SPIRV simulator: First operand is not a matrix");
    assertm(GetTypeByResultId(vector_id).kind == Type::Kind::Vector, "SPIRV simulator: Second operand is not a vector");

    const std::shared_ptr<VectorV>& vector = std::get<std::shared_ptr<VectorV>>(GetValue(vector_id));
    const std::shared_ptr<MatrixV>& matrix = std::get<std::shared_ptr<MatrixV>>(GetValue(matrix_id));

    std::vector<double> tmp_result;
    tmp_result.resize(type.vector.elem_count);

    for (uint32_t col_index = 0; col_index < matrix->cols.size(); ++col_index)
    {
        for (uint32_t row_index = 0; row_index < type.vector.elem_count; ++row_index)
        {
            assertm(std::holds_alternative<std::shared_ptr<VectorV>>(matrix->cols[col_index]),
                    "SPIRV simulator: Non-vector column value found in matrix operand");
            assertm(std::holds_alternative<double>(vector->elems[row_index]),
                    "SPIRV simulator: Non-floating point value found in vector operand");

            const std::shared_ptr<VectorV>& col_vector = std::get<std::shared_ptr<VectorV>>(matrix->cols[col_index]);
            assertm(std::holds_alternative<double>(col_vector->elems[row_index]),
                    "SPIRV simulator: Non-floating point value found in column vector operand");

            tmp_result[row_index] +=
                std::get<double>(col_vector->elems[row_index]) * std::get<double>(vector->elems[col_index]);
        }
    }

    std::shared_ptr<VectorV> result = std::make_shared<VectorV>();
    for (double result_val : tmp_result)
    {
        result->elems.push_back(result_val);
    }

    SetValue(result_id, result);

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_VectorShuffle(const Instruction& instruction)
{
    /*
    OpVectorShuffle

    Select arbitrary components from two vectors to make a new vector.

    Result Type must be an OpTypeVector.
    The number of components in Result Type must be the same as the number of Component operands.

    Vector 1 and Vector 2 must both have vector types, with the same Component Type as Result Type.
    They do not have to have the same number of components as Result Type or with each other. They are logically
    concatenated, forming a single vector with Vector 1’s components appearing before Vector 2’s. The components of this
    logical vector are logically numbered with a single consecutive set of numbers from 0 to N - 1, where N is the total
    number of components.

    Components are these logical numbers (see above), selecting which of the logically numbered components form the
    result. Each component is an unsigned 32-bit integer. They can select the components in any order and can repeat
    components. The first component of the result is selected by the first Component operand, the second component of
    the result is selected by the second Component operand, etc. A Component literal may also be FFFFFFFF, which means
    the corresponding result component has no source and is undefined. All Component literals must either be FFFFFFFF or
    in [0, N - 1] (inclusive).

    Note: A vector “swizzle” can be done by using the vector for both Vector operands, or
    using an OpUndef for one of the Vector operands.
    */
    assert(instruction.opcode == spv::Op::OpVectorShuffle);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t vec1_id   = instruction.words[3];
    uint32_t vec2_id   = instruction.words[4];

    const Type& type = GetTypeByTypeId(type_id);

    assertm(type.kind == Type::Kind::Vector, "SPIRV simulator: Non-vector result type");

    const Value& vector1_val = GetValue(vec1_id);
    const Value& vector2_val = GetValue(vec2_id);

    assertm(std::holds_alternative<std::shared_ptr<VectorV>>(vector1_val),
            "SPIRV simulator: Non-vector value in vector operand 1");
    assertm(std::holds_alternative<std::shared_ptr<VectorV>>(vector2_val),
            "SPIRV simulator: Non-vector value in vector operand 2");

    const std::shared_ptr<VectorV>& vector1 = std::get<std::shared_ptr<VectorV>>(vector1_val);
    const std::shared_ptr<VectorV>& vector2 = std::get<std::shared_ptr<VectorV>>(vector2_val);

    std::vector<Value> values;
    values.insert(values.end(), vector1->elems.begin(), vector1->elems.end());
    values.insert(values.end(), vector2->elems.begin(), vector2->elems.end());

    std::shared_ptr<VectorV> result = std::make_shared<VectorV>();
    for (uint32_t literal_index = 5; literal_index < instruction.word_count; ++literal_index)
    {
        uint32_t component_index = instruction.words[literal_index];
        assertm(component_index < values.size(), "SPIRV simulator: Literal index OOB");

        if (component_index == 0xFFFFFFFF)
        {
            Value undef_val = (uint64_t)0xFFFFFFFF;
            result->elems.push_back(undef_val);
        }
        else
        {
            result->elems.push_back(values[component_index]);
        }
    }

    SetValue(result_id, result);

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_ShiftRightLogical(const Instruction& instruction)
{
    /*
    OpShiftRightLogical

    Shift the bits in Base right by the number of bits specified in Shift.The most-significant bits are zero filled.

    Result Type must be a scalar or vector of integer type.

    The type of each Base and Shift must be a scalar or vector of integer type. Base and Shift must have the same number
    of components. The number of components and bit width of the type of Base must be the same as in Result Type.

    Shift is consumed as an unsigned integer. The resulting value is undefined if Shift is greater than or equal to the
    bit width of the components of Base.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpShiftRightLogical);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t op1_id    = instruction.words[3];
    uint32_t op2_id    = instruction.words[4];

    const Type&  type = GetTypeByTypeId(type_id);
    const Value& op1  = GetValue(op1_id);
    const Value& op2  = GetValue(op2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        auto vec1 = std::get<std::shared_ptr<VectorV>>(op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(op2);

        assertm(vec1->elems.size() == vec2->elems.size() && vec1->elems.size() == type.vector.elem_count,
                "SPIRV simulator: Vector size mismatch in Op_ShiftRightLogical");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            if (std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i]))
            {
                result_vec->elems.push_back(std::get<uint64_t>(vec1->elems[i]) >> std::get<uint64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<uint64_t>(vec1->elems[i]) &&
                     std::holds_alternative<int64_t>(vec2->elems[i]))
            {
                result_vec->elems.push_back(std::get<uint64_t>(vec1->elems[i]) >> std::get<int64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<int64_t>(vec1->elems[i]) &&
                     std::holds_alternative<uint64_t>(vec2->elems[i]))
            {
                result_vec->elems.push_back((uint64_t)std::get<int64_t>(vec1->elems[i]) >>
                                            std::get<uint64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]))
            {
                result_vec->elems.push_back(std::get<int64_t>(vec1->elems[i]) >> std::get<int64_t>(vec2->elems[i]));
            }
            else
            {
                assertx("SPIRV simulator: Invalid operand types in Op_ShiftRightLogical vector");
            }
        }
        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Int)
    {
        Value result;
        if (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2))
        {
            result = std::get<uint64_t>(op1) >> std::get<uint64_t>(op2);
        }
        else if (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<int64_t>(op2))
        {
            result = (uint64_t)std::get<uint64_t>(op1) >> std::get<int64_t>(op2);
        }
        else if (std::holds_alternative<int64_t>(op1) && std::holds_alternative<uint64_t>(op2))
        {
            result = (uint64_t)std::get<int64_t>(op1) >> std::get<uint64_t>(op2);
        }
        else if (std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2))
        {
            result = (uint64_t)std::get<int64_t>(op1) >> std::get<int64_t>(op2);
        }
        else
        {
            assertx("SPIRV simulator: Invalid operand types in Op_ShiftRightLogical");
        }
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type in Op_ShiftRightLogical, must be vector or int");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_ShiftLeftLogical(const Instruction& instruction)
{
    /*
     OpShiftLeftLogical

     Shift the bits in Base left by the number of bits specified in Shift. The most-significant bits are zero filled.
     Result Type must be a scalar or vector of integer type.
     The type of each Base and Shift must be a scalar or vector of integer type. Base and Shift must have the same
     number of components. The number of components and bit width of the type of Base must be the same as in Result
     Type. Shift is consumed as an unsigned integer. The resulting value is undefined if Shift is greater than or equal
     to the bit width of the components of Base. Results are computed per component.
     */
    assert(instruction.opcode == spv::Op::OpShiftLeftLogical);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t op1_id    = instruction.words[3];
    uint32_t op2_id    = instruction.words[4];

    const Type&  type = GetTypeByTypeId(type_id);
    const Value& op1  = GetValue(op1_id);
    const Value& op2  = GetValue(op2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        auto vec1 = std::get<std::shared_ptr<VectorV>>(op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(op2);

        assertm(vec1->elems.size() == vec2->elems.size() && vec1->elems.size() == type.vector.elem_count,
                "SPIRV simulator: Vector size mismatch in Op_ShiftLeftLogical");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            if (std::holds_alternative<uint64_t>(vec1->elems[i]) && std::holds_alternative<uint64_t>(vec2->elems[i]))
            {
                result_vec->elems.push_back(std::get<uint64_t>(vec1->elems[i]) << std::get<uint64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<int64_t>(vec1->elems[i]) &&
                     std::holds_alternative<uint64_t>(vec2->elems[i]))
            {
                result_vec->elems.push_back((uint64_t)std::get<int64_t>(vec1->elems[i])
                                            << std::get<uint64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<uint64_t>(vec1->elems[i]) &&
                     std::holds_alternative<int64_t>(vec2->elems[i]))
            {
                result_vec->elems.push_back((uint64_t)std::get<uint64_t>(vec1->elems[i])
                                            << std::get<int64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]))
            {
                result_vec->elems.push_back((uint64_t)std::get<int64_t>(vec1->elems[i])
                                            << std::get<int64_t>(vec2->elems[i]));
            }
            else
            {
                assertx("SPIRV simulator: Invalid operand types in Op_ShiftLeftLogical vector");
            }
        }
        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Int)
    {
        Value result;
        if (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<uint64_t>(op2))
        {
            result = std::get<uint64_t>(op1) << std::get<uint64_t>(op2);
        }
        else if (std::holds_alternative<int64_t>(op1) && std::holds_alternative<uint64_t>(op2))
        {
            result = (uint64_t)std::get<int64_t>(op1) << std::get<uint64_t>(op2);
        }

        else if (std::holds_alternative<uint64_t>(op1) && std::holds_alternative<int64_t>(op2))
        {
            result = (uint64_t)std::get<uint64_t>(op1) << std::get<int64_t>(op2);
        }
        else if (std::holds_alternative<int64_t>(op1) && std::holds_alternative<int64_t>(op2))
        {
            result = (uint64_t)std::get<int64_t>(op1) << std::get<int64_t>(op2);
        }
        else
        {
            assertx("SPIRV simulator: Invalid operand types in Op_ShiftLeftLogical");
        }
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type in Op_ShiftLeftLogical, must be vector or int");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_BitwiseOr(const Instruction& instruction)
{
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

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t op1_id    = instruction.words[3];
    uint32_t op2_id    = instruction.words[4];

    const Type&  type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(op1_id);
    const Value& val_op2 = GetValue(op2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        const Type& elem_type = GetTypeByTypeId(type.vector.elem_type_id);

        assertm(elem_type.kind == Type::Kind::Int, "SPIRV simulator: Vector element type is not int in OpBitwiseOr");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length");
        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            uint64_t val1;
            if (std::holds_alternative<int64_t>(vec1->elems[i]))
            {
                val1 = bit_cast<uint64_t>(std::get<int64_t>(vec1->elems[i]));
            }
            else
            {
                val1 = std::get<uint64_t>(vec1->elems[i]);
            }

            uint64_t val2;
            if (std::holds_alternative<int64_t>(vec2->elems[i]))
            {
                val2 = bit_cast<uint64_t>(std::get<int64_t>(vec2->elems[i]));
            }
            else
            {
                val2 = std::get<uint64_t>(vec2->elems[i]);
            }

            Value elem_result;
            if (elem_type.scalar.is_signed)
            {
                elem_result = (int64_t)(val1 | val2);
            }
            else
            {
                elem_result = (uint64_t)(val1 | val2);
            }
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Int)
    {
        uint64_t val1;
        if (std::holds_alternative<int64_t>(val_op1))
        {
            val1 = bit_cast<uint64_t>(std::get<int64_t>(val_op1));
        }
        else
        {
            val1 = std::get<uint64_t>(val_op1);
        }

        uint64_t val2;
        if (std::holds_alternative<int64_t>(val_op2))
        {
            val2 = bit_cast<uint64_t>(std::get<int64_t>(val_op2));
        }
        else
        {
            val2 = std::get<uint64_t>(val_op2);
        }

        Value result;
        if (type.scalar.is_signed)
        {
            result = (int64_t)(val1 | val2);
        }
        else
        {
            result = (uint64_t)(val1 | val2);
        }
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or int");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_BitwiseAnd(const Instruction& instruction)
{
    /*
    OpBitwiseAnd

    Result is 1 if both Operand 1 and Operand 2 are 1. Result is 0 if either Operand 1 or Operand 2 are 0.

    Results are computed per component, and within each component, per bit.

    Result Type must be a scalar or vector of integer type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same number of components as Result Type. They must have the same component width as Result Type.
    */
    assert(instruction.opcode == spv::Op::OpBitwiseAnd);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t op1_id    = instruction.words[3];
    uint32_t op2_id    = instruction.words[4];

    const Type&  type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(op1_id);
    const Value& val_op2 = GetValue(op2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        const Type& elem_type = GetTypeByTypeId(type.vector.elem_type_id);

        assertm(elem_type.kind == Type::Kind::Int, "SPIRV simulator: Vector element type is not int in OpBitwiseAnd");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length");
        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            uint64_t val1;
            if (std::holds_alternative<int64_t>(vec1->elems[i]))
            {
                val1 = bit_cast<uint64_t>(std::get<int64_t>(vec1->elems[i]));
            }
            else if (std::holds_alternative<uint64_t>(vec1->elems[i]))
            {
                val1 = std::get<uint64_t>(vec1->elems[i]);
            }
            else
            {
                assertx("SPIRV simulator: Invalid vector element type encountered in Op_BitwiseAnd operands");
            }

            uint64_t val2;
            if (std::holds_alternative<int64_t>(vec2->elems[i]))
            {
                val2 = bit_cast<uint64_t>(std::get<int64_t>(vec2->elems[i]));
            }
            else if (std::holds_alternative<uint64_t>(vec2->elems[i]))
            {
                val2 = std::get<uint64_t>(vec2->elems[i]);
            }
            else
            {
                assertx("SPIRV simulator: Invalid vector element type encountered in Op_BitwiseAnd operands");
            }

            Value elem_result;
            if (elem_type.scalar.is_signed)
            {
                elem_result = (int64_t)(val1 & val2);
            }
            else
            {
                elem_result = (uint64_t)(val1 & val2);
            }

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Int)
    {
        uint64_t val1;
        if (std::holds_alternative<int64_t>(val_op1))
        {
            val1 = bit_cast<uint64_t>(std::get<int64_t>(val_op1));
        }
        else if (std::holds_alternative<uint64_t>(val_op1))
        {
            val1 = std::get<uint64_t>(val_op1);
        }
        else
        {
            assertx("SPIRV simulator: Invalid type encountered in Op_BitwiseAnd operands");
        }

        uint64_t val2;
        if (std::holds_alternative<int64_t>(val_op2))
        {
            val2 = bit_cast<uint64_t>(std::get<int64_t>(val_op2));
        }
        else if (std::holds_alternative<uint64_t>(val_op2))
        {
            val2 = std::get<uint64_t>(val_op2);
        }
        else
        {
            assertx("SPIRV simulator: Invalid type encountered in Op_BitwiseAnd operands");
        }

        Value result;
        if (type.scalar.is_signed)
        {
            result = (int64_t)(val1 & val2);
        }
        else
        {
            result = (uint64_t)(val1 & val2);
        }
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or int");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_All(const Instruction& instruction)
{
    /*
    OpAll

    Result is true if all components of Vector are true, otherwise result is false.

    Result Type must be a Boolean type scalar.

    Vector must be a vector of Boolean type.
    */
    assert(instruction.opcode == spv::Op::OpAll);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t vector_id = instruction.words[3];

    const Type&  type         = GetTypeByTypeId(type_id);
    const Type&  operand_type = GetTypeByResultId(vector_id);
    const Value& vector_val   = GetValue(vector_id);

    assertm(operand_type.kind == Type::Kind::Vector, "SPIRV simulator: Operand is not of vector type");
    assertm(std::holds_alternative<std::shared_ptr<VectorV>>(vector_val),
            "SPIRV simulator: Operand is of vector type but does not hold a vector");

    const std::shared_ptr<VectorV>& vec = std::get<std::shared_ptr<VectorV>>(vector_val);

    bool result_bool = true;
    for (const auto& bool_val : vec->elems)
    {
        result_bool = result_bool && (bool)std::get<uint64_t>(bool_val);
    }

    Value result = (uint64_t)result_bool;
    SetValue(result_id, result);

    if (ValueIsArbitrary(instruction.words[3]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_Any(const Instruction& instruction)
{
    /*
    OpAny

    Result is true if any component of Vector is true, otherwise result is false.

    Result Type must be a Boolean type scalar.

    Vector must be a vector of Boolean type.
    */
    assert(instruction.opcode == spv::Op::OpAny);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t vector_id = instruction.words[3];

    const Type&  type         = GetTypeByTypeId(type_id);
    const Type&  operand_type = GetTypeByResultId(vector_id);
    const Value& vector_val   = GetValue(vector_id);

    assertm(operand_type.kind == Type::Kind::Vector, "SPIRV simulator: Operand is not of vector type");
    assertm(std::holds_alternative<std::shared_ptr<VectorV>>(vector_val),
            "SPIRV simulator: Operand is of vector type but does not hold a vector");

    const std::shared_ptr<VectorV>& vec = std::get<std::shared_ptr<VectorV>>(vector_val);

    bool result_bool = false;
    for (const auto& bool_val : vec->elems)
    {
        result_bool = result_bool || (bool)std::get<uint64_t>(bool_val);
    }

    Value result = (uint64_t)result_bool;
    SetValue(result_id, result);

    if (ValueIsArbitrary(instruction.words[3]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_BitCount(const Instruction& instruction)
{
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

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t base_id   = instruction.words[3];

    const Type&  type     = GetTypeByTypeId(type_id);
    const Value& base_val = GetValue(base_id);

    uint32_t base_type_id = GetTypeID(base_id);

    bool is_arbitrary = ValueIsArbitrary(instruction.words[3]);
    if (type.kind == Type::Kind::Vector)
    {
        const Type&                    base_type = GetTypeByTypeId(base_type_id);
        const std::shared_ptr<VectorV> vec       = std::get<std::shared_ptr<VectorV>>(base_val);

        std::shared_ptr<VectorV> result_vec = std::make_shared<VectorV>();

        bool ab_val = false;
        for (const Value& val : vec->elems)
        {
            result_vec->elems.push_back((uint64_t)CountSetBits(val, base_type.vector.elem_type_id, &ab_val));
        }

        is_arbitrary |= ab_val;

        SetValue(result_id, result_vec);
    }
    else if (type.kind == Type::Kind::Int)
    {
        bool ab_val = false;
        SetValue(result_id, (uint64_t)CountSetBits(base_val, base_type_id, &ab_val));
        is_arbitrary |= ab_val;
    }
    else
    {
        assertx("SPIRV simulator: Invalid result value, must be vector or int");
    }

    if (is_arbitrary)
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_Kill(const Instruction& instruction)
{
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

    if (verbose_)
    {
        std::cout << execIndent << "Thread killed by OpKill, ceasing all further processing" << std::endl;
    }

    call_stack_.clear();
}

void SPIRVSimulator::Op_Unreachable(const Instruction& instruction)
{
    /*
    OpUnreachable

    Behavior is undefined if this instruction is executed.

    This instruction must be the last instruction in a block.
    */
    assert(instruction.opcode == spv::Op::OpUnreachable);

    assertx("SPIRV simulator: OpUnreachable executed, this is undefined behaviour");
}

void SPIRVSimulator::Op_Undef(const Instruction& instruction)
{
    /*
    OpUndef

    Make an intermediate object whose value is undefined.

    Result Type is the type of object to make. Result Type can be any type except OpTypeVoid.

    Each consumption of Result <id> yields an arbitrary, possibly different bit pattern or abstract value
    resulting in possibly different concrete, abstract, or opaque values.
    */
    assert(instruction.opcode == spv::Op::OpUndef);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    SetValue(result_id, MakeDefault(type_id));
}

void SPIRVSimulator::Op_VectorTimesMatrix(const Instruction& instruction)
{
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

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t vector_id = instruction.words[3];
    uint32_t matrix_id = instruction.words[4];

    const Type& type = GetTypeByTypeId(type_id);
    assertm(type.kind == Type::Kind::Vector, "SPIRV simulator: Result operand is not a vector");
    assertm(GetTypeByResultId(matrix_id).kind == Type::Kind::Matrix, "SPIRV simulator: Second operand is not a matrix");
    assertm(GetTypeByResultId(vector_id).kind == Type::Kind::Vector, "SPIRV simulator: First operand is not a vector");

    const std::shared_ptr<VectorV>& vector = std::get<std::shared_ptr<VectorV>>(GetValue(vector_id));
    const std::shared_ptr<MatrixV>& matrix = std::get<std::shared_ptr<MatrixV>>(GetValue(matrix_id));

    std::vector<double> tmp_result;
    tmp_result.resize(type.vector.elem_count);

    for (uint32_t col_index = 0; col_index < matrix->cols.size(); ++col_index)
    {
        for (uint32_t row_index = 0; row_index < type.vector.elem_count; ++row_index)
        {
            assertm(std::holds_alternative<std::shared_ptr<VectorV>>(matrix->cols[col_index]),
                    "SPIRV simulator: Non-vector column value found in matrix operand");
            assertm(std::holds_alternative<double>(vector->elems[row_index]),
                    "SPIRV simulator: Non-floating point value found in vector operand");

            const std::shared_ptr<VectorV>& col_vector = std::get<std::shared_ptr<VectorV>>(matrix->cols[col_index]);
            assertm(std::holds_alternative<double>(col_vector->elems[row_index]),
                    "SPIRV simulator: Non-floating point value found in column vector operand");

            tmp_result[col_index] +=
                std::get<double>(col_vector->elems[row_index]) * std::get<double>(vector->elems[row_index]);
        }
    }

    std::shared_ptr<VectorV> result = std::make_shared<VectorV>();
    for (double result_val : tmp_result)
    {
        result->elems.push_back(result_val);
    }

    SetValue(result_id, result);

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_ULessThanEqual(const Instruction& instruction)
{
    /*
    OpULessThanEqual

    Unsigned-integer comparison if Operand 1 is less than or equal to Operand 2.

    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpULessThanEqual);

    uint32_t type_id     = instruction.words[1];
    uint32_t result_id   = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    Type         type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<uint64_t>(vec1->elems[i]) &&
                        std::holds_alternative<uint64_t>(vec2->elems[i]),
                    "SPIRV simulator: Found non-unsigned int operand in vector operands");

            Value elem_result = (uint64_t)(std::get<uint64_t>(vec1->elems[i]) <= std::get<uint64_t>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        Value result = (uint64_t)(std::get<uint64_t>(val_op1) <= std::get<uint64_t>(val_op2));
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or bool");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_SLessThanEqual(const Instruction& instruction)
{
    /*
    OpSLessThanEqual

    Signed-integer comparison if Operand 1 is less than or equal to Operand 2.

    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpSLessThanEqual);

    uint32_t type_id     = instruction.words[1];
    uint32_t result_id   = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    Type         type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]),
                    "SPIRV simulator: Found non-signed int operand in vector operands");

            Value elem_result = (uint64_t)(std::get<int64_t>(vec1->elems[i]) <= std::get<int64_t>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        Value result = (uint64_t)(std::get<int64_t>(val_op1) <= std::get<int64_t>(val_op2));
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or bool");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_SGreaterThanEqual(const Instruction& instruction)
{
    /*
    OpSGreaterThanEqual

    Signed-integer comparison if Operand 1 is greater than or equal to Operand 2.

    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpSGreaterThanEqual);

    uint32_t type_id     = instruction.words[1];
    uint32_t result_id   = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    Type         type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]),
                    "SPIRV simulator: Found non-signed int operand in vector operands");

            Value elem_result = (uint64_t)(std::get<int64_t>(vec1->elems[i]) >= std::get<int64_t>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        Value result = (uint64_t)(std::get<int64_t>(val_op1) >= std::get<int64_t>(val_op2));
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or bool");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_SGreaterThan(const Instruction& instruction)
{
    /*
    OpSGreaterThan

    Signed-integer comparison if Operand 1 is greater than Operand 2.

    Result Type must be a scalar or vector of Boolean type.

    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same component width, and they must have the same number of components as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpSGreaterThan);

    uint32_t type_id     = instruction.words[1];
    uint32_t result_id   = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    Type         type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(operand1_id);
    const Value& val_op2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]),
                    "SPIRV simulator: Found non-signed int operand in vector operands");

            Value elem_result = (uint64_t)(std::get<int64_t>(vec1->elems[i]) > std::get<int64_t>(vec2->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        Value result = (uint64_t)(std::get<int64_t>(val_op1) > std::get<int64_t>(val_op2));
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or bool");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_SDiv(const Instruction& instruction)
{
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

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];

    Type         type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(instruction.words[3]);
    const Value& val_op2 = GetValue(instruction.words[4]);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            Value elem_result;

            // TODO: Operands dont have to be signed, deal with it and remove the asserts
            assertm(std::holds_alternative<int64_t>(vec1->elems[i]) && std::holds_alternative<int64_t>(vec2->elems[i]),
                    "SPIRV simulator: Found non-signed int operand vector operands");

            int64_t op2 = std::get<int64_t>(vec2->elems[i]);
            if (op2 == 0)
            {
                if (verbose_)
                {
                    std::cout << "SPIRV simulator: Divisor in Op_SDiv is 0, this is undefined behaviour, setting to 1"
                              << std::endl;
                }

                op2 = 1;
            }

            elem_result = std::get<int64_t>(vec1->elems[i]) / op2;

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Int)
    {
        // TODO: Operands dont have to be signed, deal with it and remove the asserts
        assertm(std::holds_alternative<int64_t>(val_op1) && std::holds_alternative<int64_t>(val_op2),
                "SPIRV simulator: Found non-signed int operand");

        int64_t op2 = std::get<int64_t>(val_op2);
        if (op2 == 0)
        {
            if (verbose_)
            {
                std::cout << "SPIRV simulator: Divisor in Op_SDiv is 0, this is undefined behaviour, setting to 1"
                          << std::endl;
            }

            op2 = 1;
        }

        Value result = std::get<int64_t>(val_op1) / op2;

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or signed int");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_SNegate(const Instruction& instruction)
{
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

    uint32_t type_id    = instruction.words[1];
    uint32_t result_id  = instruction.words[2];
    uint32_t operand_id = instruction.words[3];

    const Type&  type   = GetTypeByTypeId(type_id);
    const Value& val_op = GetValue(operand_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        // TODO: Operands dont have to be signed? If so, fix it
        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op),
                "SPIRV simulator: Operand not of vector type");

        auto vec = std::get<std::shared_ptr<VectorV>>(val_op);

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            int64_t elem_result = 0 - std::get<int64_t>(vec->elems[i]);
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Int)
    {
        // TODO: Operands dont have to be signed? If so, fix it
        assertm(std::holds_alternative<int64_t>(val_op), "SPIRV simulator: Operands not of int type");

        Value result = 0 - std::get<int64_t>(val_op);

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or integer");
    }

    if (ValueIsArbitrary(instruction.words[3]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_LogicalOr(const Instruction& instruction)
{
    /*
    OpLogicalOr

    Result is true if either Operand 1 or Operand 2 is true. Result is false if both Operand 1 and Operand 2 are false.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 must be the same as Result Type.
    The type of Operand 2 must be the same as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpLogicalOr);

    uint32_t type_id     = instruction.words[1];
    uint32_t result_id   = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    const Type&  type     = GetTypeByTypeId(type_id);
    const Value& operand1 = GetValue(operand1_id);
    const Value& operand2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand1),
                "SPIRV simulator: Invalid value type for operand 1, must be vector when using vector type");
        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand2),
                "SPIRV simulator: Invalid value type for operand 2, must be vector when using vector type");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(operand1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(operand2);

        assertm(std::holds_alternative<uint64_t>(vec1->elems[0]),
                "SPIRV simulator: Invalid vector component value for operand 1, must be bool");
        assertm(std::holds_alternative<uint64_t>(vec2->elems[0]),
                "SPIRV simulator: Invalid vector component value for operand 2, must be bool");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            result_vec->elems.push_back(
                (uint64_t)(std::get<uint64_t>(vec1->elems[i]) || std::get<uint64_t>(vec2->elems[i])));
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        assertm(std::holds_alternative<uint64_t>(operand1),
                "SPIRV simulator: Invalid value for operand 1, must be bool");
        assertm(std::holds_alternative<uint64_t>(operand2),
                "SPIRV simulator: Invalid value for operand 2, must be bool");
        Value result = (uint64_t)(std::get<uint64_t>(operand1) || std::get<uint64_t>(operand2));

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or bool");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_LogicalAnd(const Instruction& instruction)
{
    /*
    OpLogicalAnd

    Result is true if both Operand 1 and Operand 2 are true. Result is false if either Operand 1 or Operand 2 are false.
    Result Type must be a scalar or vector of Boolean type.
    The type of Operand 1 must be the same as Result Type.
    The type of Operand 2 must be the same as Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpLogicalAnd);

    uint32_t type_id     = instruction.words[1];
    uint32_t result_id   = instruction.words[2];
    uint32_t operand1_id = instruction.words[3];
    uint32_t operand2_id = instruction.words[4];

    const Type&  type     = GetTypeByTypeId(type_id);
    const Value& operand1 = GetValue(operand1_id);
    const Value& operand2 = GetValue(operand2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand1),
                "SPIRV simulator: Invalid value type for operand 1, must be vector when using vector type");
        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand2),
                "SPIRV simulator: Invalid value type for operand 2, must be vector when using vector type");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(operand1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(operand2);

        assertm(std::holds_alternative<uint64_t>(vec1->elems[0]),
                "SPIRV simulator: Invalid vector component value for operand 1, must be bool");
        assertm(std::holds_alternative<uint64_t>(vec2->elems[0]),
                "SPIRV simulator: Invalid vector component value for operand 2, must be bool");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            result_vec->elems.push_back(
                (uint64_t)(std::get<uint64_t>(vec1->elems[i]) && std::get<uint64_t>(vec2->elems[i])));
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        assertm(std::holds_alternative<uint64_t>(operand1),
                "SPIRV simulator: Invalid value for operand 1, must be bool");
        assertm(std::holds_alternative<uint64_t>(operand2),
                "SPIRV simulator: Invalid value for operand 2, must be bool");
        Value result = (uint64_t)(std::get<uint64_t>(operand1) && std::get<uint64_t>(operand2));

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or bool");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_MatrixTimesMatrix(const Instruction& instruction)
{
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

    uint32_t type_id         = instruction.words[1];
    uint32_t result_id       = instruction.words[2];
    uint32_t matrix_left_id  = instruction.words[3];
    uint32_t matrix_right_id = instruction.words[4];

    const Type& type       = GetTypeByTypeId(type_id);
    const Type& left_type  = GetTypeByResultId(matrix_left_id);
    const Type& right_type = GetTypeByResultId(matrix_right_id);

    assertm(type.kind == Type::Kind::Matrix, "SPIRV simulator: Result operand is not a matrix");
    assertm(left_type.kind == Type::Kind::Matrix, "SPIRV simulator: First operand is not a matrix");
    assertm(right_type.kind == Type::Kind::Matrix, "SPIRV simulator: Second operand is not a matrix");

    const Type& left_col_type = GetTypeByTypeId(left_type.matrix.col_type_id);
    assertm(left_col_type.kind == Type::Kind::Vector, "SPIRV simulator: Left matrix col type is not vector");

    const std::shared_ptr<MatrixV>& matrix_left  = std::get<std::shared_ptr<MatrixV>>(GetValue(matrix_left_id));
    const std::shared_ptr<MatrixV>& matrix_right = std::get<std::shared_ptr<MatrixV>>(GetValue(matrix_right_id));

    assertm(type.matrix.col_count == matrix_right->cols.size(),
            "SPIRV simulator: Second operand matrix number of columns dont match the result type");

    std::shared_ptr<MatrixV> result_matrix = std::make_shared<MatrixV>();

    for (uint32_t right_col_index = 0; right_col_index < matrix_right->cols.size(); ++right_col_index)
    {
        const auto& right_col_vec = std::get<std::shared_ptr<VectorV>>(matrix_right->cols[right_col_index]);

        std::vector<double> tmp_result;
        tmp_result.resize(left_col_type.vector.elem_count);

        for (uint32_t left_col_index = 0; left_col_index < matrix_left->cols.size(); ++left_col_index)
        {
            assertm(std::holds_alternative<std::shared_ptr<VectorV>>(matrix_left->cols[left_col_index]),
                    "SPIRV simulator: Non-vector column value found in matrix operand");
            assertm(std::holds_alternative<double>(right_col_vec->elems[left_col_index]),
                    "SPIRV simulator: Non-floating point value found in vector operand");

            const std::shared_ptr<VectorV>& left_col_vector =
                std::get<std::shared_ptr<VectorV>>(matrix_left->cols[left_col_index]);

            for (uint32_t row_index = 0; row_index < left_col_vector->elems.size(); ++row_index)
            {
                assertm(std::holds_alternative<double>(left_col_vector->elems[row_index]),
                        "SPIRV simulator: Non-floating point value found in column vector operand");
                tmp_result[row_index] += std::get<double>(left_col_vector->elems[row_index]) *
                                         std::get<double>(right_col_vec->elems[left_col_index]);
            }
        }

        std::shared_ptr<VectorV> new_column = std::make_shared<VectorV>();
        for (auto col_val : tmp_result)
        {
            new_column->elems.push_back(col_val);
        }
        result_matrix->cols.push_back(new_column);
    }

    SetValue(result_id, result_matrix);

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_IsNan(const Instruction& instruction)
{
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

    Also, C++ math functions can do a lot of wild stuff here, but apart from the operands above there are no guarantees
    GPU side either so we should be good.

    NOTE: We should investigate this carefully and in depth if we ever see broken behaviour in applications that use
    OpIsNan.
    */
    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t x_id      = instruction.words[3];

    std::cout << "SPIRV simulator: WARNING: OpIsNan executed, keep this in mind if you see broken behaviour here"
              << std::endl;

    const Type&  type  = GetTypeByTypeId(type_id);
    const Value& x_val = GetValue(x_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(x_val),
                "SPIRV simulator: Invalid value type for operand 1, must be vector when using vector type");

        auto x_vec = std::get<std::shared_ptr<VectorV>>(x_val);

        assertm(std::holds_alternative<double>(x_vec->elems[0]),
                "SPIRV simulator: Invalid vector component value for operand 1, must be bool");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            result_vec->elems.push_back((uint64_t)(std::isnan(std::get<double>(x_vec->elems[i]))));
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::BoolT)
    {
        assertm(std::holds_alternative<double>(x_val), "SPIRV simulator: Invalid value for operand 1, must be bool");
        Value result = (uint64_t)(std::isnan(std::get<double>(x_val)));

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or bool");
    }

    if (ValueIsArbitrary(instruction.words[3]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_SampledImage(const Instruction& instruction)
{
    /*
    OpSampledImage

    Create a sampled image, containing both a sampler and an image.

    Result Type must be OpTypeSampledImage.

    Image is an object whose type is an OpTypeImage, whose Sampled operand is
    0 or 1, and whose Dim operand is not SubpassData. Additionally, starting with
    version 1.6, the Dim operand must not be Buffer.

    Sampler must be an object whose type is OpTypeSampler.

    If the client API does not ignore Depth, the Image Type operand of the Result
    Type must be the same as the type of Image. Otherwise, the type of Image and
    the Image Type operand of the Result Type must be two OpTypeImage with all
    operands matching each other except for Depth which can be different.
    */
    assert(instruction.opcode == spv::Op::OpSampledImage);

    uint32_t result_type_id = instruction.words[1];
    uint32_t result_id      = instruction.words[2];
    uint32_t image_id       = instruction.words[3];
    uint32_t sampler_id     = instruction.words[4];

    SampledImageV new_si{ std::get<uint64_t>(GetValue(image_id)), std::get<uint64_t>(GetValue(sampler_id)) };
    SetValue(result_id, new_si);
}

void SPIRVSimulator::Op_ImageSampleImplicitLod(const Instruction& instruction)
{
    /*
    OpImageSampleImplicitLod

    Sample an image with an implicit level of detail.

    An invocation will not execute a dynamic instance of this instruction (X') until all invocations in its
    derivative group have executed all dynamic instances that are program-ordered before X'.

    Result Type must be a vector of four components of floating-point type or integer type. Its
    components must be the same as Sampled Type of the underlying OpTypeImage (unless that
    underlying Sampled Type is OpTypeVoid).

    Sampled Image must be an object whose type is OpTypeSampledImage. Its OpTypeImage must
    not have a Dim of Buffer. The MS operand of the underlying OpTypeImage must be 0.

    Coordinate must be a scalar or vector of floating-point type. It contains (u[, v] …​ [, array layer]) as
    needed by the definition of Sampled Image. It may be a vector larger than needed, but all unused
    components appear after all used components.

    Image Operands encodes what operands follow, as per Image Operands.

    This instruction is only valid in the Fragment Execution Model. In addition, it consumes an implicit
    derivative that can be affected by code motion.
    */
    assert(instruction.opcode == spv::Op::OpImageSampleImplicitLod);

    uint32_t type_id          = instruction.words[1];
    uint32_t result_id        = instruction.words[2];
    uint32_t sampled_image_id = instruction.words[3];
    uint32_t coordinate_id    = instruction.words[4];

    uint32_t image_operand_mask = 0;
    if (instruction.word_count > 5)
    {
        image_operand_mask = instruction.words[5];
    }

    SetValue(result_id, MakeDefault(type_id));
    SetIsArbitrary(result_id);
}

void SPIRVSimulator::Op_ImageSampleExplicitLod(const Instruction& instruction)
{
    /*
    OpImageSampleExplicitLod

    Sample an image using an explicit level of detail.

    Result Type must be a vector of four components of floating-point type or integer type. Its components
    must be the same as Sampled Type of the underlying OpTypeImage (unless that underlying Sampled
    Type is OpTypeVoid).

    Sampled Image must be an object whose type is OpTypeSampledImage. Its OpTypeImage must not
    have a Dim of Buffer. The MS operand of the underlying OpTypeImage must be 0.

    Coordinate must be a scalar or vector of floating-point type or integer type. It contains (u[, v] …​ [, array
    layer]) as needed by the definition of Sampled Image. Unless the Kernel capability is declared, it must
    be floating point. It may be a vector larger than needed, but all unused components appear after all used
    components.

    Image Operands encodes what operands follow, as per Image Operands. Either Lod or Grad image
    operands must be present.
    */
    assert(instruction.opcode == spv::Op::OpImageSampleExplicitLod);

    uint32_t type_id            = instruction.words[1];
    uint32_t result_id          = instruction.words[2];
    uint32_t sampled_image_id   = instruction.words[3];
    uint32_t coordinate_id      = instruction.words[4];
    uint32_t image_operand_mask = instruction.words[5];

    SetValue(result_id, MakeDefault(type_id));
    SetIsArbitrary(result_id);
}

void SPIRVSimulator::Op_ImageFetch(const Instruction& instruction)
{
    /*
    OpImageFetch

    Fetch a single texel from an image whose Sampled operand is 1.

    Result Type must be a vector of four components of floating-point type or integer type.
    Its components must be the same as Sampled Type of the underlying OpTypeImage
    (unless that underlying Sampled Type is OpTypeVoid).

    Image must be an object whose type is OpTypeImage. Its Dim operand must not be Cube, and its Sampled operand must
    be 1.

    Coordinate must be a scalar or vector of integer type. It contains (u[, v] …​ [, array layer]) as needed
    by the definition of Sampled Image.

    Image Operands encodes what operands follow, as per Image Operands.
    */
    assert(instruction.opcode == spv::Op::OpImageFetch);

    uint32_t type_id       = instruction.words[1];
    uint32_t result_id     = instruction.words[2];
    uint32_t image_id      = instruction.words[3];
    uint32_t coordinate_id = instruction.words[4];

    uint32_t image_operand_mask = 0;
    if (instruction.word_count > 5)
    {
        image_operand_mask = instruction.words[5];
    }

    SetValue(result_id, MakeDefault(type_id));
    SetIsArbitrary(result_id);
}

void SPIRVSimulator::Op_ImageGather(const Instruction& instruction)
{
    /*
    OpImageGather

    Gathers the requested component from four texels.

    Result Type must be a vector of four components of floating-point type or integer type. Its components
    must be the same as Sampled Type of the underlying OpTypeImage (unless that underlying Sampled
    Type is OpTypeVoid). It has one component per gathered texel.

    Sampled Image must be an object whose type is OpTypeSampledImage. Its OpTypeImage must have
    a Dim of 2D, Cube, or Rect. The MS operand of the underlying OpTypeImage must be 0.

    Coordinate must be a scalar or vector of floating-point type. It contains (u[, v] …​ [, array layer]) as
    needed by the definition of Sampled Image.

    Component is the component number gathered from all four texels. It must be a 32-bit integer type
    scalar. Behavior is undefined if its value is not 0, 1, 2 or 3.

    Image Operands encodes what operands follow, as per Image Operands.
    */
    assert(instruction.opcode == spv::Op::OpImageGather);

    uint32_t type_id          = instruction.words[1];
    uint32_t result_id        = instruction.words[2];
    uint32_t sampled_image_id = instruction.words[3];
    uint32_t coordinate_id    = instruction.words[4];
    uint32_t component_id     = instruction.words[5];

    uint32_t image_operand_mask = 0;
    if (instruction.word_count > 6)
    {
        image_operand_mask = instruction.words[6];
    }

    SetValue(result_id, MakeDefault(type_id));
    SetIsArbitrary(result_id);
}

void SPIRVSimulator::Op_ImageRead(const Instruction& instruction)
{
    /*
    OpImageRead

    Read a texel from an image without a sampler.

    Result Type must be a scalar or vector of floating-point type or integer type. It must be a scalar or
    vector with component type the same as Sampled Type of the OpTypeImage (unless that Sampled
    Type is OpTypeVoid).

    Image must be an object whose type is OpTypeImage with a Sampled operand of 0 or 2. If the
    Arrayed operand is 1, then additional capabilities may be required; e.g., ImageCubeArray, or
    ImageMSArray.

    Coordinate must be a scalar or vector of floating-point type or integer type. It contains non-normalized
    texel coordinates (u[, v] …​ [, array layer]) as needed by the definition of Image. See the
    client API specification for handling of coordinates outside the image.

    If the Image Dim operand is SubpassData, Coordinate is relative to the current fragment location.
    See the client API specification for more detail on how these coordinates are applied.

    If the Image Dim operand is not SubpassData, the Image Format must not be Unknown, unless
    the StorageImageReadWithoutFormat Capability was declared.

    Image Operands encodes what operands follow, as per Image Operands.
    */
    assert(instruction.opcode == spv::Op::OpImageRead);

    uint32_t result_type_id = instruction.words[1];
    uint32_t result_id      = instruction.words[2];
    uint32_t image_id       = instruction.words[3];
    uint32_t coordinate_id  = instruction.words[4];

    uint32_t image_operand_mask = 0;
    if (instruction.word_count > 5)
    {
        image_operand_mask = instruction.words[5];
    }

    SetIsArbitrary(result_id);

    const Type& result_type     = GetTypeByTypeId(result_type_id);
    const Type& image_type      = GetTypeByResultId(image_id);
    const Type& coordinate_type = GetTypeByResultId(coordinate_id);

    const Type& sampled_type = GetTypeByTypeId(image_type.image.sampled_type_id);

    if (result_type.kind == Type::Kind::Int)
    {
        assert(sampled_type.kind == Type::Kind::Void || sampled_type.kind == Type::Kind::Int);
        if (result_type.scalar.is_signed)
        {
            SetValue(result_id, int64_t(std::numeric_limits<int64_t>::max()));
        }
        else
        {
            SetValue(result_id, uint64_t(std::numeric_limits<uint64_t>::max()));
        }
    }
    if (result_type.kind == Type::Kind::Float)
    {
        SetValue(result_id, std::numeric_limits<double>::max());
    }
    if (result_type.kind == Type::Kind::Vector)
    {
        const Type& result_elem_type = GetTypeByTypeId(result_type.vector.elem_type_id);

        std::shared_ptr<VectorV> result_value = std::make_shared<VectorV>();
        if (result_elem_type.kind == Type::Kind::Float)
        {
            result_value->elems.resize(result_type.vector.elem_count, std::numeric_limits<double>::max());
        }
        else if (result_elem_type.kind == Type::Kind::Int)
        {
            if (result_elem_type.scalar.is_signed)
            {
                result_value->elems.resize(result_type.vector.elem_count, std::numeric_limits<int64_t>::max());
            }
            else
            {
                result_value->elems.resize(result_type.vector.elem_count, std::numeric_limits<uint64_t>::max());
            }
        }
        else
        {
            assertx("SPIRV simulator: Invalid type in output vector for OpImageRead");
        }

        SetValue(result_id, result_value);
    }
}

void SPIRVSimulator::Op_ImageWrite(const Instruction& instruction)
{
    /*
    OpImageWrite

    Write a texel to an image without a sampler.

    Image must be an object whose type is OpTypeImage with a Sampled operand of 0 or 2. If the
    Arrayed operand is 1, then additional capabilities may be required; e.g., ImageCubeArray, or
    ImageMSArray. Its Dim operand must not be SubpassData.

    Coordinate must be a scalar or vector of floating-point type or integer type. It contains non-normalized
    texel coordinates (u[, v] …​ [, array layer]) as needed by the definition of Image. See
    the client API specification for handling of coordinates outside the image.

    Texel is the data to write. It must be a scalar or vector with component type the same as
    Sampled Type of the OpTypeImage (unless that Sampled Type is OpTypeVoid).

    The Image Format must not be Unknown, unless the StorageImageWriteWithoutFormat
    Capability was declared.

    Image Operands encodes what operands follow, as per Image Operands.
    */
    assert(instruction.opcode == spv::Op::OpImageWrite);

    uint32_t image_id      = instruction.words[1];
    uint32_t coordinate_id = instruction.words[2];
    uint32_t texel_id      = instruction.words[3];

    uint32_t image_operand_mask = 0;
    if (instruction.word_count > 5)
    {
        image_operand_mask = instruction.words[5];
    }
    // Currently a NOP
}

void SPIRVSimulator::Op_ImageQuerySize(const Instruction& instruction)
{
    /*
    OpImageQuerySize

    Query the dimensions of Image, with no level of detail.

    Result Type must be an integer type scalar or vector. The number of components must be:
    1 for the 1D and Buffer dimensionalities,
    2 for the 2D, Cube, and Rect dimensionalities,
    3 for the 3D dimensionality,
    plus 1 more if the image type is arrayed. This vector is filled in with (width [, height] [, elements])
    where elements is the number of layers in an image array or the number of cubes in a cube-map
    array.

    Image must be an object whose type is OpTypeImage. Its Dim operand must be one of those listed
    under Result Type, above. Additionally, if its Dim is 1D, 2D, 3D, or Cube, it must also have either an
    MS of 1 or a Sampled of 0 or 2. There is no implicit level-of-detail consumed by this instruction. See
    OpImageQuerySizeLod for querying images having level of detail. See the client API specification
    for additional image type restrictions.
    */
    assert(instruction.opcode == spv::Op::OpImageQuerySize);

    uint32_t result_type_id = instruction.words[1];
    uint32_t result_id      = instruction.words[2];
    uint32_t image_id       = instruction.words[3];

    const Type& result_type = GetTypeByTypeId(result_type_id);
    const Type& image_type  = GetTypeByResultId(image_id);

    SetIsArbitrary(result_id);

    std::vector<uint64_t> size;
    switch (image_type.image.dim)
    {
        case spv::Dim::Dim1D:
        case spv::Dim::DimBuffer:
        {
            if (image_type.image.dim == spv::Dim::Dim1D)
            {
                assert(image_type.image.multisampled == 1 || image_type.image.sampled == 0 ||
                       image_type.image.sampled == 2);
            }

            size.resize(1, 1);

            break;
        }
        case spv::Dim::Dim2D:
        case spv::Dim::DimCube:
        case spv::Dim::DimRect:
        {
            if (image_type.image.dim == spv::Dim::Dim2D || image_type.image.dim == spv::Dim::DimCube)
            {
                assert(image_type.image.multisampled == 1 || image_type.image.sampled == 0 ||
                       image_type.image.sampled == 2);
            }

            size.resize(2, 1);

            break;
        }
        case spv::Dim::Dim3D:
        {
            if (image_type.image.dim == spv::Dim::Dim3D)
            {
                assert(image_type.image.multisampled == 1 || image_type.image.sampled == 0 ||
                       image_type.image.sampled == 2);
            }

            size.resize(3, 1);

            break;
        }
        default:
        {
            assert(false); // These image dimensions are not accepted for this opcode.
        }
    }

    if (image_type.image.arrayed == 1)
    {
        size.push_back(1);
    }

    if (result_type.kind == Type::Kind::Int)
    {
        assert(size.size() == 1);

        if (result_type.scalar.is_signed)
        {
            SetValue(result_id, int64_t(size[0]));
        }
        else
        {
            SetValue(result_id, uint64_t(size[0]));
        }
    }
    else if (result_type.kind == Type::Kind::Vector)
    {
        assert(size.size() == result_type.vector.elem_count);

        const Type& result_elem_type = GetTypeByTypeId(result_type.vector.elem_type_id);
        assert(result_elem_type.kind == Type::Kind::Int);

        std::shared_ptr<VectorV> result_value = std::make_shared<VectorV>();

        result_value->elems.resize(result_type.vector.elem_count);
        for (unsigned i = 0; i < result_type.vector.elem_count; ++i)
        {
            if (result_elem_type.scalar.is_signed)
            {
                result_value->elems[i] = int64_t(size[i]);
            }
            else
            {
                result_value->elems[i] = uint64_t(size[i]);
            }
        }

        SetValue(result_id, result_value);
    }
    else
    {
        assert(false);
    }
}

void SPIRVSimulator::Op_ImageQuerySizeLod(const Instruction& instruction)
{
    /*
    OpImageQuerySizeLod

    Query the dimensions of Image for mipmap level for Level of Detail.

    Result Type must be an integer type scalar or vector. The number of components must be
    1 for the 1D dimensionality,
    2 for the 2D and Cube dimensionalities,
    3 for the 3D dimensionality,
    plus 1 more if the image type is arrayed. This vector is filled in with (width [, height] [, depth] [, elements])
    where elements is the number of layers in an image array, or the number of cubes in a cube-map array.

    Image must be an object whose type is OpTypeImage. Its Dim operand must be one of 1D, 2D, 3D, or Cube, and its MS
    must be 0. See OpImageQuerySize for querying image types without level of detail. See the client API specification
    for additional image type restrictions.

    Level of Detail is used to compute which mipmap level to query and must be a 32-bit integer type scalar.
    */
    assert(instruction.opcode == spv::Op::OpImageQuerySize);

    uint32_t result_type_id = instruction.words[1];
    uint32_t result_id      = instruction.words[2];
    uint32_t image_id       = instruction.words[3];
    uint32_t lod            = instruction.words[4];

    const Type& result_type = GetTypeByTypeId(result_type_id);
    const Type& image_type  = GetTypeByResultId(image_id);

    assertm(image_type.kind == Type::Kind::Image, "SPIRV simulator: Image type is not Image");

    SetIsArbitrary(result_id);

    std::vector<uint64_t> size;
    switch (image_type.image.dim)
    {
        case spv::Dim::Dim1D:
        case spv::Dim::DimBuffer:
        {
            if (image_type.image.dim == spv::Dim::Dim1D)
            {
                assertm(image_type.image.multisampled == 1 || image_type.image.sampled == 0 ||
                            image_type.image.sampled == 2,
                        "SPIRV simulator: Invalid image configuration for 1D image");
            }

            size.resize(1, 1);

            break;
        }
        case spv::Dim::Dim2D:
        case spv::Dim::DimCube:
        case spv::Dim::DimRect:
        {
            if (image_type.image.dim == spv::Dim::Dim2D || image_type.image.dim == spv::Dim::DimCube)
            {
                assertm(image_type.image.multisampled == 1 || image_type.image.sampled == 0 ||
                            image_type.image.sampled == 2,
                        "SPIRV simulator: Invalid image configuration for 2D image");
            }

            size.resize(2, 1);

            break;
        }
        case spv::Dim::Dim3D:
        {
            if (image_type.image.dim == spv::Dim::Dim3D)
            {
                assertm(image_type.image.multisampled == 1 || image_type.image.sampled == 0 ||
                            image_type.image.sampled == 2,
                        "SPIRV simulator: Invalid image configuration for 3D image");
            }

            size.resize(3, 1);

            break;
        }
        default:
        {
            assertm(false, "SPIRV simulator: Invalid image dimensions in Op_ImageQuerySizeLod");
        }
    }

    if (image_type.image.arrayed == 1)
    {
        size.push_back(1);
    }

    if (result_type.kind == Type::Kind::Int)
    {
        assertm(size.size() == 1, "SPIRV simulator: Calculated dim size does not match scalar return type");

        if (result_type.scalar.is_signed)
        {
            SetValue(result_id, int64_t(size[0]));
        }
        else
        {
            SetValue(result_id, uint64_t(size[0]));
        }
    }
    else if (result_type.kind == Type::Kind::Vector)
    {
        assertm(size.size() == result_type.vector.elem_count,
                "SPIRV simulator: Calculated dim size does not match return vector type size");

        const Type& result_elem_type = GetTypeByTypeId(result_type.vector.elem_type_id);
        assertm(result_elem_type.kind == Type::Kind::Int, "SPIRV simulator: Vectory element type must be int");

        std::shared_ptr<VectorV> result_value = std::make_shared<VectorV>();

        result_value->elems.resize(result_type.vector.elem_count);
        for (unsigned i = 0; i < result_type.vector.elem_count; ++i)
        {
            if (result_elem_type.scalar.is_signed)
            {
                result_value->elems[i] = int64_t(size[i]);
            }
            else
            {
                result_value->elems[i] = uint64_t(size[i]);
            }
        }

        SetValue(result_id, result_value);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type in Op_ImageQuerySizeLod");
    }
}

void SPIRVSimulator::Op_FunctionParameter(const Instruction& instruction)
{
    /*
    OpFunctionParameter

    Declare a formal parameter of the current function.

    Result Type is the type of the parameter.

    This instruction must immediately follow an OpFunction or OpFunctionParameter instruction.
    The order of contiguous OpFunctionParameter instructions is the same order arguments are listed in
    an OpFunctionCall instruction to this function.

    It is also the same order in which Parameter Type operands are listed in the OpTypeFunction of the
    Function Type operand for this function’s OpFunction instruction.
    */
    // This is a nop in our implementation (handled at parse time)
    assert(instruction.opcode == spv::Op::OpFunctionParameter);
}

void SPIRVSimulator::Op_EmitVertex(const Instruction& instruction)
{
    /*
    OpEmitVertex

    Emits the current values of all output variables to the current output primitive.
    After execution, the values of all output variables are undefined.

    This instruction must only be used when only one stream is present.
    */
    assert(instruction.opcode == spv::Op::OpEmitVertex);
    std::cout << "SPIRV simulator: WARNING: Geometry shaders not implemented, instructions are ignored" << std::endl;
}

void SPIRVSimulator::Op_EndPrimitive(const Instruction& instruction)
{
    /*
    OpEndPrimitive

    Finish the current primitive and start a new one. No vertex is emitted.

    This instruction must only be used when only one stream is present.
    */
    assert(instruction.opcode == spv::Op::OpEndPrimitive);
    std::cout << "SPIRV simulator: WARNING: Geometry shaders not implemented, instructions are ignored" << std::endl;
}

void SPIRVSimulator::Op_FConvert(const Instruction& instruction)
{
    /*
    OpFConvert

    Convert value numerically from one floating-point width to another width.

    Result Type must be a scalar or vector of floating-point type.

    Float Value must be a scalar or vector of floating-point type.
    It must have the same number of components as Result Type.
    The component type must not equal the component type in Result Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpFConvert);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t value_id  = instruction.words[3];

    // We always store as doubles, this just equates to a type change
    SetValue(result_id, GetValue(value_id));

    if (ValueIsArbitrary(value_id))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_Image(const Instruction& instruction)
{
    /*
    OpImage

    Extract the image from a sampled image.

    Result Type must be OpTypeImage.

    Sampled Image must have type OpTypeSampledImage whose Image Type is the same as Result Type.
    */
    assert(instruction.opcode == spv::Op::OpImage);

    uint32_t type_id          = instruction.words[1];
    uint32_t result_id        = instruction.words[2];
    uint32_t sampled_image_id = instruction.words[3];

    Value sampled_image = GetValue(sampled_image_id);
    assertm(std::holds_alternative<SampledImageV>(sampled_image), "SPIRV simulator: Input value is not a SampledImage");

    uint64_t result_image = (uint64_t)(std::get<SampledImageV>(sampled_image).image_handle);
    SetValue(result_id, result_image);
}

void SPIRVSimulator::Op_ConvertFToS(const Instruction& instruction)
{
    /*

    OpConvertFToS

    Convert value numerically from floating point to signed integer, with round toward 0.0.

    Result Type must be a scalar or vector of integer type. Behavior is undefined if Result Type is not wide enough to
    hold the converted value.

    Float Value must be a scalar or vector of floating-point type. It must have the same number of components as Result
    Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpConvertFToS);

    uint32_t type_id    = instruction.words[1];
    uint32_t result_id  = instruction.words[2];
    uint32_t operand_id = instruction.words[3];

    Value       operand      = GetValue(operand_id);
    const Type& type         = GetTypeByTypeId(type_id);
    const Type& operand_type = GetTypeByResultId(operand_id);

    if (operand_type.kind == Type::Kind::Vector)
    {
        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand),
                "SPIRV simulator: Operand is set to be vector type, but it is not, illegal input parameters");

        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        auto vec = std::get<std::shared_ptr<VectorV>>(operand);

        assertm((vec->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operand vector length does not match result type");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<double>(vec->elems[i]),
                    "SPIRV simulator: Non-float operand detected in vector operand for Op_ConvertFToS");
            int64_t elem_result = std::trunc(std::get<double>(vec->elems[i]));
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (operand_type.kind == Type::Kind::Float)
    {
        assertm(std::holds_alternative<double>(operand),
                "SPIRV simulator: Non-float operand detected in Op_ConvertFToS");

        int64_t result = std::trunc(std::get<double>(operand));
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or float");
    }

    if (ValueIsArbitrary(operand_id))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_ConvertFToU(const Instruction& instruction)
{
    /*
    OpConvertFToU

    Convert value numerically from floating point to unsigned integer, with round toward 0.0.

    Result Type must be a scalar or vector of integer type, whose Signedness operand is 0.
    Behavior is undefined if Result Type is not wide enough to hold the converted value.

    Float Value must be a scalar or vector of floating-point type. It must have the same number of components as Result
    Type.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpConvertFToU);

    uint32_t type_id    = instruction.words[1];
    uint32_t result_id  = instruction.words[2];
    uint32_t operand_id = instruction.words[3];

    Value       operand      = GetValue(operand_id);
    const Type& type         = GetTypeByTypeId(type_id);
    const Type& operand_type = GetTypeByResultId(operand_id);

    if (operand_type.kind == Type::Kind::Vector)
    {
        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand),
                "SPIRV simulator: Operand is set to be vector type, but it is not, illegal input parameters");

        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        auto vec = std::get<std::shared_ptr<VectorV>>(operand);

        assertm((vec->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operand vector length does not match result type");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<double>(vec->elems[i]),
                    "SPIRV simulator: Non-float operand detected in vector operand for Op_ConvertFToU");
            int64_t elem_result = std::trunc(std::get<double>(vec->elems[i]));
            elem_result         = elem_result < 0 ? 0 : elem_result;
            result_vec->elems.push_back((uint64_t)elem_result);
        }

        SetValue(result_id, result);
    }
    else if (operand_type.kind == Type::Kind::Float)
    {
        assertm(std::holds_alternative<double>(operand),
                "SPIRV simulator: Non-float operand detected in Op_ConvertFToU");

        int64_t result = std::trunc(std::get<double>(operand));
        result         = result < 0 ? 0 : result;
        SetValue(result_id, (uint64_t)result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or float");
    }

    if (ValueIsArbitrary(operand_id))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_FRem(const Instruction& instruction)
{
    /*
    OpFRem

    The floating-point remainder whose sign matches the sign of Operand 1.

    Result Type must be a scalar or vector of floating-point type.

    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component. The resulting value is undefined if Operand 2 is 0.
    Otherwise, the result is the remainder r of Operand 1 divided by Operand 2 where if r ≠ 0,
    the sign of r is the same as the sign of Operand 1.
    */
    assert(instruction.opcode == spv::Op::OpFRem);

    uint32_t type_id      = instruction.words[1];
    uint32_t result_id    = instruction.words[2];
    uint32_t operand_1_id = instruction.words[3];
    uint32_t operand_2_id = instruction.words[4];

    Value       operand_1 = GetValue(operand_1_id);
    Value       operand_2 = GetValue(operand_2_id);
    const Type& type      = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Vector)
    {
        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand_1),
                "SPIRV simulator: First operand is set to be vector type, but it is not, illegal input parameters");
        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand_2),
                "SPIRV simulator: Second operand is set to be vector type, but it is not, illegal input parameters");

        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        auto vec1 = std::get<std::shared_ptr<VectorV>>(operand_1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(operand_2);

        assertm(((vec1->elems.size() == type.vector.elem_count) && (vec1->elems.size() == vec2->elems.size())),
                "SPIRV simulator: Operand vector lengths do not match");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<double>(vec1->elems[i]),
                    "SPIRV simulator: Non-float operand detected in first vector operand for Op_FRem");
            assertm(std::holds_alternative<double>(vec2->elems[i]),
                    "SPIRV simulator: Non-float operand detected in second vector operand for Op_FRem");

            double val_1 = std::get<double>(vec1->elems[i]);
            double val_2 = std::get<double>(vec2->elems[i]);

            double elem_result = std::fmod(val_1, val_2);

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Float)
    {
        assertm(std::holds_alternative<double>(operand_1), "SPIRV simulator: First operand is non-float in Op_FRem");
        assertm(std::holds_alternative<double>(operand_2), "SPIRV simulator: Second operand is non-float in Op_FRem");

        double val_1 = std::get<double>(operand_1);
        double val_2 = std::get<double>(operand_2);

        double result = std::fmod(val_1, val_2);

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or float");
    }

    if (ValueIsArbitrary(operand_1_id) || ValueIsArbitrary(operand_2_id))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_FMod(const Instruction& instruction)
{
    /*
    OpFMod

    The floating-point remainder whose sign matches the sign of Operand 2.

    Result Type must be a scalar or vector of floating-point type.

    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component. The resulting value is undefined if Operand 2 is 0.
    Otherwise, the result is the remainder r of Operand 1 divided by Operand 2 where if r ≠ 0,
    the sign of r is the same as the sign of Operand 2.
    */
    assert(instruction.opcode == spv::Op::OpFMod);

    uint32_t type_id      = instruction.words[1];
    uint32_t result_id    = instruction.words[2];
    uint32_t operand_1_id = instruction.words[3];
    uint32_t operand_2_id = instruction.words[4];

    Value       operand_1 = GetValue(operand_1_id);
    Value       operand_2 = GetValue(operand_2_id);
    const Type& type      = GetTypeByTypeId(type_id);

    if (type.kind == Type::Kind::Vector)
    {
        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand_1),
                "SPIRV simulator: First operand is set to be vector type, but it is not, illegal input parameters");
        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand_2),
                "SPIRV simulator: Second operand is set to be vector type, but it is not, illegal input parameters");

        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        auto vec1 = std::get<std::shared_ptr<VectorV>>(operand_1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(operand_2);

        assertm(((vec1->elems.size() == type.vector.elem_count) && (vec1->elems.size() == vec2->elems.size())),
                "SPIRV simulator: Operand vector lengths do not match");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<double>(vec1->elems[i]),
                    "SPIRV simulator: Non-float operand detected in first vector operand for Op_FMod");
            assertm(std::holds_alternative<double>(vec2->elems[i]),
                    "SPIRV simulator: Non-float operand detected in second vector operand for Op_FMod");

            double val_1 = std::get<double>(vec1->elems[i]);
            double val_2 = std::get<double>(vec2->elems[i]);

            double elem_result = std::fmod(val_1, val_2);

            if ((elem_result != 0.0) && (std::signbit(elem_result) != std::signbit(val_2)))
            {
                elem_result += val_2;
            }

            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Float)
    {
        assertm(std::holds_alternative<double>(operand_1), "SPIRV simulator: First operand is non-float in Op_FMod");
        assertm(std::holds_alternative<double>(operand_2), "SPIRV simulator: Second operand is non-float in Op_FMod");

        double val_1 = std::get<double>(operand_1);
        double val_2 = std::get<double>(operand_2);

        double result = std::fmod(val_1, val_2);

        if ((result != 0.0) && (std::signbit(result) != std::signbit(val_2)))
        {
            result += val_2;
        }

        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or float");
    }

    if (ValueIsArbitrary(operand_1_id) || ValueIsArbitrary(operand_2_id))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_AtomicOr(const Instruction& instruction)
{
    /*
    OpAtomicOr

    Perform the following steps atomically with respect to any other atomic accesses within Memory to the same location:
    1) load through Pointer to get an Original Value,
    2) get a New Value by the bitwise OR of Original Value and Value, and
    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

    The type of Value must be the same as Result Type. The type of the value pointed to by Pointer must be the same as
    Result Type.

    Memory is a memory Scope.
    */
    assert(instruction.opcode == spv::Op::OpAtomicOr);

    uint32_t type_id    = instruction.words[1];
    uint32_t result_id  = instruction.words[2];
    uint32_t pointer_id = instruction.words[3];
    uint32_t scope_id   = instruction.words[4];
    uint32_t sem_id     = instruction.words[5];
    uint32_t value_id   = instruction.words[6];

    const Type&  type        = GetTypeByTypeId(type_id);
    const Value& pointer_val = GetValue(pointer_id);
    const Value& value       = GetValue(value_id);

    assertm(std::holds_alternative<PointerV>(pointer_val),
            "SPIRV simulator: Pointer operand is not a pointer in Op_AtomicOr");
    assertm(type.kind == Type::Kind::Int, "SPIRV simulator: Result type is not int in Op_AtomicOr");

    const PointerV& pointer     = std::get<PointerV>(pointer_val);
    const Value&    pointee_val = ReadPointer(pointer);

    assertm(std::holds_alternative<uint64_t>(pointee_val) || std::holds_alternative<int64_t>(pointee_val),
            "SPIRV simulator: Operand type is not int in Op_AtomicOr");

    SetValue(result_id, pointee_val);

    if (ValueIsArbitrary(pointer_id) || PointeeValueIsArbitrary(pointer))
    {
        SetIsArbitrary(result_id);
    }

    if (std::holds_alternative<uint64_t>(pointee_val))
    {
        Value result = (uint64_t)(std::get<uint64_t>(pointee_val) | std::get<uint64_t>(value));
        WritePointer(pointer, result);
    }
    else
    {
        Value result = (int64_t)(std::get<int64_t>(pointee_val) | std::get<int64_t>(value));
        WritePointer(pointer, result);
    }
}

void SPIRVSimulator::Op_AtomicUMax(const Instruction& instruction)
{
    /*
    OpAtomicUMax

    Perform the following steps atomically with respect to any other atomic accesses within Memory to the same location:
    1) load through Pointer to get an Original Value,
    2) get a New Value by finding the largest unsigned integer of Original Value and Value, and
    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

    The type of Value must be the same as Result Type. The type of the value pointed to by Pointer must be the same as
    Result Type.

    Memory is a memory Scope.
    */
    assert(instruction.opcode == spv::Op::OpAtomicUMax);

    uint32_t type_id    = instruction.words[1];
    uint32_t result_id  = instruction.words[2];
    uint32_t pointer_id = instruction.words[3];
    uint32_t scope_id   = instruction.words[4];
    uint32_t sem_id     = instruction.words[5];
    uint32_t value_id   = instruction.words[6];

    const Type&  type        = GetTypeByTypeId(type_id);
    const Value& pointer_val = GetValue(pointer_id);
    const Value& value       = GetValue(value_id);

    assertm(std::holds_alternative<PointerV>(pointer_val),
            "SPIRV simulator: Pointer operand is not a pointer in Op_AtomicUMax");
    assertm(type.kind == Type::Kind::Int, "SPIRV simulator: Result type is not int in Op_AtomicUMax");

    const PointerV& pointer     = std::get<PointerV>(pointer_val);
    const Value&    pointee_val = ReadPointer(pointer);

    assertm(std::holds_alternative<uint64_t>(pointee_val) || std::holds_alternative<int64_t>(pointee_val),
            "SPIRV simulator: Operand type is not int in Op_AtomicUMax");

    SetValue(result_id, pointee_val);

    if (ValueIsArbitrary(pointer_id) || PointeeValueIsArbitrary(pointer))
    {
        SetIsArbitrary(result_id);
    }

    if (std::holds_alternative<uint64_t>(pointee_val))
    {
        Value result = (uint64_t)std::max(std::get<uint64_t>(pointee_val), std::get<uint64_t>(value));
        WritePointer(pointer, result);
    }
    else
    {
        Value result = (int64_t)std::max(std::get<int64_t>(pointee_val), std::get<int64_t>(value));
        WritePointer(pointer, result);
    }
}

void SPIRVSimulator::Op_AtomicUMin(const Instruction& instruction)
{
    /*

    OpAtomicUMin

    Perform the following steps atomically with respect to any other atomic accesses within Memory to the same location:
    1) load through Pointer to get an Original Value,
    2) get a New Value by finding the smallest unsigned integer of Original Value and Value, and
    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

    The type of Value must be the same as Result Type. The type of the value pointed to by Pointer must be the same as
    Result Type.

    Memory is a memory Scope.
    */
    assert(instruction.opcode == spv::Op::OpAtomicUMin);

    uint32_t type_id    = instruction.words[1];
    uint32_t result_id  = instruction.words[2];
    uint32_t pointer_id = instruction.words[3];
    // uint32_t scope_id      = instruction.words[4];
    // uint32_t sem_id        = instruction.words[5];
    uint32_t value_id = instruction.words[6];

    const Type&  type        = GetTypeByTypeId(type_id);
    const Value& pointer_val = GetValue(pointer_id);
    const Value& value       = GetValue(value_id);

    assertm(std::holds_alternative<PointerV>(pointer_val),
            "SPIRV simulator: Pointer operand is not a pointer in Op_AtomicUMin");
    assertm(type.kind == Type::Kind::Int, "SPIRV simulator: Result type is not int in Op_AtomicUMin");

    const PointerV& pointer     = std::get<PointerV>(pointer_val);
    const Value&    pointee_val = ReadPointer(pointer);

    assertm(std::holds_alternative<uint64_t>(pointee_val) || std::holds_alternative<int64_t>(pointee_val),
            "SPIRV simulator: Operand type is not int in Op_AtomicUMin");

    SetValue(result_id, pointee_val);

    if (ValueIsArbitrary(pointer_id) || PointeeValueIsArbitrary(pointer))
    {
        SetIsArbitrary(result_id);
    }

    if (std::holds_alternative<uint64_t>(pointee_val))
    {
        Value result = (uint64_t)std::min(std::get<uint64_t>(pointee_val), std::get<uint64_t>(value));
        WritePointer(pointer, result);
    }
    else
    {
        Value result = (int64_t)std::min(std::get<int64_t>(pointee_val), std::get<int64_t>(value));
        WritePointer(pointer, result);
    }
}

void SPIRVSimulator::Op_BitReverse(const Instruction& instruction)
{
    /*
    OpBitReverse

    Reverse the bits in an object.

    Results are computed per component.

    Result Type must be a scalar or vector of integer type.

    The type of Base must be the same as Result Type.

    The bit-number n of the result is taken from bit-number Width - 1 - n of Base, where Width is the OpTypeInt operand
    of the Result Type.
    */
    assert(instruction.opcode == spv::Op::OpBitReverse);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t base_id   = instruction.words[3];

    const Type& type = GetTypeByTypeId(type_id);
    assertm(type.kind == Type::Kind::Int, "SPIRV simulator: Non-integer type in Op_BitReverse result type");

    Value operand = GetValue(base_id);

    if (type.kind == Type::Kind::Vector)
    {
        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(operand),
                "SPIRV simulator: Non-vector type found in Op_BitReverse operand");

        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        const auto& vec = std::get<std::shared_ptr<VectorV>>(operand);

        assertm((vec->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operand vector length do not match result type");

        const Type& elem_type = GetTypeByTypeId(type.vector.elem_type_id);

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            assertm(std::holds_alternative<uint64_t>(vec->elems[i]),
                    "SPIRV simulator: Non-integer type in Op_BitReverse operand");

            uint64_t operand_val;
            if (std::holds_alternative<uint64_t>(vec->elems[i]))
            {
                operand_val = std::get<uint64_t>(vec->elems[i]);
            }
            else
            {
                operand_val = bit_cast<uint64_t>(std::get<int64_t>(vec->elems[i]));
            }

            uint64_t elem_result = ReverseBits(operand_val, type.scalar.width);

            if (elem_type.scalar.is_signed)
            {
                result_vec->elems.push_back(bit_cast<int64_t>(elem_result));
            }
            else
            {
                result_vec->elems.push_back(elem_result);
            }
        }

        SetValue(result_id, result);
    }
    else
    {
        assertm(std::holds_alternative<uint64_t>(operand) || std::holds_alternative<int64_t>(operand),
                "SPIRV simulator: Non-integer type in Op_BitReverse operand");

        uint64_t operand_val;
        if (std::holds_alternative<uint64_t>(operand))
        {
            operand_val = std::get<uint64_t>(operand);
        }
        else
        {
            operand_val = bit_cast<uint64_t>(std::get<int64_t>(operand));
        }

        uint64_t result = ReverseBits(operand_val, type.scalar.width);

        if (type.scalar.is_signed)
        {
            SetValue(result_id, bit_cast<int64_t>(result));
        }
        else
        {
            SetValue(result_id, result);
        }
    }

    if (ValueIsArbitrary(base_id))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_BitwiseXor(const Instruction& instruction)
{
    /*
    OpBitwiseXor

    Result is 1 if exactly one of Operand 1 or Operand 2 is 1. Result is 0 if Operand 1 and Operand 2 have the same
    value.

    Results are computed per component, and within each component, per bit.

    Result Type must be a scalar or vector of integer type.
    The type of Operand 1 and Operand 2 must be a scalar or vector of integer type.
    They must have the same number of components as Result Type.
    They must have the same component width as Result Type.
    */
    assert(instruction.opcode == spv::Op::OpBitwiseXor);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t op1_id    = instruction.words[3];
    uint32_t op2_id    = instruction.words[4];

    const Type&  type    = GetTypeByTypeId(type_id);
    const Value& val_op1 = GetValue(op1_id);
    const Value& val_op2 = GetValue(op2_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        assertm(std::holds_alternative<std::shared_ptr<VectorV>>(val_op1) &&
                    std::holds_alternative<std::shared_ptr<VectorV>>(val_op2),
                "SPIRV simulator: Operands set to be vector type, but they are not, illegal input parameters");

        const Type& elem_type = GetTypeByTypeId(type.vector.elem_type_id);

        assertm(elem_type.kind == Type::Kind::Int, "SPIRV simulator: Vector element type is not int in Op_BitwiseXor");

        auto vec1 = std::get<std::shared_ptr<VectorV>>(val_op1);
        auto vec2 = std::get<std::shared_ptr<VectorV>>(val_op2);

        assertm((vec1->elems.size() == vec2->elems.size()) && (vec1->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Operands are vector type but not of equal length");
        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            uint64_t val1;
            if (std::holds_alternative<int64_t>(vec1->elems[i]))
            {
                val1 = bit_cast<uint64_t>(std::get<int64_t>(vec1->elems[i]));
            }
            else
            {
                val1 = std::get<uint64_t>(vec1->elems[i]);
            }

            uint64_t val2;
            if (std::holds_alternative<int64_t>(vec2->elems[i]))
            {
                val2 = bit_cast<uint64_t>(std::get<int64_t>(vec2->elems[i]));
            }
            else
            {
                val2 = std::get<uint64_t>(vec2->elems[i]);
            }

            Value elem_result;
            if (elem_type.scalar.is_signed)
            {
                elem_result = (int64_t)(val1 ^ val2);
            }
            else
            {
                elem_result = (uint64_t)(val1 ^ val2);
            }
            result_vec->elems.push_back(elem_result);
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Int)
    {
        uint64_t val1;
        if (std::holds_alternative<int64_t>(val_op1))
        {
            val1 = bit_cast<uint64_t>(std::get<int64_t>(val_op1));
        }
        else
        {
            val1 = std::get<uint64_t>(val_op1);
        }

        uint64_t val2;
        if (std::holds_alternative<int64_t>(val_op2))
        {
            val2 = bit_cast<uint64_t>(std::get<int64_t>(val_op2));
        }
        else
        {
            val2 = std::get<uint64_t>(val_op2);
        }

        Value result;
        if (type.scalar.is_signed)
        {
            result = (int64_t)(val1 ^ val2);
        }
        else
        {
            result = (uint64_t)(val1 ^ val2);
        }
        SetValue(result_id, result);
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type, must be vector or int");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_ControlBarrier(const Instruction& instruction)
{
    /*
    OpControlBarrier

    Wait for all invocations in the scope restricted tangle to reach the current point of execution before executing
    further instructions.

    Execution is the scope defining the scope restricted tangle affected by this command.

    An invocation will not execute a dynamic instance of this instruction (X') until all invocations in its scope
    restricted tangle have executed all dynamic instances that are program-ordered before X'.

    An invocation will not execute dynamic instances that are program-ordered after a dynamic instance of this
    instruction (X') until all invocations in its scope restricted tangle have executed X'.

    When Execution is Workgroup or larger, behavior is undefined unless all invocations within Execution execute the
    same dynamic instance of this instruction.

    If Semantics is not None, this instruction also serves as an OpMemoryBarrier instruction,
    and also performs and adheres to the description and semantics of an OpMemoryBarrier instruction with the same
    Memory and Semantics operands. This allows atomically specifying both a control barrier and a memory barrier (that
    is, without needing two instructions). If Semantics is None, Memory is ignored.

    Before version 1.3, it is only valid to use this instruction with TessellationControl, GLCompute,
    or Kernel execution models. There is no such restriction starting with version 1.3.

    If used with the TessellationControl execution model, it also implicitly synchronizes the
    Output Storage Class: Writes to Output variables performed by any invocation executed prior to a
    OpControlBarrier are visible to any other invocation proceeding beyond that OpControlBarrier.
    */
    assert(instruction.opcode == spv::Op::OpControlBarrier);

    // This is a nop in our current implementation
}

void SPIRVSimulator::Op_ShiftRightArithmetic(const Instruction& instruction)
{
    /*
    OpShiftRightArithmetic

    Shift the bits in Base right by the number of bits specified in Shift.
    The most-significant bits are filled with the most-significant bit from Base.

    Result Type must be a scalar or vector of integer type.

    The type of each Base and Shift must be a scalar or vector of integer type.
    Base and Shift must have the same number of components.
    The number of components and bit width of the type of Base must be the same as in Result Type.

    Shift is treated as unsigned. The resulting value is undefined if Shift is greater than or equal to the bit
    width of the components of Base.

    Results are computed per component.
    */
    assert(instruction.opcode == spv::Op::OpShiftRightArithmetic);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t base_id   = instruction.words[3];
    uint32_t shift_id  = instruction.words[4];

    const Type&  type      = GetTypeByTypeId(type_id);
    const Type&  base_type = GetTypeByResultId(base_id);
    const Value& base_val  = GetValue(base_id);
    const Value& shift_val = GetValue(shift_id);

    if (type.kind == Type::Kind::Vector)
    {
        Value result     = std::make_shared<VectorV>();
        auto  result_vec = std::get<std::shared_ptr<VectorV>>(result);

        auto vec  = std::get<std::shared_ptr<VectorV>>(base_val);
        auto svec = std::get<std::shared_ptr<VectorV>>(shift_val);

        assertm((vec->elems.size() == type.vector.elem_count) && (svec->elems.size() == type.vector.elem_count),
                "SPIRV simulator: Vector size mismatch in Op_ShiftRightArithmetic");

        const Type& elem_type = GetTypeByTypeId(type.vector.elem_type_id);
        assertm(elem_type.kind == Type::Kind::Int,
                "SPIRV simulator: Element type of vector operand is not int in Op_ShiftRightArithmetic");

        for (uint32_t i = 0; i < type.vector.elem_count; ++i)
        {
            uint64_t shift;
            if (std::holds_alternative<uint64_t>(svec->elems[i]))
            {
                shift = std::get<uint64_t>(svec->elems[i]);
            }
            else
            {
                int64_t s_shift = std::get<int64_t>(svec->elems[i]);
                assertm(s_shift >= 0, "SPIRV simulator: Shift value is less than zero, this is undefined behaviour");
                shift = (uint64_t)s_shift;
            }

            assertm(
                shift <= elem_type.scalar.width,
                "SPIRV simulator: Shift operand is greater than the bit width of base. This is undefined behaviour");

            uint64_t elem_result;
            if (std::holds_alternative<uint64_t>(vec->elems[i]))
            {
                elem_result = std::get<uint64_t>(vec->elems[i]);
            }
            else if (std::holds_alternative<int64_t>(vec->elems[i]))
            {
                elem_result = bit_cast<uint64_t>(std::get<int64_t>(vec->elems[i]));
            }
            else
            {
                assertx("SPIRV simulator: Invalid operand types in Op_ShiftRightArithmetic vector");
            }

            elem_result = ArithmeticRightShiftUnsigned(elem_result, shift, elem_type.scalar.width);

            if (elem_type.scalar.is_signed)
            {
                result_vec->elems.push_back(bit_cast<int64_t>(elem_result));
            }
            else
            {
                result_vec->elems.push_back(elem_result);
            }
        }

        SetValue(result_id, result);
    }
    else if (type.kind == Type::Kind::Int)
    {
        uint64_t shift;
        if (std::holds_alternative<uint64_t>(shift_val))
        {
            shift = std::get<uint64_t>(shift_val);
        }
        else
        {
            int64_t s_shift = std::get<int64_t>(shift_val);
            assertm(s_shift >= 0, "SPIRV simulator: Shift value is less than zero, this is undefined behaviour");
            shift = (uint64_t)s_shift;
        }

        assertm(shift <= base_type.scalar.width,
                "SPIRV simulator: Shift operand is greater than the bit width of base. This is undefined behaviour");

        uint64_t result;
        if (std::holds_alternative<uint64_t>(base_val))
        {
            result = ArithmeticRightShiftUnsigned(std::get<uint64_t>(base_val), shift, base_type.scalar.width);
        }
        else if (std::holds_alternative<int64_t>(base_val))
        {
            result = ArithmeticRightShiftUnsigned(
                bit_cast<uint64_t>(std::get<int64_t>(base_val)), shift, base_type.scalar.width);
        }
        else
        {
            assertx("SPIRV simulator: Invalid operand types in Op_ShiftRightArithmetic");
        }

        if (type.scalar.is_signed)
        {
            SetValue(result_id, bit_cast<int64_t>(result));
        }
        else
        {
            SetValue(result_id, result);
        }
    }
    else
    {
        assertx("SPIRV simulator: Invalid result type in Op_ShiftRightArithmetic, must be vector or int");
    }

    if (ValueIsArbitrary(instruction.words[3]) || ValueIsArbitrary(instruction.words[4]))
    {
        SetIsArbitrary(result_id);
    }
}

void SPIRVSimulator::Op_GroupNonUniformAll(const Instruction& instruction)
{
    /*
    OpGroupNonUniformAll

    Evaluates a predicate for all tangled invocations within the Execution scope,
    resulting in true if predicate evaluates to true for all tangled invocations within the Execution scope, otherwise
    the result is false.

    Result Type must be a Boolean type.

    Execution is the scope defining the scope restricted tangle affected by this command. It must be Subgroup.

    Predicate must be a Boolean type.

    An invocation will not execute a dynamic instance of this instruction (X') until all
    invocations in its scope restricted tangle have executed all dynamic instances that are program-ordered before X'.
    */
    assert(instruction.opcode == spv::Op::OpGroupNonUniformAll);

    uint32_t type_id      = instruction.words[1];
    uint32_t result_id    = instruction.words[2];
    uint32_t exec_id      = instruction.words[3];
    uint32_t predicate_id = instruction.words[4];

    // TODO: Group op warnings

    SetIsArbitrary(result_id);
    SetValue(result_id, GetValue(predicate_id));
}

void SPIRVSimulator::Op_GroupNonUniformAny(const Instruction& instruction)
{
    /*
    OpGroupNonUniformAny

    Evaluates a predicate for all tangled invocations within the Execution scope, resulting in
    true if predicate evaluates to true for any tangled invocations within the Execution scope, otherwise the result is
    false.

    Result Type must be a Boolean type.

    Execution is the scope defining the scope restricted tangle affected by this command. It must be Subgroup.

    Predicate must be a Boolean type.

    An invocation will not execute a dynamic instance of this instruction (X') until all invocations in its scope
    restricted tangle have executed all dynamic instances that are program-ordered before X'.
    */
    assert(instruction.opcode == spv::Op::OpGroupNonUniformAny);

    uint32_t type_id      = instruction.words[1];
    uint32_t result_id    = instruction.words[2];
    uint32_t exec_id      = instruction.words[3];
    uint32_t predicate_id = instruction.words[4];

    // TODO: Group op warnings

    SetIsArbitrary(result_id);
    SetValue(result_id, GetValue(predicate_id));
}

void SPIRVSimulator::Op_GroupNonUniformBallot(const Instruction& instruction)
{
    /*
    OpGroupNonUniformBallot

    Result is a bitfield value combining the Predicate value from all tangled invocations within the Execution scope
    that execute the same dynamic instance of this instruction. The bit is set to 1 if the corresponding invocation is
    part of the tangled invocations within the Execution scope and the Predicate for that invocation evaluated to true;
    otherwise, it is set to 0.

    Result Type must be a vector of four components of integer type scalar, whose Width operand is 32 and whose
    Signedness operand is 0.

    Result is a set of bitfields where the first invocation is represented in the lowest bit of the first vector
    component and the last (up to the size of the scope) is the higher bit number of the last bitmask needed to
    represent all bits of the invocations in the scope restricted tangle.

    Execution is the scope defining the scope restricted tangle affected by this command.

    Predicate must be a Boolean type.

    An invocation will not execute a dynamic instance of this instruction (X') until all invocations in its scope
    restricted tangle have executed all dynamic instances that are program-ordered before X'.
    */
    assert(instruction.opcode == spv::Op::OpGroupNonUniformBallot);

    uint32_t type_id      = instruction.words[1];
    uint32_t result_id    = instruction.words[2];
    uint32_t exec_id      = instruction.words[3];
    uint32_t predicate_id = instruction.words[4];

    // TODO: Group op warnings

    const Type& type = GetTypeByTypeId(type_id);
    assertm(type.kind == Type::Kind::Vector, "SPIRV simulator: Op_GroupNonUniformBallot output must be a vector");
    assertm(type.vector.elem_count == 4,
            "SPIRV simulator: Op_GroupNonUniformBallot output vector must have 4 elements");

    const Type& elem_type = GetTypeByTypeId(type.vector.elem_type_id);
    assertm(elem_type.kind == Type::Kind::Int,
            "SPIRV simulator: Op_GroupNonUniformBallot output vector element type must be int");
    assertm(!elem_type.scalar.is_signed,
            "SPIRV simulator: Op_GroupNonUniformBallot output vector element type must be unsigned");
    assertm(elem_type.scalar.width == 32,
            "SPIRV simulator: Op_GroupNonUniformBallot output vector element type have 32 bit width");

    const Value& predicate_val = GetValue(predicate_id);
    assertm(std::holds_alternative<uint64_t>(predicate_val), "SPIRV simulator: Invalid type for boolean predicate");

    Value result  = MakeDefault(type_id);
    auto& vec     = std::get<std::shared_ptr<VectorV>>(result);
    vec->elems[3] = std::get<uint64_t>(predicate_val);

    SetValue(result_id, result);
    SetIsArbitrary(result_id);
}

void SPIRVSimulator::Op_GroupNonUniformBallotBitCount(const Instruction& instruction)
{
    /*
    OpGroupNonUniformBallotBitCount

    Result is the number of bits that are set to 1 in Value, considering only the bits in Value required to represent
    all bits of the scope restricted tangle.

    Result Type must be a scalar of integer type, whose Signedness operand is 0.

    Execution is the scope defining the scope restricted tangle affected by this command. It must be Subgroup.

    The identity I for Operation is 0.

    Value must be a vector of four components of integer type scalar, whose Width operand is 32 and whose Signedness
    operand is 0.

    Value is a set of bitfields where the first invocation is represented in the lowest bit of the first vector
    component and the last (up to the size of the scope) is the higher bit number of the last bitmask needed to
    represent all bits of the invocations in the scope restricted tangle.

    An invocation will not execute a dynamic instance of this instruction (X') until all invocations in its scope
    restricted tangle have executed all dynamic instances that are program-ordered before X'.
    */
    assert(instruction.opcode == spv::Op::OpGroupNonUniformBallotBitCount);

    uint32_t type_id     = instruction.words[1];
    uint32_t result_id   = instruction.words[2];
    uint32_t exec_id     = instruction.words[3];
    uint32_t group_op_id = instruction.words[4];
    uint32_t value_id    = instruction.words[5];

    // TODO: Group op warnings

    const Value& value    = GetValue(value_id);
    const Type&  type     = GetTypeByTypeId(type_id);
    const Type&  val_type = GetTypeByResultId(value_id);

    bool arb_count = false;
    SetValue(result_id, (uint64_t)CountSetBits(value, GetTypeID(value_id), &arb_count));
    SetIsArbitrary(result_id);
}

void SPIRVSimulator::Op_GroupNonUniformBroadcastFirst(const Instruction& instruction)
{
    /*
    OpGroupNonUniformBroadcastFirst

    Result is the Value of the invocation from the tangled invocations with the lowest id within the Execution scope
    to all tangled invocations within the Execution scope.

    Result Type must be a scalar or vector of floating-point type, integer type, or Boolean type.

    Execution is the scope defining the scope restricted tangle affected by this command. It must be Subgroup.

    The type of Value must be the same as Result Type.

    An invocation will not execute a dynamic instance of this instruction (X') until all invocations in its scope
    restricted tangle have executed all dynamic instances that are program-ordered before X'.
    */
    assert(instruction.opcode == spv::Op::OpGroupNonUniformBroadcastFirst);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t exec_id   = instruction.words[3];
    uint32_t value_id  = instruction.words[4];

    // TODO: Group op warnings

    SetIsArbitrary(result_id);
    SetValue(result_id, GetValue(value_id));
}

void SPIRVSimulator::Op_GroupNonUniformElect(const Instruction& instruction)
{
    /*
    OpGroupNonUniformElect

    Result is true only in the tangled invocation with the lowest id within the Execution scope, otherwise result is
    false.

    Result Type must be a Boolean type.

    Execution is the scope defining the scope restricted tangle affected by this command. It must be Subgroup.

    An invocation will not execute a dynamic instance of this instruction (X') until all invocations in its scope
    restricted tangle have executed all dynamic instances that are program-ordered before X'.
    */
    assert(instruction.opcode == spv::Op::OpGroupNonUniformElect);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t exec_id   = instruction.words[3];
    uint32_t value_id  = instruction.words[4];

    // TODO: Group op warnings

    SetIsArbitrary(result_id);
    SetValue(result_id, (uint64_t)1);
}

void SPIRVSimulator::Op_GroupNonUniformFMax(const Instruction& instruction)
{
    /*
    OpGroupNonUniformFMax

    A floating point maximum group operation of all Value operands contributed by all tangled invocations within the
    Execution scope.

    Result Type must be a scalar or vector of floating-point type.

    Execution is the scope defining the scope restricted tangle affected by this command. It must be Subgroup.

    The identity I for Operation is -INF. If Operation is ClusteredReduce, ClusterSize must be present.

    The type of Value must be the same as Result Type. The method used to perform the group operation on the contributed
    Value(s) from the tangled invocations is implementation defined. From the set of Value(s) provided by the tangled
    invocations within a subgroup, if for any two Values one of them is a NaN, the other is chosen. If all Value(s) that
    are used by the current invocation are NaN, then the result is an undefined value.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar of integer type, whose Signedness operand is
    0. ClusterSize must come from a constant instruction. Behavior is undefined unless ClusterSize is at least 1 and a
    power of 2. If ClusterSize is greater than the size of the scope, executing this instruction results in undefined
    behavior.

    An invocation will not execute a dynamic instance of this instruction (X') until all invocations in its
    scope restricted tangle have executed all dynamic instances that are program-ordered before X'.
    */
    assert(instruction.opcode == spv::Op::OpGroupNonUniformFMax);

    uint32_t type_id        = instruction.words[1];
    uint32_t result_id      = instruction.words[2];
    uint32_t exec_id        = instruction.words[3];
    uint32_t Operation      = instruction.words[4];
    uint32_t value_id       = instruction.words[5];
    uint32_t clustersize_id = instruction.words[6];

    // TODO: Group op warnings

    SetIsArbitrary(result_id);
    SetValue(result_id, GetValue(value_id));
}

void SPIRVSimulator::Op_GroupNonUniformFMin(const Instruction& instruction)
{
    /*
    OpGroupNonUniformFMin

    A floating point minimum group operation of all Value operands contributed by all tangled invocations within the
    Execution scope.

    Result Type must be a scalar or vector of floating-point type.

    Execution is the scope defining the scope restricted tangle affected by this command. It must be Subgroup.

    The identity I for Operation is +INF. If Operation is ClusteredReduce, ClusterSize must be present.

    The type of Value must be the same as Result Type. The method used to perform the group operation on the contributed
    Value(s) from the tangled invocations is implementation defined. From the set of Value(s) provided by the tangled
    invocations within a subgroup, if for any two Values one of them is a NaN, the other is chosen. If all Value(s) that
    are used by the current invocation are NaN, then the result is an undefined value.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar of integer type, whose Signedness operand is
    0. ClusterSize must come from a constant instruction. Behavior is undefined unless ClusterSize is at least 1 and a
    power of 2. If ClusterSize is greater than the size of the scope, executing this instruction results in undefined
    behavior.

    An invocation will not execute a dynamic instance of this instruction (X') until all invocations in its scope
    restricted tangle have executed all dynamic instances that are program-ordered before X'.
    */
    assert(instruction.opcode == spv::Op::OpGroupNonUniformFMin);

    uint32_t type_id        = instruction.words[1];
    uint32_t result_id      = instruction.words[2];
    uint32_t exec_id        = instruction.words[3];
    uint32_t Operation      = instruction.words[4];
    uint32_t value_id       = instruction.words[5];
    uint32_t clustersize_id = instruction.words[6];

    // TODO: Group op warnings

    SetIsArbitrary(result_id);
    SetValue(result_id, GetValue(value_id));
}

void SPIRVSimulator::Op_GroupNonUniformIAdd(const Instruction& instruction)
{
    /*
    OpGroupNonUniformIAdd

    An integer add group operation of all Value operands contributed by all tangled invocations within the Execution
    scope.

    Result Type must be a scalar or vector of integer type.

    Execution is the scope defining the scope restricted tangle affected by this command. It must be Subgroup.

    The identity I for Operation is 0. If Operation is ClusteredReduce, ClusterSize must be present.

    The type of Value must be the same as Result Type.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar of integer type, whose Signedness operand is
    0. ClusterSize must come from a constant instruction. Behavior is undefined unless ClusterSize is at least 1 and a
    power of 2. If ClusterSize is greater than the size of the scope, executing this instruction results in undefined
    behavior.

    An invocation will not execute a dynamic instance of this instruction (X') until all invocations in its scope
    restricted tangle have executed all dynamic instances that are program-ordered before X'.
    */
    assert(instruction.opcode == spv::Op::OpGroupNonUniformIAdd);

    uint32_t type_id        = instruction.words[1];
    uint32_t result_id      = instruction.words[2];
    uint32_t exec_id        = instruction.words[3];
    uint32_t Operation      = instruction.words[4];
    uint32_t value_id       = instruction.words[5];
    uint32_t clustersize_id = instruction.words[6];

    // TODO: Group op warnings

    SetIsArbitrary(result_id);
    SetValue(result_id, GetValue(value_id));
}

void SPIRVSimulator::Op_GroupNonUniformShuffle(const Instruction& instruction)
{
    /*
    OpGroupNonUniformShuffle

    Result is the Value of the invocation identified by the id Id.

    Result Type must be a scalar or vector of floating-point type, integer type, or Boolean type.

    Execution is the scope defining the scope restricted tangle affected by this command.

    The type of Value must be the same as Result Type.

    Id must be a scalar of integer type, whose Signedness operand is 0.

    The resulting value is undefined if Id is not part of the scope restricted tangle, or is greater than or equal to
    the size of the scope.

    An invocation will not execute a dynamic instance of this instruction (X') until all invocations in its scope
    restricted tangle have executed all dynamic instances that are program-ordered before X'.
    */
    assert(instruction.opcode == spv::Op::OpGroupNonUniformShuffle);

    uint32_t type_id   = instruction.words[1];
    uint32_t result_id = instruction.words[2];
    uint32_t exec_id   = instruction.words[3];
    uint32_t value_id  = instruction.words[4];
    uint32_t id_id     = instruction.words[5];

    // TODO: Group op warnings

    SetIsArbitrary(result_id);
    SetValue(result_id, GetValue(value_id));
}

void SPIRVSimulator::Op_GroupNonUniformUMax(const Instruction& instruction)
{
    /*
    OpGroupNonUniformUMax

    An unsigned integer maximum group operation of all Value operands contributed by all tangled invocations within the
    Execution scope.

    Result Type must be a scalar or vector of integer type, whose Signedness operand is 0.

    Execution is the scope defining the scope restricted tangle affected by this command. It must be Subgroup.

    The identity I for Operation is 0. If Operation is ClusteredReduce, ClusterSize must be present.

    The type of Value must be the same as Result Type.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar of integer type, whose Signedness operand is
    0. ClusterSize must come from a constant instruction. Behavior is undefined unless ClusterSize is at least 1 and a
    power of 2. If ClusterSize is greater than the size of the scope, executing this instruction results in undefined
    behavior.

    An invocation will not execute a dynamic instance of this instruction (X') until all invocations in its scope
    restricted tangle have executed all dynamic instances that are program-ordered before X'.
    */
    assert(instruction.opcode == spv::Op::OpGroupNonUniformUMax);

    uint32_t type_id        = instruction.words[1];
    uint32_t result_id      = instruction.words[2];
    uint32_t exec_id        = instruction.words[3];
    uint32_t Operation      = instruction.words[4];
    uint32_t value_id       = instruction.words[5];
    uint32_t clustersize_id = instruction.words[6];

    // TODO: Group op warnings

    SetIsArbitrary(result_id);
    SetValue(result_id, GetValue(value_id));
}

void SPIRVSimulator::Op_RayQueryGetIntersectionBarycentricsKHR(const Instruction& instruction)
{
    /*
    OpRayQueryGetIntersectionBarycentricsKHR

    Reserved.
    */
    assert(instruction.opcode == spv::Op::OpRayQueryGetIntersectionBarycentricsKHR);

    uint32_t type_id         = instruction.words[1];
    uint32_t result_id       = instruction.words[2];
    uint32_t ray_query_id    = instruction.words[3];
    uint32_t intersection_id = instruction.words[4];

    if (verbose_)
    {
        std::cout << "SPIRV simulator: Ray Op_RayQueryGetIntersectionBarycentricsKHR is pass-through, creating "
                     "arbitrary dummy value"
                  << std::endl;
    }

    Value result = MakeDefault(type_id);

    SetIsArbitrary(result_id);
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_RayQueryGetIntersectionFrontFaceKHR(const Instruction& instruction)
{
    /*
    OpRayQueryGetIntersectionFrontFaceKHR

    Reserved.
    */
    assert(instruction.opcode == spv::Op::OpRayQueryGetIntersectionFrontFaceKHR);

    uint32_t type_id         = instruction.words[1];
    uint32_t result_id       = instruction.words[2];
    uint32_t ray_query_id    = instruction.words[3];
    uint32_t intersection_id = instruction.words[4];

    if (verbose_)
    {
        std::cout << "SPIRV simulator: Ray Op_RayQueryGetIntersectionFrontFaceKHR is pass-through, creating arbitrary "
                     "dummy value"
                  << std::endl;
    }

    Value result = MakeDefault(type_id);

    SetIsArbitrary(result_id);
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_RayQueryGetIntersectionGeometryIndexKHR(const Instruction& instruction)
{
    /*
    OpRayQueryGetIntersectionGeometryIndexKHR

    Reserved.
    */
    assert(instruction.opcode == spv::Op::OpRayQueryGetIntersectionGeometryIndexKHR);

    uint32_t type_id         = instruction.words[1];
    uint32_t result_id       = instruction.words[2];
    uint32_t ray_query_id    = instruction.words[3];
    uint32_t intersection_id = instruction.words[4];

    if (verbose_)
    {
        std::cout << "SPIRV simulator: Ray Op_RayQueryGetIntersectionGeometryIndexKHR is pass-through, creating "
                     "arbitrary dummy value"
                  << std::endl;
    }

    Value result = MakeDefault(type_id);

    SetIsArbitrary(result_id);
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_RayQueryGetIntersectionInstanceCustomIndexKHR(const Instruction& instruction)
{
    /*
    OpRayQueryGetIntersectionInstanceCustomIndexKHR

    Reserved.
    */
    assert(instruction.opcode == spv::Op::OpRayQueryGetIntersectionInstanceCustomIndexKHR);

    uint32_t type_id         = instruction.words[1];
    uint32_t result_id       = instruction.words[2];
    uint32_t ray_query_id    = instruction.words[3];
    uint32_t intersection_id = instruction.words[4];

    if (verbose_)
    {
        std::cout << "SPIRV simulator: Ray Op_RayQueryGetIntersectionInstanceCustomIndexKHR is pass-through, creating "
                     "arbitrary dummy value"
                  << std::endl;
    }

    Value result = MakeDefault(type_id);

    SetIsArbitrary(result_id);
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_RayQueryGetIntersectionInstanceIdKHR(const Instruction& instruction)
{
    /*
    OpRayQueryGetIntersectionInstanceIdKHR

    Reserved.
    */
    assert(instruction.opcode == spv::Op::OpRayQueryGetIntersectionInstanceIdKHR);

    uint32_t type_id         = instruction.words[1];
    uint32_t result_id       = instruction.words[2];
    uint32_t ray_query_id    = instruction.words[3];
    uint32_t intersection_id = instruction.words[4];

    if (verbose_)
    {
        std::cout << "SPIRV simulator: Ray Op_RayQueryGetIntersectionInstanceIdKHR is pass-through, creating arbitrary "
                     "dummy value"
                  << std::endl;
    }

    Value result = MakeDefault(type_id);

    SetIsArbitrary(result_id);
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_RayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR(const Instruction& instruction)
{
    /*
    OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR

    Reserved.
    */
    assert(instruction.opcode == spv::Op::OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR);

    uint32_t type_id         = instruction.words[1];
    uint32_t result_id       = instruction.words[2];
    uint32_t ray_query_id    = instruction.words[3];
    uint32_t intersection_id = instruction.words[4];

    if (verbose_)
    {
        std::cout << "SPIRV simulator: Ray Op_RayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR is "
                     "pass-through, creating arbitrary dummy value"
                  << std::endl;
    }

    Value result = MakeDefault(type_id);

    SetIsArbitrary(result_id);
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_RayQueryGetIntersectionPrimitiveIndexKHR(const Instruction& instruction)
{
    /*
    OpRayQueryGetIntersectionPrimitiveIndexKHR

    Reserved.
    */
    assert(instruction.opcode == spv::Op::OpRayQueryGetIntersectionPrimitiveIndexKHR);

    uint32_t type_id         = instruction.words[1];
    uint32_t result_id       = instruction.words[2];
    uint32_t ray_query_id    = instruction.words[3];
    uint32_t intersection_id = instruction.words[4];

    if (verbose_)
    {
        std::cout << "SPIRV simulator: Ray Op_RayQueryGetIntersectionPrimitiveIndexKHR is pass-through, creating "
                     "arbitrary dummy value"
                  << std::endl;
    }

    Value result = MakeDefault(type_id);

    SetIsArbitrary(result_id);
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_RayQueryGetIntersectionTKHR(const Instruction& instruction)
{
    /*
    OpRayQueryGetIntersectionTKHR

    Reserved.
    */
    assert(instruction.opcode == spv::Op::OpRayQueryGetIntersectionTKHR);

    uint32_t type_id         = instruction.words[1];
    uint32_t result_id       = instruction.words[2];
    uint32_t ray_query_id    = instruction.words[3];
    uint32_t intersection_id = instruction.words[4];

    if (verbose_)
    {
        std::cout
            << "SPIRV simulator: Ray Op_RayQueryGetIntersectionTKHR is pass-through, creating arbitrary dummy value"
            << std::endl;
    }

    Value result = MakeDefault(type_id);

    SetIsArbitrary(result_id);
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_RayQueryGetIntersectionTypeKHR(const Instruction& instruction)
{
    /*
    OpRayQueryGetIntersectionTypeKHR

    Reserved.
    */
    assert(instruction.opcode == spv::Op::OpRayQueryGetIntersectionTypeKHR);

    uint32_t type_id         = instruction.words[1];
    uint32_t result_id       = instruction.words[2];
    uint32_t ray_query_id    = instruction.words[3];
    uint32_t intersection_id = instruction.words[4];

    if (verbose_)
    {
        std::cout
            << "SPIRV simulator: Ray Op_RayQueryGetIntersectionTypeKHR is pass-through, creating arbitrary dummy value"
            << std::endl;
    }

    Value result = MakeDefault(type_id);

    SetIsArbitrary(result_id);
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_RayQueryGetIntersectionWorldToObjectKHR(const Instruction& instruction)
{
    /*
    OpRayQueryGetIntersectionWorldToObjectKHR

    Reserved.
    */
    assert(instruction.opcode == spv::Op::OpRayQueryGetIntersectionWorldToObjectKHR);

    uint32_t type_id         = instruction.words[1];
    uint32_t result_id       = instruction.words[2];
    uint32_t ray_query_id    = instruction.words[3];
    uint32_t intersection_id = instruction.words[4];

    if (verbose_)
    {
        std::cout << "SPIRV simulator: Ray Op_RayQueryGetIntersectionWorldToObjectKHR is pass-through, creating "
                     "arbitrary dummy value"
                  << std::endl;
    }

    Value result = MakeDefault(type_id);

    SetIsArbitrary(result_id);
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_RayQueryGetWorldRayDirectionKHR(const Instruction& instruction)
{
    /*
    OpRayQueryGetWorldRayDirectionKHR

    Reserved.
    */
    assert(instruction.opcode == spv::Op::OpRayQueryGetWorldRayDirectionKHR);

    uint32_t type_id      = instruction.words[1];
    uint32_t result_id    = instruction.words[2];
    uint32_t ray_query_id = instruction.words[3];

    if (verbose_)
    {
        std::cout
            << "SPIRV simulator: Ray Op_RayQueryGetWorldRayDirectionKHR is pass-through, creating arbitrary dummy value"
            << std::endl;
    }

    Value result = MakeDefault(type_id);

    SetIsArbitrary(result_id);
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_RayQueryInitializeKHR(const Instruction& instruction)
{
    /*
    OpRayQueryInitializeKHR

    Reserved.
    */
    assert(instruction.opcode == spv::Op::OpRayQueryInitializeKHR);

    if (verbose_)
    {
        std::cout << "SPIRV simulator: Ray Op_RayQueryInitializeKHR is pass-through, treating as NOP" << std::endl;
    }
}

void SPIRVSimulator::Op_RayQueryProceedKHR(const Instruction& instruction)
{
    /*
    OpRayQueryProceedKHR

    Reserved.
    */
    assert(instruction.opcode == spv::Op::OpRayQueryProceedKHR);

    uint32_t type_id      = instruction.words[1];
    uint32_t result_id    = instruction.words[2];
    uint32_t ray_query_id = instruction.words[3];

    if (verbose_)
    {
        std::cout << "SPIRV simulator: Ray Op_RayQueryProceedKHR is pass-through, creating arbitrary dummy value"
                  << std::endl;
    }

    Value result = MakeDefault(type_id);

    SetIsArbitrary(result_id);
    SetValue(result_id, result);
}

void SPIRVSimulator::Op_DecorateString(const Instruction& instruction)
{
    /*
    OpDecorateString (OpDecorateStringGOOGLE)

    Add a string Decoration to another <id>.

    Target is the <id> to decorate. It can potentially be any <id> that is a forward reference,
    except it must not be the <id> of an OpDecorationGroup.

    Decoration is a decoration that takes at least one Literal operand, and has only Literal string operands.
    */
    assert(instruction.opcode == spv::Op::OpDecorateString);

    uint32_t type_id    = instruction.words[1];
    uint32_t decoration = instruction.words[2];
    uint32_t literal    = instruction.words[3];
    // uint32_t literal_opt       = instruction.words[4];

    // This is currently a nop, but can be used for debugging later
}

} // namespace SPIRVSimulator
