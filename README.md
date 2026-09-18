# spirv-simulator

This repo implements a SPIR-V simulator that can be used to detect and mark resource types in memory that needs to be
tracked for special handling from API tools.

**Still in very early stage development -- please do not use unless you have an ongoing dialogue with the main
developers of the tool.**

Types we currently track:

- device memory addresses (pointers)
- shader group handles
- opaque descriptors
- offsets to and sizes of any of the above

The intended use case is for it to be integrated with tracing and post processing graphics API inspection tools to find
and handle these types difficult to track resources that may vary between runs and platforms.

It works by taking SPIR-V binary code and host side mappings as an inputs alongside origin provenance data on all such
mappings. It then simulates relevant portions of command execution and detect where types in need of tracking are
consumed, which gives us their location and type, then correlate this with their origin provenance data, which gives us
from where the data was originally provided from the host. This allows us to modify host updates so that we can replace
the types, eg swap out captured pointers with replay pointers.

It also populates output structures with relevant data that can be used to track persistent state across multiple
passes/draw calls/compute dispatches. We consider data relevant if it _could_ be used to construct, populate or fetch
any tracked type. We try to skip the computation of any data that is not relevant in this way to speed up the
simulation.

The simulator tries very hard to only do the minimum amount of computation and tracking that is needed to accomplish our
stated goals above. We try to unroll all loops, take all branches, and skip anything that ends up in data types that are
not likely to contain the tracked types, such as images.

Read [Terminology.md](Terminology.md) for terminology list.

## Dependencies

Requires a C++ 20 compatible compiler. g++-13 or clang-19 or newer should do the trick.

To build:

```sh
git submodule update --init --recursive
cmake -S . -B build
cmake --build build
```

To run tests:

```sh
ctest --test-dir build
```

To use a system-installed SPIR-V header instead of the bundled header, define the `SPIRV_HEADERS_PRESENT` preprocessor
macro when configuring the build:

```sh
cmake -S . -B build -DCMAKE_CXX_FLAGS="-DSPIRV_HEADERS_PRESENT=1"
```

## Execution framework

The main files of interest are `framework/spirv_simulator.hpp` and `framework/spirv_simulator.cpp`, they contain all the
relevant code.

The main framework is implemented in the `SPIRVSimulator` class and the `SimulationData` structure.

It can be used as follows:

```cpp
#include <spirv_simulator.hpp>
#include <util.hpp>

std::string spirv_filepath = "myshader.spirv";
SPIRVSimulator::MemoryFlagTracker memory_flag_tracker;
SPIRVSimulator::SimulationData simulation_data;
SPIRVSimulator::SimulationResults simulation_results;
bool verbose = true;
// Populate simulation_data and memory_flag_tracker here.

SPIRVSimulator::SPIRVSimulator sim(
    util::ReadFile(spirv_filepath),
    &memory_flag_tracker,
    &simulation_data,
    &simulation_results,
    nullptr,
    verbose,
    ERROR_PRINT_CONTEXT);
sim.Run();

const auto& physical_address_data = simulation_results.physical_address_data;
// Work with the outputs here.
```

### Populating the inputs

The input structure has the following format:

```cpp
struct SimulationData
{
    uint32_t entry_point_id = 0;
    std::string entry_point_op_name = "";

    UnorderedMap<uint64_t, UnorderedMap<size_t, size_t>> rt_array_lengths;
    UnorderedMap<uint32_t, size_t> specialization_constant_offsets;
    const void* specialization_constants = nullptr;
    const void* push_constants = nullptr;
    UnorderedMap<uint64_t, UnorderedMap<uint64_t, void*>> bindings;
    UnorderedMap<uint64_t, std::pair<size_t, void*>> physical_address_buffers;
    UnorderedMap<const void*, std::vector<DescriptorCandidate>> descriptor_candidates;

    bool has_compute_num_workgroups = false;
    ComputeDispatchDimensions compute_num_workgroups;
    ComputeDispatchDimensions base_workgroups = { 0, 0, 0 };
    ComputeLocalSize compute_local_size;

    uint64_t shader_id = UINT64_MAX;
};
```

The format of the data must be what the shader expects, eg. if a buffer is bound to a binding with the std430 layout,
the supplied host data block must obey the rules of that layout.

And each shader input should be mapped to a compatible member in the input structure.

`entry_point_op_name` should be set to the name of the entry point in the shader.

`entry_point_id` is optional and is ignored when `entry_point_op_name` is set. It is the entry-point function ID operand
of an `OpEntryPoint` instruction.

`specialization_constants` should be set to a pointer to the full spec constant block.

`specialization_constant_offsets` should contain one entry per spec constant in the shader. The key should be the SpecId
matching the specialization constant ID as defined by the API. The offset value is the offset (in bytes) into the
specialization_constants pointer where the given spec constant value can be found.

`push_constants` should be set to a pointer to the full push constant block.

`bindings` should contain a key for each descriptorset ID, then for each key one map value mapping binding ID's to
pointers that point to the full host side data block for the given binding.

`physical_address_buffers` should contain a `uint64_t` key holding the GPU address at which the buffer is visible to the
shader. The value should be a pair, where the first entry is the size of the buffer in bytes, and the second value is a
pointer to the host side memory containing all the values in the buffer.

`rt_array_lengths` should contain a `uint64_t` outer key holding the bit-cast host pointer of a data block supplied
through `bindings`, `push_constants`, `specialization_constants`, or `physical_address_buffers`. Each value is a map
from the byte offset of a runtime array within that block to the array length in elements.

## Framework details

The main framework is based around 3 structures:

- `Value`: This encapsulates all SPIR-V valid type values.
- `Type`: Describes the type details of a SPIR-V value. All values have an associated Type.
- `Instruction`: Encapsulates a decoded instruction.

And these access member functions:

- `GetValue(<ID>)`: Fetches the Value associated with a SPIR-V result ID.
- `SetValue(<ID>, Value)`: Writes a Value to the specified SPIR-V result ID.
- `ReadPointer(PointerV)`: Reads through a framework pointer and returns an optional Value.
- `WritePointer(PointerV, Value)`: Writes a Value through a framework pointer and reports whether execution can
  continue.

The Value structure is essentially just a C++ variant, the underlying value can be queried with the standard C++ variant
functionality, eg:

```cpp
SPIRVSimulator::Value value = uint64_t{42};
if (std::holds_alternative<uint64_t>(value))
{
    uint64_t scalar = std::get<uint64_t>(value);
}
```

The `Type` structure holds 2 member variables, one is a Enum describing the kind of type it represents, the other is a
union with metadata for the given type.

The `Instruction` struct encapsulates decoded instructions. It contains a span of all the words in the instruction, plus
the opcode and word count in a decoded, easy-to-access format.

You can fetch the opcode and operands from the `Instruction` instance passed to a handler after adding the corresponding
case to `SPIRVSimulator::ExecuteInstruction`.

The framework uses the result ID's of every instruction that has a result as access handles to the data they returned.

You can get and set the results tied to a result ID by using `GetValue(result_id)` or `SetValue(result_id, value)`.

Scopes and access handling is taken care of by the framework.

For accessing allocated data through pointers, use `ReadPointer(pointer)` and `WritePointer(pointer, value)`.

## Adding support for more opcodes

1. Add a member function to the `SPIRVSimulator` class that takes an `Instruction` instance as a parameter, for example:
   `void Op_FAdd(const Instruction&);`

1. Add the opcode to the switch in `SPIRVSimulator::ExecuteInstruction`, for example:

   ```cpp
   case spv::Op::OpFAdd:
       R(Op_FAdd)
   ```

Then you should implement the member function so that it performs/simulates the operations performed by the SPIR-V
instruction matching the given OpCode.

If the instruction has a result ID, then it needs to write the result Value to the given result ID using
`SetValue(...)`.

If the instruction reads or writes through pointers, it needs to use `ReadPointer(...)` or `WritePointer(...)` to access
the correct backing storage (see `SPIRVSimulator::Op_Load` and `SPIRVSimulator::Op_Store` in
`framework/spirv_simulator.cpp` for examples).

See the existing opcode declarations in `framework/spirv_simulator.hpp` and implementations in
`framework/spirv_simulator.cpp` for more examples.
