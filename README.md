# DO NOT USE

Unless you have an ongoing dialogue with the main developers of the tool.

Currently under active, early development. The whole project and the tool interface is subject to large changes on a day-to-day basis, and can break at any time.


# spirv-simulator

This repo implements a SPIRV simulator that can be used to detect and extract pointers to physical storage buffers in API streams.

The intended use case is for it to be integrated with tracing and post processing graphics API inspection tools to find and handle pointers to GPU memory embedded in other data sources.


It is not intended for simulating full dispatches of all threads in a batch of GPU work (although it could technically do this to some extent, it is not optimized for it).


## dependencies

Requires a C++ 20 compatible compiler.
G++ 10 or newer and Clang 10 or newer should do the trick.

To build:

```
cmake -H. -Bbuild
cd build
make
```

To run a test:

```
./spirv_simulator ../test_shaders/<shader_name>.spirv
```


If you wish to include the SPIRV dependencies from your system, set:
```
SPIRV_HEADERS_PRESENT=1
```
Before compiling, if this is not set it will use the spirv.hpp file instead.

To run compilation unit tests:
```
cmake -H. -Bbuild
cd build
make run_tests
```


## Execution framework (WIP)

The main files of interest are spirv_simulator.hpp and spirv_simulator.cpp, they contain all the relevant code.

The main framework is implemented in the SPIRVSimulator class and the InputData class.

It can be used as follows:

```
std::string spirv_filepath = "myshader.spirv";
SPIRVSimulator::InputData input_data;
bool verbose = true;
<Initialize and write to the input data here>

SPIRVSimulator::SPIRVSimulator sim(ReadFile(spirv_filepath.c_str()), input_data, verbose);
sim.Run();

auto physical_address_data = sim.GetPhysicalAddressData();
<Work with the outputs here>
```

### Populating the inputs (WIP)

The input structure has the following format:

```
struct InputData{
    uint32_t entry_point_id = 0;
    std::string entry_point_op_name = "";

    std::unordered_map<uint64_t, std::pair<size_t, size_t>> rt_array_lengths;
    std::unordered_map<uint32_t, size_t> specialization_constant_offsets;
    void* specialization_constants = nullptr;
    void* push_constants = nullptr;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, void*>> bindings;
    std::unordered_map<uint64_t, std::pair<size_t, void*>> physical_address_buffers;
};
```

The format of the data must be what the shader expects, eg. if a buffer is bound to a binding with the std430 layout, the data in the byte vectors must obey the rules of that layout.

And each shader input should be mapped to a compatible member in the input structure.

entry_point_op_name should be set to the name of the entry point in the shader.

entry_point_id is optional, and will be ignored if entry_point_op_name. It represents the result ID of a OpEntryPoint instruction.

specialization_constants should be set to a pointer to the full spec constant block.

specialization_constant_offsets should contain one entry per spec constant in the shader.
The key should be the SpecId matching the specialization constant ID as defined by the API.
The offset value is the offset (in bytes) into the specialization_constants pointer where the given spec constant value can be found.

push_constants should be set to a pointer to the full push constant block.

bindings should contain a key for each descriptorset ID, then for each key one map value mapping binding ID's to pointers that point to the full host side data block for the given binding.

physical_address_buffers should contain a uint64_t key which represents the physical address buffer pointer cast (in a manner that preserves the bit pattern) to a 64 bit unsigned integer. The value should be a pair, where the first entry is the size of the buffer in bytes, and the second value is a pointer to the host side memory containing all the values in the buffer.

rt_array_lengths should contain a uint64_t key which represents the host side pointer to the runtime array buffer cast (in a manner that preserves the bit pattern) to a 64 bit unsigned integer. The value should be a pair, where the first entry is the offset into the buffer pointer stored in the uint64_t key to the point where the runtime array can be found. The second entry is the array length (in bytes).


## Framework details (WIP)

The main framework is based around 3 structures:
```
- Value:
	- This encapsulates all SPIRV valid type values.

- Type:
	- Describes the type details of a SPIRV value. All values have an associated Type.

- Instruction:
	- Encapsulates a decoded instruction.
```

And these three access member functions:
```
- GetValue(<ID>):
	- Fetches the Value associated with a SPIRV result ID.

- SetValue(<ID>, Value):
	- Writes a Value to the specified SPIRV result ID.

- Deref(Value(PointerV)):
	- Returns a reference to the Value pointed to by a framework Pointer Value, this can then be written to or read from as needed.
```

The Value structure is essentially just a C++ variant, the underlying value can be queried with the standard C++ variant functionality, eg:
```
std::holds_alternative<T>(Value);
std::get<T>(Value);
```

The Type structure holds 2 member variables, one is a Enum describing the kind of type it represents, the other is a union with metadata for the given type.

The Instruction struct encapsulates decoded instructions, it contains a vector of all the words in the instruction, plus the opcode and wordcount in a decoded, easy-to-access format.

You can fetch the OpCode metadata and operands from the input instruction instances passed to your member function as long as you registered it with the correct OpCode.


The framework uses the result ID's of every instruction that has a result as access handles to the data they returned.

You can get and set the results tied to a result ID by using GetValue(result_id) or SetValue(result_id, Value).

Scopes and access handling is taken care of by the framework.

For accessing allocated data through pointers, Deref(PointerV) can be used, the returned reference can be read from or written to as needed.


## Adding support for more opcodes

```
1. Add a member function to the SPIRVSimulator class that takes an Instuction instance as a parameter, example:
	- void Op_FAdd(const Instruction&);
2. In RegisterOpcodeHandlers, register your member function to a given OpCode, example:
	- R(spv::Op::OpFAdd,              [this](const Instruction& i){Op_FAdd(i);});
```

Then you should implement the member function so that it performs/simulates the operations performed by the SPIRV instruction matching the given OpCode.

If the instruction has a result ID, then it needs to write the result Value to the given result ID using SetValue(...).

If the instruction reads or writes to pointers, it needs to use the Deref(...) method to access the correct heap (see Op_Load and Op_Store for examples).

See the existing OpCode implementations in spirv_simulator.cpp for examples.
