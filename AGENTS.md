# Repository Guidelines

## Project Structure & Module Organization
- Core simulator: `framework/` (`spirv_simulator.cpp/.hpp`, `util.*`, opcode helpers, exceptions).
- CLI entry points: `main.cpp` (simulator) and `main_opcode_checker.cpp` (opcode utility).
- Build scaffolding: root `CMakeLists.txt` plus `cmake/` modules and `external/` vendored deps.
- Tests: `test/` (gtest/gmock suites and harness) and sample shaders in `test_shaders/`.
- Generated artifacts live in `build/` (not committed).

## Coding Style & Naming Conventions
- C++20, 4-space indentation, braces on new lines for control blocks as in existing sources.
- Keep header guards/`#pragma once`, grouped includes, and minimal headers in headers.
- Types/classes use PascalCase (`SPIRVSimulator`); functions/methods are PascalCase; variables and struct fields are lower_snake_case.
- Mirror existing patterns in `framework/spirv_simulator.*` and `test/testing_common.*`; prefer standard library utilities and assertions already in use.

## Notes & Configuration Tips
- The SPIRV simulator implements the SPIRV specification and parses SPIRV code to extract information about pointers and memory usage.

## Code reviews
- Do not run compilation or tests as part of code reviews.

## Adding support for a new SPIRV OpCode
- Opcodes are handled in the c++ files `framework/spirv_simulator.cpp` and `framework/spirv_simulator.hpp`, and the class `SPIRVSimulator`
- Each Opcode has a function handler of the form `void Op_<name>(const Instruction&);`
- Build with `make -j`
- Run tests with `ctest --output-on-failure`

Minimal function implementation:
```cpp
void SPIRVSimulator::Op_<name>(const Instruction& instruction)
{
    /*
    Op<name>

    Some explanation goes here
    */
    assert(instruction.opcode == spv::Op::Op<name>);

    uint32_t type_id          = instruction.words[1];

    // do more work here
}
```

## Adding Unit Tests
- Framework: gtest + gmock (`TEST`, `TEST_P`).
- Place new cases in the closest suite under `test/` (use `misc_test.cpp` as the fallback if no other suite fits)
- Build with `make -j`. Fix any errors shown.
- Run tests with `ctest --output-on-failure`. Fix any errors shown.
- When adding opcodes, include positive/edge cases and pointer/descriptor coverage where applicable.
- For shader-driven scenarios, place inputs in `test_shaders/` and reference them via `./spirv_simulator` in tests.

## Performance measurements
- For performance runs, make sure you build in release mode (made with `-DCMAKE_BUILD_TYPE=Release`, eg in a `build_release` folder).
- Performance can be measured like this: `taskset 1 perf stat -r 100 build_release/spirv_simulator test_shaders/aztec_ruins_shader_4.spv`
