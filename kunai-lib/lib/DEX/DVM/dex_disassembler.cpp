//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file dex_disassembler.cpp

#include "Kunai/DEX/DVM/dex_disassembler.hpp"

using namespace KUNAI::DEX;

void DexDisassembler::disassembly_dex()
{
    auto logger = LOGGER::logger();

    // set disassembly as correct, this will be fixed in case of an error
    disassembly_correct = true;

    logger->debug("disassembly_dex: started disassembly of dex file");

    auto &classes = parser->get_classes();

    auto &class_defs = classes.get_classdefs();

    for (auto &class_def : class_defs)
    {
        // get the ClassDataItem
        auto &class_data_item = class_def->get_class_data_item();

        auto &methods = class_data_item.get_methods();

        for (auto method : methods)
        {
            auto &code_item_struct = method->get_code_item();

            auto &buffer_instructions = code_item_struct.get_bytecode();

            std::vector<std::unique_ptr<Instruction>> instructions;

            try
            {
                if (algorithm == disassembly_algorithm::LINEAR_SWEEP_ALGORITHM)
                    linear_sweep.disassembly(buffer_instructions, instructions);
                else if (algorithm == disassembly_algorithm::RECURSIVE_TRAVERSAL_ALGORITHM)
                    recursive_traversal.disassembly(buffer_instructions, method, instructions);

                dex_instructions[method] = std::move(instructions);
            }
            catch (const std::exception &e)
            {
                disassembly_correct = false;
                instructions.clear();
                dex_instructions[method] = std::move(instructions);
            }
        }
    }

    logger->debug("disassembly_dex: finished disassembly of dex file");
}

std::vector<std::unique_ptr<Instruction>>
DexDisassembler::disassembly_buffer(std::vector<std::uint8_t> &buffer)
{
    std::vector<std::unique_ptr<Instruction>> instructions;

    // since we don't know about the method, we use a simple
    // linear sweep disassembly
    linear_sweep.disassembly(buffer, instructions);

    return instructions;
}