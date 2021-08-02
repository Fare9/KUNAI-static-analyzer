/**
 * @file dex_disassembler.hpp
 * @author Fare9
 * 
 * @brief Class to manage the disassembly process of
 *        the DEX file.
 */

#ifndef DEX_DISASSEMBLER_HPP
#define DEX_DISASSEMBLER_HPP

#include <iostream>
#include <memory>

#include "dex_parser.hpp"
#include "dex_instructions.hpp"
#include "dex_linear_sweep_disassembly.hpp"

namespace KUNAI
{
    namespace DEX
    {
        class DexDisassembler
        {
        public:
            DexDisassembler(bool parsing_correct, std::shared_ptr<DexParser> dex_parser, std::shared_ptr<DalvikOpcodes> dalvik_opcodes);
            ~DexDisassembler();

            bool get_disassembly_correct();
            void disassembly_analysis();
            std::shared_ptr<LinearSweepDisassembler> get_linear_sweep_disassembler();
            std::map<std::tuple<std::shared_ptr<ClassDef>, std::shared_ptr<EncodedMethod>>,
                     std::map<std::uint64_t, std::shared_ptr<Instruction>>> get_instructions();

            friend std::ostream& operator<<(std::ostream& os, const DexDisassembler& entry);
        private:
            void disassembly_methods();

            std::shared_ptr<DexParser> dex_parser;
            std::shared_ptr<DalvikOpcodes> dalvik_opcodes;
            std::shared_ptr<LinearSweepDisassembler> dalvik_disassembler;

            std::map<std::tuple<std::shared_ptr<ClassDef>, std::shared_ptr<EncodedMethod>>,
                     std::map<std::uint64_t, std::shared_ptr<Instruction>>>
                method_instructions;
            bool parsing_correct;
            bool disassembly_correct;

        };
    } // namespace DEX
} // namespace KUNAI

#endif