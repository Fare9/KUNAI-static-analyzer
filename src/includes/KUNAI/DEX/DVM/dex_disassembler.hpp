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
            ~DexDisassembler() = default;

            /**
             * @brief get if the disassembly was correct or not.
             * @return bool.
             */
            bool get_disassembly_correct()
            {
                return disassembly_correct;
            }

            /**
             * @brief Public method for disassembly with error checking.
             * @return void.
             */
            void disassembly_analysis();

            /**
             * @brief Get the linear sweep disassembler object.
             * @return std::shared_ptr<LinearSweepDisassembler>
             */
            std::shared_ptr<LinearSweepDisassembler> get_linear_sweep_disassembler()
            {
                return dalvik_disassembler;
            }

            /**
             * @brief Return all the instructions from the disassembled DEX.
             * @return std::map<std::tuple<std::shared_ptr<ClassDef>, std::shared_ptr<EncodedMethod>>, std::map<std::uint64_t, std::shared_ptr<Instruction>>>
             */
            std::map<std::tuple<std::shared_ptr<ClassDef>, std::shared_ptr<EncodedMethod>>,
                     std::map<std::uint64_t, std::shared_ptr<Instruction>>> &
            get_instructions()
            {
                return method_instructions;
            }

            void add_disassembly(std::shared_ptr<DexDisassembler> disas);

            friend std::ostream &operator<<(std::ostream &os, const DexDisassembler &entry);

            friend DexDisassembler& operator+(DexDisassembler& first_disassembler, DexDisassembler& other_disassembler);

        private:
            /**
             * @brief Disassembly DEX methods using linear sweep disassembly, store instructions with its class and method.
             * @return void.
             */
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