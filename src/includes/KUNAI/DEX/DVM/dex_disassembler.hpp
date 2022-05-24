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
        class DexDisassembler;

        /**
         * @brief std::shared_ptr of DexDisassembler object which manages the disassembly
         * of the DEX bytecode, currently it implements a LinearSweepDisassembly which
         * will take methods and will disassemble them, it returns the instructions as a map.
         */
        using dexdisassembler_t = std::shared_ptr<DexDisassembler>;

        using instruction_map_t = std::map<std::tuple<classdef_t, encodedmethod_t>, std::map<std::uint64_t, instruction_t>>;

        class DexDisassembler
        {
        public:
            DexDisassembler(bool parsing_correct, dexparser_t dex_parser, dalvikopcodes_t dalvik_opcodes);
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
             * @return linearsweepdisassembler_t
             */
            linearsweepdisassembler_t &get_linear_sweep_disassembler()
            {
                return dalvik_disassembler;
            }

            /**
             * @brief Return all the instructions from the disassembled DEX.
             * @return instruction_map_t
             */
            instruction_map_t &
            get_instructions()
            {
                return method_instructions;
            }

            void add_disassembly(dexdisassembler_t disas);

            friend std::ostream &operator<<(std::ostream &os, const DexDisassembler &entry);

            friend DexDisassembler &operator+(DexDisassembler &first_disassembler, DexDisassembler &other_disassembler);

        private:
            /**
             * @brief Disassembly DEX methods using linear sweep disassembly, store instructions with its class and method.
             * @return void.
             */
            void disassembly_methods();

            dexparser_t dex_parser;
            dalvikopcodes_t dalvik_opcodes;
            linearsweepdisassembler_t dalvik_disassembler;

            std::map<std::tuple<classdef_t, encodedmethod_t>,
                     std::map<std::uint64_t, instruction_t>>
                method_instructions;
            bool parsing_correct;
            bool disassembly_correct;
        };
    } // namespace DEX
} // namespace KUNAI

#endif