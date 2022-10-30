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
#include <optional>

#include "KUNAI/DEX/parser/dex_parser.hpp"
#include "KUNAI/DEX/DVM/dex_instructions.hpp"
#include "KUNAI/DEX/DVM/dex_linear_sweep_disassembly.hpp"
#include "KUNAI/DEX/DVM/dex_recursive_traversal_disassembly.hpp"

namespace KUNAI
{
    namespace DEX
    {
        class DexDisassembler;

        /**
         * @brief std::unique_ptr of DexDisassembler object which manages the disassembly
         * of the DEX bytecode, currently it implements a LinearSweepDisassembly which
         * will take methods and will disassemble them, it returns the instructions as a map.
         */
        using dexdisassembler_t = std::unique_ptr<DexDisassembler>;

        /**
         * @brief Type which contains the instructions of the disassembler.
         */
        using instruction_map_t = std::map<std::tuple<ClassDef*, EncodedMethod*>, std::map<std::uint64_t, instruction_t>>;

        enum disassembler_t
        {
            LINEAR_SWEEP_DISASSEMBLER,
            RECURSIVE_TRAVERSAL_DISASSEMBLER
        };

        class DexDisassembler
        {
        public:
            /**
             * @brief Construct a new Dex Disassembler object
             *        use by default the Linear Sweep Algorithm
             *
             * @param parsing_correct
             * @param dex_parser
             * @param dalvik_opcodes
             */
            DexDisassembler(bool parsing_correct, DexParser* dex_parser, DalvikOpcodes* dalvik_opcodes);

            /**
             * @brief Destroy the Dex Disassembler object
             *
             */
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
                return linear_dalvik_disassembler;
            }

            /**
             * @brief Get the recursive traversal disassembler object
             *
             * @return recursivetraversaldisassembler_t&
             */
            recursivetraversaldisassembler_t &get_recursive_traversal_disassembler()
            {
                return recursive_dalvik_disassembler;
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

            /**
             * @brief We can include in one disassembler the disassembly from
             *        many others, this will allow to disassembly more than one
             *        DEX file as one.
             *
             * @param disas disassembler object to include to current one.
             */
            void add_disassembly(DexDisassembler *disas);

            /**
             * @brief Set the disassembler type: LINEAR_SWEEP_DISASSEMBLER, RECURSIVE_TRAVERSAL_DISASSEMBLER
             *
             * @param type
             */
            void set_disassembler_type(disassembler_t type)
            {
                disas_type = type;
            }

            /**
             * @brief Write into the std::ostream the disassembly of the whole file.
             *
             * @param os
             * @param entry
             * @return std::ostream&
             */
            friend std::ostream &operator<<(std::ostream &os, const DexDisassembler &entry);

            /**
             * @brief Operator + to join two disassemblers, this will join the two maps
             * as well as we will & the disassembly_correct variable.
             *
             * @param other_disassembler
             * @return DexDisassembler&
             */
            friend DexDisassembler &operator+(DexDisassembler &first_disassembler, DexDisassembler &other_disassembler);

        private:
            /**
             * @brief Disassembly DEX methods using linear sweep disassembly, store instructions with its class and method.
             * @return void.
             */
            void disassembly_methods();

            DexParser* dex_parser;
            DalvikOpcodes* dalvik_opcodes;
            disassembler_t disas_type;

            linearsweepdisassembler_t linear_dalvik_disassembler;
            recursivetraversaldisassembler_t recursive_dalvik_disassembler;

            instruction_map_t method_instructions;
            bool parsing_correct;
            bool disassembly_correct;
        };
    } // namespace DEX
} // namespace KUNAI

#endif