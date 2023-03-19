/**
 * @file dex_linear_sweep_disassembly.hpp
 * @author @Farenain
 * 
 * @brief Linear sweep disassembly for DEX file
 *        we will create a disassembly method that
 *        users can also use to disassembly raw
 *        hex bytes.
 */

#ifndef DEX_LINEAR_SWEEP_DISASSEMBLY_HPP
#define DEX_LINEAR_SWEEP_DISASSEMBLY_HPP

#include <iostream>
#include <vector>
#include <istream>
#include <sstream>
#include <iterator>

#include "KUNAI/DEX/DVM/dex_instructions.hpp"

namespace KUNAI {
    namespace DEX {
        class LinearSweepDisassembler;

        using linearsweepdisassembler_t = std::unique_ptr<LinearSweepDisassembler>;

        class LinearSweepDisassembler
        {
        public:
            LinearSweepDisassembler(DalvikOpcodes* dalvik_opcodes);
            ~LinearSweepDisassembler() = default;

            std::map<std::uint64_t, instruction_t> disassembly(const std::vector<std::uint8_t>& byte_buffer);
        private:
            void assign_switch_if_any(std::map<std::uint64_t, instruction_t>& instrs);

            DalvikOpcodes* dalvik_opcodes;
        };
    }
}


#endif