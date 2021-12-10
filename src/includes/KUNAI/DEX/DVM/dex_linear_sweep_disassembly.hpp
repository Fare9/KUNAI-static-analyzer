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


#include "dex_instructions.hpp"

namespace KUNAI {
    namespace DEX {
        class LinearSweepDisassembler
        {
        public:
            LinearSweepDisassembler(std::shared_ptr<DalvikOpcodes> dalvik_opcodes);
            ~LinearSweepDisassembler();

            std::map<std::uint64_t, std::shared_ptr<Instruction>> disassembly(std::vector<std::uint8_t> byte_buffer);
        private:
            void assign_switch_if_any(std::map<std::uint64_t, std::shared_ptr<Instruction>> instrs);

            std::shared_ptr<DalvikOpcodes> dalvik_opcodes;
        };
    }
}


#endif