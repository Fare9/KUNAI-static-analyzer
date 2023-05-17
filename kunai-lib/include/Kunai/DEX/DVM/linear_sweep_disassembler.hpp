//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file linear_sweep_disassembler.hpp
// @brief Class for managing the Linear Sweep Disassembly algorithm.

#ifndef KUNAI_DEX_DVM_LINEAR_SWEEP_DISASSEMBLER_HPP
#define KUNAI_DEX_DVM_LINEAR_SWEEP_DISASSEMBLER_HPP

#include "Kunai/DEX/DVM/disassembler.hpp"

namespace KUNAI
{
namespace DEX
{
    /// @brief LinearSweepDisassembler is one of the DEX disassembly
    /// algorithms implemented by Kunai, this algorithm will go from
    /// the first byte of a buffer to the last one disassemblying all
    /// the found instructions.
    class LinearSweepDisassembler
    {
        /// @brief Internal disassembler
        Disassembler *disassembler;

        /// @brief If there's any switch in code, we will assign to some instructions
        /// the PackedSwitch or the SparswSwitch value
        /// @param instructions all the buffer with the instructions from a method.
        /// @param cache_instructions cache of instructions for avoiding searching
        /// always in the vector
        void assign_switch_if_any(
            std::vector<std::unique_ptr<Instruction>> &instructions,
            std::unordered_map<std::uint64_t, Instruction *> &cache_instructions);

    public:
        LinearSweepDisassembler() = default;

        void set_internal_disassembler(Disassembler *disassembler)
        {
            this->disassembler = disassembler;
        }

        /// @brief This function implements the algorithm of disassembly
        /// of linear sweep, the function receives a buffer of bytes and
        /// the buffer is traversed in a linear way disassembling each
        /// byte
        /// @param buffer_bytes bytes to disassembly
        /// @param instructions vector where to store the instructions
        void disassembly(std::vector<std::uint8_t> &buffer_bytes,
                            std::vector<std::unique_ptr<Instruction>> &instructions);
    };
} // DEX
} // KUNAI

#endif
