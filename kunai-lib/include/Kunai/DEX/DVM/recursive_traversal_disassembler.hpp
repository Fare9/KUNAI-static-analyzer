//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file recursive_traversal_disassembler.hpp
// @brief Class for managing the Recursive Traversal Disassembly algorithm.

#ifndef KUNAI_DEX_DVM_RECURSIVE_TRAVERSAL_DISASSEMBLER_HPP
#define KUNAI_DEX_DVM_RECURSIVE_TRAVERSAL_DISASSEMBLER_HPP

#include "Kunai/DEX/DVM/disassembler.hpp"
#include <queue>

namespace KUNAI
{
namespace DEX
{
    /// @brief RecursiveTraversalDisassembler is one of the DEX disassembly
    /// algorithms implemented by Kunai, this algorithm will follow the control
    /// flow to disassemble the bytecode.
    class RecursiveTraversalDisassembler
    {
        /// @brief Internal disassembler
        Disassembler *disassembler;

        /// @brief Queue for keeping track of followed paths
        std::queue<std::uint64_t> Q;

        /// @brief If there's any switch in code, we will assign to some instructions
        /// the PackedSwitch or the SparswSwitch value
        /// @param ordered_instructions pointer to a map where the instructions are
        /// stored sorted by idx, here is mandatory.
        void analyze_switch(
            std::vector<std::unique_ptr<Instruction>> &instructions,
            std::unordered_map<std::uint64_t, Instruction *> &cache_instrs,
            std::vector<std::uint8_t> &buffer_bytes);

    public:
        RecursiveTraversalDisassembler() = default;

        void set_internal_disassembler(Disassembler *disassembler)
        {
            this->disassembler = disassembler;
        }

        /// @brief This function implements the algorithm of disassembly
        /// of linear sweep, the function receives a buffer of bytes and
        /// the buffer is traversed in a linear way disassembling each
        /// byte
        /// @param buffer_bytes bytes to disassembly
        /// @param method the method to determine the exceptions
        /// @param instructions vector where to store the instructions
        void disassembly(std::vector<std::uint8_t> &buffer_bytes,
                            EncodedMethod *method,
                            std::vector<std::unique_ptr<Instruction>> &instructions);
    };
} // namespace DEX
} // namespace KUNAI

#endif
