/**
 * @file single_block_optimizations.hpp
 * @author Farenain
 * @brief There are optimizations that can be applied to
 *        whole blocks, these will return a modified block
 *        or the same as previous one if no optimization is
 *        applied.
 */

#ifndef SINGLE_BLOCK_OPTIMIZATIONS_HPP
#define SINGLE_BLOCK_OPTIMIZATIONS_HPP

#include <memory>
#include <optional>

#include "KUNAI/mjolnIR/ir_graph.hpp"
#include "KUNAI/mjolnIR/ir_grammar.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        /**
         * @brief Remove the NOP instructions from a block
         *        then return the block without any of those
         *        NOP instructions.
         *
         * @param block
         * @return std::optional<irblock_t>
         */
        std::optional<irblock_t> nop_removal(irblock_t &block);
    }
}

#endif