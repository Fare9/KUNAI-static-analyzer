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
#include <string>

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
         * @param graph
         * @return std::optional<irblock_t>
         */
        std::optional<irblock_t> nop_removal(irblock_t &block, irgraph_t &graph);

        /**
         * @brief We can find some obfuscations that rewrite some
         *        simple expressions in order to make them more
         *        difficult to follow, we can go block by block
         *        discovering them, and reducing to simplified expressions.
         * 
         * @param block 
         * @param grap
         * @return std::optional<irblock_t> 
         */
        std::optional<irblock_t> expression_simplifier(irblock_t &block, irgraph_t &graph);

        /**
         * @brief Some complex expressions can be combined in something
         *        simpler, that make the expression simpler to understand.
         * 
         * @param block 
         * @param graph
         * @return std::optional<irblock_t> 
         */
        std::optional<irblock_t> instruction_combining(irblock_t &block, irgraph_t &graph);
    }
}

#endif