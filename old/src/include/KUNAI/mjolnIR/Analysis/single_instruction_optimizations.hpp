/**
 * @file single_instruction_optimizations.hpp
 * @author Farenain
 * @brief All those optimizers that will be applied to only one instruction
 *        will be written in this file, so we can have all of them separated.
 */

#ifndef SINGLE_INSTRUCTION_OPTIMIZATIONS
#define SINGLE_INSTRUCTION_OPTIMIZATIONS

#include <memory>
#include <optional>

#include "KUNAI/mjolnIR/ir_graph.hpp"
#include "KUNAI/mjolnIR/ir_grammar.hpp"

namespace KUNAI 
{
    namespace MJOLNIR
    {

        /**
         * @brief Optimizations applied directly to one statement
         *        this in opposite to other optimizations will be
         *        applied only to the current instruction, the others
         *        would need to check a whole block or even whole graph.
         */

        /**
         * @brief Apply constant folding optimization to the
         *        given instruction, we can have different operations
         *        where we can apply constant folding:
         *        IRExpr <- IRConstInt IRBinOp IRConstInt
         *        IRExpr <- IRUnaryOp IRConstInt
         *
         * @param instr
         * @return std::optional<irstmnt_t>
         */
        std::optional<irstmnt_t> constant_folding(irstmnt_t &instr);

    }
}

#endif