#include "KUNAI/mjolnIR/Analysis/reachingDefinition.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        /**
         * Pseudo-code of the algorithm implemented for the Reaching definition
         * analysis.
         * 
         * Algorithm ReachingDefinition
         * 
         * procedure reaching_definition
         * Input
         *      graph ← Control-flow graph with method blocks
         * BEGIN
         *      change ← true
         *      while change do
         *          change ← f alse
         *          for block ∈ graph.blocks do
         *              change ← REACHING_DEFINITION_BLOCK(block)
         * END
         * 
         * procedure REACHING_DEFINITION_BLOCK
         * Input
         *     block ← Block of instructions.
         * BEGIN
         *     old ← in(block)
         *     in(block) ← ⋃ out(p)
         *             p∈pred(block)
         *     modified ← old != in(block)
         *     if modified = false then
         *         return false
         *         close;
         *     for i ∈ length(block.instrs) do
         *         modified ← REACHING_DEFINITION_INSTR(block, i)
         *     out(block) ← out(length(block.instrs))
         *     return modified
         * END
         * 
         * procedure REACHING_DEFINITION_INSTR
         * Input
         *     block ← Block of instructions.
         *     instr ← Index of instruction to analyze.
         * BEGIN
         *     old ← out(instr)
         *     //kill(instr) ← gen(instr) ⋂ in(instr)
         *     out(instr) ← gen(instr) ⋃(in(instr) − kill(instr)
         *     modified ← old ̸ = out(instr)
         *     if modif ied then
         *         in(instr + 1) ← out(instr)
         *         close;
         *     return modified
         * END
        */

        ReachingDefinition::ReachingDefinition(irgraph_t &graph) : graph(graph)
        {
        }

        ReachingDefinition::~ReachingDefinition() = default;

        
        void ReachingDefinition::compute()
        {
            bool change = true;

            while (change)
            {
                change = false;
                for (auto &block : graph->get_nodes())
                    change |= analyze_block(block);
            }
        }

        std::optional<regdefinitionset_t> ReachingDefinition::get_reach_definition_point(std::uint64_t id_block, std::uint64_t id_instr)
        {
            auto it = reaching_definitions.find(std::make_tuple(id_block, id_instr));

            if (it != reaching_definitions.end())
            {
                return it->second;
            }

            return std::nullopt;
        }

        std::ostream &operator<<(std::ostream &os, const ReachingDefinition &entry)
        {
            for (const auto &value : entry.reaching_definitions)
            {
                const auto &block_instr_key = value.first;

                os << "{Block: " << std::get<0>(block_instr_key) 
                    << " - Instr: " << std::get<1>(block_instr_key) << "}\n";

                for (const auto &definitions : value.second)
                {
                    for (const auto &definition : definitions)
                    {
                        const auto& reg = definition.first;
                        os << "\t{Value: " << reg->to_string()
                            << " - Block: " << std::get<0>(definition.second) 
                            << " - Instr: " << std::get<1>(definition.second) << "}\n";
                    }
                }
            }

            return os;
        }

        bool ReachingDefinition::analyze_block(irblock_t &block)
        {
            regdefinitionset_t predecesor_state;
            bool modified;

            // Go through each predecessor of the current block
            // in(w) = U out(p) for p in pred(w)
            for (auto pred : graph->get_predecessors(block))
            {
                // take the set of definitions from the last instruction of previous block
                auto &lval_definitions = reaching_definitions[std::make_tuple(pred->get_start_idx(), pred->get_number_of_statements())];

                // add it to predecesor state
                for (auto &lval_definition : lval_definitions)
                    predecesor_state.insert(lval_definition);
            }

            // check if our current value of reaching definitions
            // and the incomings from previous blocks
            modified = (reaching_definitions.find(std::make_tuple(block->get_start_idx(), 0)) == reaching_definitions.end()) || (reaching_definitions[std::make_tuple(block->get_start_idx(), 0)] != predecesor_state);

            if (!modified)
                return false;

            // save in(w)
            reaching_definitions[std::make_tuple(block->get_start_idx(), 0)] = predecesor_state;

            // calculate reach definition for each instruction
            // last instruction will contain out(w)
            for (size_t index = 0, size = block->get_number_of_statements(); index < size; index++)
                modified |= analyze_instruction(block, index);

            return modified;
        }

        bool ReachingDefinition::analyze_instruction(irblock_t &block, std::uint32_t instruction_id)
        {
            bool modified;

            irstmnt_t &instr = block->get_statements().at(instruction_id);
            // defs = in(instr)
            auto defs = reaching_definitions[std::make_tuple(block->get_start_idx(), instruction_id)];

            auto reg_defined = is_reg_defined(instr);

            if (reg_defined)
            {
                // out(instr) =  gen(instr) U (in(instr) - kill(instr))

                // in(instr) - kill(instr)
                for (auto it = defs.begin(); it != defs.end(); it++)
                {
                    const auto& map_value = *it;

                    if (map_value.find(reg_defined.value()) != map_value.end())
                        // we are going to remove previous definitions
                        // of the same registers
                        it = defs.erase(it);
                }

                // gen(instr)
                defs.insert({{reg_defined.value(),
                              std::make_tuple(block->get_start_idx(), instruction_id)}});
            }
            
            // old out(instr) == out(instr)?
            modified = (reaching_definitions.find(std::make_tuple(block->get_start_idx(), instruction_id + 1)) == reaching_definitions.end()) || reaching_definitions[std::make_tuple(block->get_start_idx(), instruction_id + 1)] != defs;

            if (modified)
                // in(instr+1) = out(instr)
                reaching_definitions[std::make_tuple(block->get_start_idx(), instruction_id + 1)] = defs;

            return modified;
        }

        std::optional<irexpr_t> ReachingDefinition::is_reg_defined(irstmnt_t &instr)
        {
            irexpr_t reg;
            // A = B
            if (auto assign_instr = assign_ir(instr))
            {
                reg = assign_instr->get_destination();
            }
            // A = IRUnaryOp B
            else if (auto unary_instr = unary_op_ir(instr))
            {
                reg = unary_instr->get_result();
            }
            // A = B IRBinaryOp C
            else if (auto bin_instr = bin_op_ir(instr))
            {
                reg = bin_instr->get_result();
            }
            // A = load(MEM)
            else if (auto load_instr = load_ir(instr))
            {
                reg = load_instr->get_destination();
            }
            // A = New Class
            else if (auto new_instr = new_ir(instr))
            {
                reg = load_instr->get_destination();
            }
            else
                return std::nullopt;

            return reg;
        }

    } // namespace MJOLNIR
} // namespace KUNAI