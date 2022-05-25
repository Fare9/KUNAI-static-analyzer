/**
 * @file ReachingDefinition.hpp
 * @author Farenain
 * @brief Class that represents a Reaching definition analysis inside
 *        of an IRGraph.
 *
 */

#ifndef REACHINGDEFINITION_HPP
#define REACHINGDEFINITION_HPP

#include <memory>
#include <optional>

#include "utils.hpp"
#include "ir_graph.hpp"
#include "ir_grammar.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        class ReachingDefinition;

        using reachingdefinition_t = std::shared_ptr<ReachingDefinition>;

        using blockinstrtuple_t = std::tuple<
            std::uint32_t, // block id
            std::uint32_t  // instruction id
            >;

        using regdefinitionmap_t = std::map<
            irexpr_t, // reg or temp reg
            blockinstrtuple_t>;

        using regdefinitionset_t = std::set<regdefinitionmap_t>;

        using reachingdeftype_t = std::map<
            blockinstrtuple_t,
            regdefinitionset_t>;

        class ReachingDefinition
        {
        public:
            /**
             * @brief Construct a new Reaching Definition object, it receives
             *        the graph to analyze.
             *
             * @param graph graph to apply the ReachingDefinition analysis
             */
            ReachingDefinition(irgraph_t &graph);

            ~ReachingDefinition();

            /**
             * @brief Compute the reaching definition of the IRGraph given
             *        for each line we will have a definition of where variables
             *        are defined, and this is nice, because it will be useful
             *        to calculate the def-use and use-def chains.
             *
             */
            void compute();

            /**
             * @brief Get the reach definition point object from an specific basic block
             *        and instruction
             *
             * @param id_block block to retrieve the regdefinitionmap
             * @param id_instr instruction to retrieve the regdefinitionmap
             * @return std::optional<regdefinitionset_t>
             */
            std::optional<regdefinitionset_t> get_reach_definition_point(std::uint64_t id_block, std::uint64_t id_instr);

            friend std::ostream &operator<<(std::ostream &os, const ReachingDefinition &entry);

        private:
            irgraph_t &graph;

            /**
             * @brief Analyze a block looking for definitions and updating the global
             *        reaching definitions object.
             *
             * @param block block to analyze.
             */
            bool analyze_block(irblock_t &block);

            /**
             * @brief Analyze the given instruction, checking if the instruction create
             *        a definition, in case a definition exists, update the set of definitions.
             *
             * @param block current analyzed block
             * @param instruction_id index of the instruction to analyze
             * @return true
             * @return false
             */
            bool analyze_instruction(irblock_t &block, std::uint32_t instruction_id);

            /**
             * @brief Check if the given instruction is an instruction where there is some
             *        kind of definition or redefinition of a register, in that case return
             *        the reference of the register, in other case use optional to return
             *        a std::nullopt value.
             *
             * @param instr
             * @return std::optional<irexpr_t&>
             */
            std::optional<irexpr_t> is_reg_defined(irstmnt_t &instr);

            //! information of in/out/gen/kill for each block.
            reachingdeftype_t reaching_definitions;
        };
    } // namespace MJOLNIR
} // namespace KUNAI

#endif