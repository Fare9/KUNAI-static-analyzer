/**
 * @file ir_graph_ssa.hpp
 * @author Farenain
 * @brief This is an extension of a IRGraph class, but in this case
 *        the nodes will contain a SSA form of the graph, where each
 *        variable will be assigned only once, as parameter the constructor
 *        will receive an IRGraph and this will be transformed to the
 *        IRGraphSSA.
 */

#include <optional>

#include "KUNAI/mjolnIR/ir_graph.hpp"


namespace KUNAI
{
    namespace MJOLNIR
    {
        class IRGraphSSA;

        using irgraphssa_t = std::shared_ptr<IRGraphSSA>;

        class IRGraphSSA : public IRGraph
        {
        public:
            /**
             * @brief Construct a new IRGraphSSA object for this it will
             *        be necessary a code IRGraph.
             * 
             * @param code_graph graph used for constructing the SSA Graph.
             */
            IRGraphSSA(irgraph_t& code_graph);

            /**
             * @brief Destroy the IRGraphSSA object
             */
            ~IRGraphSSA() = default;

        private:
            std::map<irreg_t, std::uint32_t> reg_last_index;

            std::optional<irblock_t> translate_ir_block(irblock_t& current_block);
            
            
        };
    } //! MJOLNIR
} //! KUNAI