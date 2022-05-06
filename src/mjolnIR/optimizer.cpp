#include "optimizer.hpp"

namespace KUNAI {
    namespace MJOLNIR {

        void Optimizer::fallthrough_target_analysis(MJOLNIR::irgraph_t& ir_graph)
        {
            auto nodes = ir_graph->get_nodes();

            for (auto node : nodes)
            {

                if (node->get_number_of_statements() == 0) // security check
                    continue;

                auto last_inst = node->get_statements().back();

                if (MJOLNIR::is_conditional_jump(last_inst) || MJOLNIR::is_unconditional_jump(last_inst) || MJOLNIR::is_ret(last_inst))
                    continue;

                for (auto aux : nodes)
                {
                    if (node->get_end_idx() == aux->get_start_idx())
                    {
                        ir_graph->add_uniq_edge(node, aux);
                    }
                }
            }
        }

        
    }
}