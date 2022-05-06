/**
 * @file optimizer.hpp
 * @author Farenain
 * @brief Class for applying optimization passes to IR, working in IR
 *        we will be able to create more generic optimizations, and having
 *        an optimizer will allow us to have different optimizatio passes.
 *
 */

#include <memory>

#include "utils.hpp"
#include "ir_graph.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        class Optimizer;

        using optimizer_t = std::shared_ptr<Optimizer>;

        class Optimizer
        {
        public:
            /**
             * @brief Analyze the basic blocks of the graph in order to create the
             *        fallthrough edges between blocks which are from conditional
             *        jumps, these has no edges so for the moment one block looks
             *        like goes nowehere:
             *
             *                          +----------------+
             *                          |                |
             *                          |  jcc           |
             *                          +----------------+
             *           fallthrough   /                  \  target
             *                        /                    \
             *               +----------------+        +----------------+
             *               | points         |        |                |
             *               | nowhere        |        |  jmp           |
             *               +----------------+        +----------------+
             *                                               |
             *                                               |
             *                                         +----------------+
             *                                         |                |
             *                                         |                |
             *                                         +----------------+
             *
             *         The one on the left points nowhere but this is the real previous
             *         block before the last block, but last block is divided because
             *         there's a jump to it, so we will create an edge between, the one
             *         on the left, and the last one, as it's a continuation.
             *
             *                          +----------------+
             *                          |                |
             *                          |  jcc           |
             *                          +----------------+
             *           fallthrough   /                  \  target
             *                        /                    \
             *               +----------------+        +----------------+
             *               | points         |        |                |
             *               | nowhere        |        |  jmp           |
             *               +----------------+        +----------------+
             *                      \                         /
             *                       \                       /
             *                        \                     /
             *                         \                   /
             *                          \                 /
             *                           +----------------+
             *                           |                |
             *                           |                |
             *                           +----------------+
             * @param ir_graph
             */
            void fallthrough_target_analysis(MJOLNIR::irgraph_t &ir_graph);
        };
    }
}