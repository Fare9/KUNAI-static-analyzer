/**
 * @file ir_graph.hpp
 * @author Miasm project.
 * 
 * @brief DiGraph class from miasm implemented in C++ using the classes
 *        from MjolnIR, some of basic graph analysis will be implemented
 *        in this class.
 */

#include <iostream>
#include <set>
#include "ir_blocks.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        typedef std::set<std::shared_ptr<IRBlock>> Nodes;
        typedef std::vector<std::pair<std::shared_ptr<IRBlock>, std::shared_ptr<IRBlock>>> Edges;

        class IRGraph
        {
        public:
            IRGraph();
            ~IRGraph();

            bool add_node(std::shared_ptr<IRBlock> node);
            void add_edge(std::shared_ptr<IRBlock> src, std::shared_ptr<IRBlock> dst);

            Nodes get_nodes();
            Edges get_edges();

            void merge_graph(std::shared_ptr<IRGraph> graph);

            void del_edge(std::shared_ptr<IRBlock> src, std::shared_ptr<IRBlock> dst);
            void del_node(std::shared_ptr<IRBlock> node);
        private:
            Nodes nodes;
            Edges edges;
        };
    } // namespace MJOLNIR
    
} // namespace KUNAI
