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
        typedef std::vector<std::vector<std::shared_ptr<IRBlock>>> Paths;

        class IRGraph
        {
        public:
            IRGraph();
            ~IRGraph();

            bool add_node(std::shared_ptr<IRBlock> node);
            void add_edge(std::shared_ptr<IRBlock> src, std::shared_ptr<IRBlock> dst);
            void add_uniq_edge(std::shared_ptr<IRBlock> src, std::shared_ptr<IRBlock> dst);

            Nodes get_nodes();
            Edges get_edges();

            void merge_graph(std::shared_ptr<IRGraph> graph);

            void del_edge(std::shared_ptr<IRBlock> src, std::shared_ptr<IRBlock> dst);
            void del_node(std::shared_ptr<IRBlock> node);

            std::vector<std::shared_ptr<IRBlock>> get_leaves();
            std::vector<std::shared_ptr<IRBlock>> get_heads();

            Paths find_path(std::shared_ptr<IRBlock> src,
                            std::shared_ptr<IRBlock> dst,
                            size_t cycles_count,
                            std::map<std::shared_ptr<IRBlock>, size_t> done);

            Paths find_path_from_src(std::shared_ptr<IRBlock> src,
                                     std::shared_ptr<IRBlock> dst,
                                     size_t cycles_count,
                                     std::map<std::shared_ptr<IRBlock>, size_t> done);

            Nodes reachable_sons(std::shared_ptr<IRBlock> head);
            Nodes reachable_parents(std::shared_ptr<IRBlock> leaf);

            std::shared_ptr<IRGraph> copy();


            // static methods that can be used in general with IRGraph class
            static Nodes reachable_nodes(std::shared_ptr<IRBlock> head, bool go_successors);
            static Nodes build_ebb(std::shared_ptr<IRBlock> r);
            
        private:
            Nodes nodes;
            Edges edges;

            // static methods that can be used in general with IRGraph class
            
            static void add_bbs(std::shared_ptr<IRBlock> r, Nodes ebb);
            
        };
    } // namespace MJOLNIR

} // namespace KUNAI
