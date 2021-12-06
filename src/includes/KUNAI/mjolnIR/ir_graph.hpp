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
#include <algorithm>
#include <map>
#include "ir_grammar.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        typedef std::vector<std::shared_ptr<IRBlock>> Nodes;
        typedef std::pair<std::shared_ptr<IRBlock>, std::shared_ptr<IRBlock>> Edge;
        typedef std::vector<Edge> Edges;
        typedef std::vector<std::vector<std::shared_ptr<IRBlock>>> Paths;
        

        class IRGraph
        {
        public:
            enum node_type_t
            {
                JOIN_NODE = 0,  // type of node with len(predecessors) > 1
                BRANCH_NODE,     // type of node with len(successors) > 1
                REGULAR_NODE,   // other cases
            };

            IRGraph();
            ~IRGraph();

            bool add_node(std::shared_ptr<IRBlock> node);
            void add_edge(std::shared_ptr<IRBlock> src, std::shared_ptr<IRBlock> dst);
            void add_uniq_edge(std::shared_ptr<IRBlock> src, std::shared_ptr<IRBlock> dst);
            void add_block_to_sucessors(std::shared_ptr<IRBlock> node, std::shared_ptr<IRBlock> successor);
            void add_block_to_predecessors(std::shared_ptr<IRBlock> node, std::shared_ptr<IRBlock> predecessor);

            Nodes get_nodes();
            Edges get_edges();

            void merge_graph(std::shared_ptr<IRGraph> graph);

            void del_edge(std::shared_ptr<IRBlock> src, std::shared_ptr<IRBlock> dst);
            void del_node(std::shared_ptr<IRBlock> node);
            void delete_block_from_sucessors(std::shared_ptr<IRBlock> node, std::shared_ptr<IRBlock> block);
            void delete_block_from_precessors(std::shared_ptr<IRBlock> node, std::shared_ptr<IRBlock> block);

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
            std::map<std::shared_ptr<IRBlock>, Nodes> compute_dominators(std::shared_ptr<IRBlock> head);
            std::map<std::shared_ptr<IRBlock>, Nodes> compute_postdominators(std::shared_ptr<IRBlock> leaf);

            std::map<std::shared_ptr<IRBlock>, std::shared_ptr<IRBlock>> compute_immediate_dominators();

            std::shared_ptr<IRGraph> copy();


            // node information
            size_t get_number_of_successors(std::shared_ptr<IRBlock> node);
            Nodes get_successors(std::shared_ptr<IRBlock> node);

            size_t get_number_of_predecessors(std::shared_ptr<IRBlock> node);
            Nodes get_predecessors(std::shared_ptr<IRBlock> node);

            node_type_t get_type_of_node(std::shared_ptr<IRBlock> node);

            // algorithms from Advanced Compiler Design and Implementation
            Nodes reachable_nodes_forward(std::shared_ptr<IRBlock> head);
            Nodes reachable_nodes_backward(std::shared_ptr<IRBlock> leaf);
            Nodes build_ebb(std::shared_ptr<IRBlock> r);
            Nodes Deep_First_Search(std::shared_ptr<IRBlock> head);
            Nodes Breadth_First_Search(std::shared_ptr<IRBlock> head);

            void generate_dot_file(std::string name);
            void generate_dominator_tree(std::string name);
        private:
            Nodes nodes;
            Edges edges;

            std::map<std::shared_ptr<IRBlock>, Nodes> successors;
            std::map<std::shared_ptr<IRBlock>, Nodes> predecessors;
            
            void add_bbs(std::shared_ptr<IRBlock> r, Nodes ebb);
            
        };
    } // namespace MJOLNIR

} // namespace KUNAI
