/**
 * @file ir_graph.hpp
 * @author Miasm project.
 * 
 * @brief DiGraph class from miasm implemented in C++ using the classes
 *        from MjolnIR, some of basic graph analysis will be implemented
 *        in this class.
 */

#ifndef IR_GRAPH_HPP
#define IR_GRAPH_HPP

#include <iostream>
#include <set>
#include <algorithm>
#include <map>
#include <optional>

#include "utils.hpp"
#include "ir_grammar.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        using Nodes = std::vector<irblock_t>;
        using Edge  = std::pair<irblock_t, irblock_t>;
        using Edges = std::vector<Edge>;
        using Paths = std::vector<std::vector<irblock_t>>;
        
        class IRGraph;

        using irgraph_t = std::shared_ptr<IRGraph>;

        class IRGraph
        {
        public:
            enum node_type_t
            {
                JOIN_NODE = 0,  // type of node with len(predecessors) > 1
                BRANCH_NODE,     // type of node with len(successors) > 1
                REGULAR_NODE,   // other cases
            };

            /**
             * @brief Constructor of IRGraph, this class represent a control-flow graph of IRBlock objects.
             * @return void
             */
            IRGraph();

            /**
             * @brief Destructor of IRGraph, nothing to be done.
             * @return void
             */
            ~IRGraph() = default;

            /**
             * @brief Add a node to the IRGraph.
             * @param node: new node to insert in set.
             * @return bool
             */
            bool add_node(irblock_t node);

            /**
             * @brief Add a new edge to the IRGraph.
             * @param src: source node, will be inserted if it does not exists.
             * @param dst: destination node, will be inserted if it does not exists.
             * @return void
             */
            void add_edge(irblock_t src, irblock_t dst);
            
            /**
             * @brief Add a new edge if it doesn't exists already.
             * @param src: source node, will be inserted if it does not exists.
             * @param dst: destination node, will be inserted if it does not exists.
             * @return void
             */
            void add_uniq_edge(irblock_t src, irblock_t dst);
            
            /**
             * @brief Add a block as one of the successors of the given block.
             * @param node: node to add its successor.
             * @param successor: successor block to add.
             * @return void
             */
            void add_block_to_sucessors(irblock_t node, irblock_t successor);
            
            /**
             * @brief Add a block as on of the precessors of the given block.
             * @param node: node to add its predecessor.
             * @param predecessor: predecessor block to add.
             * @return void
             */
            void add_block_to_predecessors(irblock_t node, irblock_t predecessor);

            /**
             * @brief Get the set of nodes from the current graph.
             * @return Nodes
             */
            Nodes& get_nodes()
            {
                return nodes;
            }

            /**
             * @brief Get the vector with pair of edges.
             * @return Edges
             */
            Edges& get_edges()
            {
                return edges;
            }

            /**
             * @brief Get the node by start idx object
             * 
             * @param idx 
             * @return std::optional<irblock_t> 
             */
            std::optional<irblock_t> get_node_by_start_idx(std::uint64_t idx);

            /**
             * @brief Merge one given graph with current one.
             * @param graph: graph to merge with current one.
             * @return void
             */
            void merge_graph(irgraph_t graph);

            /**
             * @brief Delete an edge given two blocks.
             * @param src: source node from the edge.
             * @param dst: destination node from the edge.
             * @return void
             */
            void del_edge(irblock_t src, irblock_t dst);
            
            /**
             * @brief Delete a node from the graph, 
             *        remove all the edges that are connected
             *        to the node.
             * @param node: node to remove from the graph.
             * @return void
             */
            void del_node(irblock_t node);
            
            /**
             * @brief Remove one of the sucessors given an IRBlock, if it's inside of the vector.
             * @param node: node where to remove its successors.
             * @param block: Block to remove from the successors.
             * @return void
             */
            void delete_block_from_sucessors(irblock_t node, irblock_t block);
            
            /**
             * @brief Remove on of the predecessors given an IRBlock, if it's inside of the vector.
             * @param node: node where to remove its predecessor.
             * @param block: Block to remove from the predecessors.
             * @return void
             */
            void delete_block_from_precessors(irblock_t node, irblock_t block);

            /**
             * @brief Get all those nodes which do not have successors.
             * @return std::vector<irblock_t>
             */
            std::vector<irblock_t> get_leaves();

            /**
             * @brief Get all those nodes which are head of graph, which do not have predecessors.
             * @return std::vector<irblock_t>
             */
            std::vector<irblock_t> get_heads();

            /**
             * @brief Find a different paths between a source block and destination block from the destination to backward
             *        in the graph. 
             * @param src: source block from the graph.
             * @param dst: destination block from the graph.
             * @param cycles_count: maximum number of times a basic block can be processed.
             * @param done: dictionary of already processed loc_keys, it's value is number of times it was processed.
             * @return Paths (std::vector<std::vector<irblock_t>> Paths)
             */
            Paths find_path(irblock_t src,
                            irblock_t dst,
                            size_t cycles_count,
                            std::map<irblock_t, size_t> done);

            /**
             * @brief Find a different paths between a source block and destination block from the source to forward
             *        in the graph. 
             * @param src: source block from the graph.
             * @param dst: destination block from the graph.
             * @param cycles_count: maximum number of times a basic block can be processed.
             * @param done: dictionary of already processed loc_keys, it's value is number of times it was processed.
             * @return Paths (std::vector<std::vector<irblock_t>> Paths)
             */
            Paths find_path_from_src(irblock_t src,
                                     irblock_t dst,
                                     size_t cycles_count,
                                     std::map<irblock_t, size_t> done);

            /**
             * @brief Get the reachable sons of a given block.
             * @param head: head block to calculate reachable sons.
             * @return Nodes
             */
            Nodes reachable_sons(irblock_t head);

            /**
             * @brief Get the reachable parent nodes of a given block.
             * @param leaf: leaf block to calculate reachable parents.
             * @return Nodes
             */
            Nodes reachable_parents(irblock_t leaf);
            
            /**
             * @brief Compute the dominators of the graph.
             * @param head: head used as starting point to calculate the dominators in the graph.
             * @return std::map<irblock_t, Nodes>
             */
            std::map<irblock_t, Nodes> compute_dominators(irblock_t head);
            
            /**
             * @brief Compute the postdominators of the graph.
             * @param leaf: node to get its postdominators.
             * @return std::map<irblock_t, Nodes>
             */
            std::map<irblock_t, Nodes> compute_postdominators(irblock_t leaf);

            /**
             * @brief Compute the immediate dominators algorithm from:
             *        "Advanced Compiler Design and Implementation"
             * 
             * @return std::map<irblock_t, irblock_t> 
             */
            std::map<irblock_t, irblock_t> compute_immediate_dominators();

            /**
             * @brief Create a copy of the IRGraph as a smart pointer.
             * @return irgraph_t
             */
            irgraph_t copy();


            // node information
            /**
             * @brief Get the number of successor blocks from the given block.
             * @param node: block to get its number of successors.
             * @return size_t
             */
            size_t get_number_of_successors(irblock_t node);
            
            /**
             * @brief Get the list of successor blocks.
             * @param node: block to get its successors.
             * @return std::vector<irblock_t>
             */
            Nodes get_successors(irblock_t node);

            /**
             * @brief Get the number of predecessor blocks from the given block.
             * @param node: block to get its number of predecessors.
             * @return size_t
             */
            size_t get_number_of_predecessors(irblock_t node);
            
            /**
             * @brief Get the list of predecessor blocks.
             * @param node: block to get its predecessors.
             * @return std::vector<irblock_t>
             */
            Nodes get_predecessors(irblock_t node);

            /**
             * @brief Get type of node depending on number of successors and predecessors.
             * @param node: node to check its type.
             * @return node_type_t
             */
            node_type_t get_type_of_node(irblock_t node);

            // algorithms from Advanced Compiler Design and Implementation
            
            /**
             * @brief Get all the reachable nodes from a given head node.
             * @param head: head of the set we want to get all its reachable nodes.
             * @return Nodes
             */
            Nodes reachable_nodes_forward(irblock_t head);
            
            /**
             * @brief Get all the reachable up nodes from a given leaf node.
             * @param leaf: leaf of the set we of nodes we want to get all its reachable nodes.
             * @return Nodes
             */
            Nodes reachable_nodes_backward(irblock_t leaf);
            
            /**
             * @brief An extended basic block is a maximal sequence of instructions beginning with a leader
             *        that contains no join nodes (those with more than 1 successor) other than its first node
             *        extended basic block has single entry and possible multiply entries.
             *        Algorithm taken from "Advanced Compiler Design and Implementation" by Steven Muchnick
             * @param r: entry block to do the extended basic block.
             * @return Nodes
             */
            Nodes build_ebb(irblock_t r);
            
            /**
             * @brief Given a head node, give the tree of nodes using a Depth First Search algorithm.
             * @param head: node where to start the search.
             * @return Nodes
             */
            Nodes Deep_First_Search(irblock_t head);
            
            /**
             * @brief Given a head node, give the tree of nodes using a Breadth First Search algorithm.
             * @param head: node where to start the search.
             * @return Nodes
             */
            Nodes Breadth_First_Search(irblock_t head);

            /**
             * @brief Generate a dot file with the CFG, this will include the IR
             *        code, and each block as graph nodes.
             * 
             * @param name: name for the generated dot file.
             */
            void generate_dot_file(std::string name);

            /**
             * @brief Generate a dot file with the dominator tree.
             * 
             * @param name 
             */
            void generate_dominator_tree(std::string name);

            /**
             * @brief Get the cyclomatic complexity of the graph,
             *        this will be useful to calculate the complexity
             *        of a function regarding its CFG.
             * @return std::uint64_t
             */
            const std::uint64_t get_cyclomatic_complexity();
        private:
            Nodes nodes;
            Edges edges;

            std::map<irblock_t, Nodes> successors;
            std::map<irblock_t, Nodes> predecessors;
            
            void add_bbs(irblock_t r, Nodes ebb);

            std::uint64_t cyclomatic_complexity = -1;
            
        };
    } // namespace MJOLNIR

} // namespace KUNAI

#endif