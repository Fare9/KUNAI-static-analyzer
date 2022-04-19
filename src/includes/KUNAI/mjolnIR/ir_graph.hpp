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

#include "utils.hpp"
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
            bool add_node(std::shared_ptr<IRBlock> node);

            /**
             * @brief Add a new edge to the IRGraph.
             * @param src: source node, will be inserted if it does not exists.
             * @param dst: destination node, will be inserted if it does not exists.
             * @return void
             */
            void add_edge(std::shared_ptr<IRBlock> src, std::shared_ptr<IRBlock> dst);
            
            /**
             * @brief Add a new edge if it doesn't exists already.
             * @param src: source node, will be inserted if it does not exists.
             * @param dst: destination node, will be inserted if it does not exists.
             * @return void
             */
            void add_uniq_edge(std::shared_ptr<IRBlock> src, std::shared_ptr<IRBlock> dst);
            
            /**
             * @brief Add a block as one of the successors of the given block.
             * @param node: node to add its successor.
             * @param successor: successor block to add.
             * @return void
             */
            void add_block_to_sucessors(std::shared_ptr<IRBlock> node, std::shared_ptr<IRBlock> successor);
            
            /**
             * @brief Add a block as on of the precessors of the given block.
             * @param node: node to add its predecessor.
             * @param predecessor: predecessor block to add.
             * @return void
             */
            void add_block_to_predecessors(std::shared_ptr<IRBlock> node, std::shared_ptr<IRBlock> predecessor);

            /**
             * @brief Get the set of nodes from the current graph.
             * @return Nodes
             */
            const Nodes& get_nodes() const
            {
                return nodes;
            }

            /**
             * @brief Get the vector with pair of edges.
             * @return Edges
             */
            const Edges& get_edges() const
            {
                return edges;
            }

            /**
             * @brief Merge one given graph with current one.
             * @param graph: graph to merge with current one.
             * @return void
             */
            void merge_graph(std::shared_ptr<IRGraph> graph);

            /**
             * @brief Delete an edge given two blocks.
             * @param src: source node from the edge.
             * @param dst: destination node from the edge.
             * @return void
             */
            void del_edge(std::shared_ptr<IRBlock> src, std::shared_ptr<IRBlock> dst);
            
            /**
             * @brief Delete a node from the graph, 
             *        remove all the edges that are connected
             *        to the node.
             * @param node: node to remove from the graph.
             * @return void
             */
            void del_node(std::shared_ptr<IRBlock> node);
            
            /**
             * @brief Remove one of the sucessors given an IRBlock, if it's inside of the vector.
             * @param node: node where to remove its successors.
             * @param block: Block to remove from the successors.
             * @return void
             */
            void delete_block_from_sucessors(std::shared_ptr<IRBlock> node, std::shared_ptr<IRBlock> block);
            
            /**
             * @brief Remove on of the predecessors given an IRBlock, if it's inside of the vector.
             * @param node: node where to remove its predecessor.
             * @param block: Block to remove from the predecessors.
             * @return void
             */
            void delete_block_from_precessors(std::shared_ptr<IRBlock> node, std::shared_ptr<IRBlock> block);

            /**
             * @brief Get all those nodes which do not have successors.
             * @return std::vector<std::shared_ptr<IRBlock>>
             */
            std::vector<std::shared_ptr<IRBlock>> get_leaves();

            /**
             * @brief Get all those nodes which are head of graph, which do not have predecessors.
             * @return std::vector<std::shared_ptr<IRBlock>>
             */
            std::vector<std::shared_ptr<IRBlock>> get_heads();

            /**
             * @brief Find a different paths between a source block and destination block from the destination to backward
             *        in the graph. 
             * @param src: source block from the graph.
             * @param dst: destination block from the graph.
             * @param cycles_count: maximum number of times a basic block can be processed.
             * @param done: dictionary of already processed loc_keys, it's value is number of times it was processed.
             * @return Paths (std::vector<std::vector<std::shared_ptr<IRBlock>>> Paths)
             */
            Paths find_path(std::shared_ptr<IRBlock> src,
                            std::shared_ptr<IRBlock> dst,
                            size_t cycles_count,
                            std::map<std::shared_ptr<IRBlock>, size_t> done);

            /**
             * @brief Find a different paths between a source block and destination block from the source to forward
             *        in the graph. 
             * @param src: source block from the graph.
             * @param dst: destination block from the graph.
             * @param cycles_count: maximum number of times a basic block can be processed.
             * @param done: dictionary of already processed loc_keys, it's value is number of times it was processed.
             * @return Paths (std::vector<std::vector<std::shared_ptr<IRBlock>>> Paths)
             */
            Paths find_path_from_src(std::shared_ptr<IRBlock> src,
                                     std::shared_ptr<IRBlock> dst,
                                     size_t cycles_count,
                                     std::map<std::shared_ptr<IRBlock>, size_t> done);

            /**
             * @brief Get the reachable sons of a given block.
             * @param head: head block to calculate reachable sons.
             * @return Nodes
             */
            Nodes reachable_sons(std::shared_ptr<IRBlock> head);

            /**
             * @brief Get the reachable parent nodes of a given block.
             * @param leaf: leaf block to calculate reachable parents.
             * @return Nodes
             */
            Nodes reachable_parents(std::shared_ptr<IRBlock> leaf);
            
            /**
             * @brief Compute the dominators of the graph.
             * @param head: head used as starting point to calculate the dominators in the graph.
             * @return std::map<std::shared_ptr<IRBlock>, Nodes>
             */
            std::map<std::shared_ptr<IRBlock>, Nodes> compute_dominators(std::shared_ptr<IRBlock> head);
            
            /**
             * @brief Compute the postdominators of the graph.
             * @param leaf: node to get its postdominators.
             * @return std::map<std::shared_ptr<IRBlock>, Nodes>
             */
            std::map<std::shared_ptr<IRBlock>, Nodes> compute_postdominators(std::shared_ptr<IRBlock> leaf);

            /**
             * @brief Compute the immediate dominators algorithm from:
             *        "Advanced Compiler Design and Implementation"
             * 
             * @return std::map<std::shared_ptr<IRBlock>, std::shared_ptr<IRBlock>> 
             */
            std::map<std::shared_ptr<IRBlock>, std::shared_ptr<IRBlock>> compute_immediate_dominators();

            /**
             * @brief Create a copy of the IRGraph as a smart pointer.
             * @return std::shared_ptr<IRGraph>
             */
            std::shared_ptr<IRGraph> copy();


            // node information
            /**
             * @brief Get the number of successor blocks from the given block.
             * @param node: block to get its number of successors.
             * @return size_t
             */
            size_t get_number_of_successors(std::shared_ptr<IRBlock> node);
            
            /**
             * @brief Get the list of successor blocks.
             * @param node: block to get its successors.
             * @return std::vector<std::shared_ptr<IRBlock>>
             */
            Nodes get_successors(std::shared_ptr<IRBlock> node);

            /**
             * @brief Get the number of predecessor blocks from the given block.
             * @param node: block to get its number of predecessors.
             * @return size_t
             */
            size_t get_number_of_predecessors(std::shared_ptr<IRBlock> node);
            
            /**
             * @brief Get the list of predecessor blocks.
             * @param node: block to get its predecessors.
             * @return std::vector<std::shared_ptr<IRBlock>>
             */
            Nodes get_predecessors(std::shared_ptr<IRBlock> node);

            /**
             * @brief Get type of node depending on number of successors and predecessors.
             * @param node: node to check its type.
             * @return node_type_t
             */
            node_type_t get_type_of_node(std::shared_ptr<IRBlock> node);

            // algorithms from Advanced Compiler Design and Implementation
            
            /**
             * @brief Get all the reachable nodes from a given head node.
             * @param head: head of the set we want to get all its reachable nodes.
             * @return Nodes
             */
            Nodes reachable_nodes_forward(std::shared_ptr<IRBlock> head);
            
            /**
             * @brief Get all the reachable up nodes from a given leaf node.
             * @param leaf: leaf of the set we of nodes we want to get all its reachable nodes.
             * @return Nodes
             */
            Nodes reachable_nodes_backward(std::shared_ptr<IRBlock> leaf);
            
            /**
             * @brief An extended basic block is a maximal sequence of instructions beginning with a leader
             *        that contains no join nodes (those with more than 1 successor) other than its first node
             *        extended basic block has single entry and possible multiply entries.
             *        Algorithm taken from "Advanced Compiler Design and Implementation" by Steven Muchnick
             * @param r: entry block to do the extended basic block.
             * @return Nodes
             */
            Nodes build_ebb(std::shared_ptr<IRBlock> r);
            
            /**
             * @brief Given a head node, give the tree of nodes using a Depth First Search algorithm.
             * @param head: node where to start the search.
             * @return Nodes
             */
            Nodes Deep_First_Search(std::shared_ptr<IRBlock> head);
            
            /**
             * @brief Given a head node, give the tree of nodes using a Breadth First Search algorithm.
             * @param head: node where to start the search.
             * @return Nodes
             */
            Nodes Breadth_First_Search(std::shared_ptr<IRBlock> head);

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

            std::map<std::shared_ptr<IRBlock>, Nodes> successors;
            std::map<std::shared_ptr<IRBlock>, Nodes> predecessors;
            
            void add_bbs(std::shared_ptr<IRBlock> r, Nodes ebb);

            std::uint64_t cyclomatic_complexity = -1;
            
        };
    } // namespace MJOLNIR

} // namespace KUNAI
