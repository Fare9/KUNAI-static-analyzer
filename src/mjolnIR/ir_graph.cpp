#include "ir_graph.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        /**
         * IRGraph class
         */

        /**
         * @brief Constructor of IRGraph, this class represent a control-flow graph of IRBlock objects.
         * @return void
         */
        IRGraph::IRGraph() {}

        /**
         * @brief Destructor of IRGraph, nothing to be done.
         * @return void
         */
        IRGraph::~IRGraph() {}

        /**
         * @brief Add a node to the IRGraph.
         * @param node: new node to insert in set.
         * @return bool
         */
        bool IRGraph::add_node(std::shared_ptr<IRBlock> node)
        {
            if (nodes.find(node) != nodes.end())
                return false; // node already exists, return false.
            nodes.insert(node);
            return true;
        }

        /**
         * @brief Add a new edge to the IRGraph.
         * @param src: source node, will be inserted if it does not exists.
         * @param dst: destination node, will be inserted if it does not exists.
         * @return void
         */
        void IRGraph::add_edge(std::shared_ptr<IRBlock> src, std::shared_ptr<IRBlock> dst)
        {
            if (nodes.find(src) == nodes.end())
                add_node(src);
            if (nodes.find(dst) == nodes.end())
                add_node(dst);
            edges.push_back(std::make_pair(src, dst));
            src->add_block_to_sucessors(dst);
            dst->add_block_to_predecessors(src);
        }

        /**
         * @brief Get the set of nodes from the current graph.
         * @return Nodes
         */
        Nodes IRGraph::get_nodes()
        {
            return nodes;
        }

        /**
         * @brief Get the vector with pair of edges.
         * @return Edges
         */
        Edges IRGraph::get_edges()
        {
            return edges;
        }

        /**
         * @brief Merge one given graph with current one.
         * @param graph: graph to merge with current one.
         * @return void
         */
        void IRGraph::merge_graph(std::shared_ptr<IRGraph> graph)
        {
            auto nodes = graph->get_nodes();
            auto edges = graph->get_edges();

            for (auto node = nodes.begin(); node != nodes.end(); node++)
                add_node(*node);
            for (auto edge = edges.begin(); edge != edges.end(); edge++)
                add_edge(edge->first, edge->second);
        }

        /**
         * @brief Delete an edge given two blocks.
         * @param src: source node from the edge.
         * @param dst: destination node from the edge.
         * @return void
         */
        void IRGraph::del_edge(std::shared_ptr<IRBlock> src, std::shared_ptr<IRBlock> dst)
        {
            auto edge = std::find(edges.begin(), edges.end(), std::make_pair(src, dst));
            if (edge == edges.end())
                return; // if edge does not exist, return
            edges.erase(edge);
            // once the edge has been deleted
            // remove successors and predecessors
            src->delete_block_from_sucessors(dst);
            dst->delete_block_from_precessors(src);   
        }

        /**
         * @brief Delete a node from the graph, 
         *        remove all the edges that are connected
         *        to the node.
         * @param node: node to remove from the graph.
         * @return void
         */
        void IRGraph::del_node(std::shared_ptr<IRBlock> node)
        {
            auto node_set = nodes.find(node);
            auto predecessors = node->get_predecessors();
            auto successors = node->get_successors();

            if (node_set == nodes.end())
                return; // if node does not exist, return.
            
            nodes.erase(node_set);
            for (auto pred = predecessors.begin(); pred != predecessors.end(); pred++)
                del_edge(*pred, node);
            for (auto succ = successors.begin(); succ != successors.end(); succ++)
                del_edge(node, *succ);
        }
    } // namespace MJOLNIR
} // namespace KUNAI