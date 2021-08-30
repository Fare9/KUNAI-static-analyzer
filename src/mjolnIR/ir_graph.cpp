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
         * @brief Add a new edge if it doesn't exists already.
         * @param src: source node, will be inserted if it does not exists.
         * @param dst: destination node, will be inserted if it does not exists.
         * @return void
         */
        void IRGraph::add_uniq_edge(std::shared_ptr<IRBlock> src, std::shared_ptr<IRBlock> dst)
        {
            // if src is not in nodes, or if
            // dst is not in the successors from source
            if ((std::find(nodes.begin(), nodes.end(), src) == nodes.end()) ||
                (std::find(src->get_successors().begin(), src->get_successors().end(), dst) == src->get_successors().end()))
                add_edge(src, dst);
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

        /**
         * @brief Get all those nodes which do not have successors.
         * @return std::vector<std::shared_ptr<IRBlock>>
         */
        std::vector<std::shared_ptr<IRBlock>> IRGraph::get_leaves()
        {
            std::vector<std::shared_ptr<IRBlock>> leaves;

            for (auto node : nodes)
            {
                if (node->get_number_of_successors() == 0)
                    leaves.push_back(node);
            }

            return leaves;
        }

        /**
         * @brief Get all those nodes which are head of graph, which do not have predecessors.
         * @return std::vector<std::shared_ptr<IRBlock>>
         */
        std::vector<std::shared_ptr<IRBlock>> IRGraph::get_heads()
        {
            std::vector<std::shared_ptr<IRBlock>> heads;

            for (auto node : nodes)
            {
                if (node->get_number_of_predecessors() == 0)
                    heads.push_back(node);
            }

            return heads;
        }

        /**
         * @brief Find a different paths between a source block and destination block from the destination to backward
         *        in the graph. 
         * @param src: source block from the graph.
         * @param dst: destination block from the graph.
         * @param cycles_count: maximum number of times a basic block can be processed.
         * @param done: dictionary of already processed loc_keys, it's value is number of times it was processed.
         * @return Paths (std::vector<std::vector<std::shared_ptr<IRBlock>>> Paths)
         */
        Paths IRGraph::find_path(std::shared_ptr<IRBlock> src,
                                 std::shared_ptr<IRBlock> dst,
                                 size_t cycles_count,
                                 std::map<std::shared_ptr<IRBlock>, size_t> done)
        {
            if (done.find(dst) != done.end() && done[dst] > cycles_count)
                return {{}};
            if (src == dst)
                return {{src}};
                
            Paths out;

            for (auto node : dst->get_predecessors())
            {
                if (done.find(dst) == done.end())
                    done[dst] = 0;
                else
                    done[dst] += 1;

                for (auto path : find_path(src, node, cycles_count, done))
                {
                    if (!path.empty() && path[0] == src)
                    {
                        path.push_back(dst);
                        out.push_back(path);
                    }
                }
            }

            return out;
        }

        Paths IRGraph::find_path_from_src(std::shared_ptr<IRBlock> src,
                                          std::shared_ptr<IRBlock> dst,
                                          size_t cycles_count,
                                          std::map<std::shared_ptr<IRBlock>, size_t> done)
        {
            if (src == dst)
                return {{src}};

            if (done.find(src) != done.end() && done[src] > cycles_count)
                return {{}};

            Paths out;

            for (auto node : src->get_successors())
            {
                if (done.find(src) == done.end())
                    done[src] = 0;
                else
                    done[src] += 1;
                
                for (auto path : find_path(src, node, cycles_count, done))
                {
                    if (!path.empty() && path[path.size()-1] == dst)
                    {
                        path.insert(path.begin(), src);
                        out.push_back(path);
                    }
                }
            }

            return out;
        }

        /**
         * @brief Get the reachable sons of a given block.
         * @param head: head block to calculate reachable sons.
         * @return Nodes
         */
        Nodes IRGraph::reachable_sons(std::shared_ptr<IRBlock> head)
        {
            return IRGraph::reachable_nodes(head, true);
        }


        /**
         * @brief Get the reachable parent nodes of a given block.
         * @param leaf: leaf block to calculate reachable parents.
         * @return Nodes
         */
        Nodes IRGraph::reachable_parents(std::shared_ptr<IRBlock> leaf)
        {
            return IRGraph::reachable_nodes(leaf, false);
        }

        /**
         * @brief Create a copy of the IRGraph as a smart pointer.
         * @return std::shared_ptr<IRGraph>
         */
        std::shared_ptr<IRGraph> IRGraph::copy()
        {
            auto new_graph = std::make_shared<IRGraph>();

            auto nodes = get_nodes();
            auto edges = get_edges();

            for (auto node = nodes.begin(); node != nodes.end(); node++)
                new_graph->add_node(*node);
            for (auto edge = edges.begin(); edge != edges.end(); edge++)
                new_graph->add_edge(edge->first, edge->second);

            return new_graph;
        }

        // static methods

        /**
         * @brief Get all the reachable nodes from a given head node.
         * @param head: head of the set we want to get all its reachable nodes.
         * @param go_successors: use successors to calculate the reachable nodes, in case of false, use the predecessors.
         * @return Nodes
         */
        Nodes IRGraph::reachable_nodes(std::shared_ptr<IRBlock> head, bool go_successors)
        {
            Nodes todo;
            Nodes reachable;

            while (!todo.empty())
            {
                // similar to python pop.
                auto node = todo.begin();
                todo.erase(node);                

                // node already in reachable
                if (reachable.find((*node)) != reachable.end())
                    continue;
                
                reachable.insert(*node);

                if (go_successors)
                {
                    for (auto next_node : (*node)->get_successors())
                        todo.insert(next_node);
                }else
                {
                    for (auto next_node : (*node)->get_predecessors())
                        todo.insert(next_node);
                }
                
            }
            
            return reachable;
        }

        /**
         * @brief An extended basic block is a maximal sequence of instructions beginning with a leader
         *        that contains no join nodes (those with more than 1 successor) other than its first node
         *        extended basic block has single entry and possible multiply entries.
         *        Algorithm taken from "Advanced Compiler Design and Implementation" by Steven Muchnick
         * @param r: entry block to do the extended basic block.
         * @return Nodes
         */
        Nodes IRGraph::build_ebb(std::shared_ptr<IRBlock> r)
        {
            Nodes ebb;

            add_bbs(r, ebb);

            return ebb;
        }


        void IRGraph::add_bbs(std::shared_ptr<IRBlock> r, Nodes ebb)
        {
            // insert given block
            ebb.insert(r);

            for (auto x : r->get_successors())
            {
                if (x->get_number_of_predecessors() == 1 && ebb.find(x) == ebb.end())
                    add_bbs(x, ebb);
            }

            return;
        }

    } // namespace MJOLNIR
} // namespace KUNAI