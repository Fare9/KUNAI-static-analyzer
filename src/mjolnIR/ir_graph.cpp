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
            if (std::find(nodes.begin(), nodes.end(), node) != nodes.end())
                return false; // node already exists, return false.
            nodes.push_back(node);
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
            if (std::find(nodes.begin(), nodes.end(), src) != nodes.end())
                add_node(src);
            if (std::find(nodes.begin(), nodes.end(), dst) != nodes.end())
                add_node(dst);
            edges.push_back(std::make_pair(src, dst));
            add_block_to_sucessors(src, dst);
            add_block_to_predecessors(dst, src);
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
                (std::find(successors[src].begin(), successors[src].end(), dst) != successors[src].end()))
                add_edge(src, dst);
        }

        /**
         * @brief Add a block as one of the successors of the given block.
         * @param node: node to add its successor.
         * @param successor: successor block to add.
         * @return void
         */
        void IRGraph::add_block_to_sucessors(std::shared_ptr<IRBlock> node, std::shared_ptr<IRBlock> successor)
        {
            successors[node].push_back(successor);
        }

        /**
         * @brief Add a block as on of the precessors of the given block.
         * @param node: node to add its predecessor.
         * @param predecessor: predecessor block to add.
         * @return void
         */
        void IRGraph::add_block_to_predecessors(std::shared_ptr<IRBlock> node, std::shared_ptr<IRBlock> predecessor)
        {
            predecessors[node].push_back(predecessor);
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

            for (auto node : nodes)
                add_node(node);
            for (auto edge : edges)
                add_edge(edge.first, edge.second);
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
            delete_block_from_sucessors(src, dst);
            delete_block_from_precessors(dst, src);
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
            auto node_set = std::find(nodes.begin(), nodes.end(), node);
            auto predecessors = get_predecessors(node);
            auto successors = get_successors(node);

            if (node_set == nodes.end())
                return; // if node does not exist, return.

            nodes.erase(node_set);
            for (auto pred : predecessors)
                del_edge(pred, node);
            for (auto succ : successors)
                del_edge(node, succ);
        }

        /**
         * @brief Remove one of the sucessors given an IRBlock, if it's inside of the vector.
         * @param node: node where to remove its successors.
         * @param block: Block to remove from the successors.
         * @return void
         */
        void IRGraph::delete_block_from_sucessors(std::shared_ptr<IRBlock> node, std::shared_ptr<IRBlock> block)
        {
            auto node_it = successors.find(node);
            if (node_it != successors.end())
            {
                auto succ_it = std::find(node_it->second.begin(), node_it->second.end(), block);
                node_it->second.erase(succ_it);
            }
        }

        /**
         * @brief Remove on of the predecessors given an IRBlock, if it's inside of the vector.
         * @param node: node where to remove its predecessor.
         * @param block: Block to remove from the predecessors.
         * @return void
         */
        void IRGraph::delete_block_from_precessors(std::shared_ptr<IRBlock> node, std::shared_ptr<IRBlock> block)
        {
            auto node_it = predecessors.find(node);
            if (node_it != predecessors.end())
            {
                auto pred_it = std::find(node_it->second.begin(), node_it->second.end(), block);
                node_it->second.erase(pred_it);
            }
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
                if (get_number_of_successors(node) == 0)
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
                if (get_number_of_predecessors(node) == 0)
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

            for (auto node : get_predecessors(dst))
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

        /**
         * @brief Find a different paths between a source block and destination block from the source to forward
         *        in the graph. 
         * @param src: source block from the graph.
         * @param dst: destination block from the graph.
         * @param cycles_count: maximum number of times a basic block can be processed.
         * @param done: dictionary of already processed loc_keys, it's value is number of times it was processed.
         * @return Paths (std::vector<std::vector<std::shared_ptr<IRBlock>>> Paths)
         */
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

            for (auto node : get_successors(src))
            {
                if (done.find(src) == done.end())
                    done[src] = 0;
                else
                    done[src] += 1;

                for (auto path : find_path(src, node, cycles_count, done))
                {
                    if (!path.empty() && path[path.size() - 1] == dst)
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
            return IRGraph::reachable_nodes_forward(head);
        }

        /**
         * @brief Get the reachable parent nodes of a given block.
         * @param leaf: leaf block to calculate reachable parents.
         * @return Nodes
         */
        Nodes IRGraph::reachable_parents(std::shared_ptr<IRBlock> leaf)
        {
            return IRGraph::reachable_nodes_backward(leaf);
        }

        /**
         * @brief Compute the dominators of the graph.
         * @param head: head used as starting point to calculate the dominators in the graph.
         * @return std::map<std::shared_ptr<IRBlock>, Nodes>
         */
        std::map<std::shared_ptr<IRBlock>, Nodes> IRGraph::compute_dominators(std::shared_ptr<IRBlock> head)
        {
            std::map<std::shared_ptr<IRBlock>, Nodes> dominators;

            auto nodes = reachable_sons(head);

            for (auto node : nodes)
                dominators[node] = nodes;

            dominators[head] = {head};

            Nodes todo = nodes;

            while (!todo.empty())
            {
                auto node = todo.begin();
                todo.erase(node);

                if ((*node) == head)
                    // do not use head for computing dominators
                    continue;

                // computer intersection of all predecessors'dominators
                Nodes new_dom = {};
                for (auto pred : get_predecessors(*node))
                {
                    if (std::find(nodes.begin(), nodes.end(), pred) == nodes.end()) // pred is not in nodes
                        continue;
                    if (new_dom.empty())
                        new_dom = dominators[pred];

                    Nodes intersect_aux;
                    std::set_intersection(new_dom.begin(), new_dom.end(),
                                          dominators[pred].begin(), dominators[pred].end(),
                                          std::inserter(intersect_aux, intersect_aux.begin()));
                    new_dom = intersect_aux;
                }

                new_dom.push_back(*node);

                if (new_dom == dominators[*node])
                    continue;

                dominators[*node] = new_dom;
                for (auto succ : get_successors(*node))
                    todo.push_back(succ);
            }

            return dominators;
        }

        /**
         * @brief Compute the postdominators of the graph.
         * @param leaf: node to get its postdominators.
         * @return std::map<std::shared_ptr<IRBlock>, Nodes>
         */
        std::map<std::shared_ptr<IRBlock>, Nodes> IRGraph::compute_postdominators(std::shared_ptr<IRBlock> leaf)
        {
            std::map<std::shared_ptr<IRBlock>, Nodes> postdominators;

            auto nodes = reachable_parents(leaf);

            for (auto node : nodes)
                postdominators[node] = nodes;

            postdominators[leaf] = {leaf};

            Nodes todo = nodes;

            while (!todo.empty())
            {
                auto node = todo.begin();
                todo.erase(node);

                if ((*node) == leaf)
                    // do not use head for computing dominators
                    continue;

                // computer intersection of all predecessors'dominators
                Nodes new_dom = {};
                for (auto succ : get_successors(*node))
                {
                    if (std::find(nodes.begin(), nodes.end(), succ) == nodes.end()) // pred is not in nodes
                        continue;
                    if (new_dom.empty())
                        new_dom = postdominators[succ];

                    Nodes intersect_aux;
                    std::set_intersection(new_dom.begin(), new_dom.end(),
                                          postdominators[succ].begin(), postdominators[succ].end(),
                                          std::inserter(intersect_aux, intersect_aux.begin()));
                    new_dom = intersect_aux;
                }

                new_dom.push_back(*node);

                if (new_dom == postdominators[*node])
                    continue;

                postdominators[*node] = new_dom;
                for (auto pred : get_predecessors(*node))
                    todo.push_back(pred);
            }

            return postdominators;
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

            for (auto node : nodes)
                new_graph->add_node(node);
            for (auto edge : edges)
                new_graph->add_edge(edge.first, edge.second);

            return new_graph;
        }

        // node information
        /**
         * @brief Get the number of successor blocks from the given block.
         * @param node: block to get its number of successors.
         * @return size_t
         */
        size_t IRGraph::get_number_of_successors(std::shared_ptr<IRBlock> node)
        {
            if (successors.find(node) == successors.end())
                return 0;
            return successors[node].size();
        }

        /**
         * @brief Get the list of successor blocks.
         * @param node: block to get its successors.
         * @return std::vector<std::shared_ptr<IRBlock>>
         */
        Nodes IRGraph::get_successors(std::shared_ptr<IRBlock> node)
        {
            if (successors.find(node) == successors.end())
                return {};
            return successors[node];
        }

        /**
         * @brief Get the number of predecessor blocks from the given block.
         * @param node: block to get its number of predecessors.
         * @return size_t
         */
        size_t IRGraph::get_number_of_predecessors(std::shared_ptr<IRBlock> node)
        {
            if (predecessors.find(node) == predecessors.end())
                return 0;
            return predecessors[node].size();
        }

        /**
         * @brief Get the list of predecessor blocks.
         * @param node: block to get its predecessors.
         * @return std::vector<std::shared_ptr<IRBlock>>
         */
        Nodes IRGraph::get_predecessors(std::shared_ptr<IRBlock> node)
        {
            if (predecessors.find(node) == predecessors.end())
                return {};
            return predecessors[node];
        }

        /**
         * @brief Get type of node depending on number of successors and predecessors.
         * @param node: node to check its type.
         * @return node_type_t
         */
        IRGraph::node_type_t IRGraph::get_type_of_node(std::shared_ptr<IRBlock> node)
        {
            if (get_number_of_successors(node) > 1)
                return BRANCH_NODE;
            else if (get_number_of_predecessors(node) > 1)
                return JOIN_NODE;
            else
                return REGULAR_NODE;
        }

        // static methods

        /**
         * @brief Get all the reachable nodes from a given head node.
         * @param head: head of the set we want to get all its reachable nodes.
         * @return Nodes
         */
        Nodes IRGraph::reachable_nodes_forward(std::shared_ptr<IRBlock> head)
        {
            Nodes todo;
            Nodes reachable;

            todo.push_back(head);

            while (!todo.empty())
            {
                // similar to python pop.
                auto node = todo.begin();
                todo.erase(node);

                // node already in reachable
                if (std::find(reachable.begin(), reachable.end(), *node) != reachable.end())
                    continue;

                reachable.push_back(*node);

                for (auto next_node : get_successors(*node))
                    todo.push_back(next_node);
            }

            return reachable;
        }

        /**
         * @brief Get all the reachable up nodes from a given leaf node.
         * @param leaf: leaf of the set we of nodes we want to get all its reachable nodes.
         * @return Nodes
         */
        Nodes IRGraph::reachable_nodes_backward(std::shared_ptr<IRBlock> leaf)
        {
            Nodes todo;
            Nodes reachable;

            todo.push_back(leaf);

            while (!todo.empty())
            {
                // similar to python pop.
                auto node = todo.begin();
                todo.erase(node);

                // node already in reachable
                if (std::find(reachable.begin(), reachable.end(), *node) != reachable.end())
                    continue;

                reachable.push_back(*node);

                for (auto next_node : get_predecessors(*node))
                    todo.push_back(next_node);
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

        /**
         * @brief Given a head node, give the tree of nodes using a Depth First Search algorithm.
         * @param head: node where to start the search.
         * @return Nodes
         */
        Nodes IRGraph::Deep_First_Search(std::shared_ptr<IRBlock> head)
        {
            Nodes todo;
            Nodes done;

            todo.push_back(head);

            while (!todo.empty())
            {
                // pop last element
                auto node = todo.end();
                node--;
                todo.erase(node);

                if (std::find(done.begin(), done.end(), *node) != done.end())            
                    continue;

                done.push_back(*node);

                for (auto succ : get_successors(*node))
                    todo.push_back(succ);
            }

            return done;
        }

        /**
         * @brief Given a head node, give the tree of nodes using a Breadth First Search algorithm.
         * @param head: node where to start the search.
         * @return Nodes
         */
        Nodes IRGraph::Breadth_First_Search(std::shared_ptr<IRBlock> head)
        {
            Nodes todo;
            Nodes done;

            todo.push_back(head);

            while (!todo.empty())
            {
                // pop first element
                auto node = todo.begin();
                todo.erase(node);

                if (std::find(done.begin(), done.end(), *node) != done.end())
                    continue;

                done.push_back(*node);

                for (auto succ : get_successors(*node))
                    todo.push_back(succ);
            }

            return done;
        }

        void IRGraph::add_bbs(std::shared_ptr<IRBlock> r, Nodes ebb)
        {
            // insert given block
            ebb.push_back(r);

            for (auto x : get_successors(r))
            {
                if (get_number_of_predecessors(x) == 1 && 
                    std::find(ebb.begin(), ebb.end(), x) == ebb.end())
                    add_bbs(x, ebb);
            }

            return;
        }

    } // namespace MJOLNIR
} // namespace KUNAI