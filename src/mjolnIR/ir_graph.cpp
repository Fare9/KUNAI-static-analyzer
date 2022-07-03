#include "KUNAI/mjolnIR/ir_graph.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        std::unique_ptr<IRGraph> get_unique_empty_graph()
        {
            return std::make_unique<IRGraph>();
        }

        irgraph_t get_shared_empty_graph()
        {
            return std::make_shared<IRGraph>();
        }

        /**
         * IRGraph class
         */

        IRGraph::IRGraph() {}

        
        bool IRGraph::add_node(irblock_t node)
        {
            if (std::find(nodes.begin(), nodes.end(), node) != nodes.end())
                return false; // node already exists, return false.
            nodes.push_back(node);
            return true;
        }

        
        void IRGraph::add_edge(irblock_t src, irblock_t dst)
        {
            if (std::find(nodes.begin(), nodes.end(), src) != nodes.end())
                add_node(src);
            if (std::find(nodes.begin(), nodes.end(), dst) != nodes.end())
                add_node(dst);
            edges.push_back(std::make_pair(src, dst));
            add_block_to_sucessors(src, dst);
            add_block_to_predecessors(dst, src);
        }

        
        void IRGraph::add_uniq_edge(irblock_t src, irblock_t dst)
        {
            // if src is not in nodes, or if
            // dst is not in the successors from source
            if ((std::find(nodes.begin(), nodes.end(), src) == nodes.end()) ||
                (std::find(successors[src].begin(), successors[src].end(), dst) == successors[src].end()))
                add_edge(src, dst);
        }

        
        void IRGraph::add_block_to_sucessors(irblock_t node, irblock_t successor)
        {
            successors[node].push_back(successor);
        }

        
        void IRGraph::add_block_to_predecessors(irblock_t node, irblock_t predecessor)
        {
            predecessors[node].push_back(predecessor);
        }


        std::optional<irblock_t> IRGraph::get_node_by_start_idx(std::uint64_t idx)
        {
            for (auto& node : nodes)
            {
                if (idx == node->get_start_idx())
                    return node;
            }

            return std::nullopt;
        }

        void IRGraph::merge_graph(irgraph_t graph)
        {
            auto nodes = graph->get_nodes();
            auto edges = graph->get_edges();

            for (auto node : nodes)
                add_node(node);
            for (auto edge : edges)
                add_edge(edge.first, edge.second);
        }

    
        void IRGraph::del_edge(irblock_t src, irblock_t dst)
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

        
        void IRGraph::del_node(irblock_t node)
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

        
        void IRGraph::delete_block_from_sucessors(irblock_t node, irblock_t block)
        {
            auto node_it = successors.find(node);
            if (node_it != successors.end())
            {
                auto succ_it = std::find(node_it->second.begin(), node_it->second.end(), block);
                node_it->second.erase(succ_it);
            }
        }

        
        void IRGraph::delete_block_from_precessors(irblock_t node, irblock_t block)
        {
            auto node_it = predecessors.find(node);
            if (node_it != predecessors.end())
            {
                auto pred_it = std::find(node_it->second.begin(), node_it->second.end(), block);
                node_it->second.erase(pred_it);
            }
        }

        
        std::vector<irblock_t> IRGraph::get_leaves()
        {
            std::vector<irblock_t> leaves;

            for (auto node : nodes)
            {
                if (get_number_of_successors(node) == 0)
                    leaves.push_back(node);
            }

            return leaves;
        }

        
        std::vector<irblock_t> IRGraph::get_heads()
        {
            std::vector<irblock_t> heads;

            for (auto node : nodes)
            {
                if (get_number_of_predecessors(node) == 0)
                    heads.push_back(node);
            }

            return heads;
        }

        
        Paths IRGraph::find_path(irblock_t src,
                                 irblock_t dst,
                                 size_t cycles_count,
                                 std::map<irblock_t, size_t> done)
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

        
        Paths IRGraph::find_path_from_src(irblock_t src,
                                          irblock_t dst,
                                          size_t cycles_count,
                                          std::map<irblock_t, size_t> done)
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

        
        Nodes IRGraph::reachable_sons(irblock_t head)
        {
            return IRGraph::reachable_nodes_forward(head);
        }

        
        Nodes IRGraph::reachable_parents(irblock_t leaf)
        {
            return IRGraph::reachable_nodes_backward(leaf);
        }

        
        std::map<irblock_t, Nodes> IRGraph::compute_dominators(irblock_t head)
        {
            std::map<irblock_t, Nodes> dominators;

            auto nodes = reachable_sons(head);

            for (auto node : nodes)
                dominators[node] = nodes;

            dominators[head] = {head};

            Nodes todo = nodes;

            while (!todo.empty())
            {
                auto node_it = todo.begin();
                auto node = *node_it;
                todo.erase(node_it);

                if (node == head)
                    // do not use head for computing dominators
                    continue;

                // computer intersection of all predecessors'dominators
                Nodes new_dom = {};
                for (auto pred : get_predecessors(node))
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

                new_dom.push_back(node);

                if (new_dom == dominators[node])
                    continue;

                dominators[node] = new_dom;
                for (auto succ : get_successors(node))
                    todo.push_back(succ);
            }

            return dominators;
        }

        
        std::map<irblock_t, Nodes> IRGraph::compute_postdominators(irblock_t leaf)
        {
            std::map<irblock_t, Nodes> postdominators;

            auto nodes = reachable_parents(leaf);

            for (auto node : nodes)
                postdominators[node] = nodes;

            postdominators[leaf] = {leaf};

            Nodes todo = nodes;

            while (!todo.empty())
            {
                auto node_it = todo.begin();
                auto node = *node_it;
                todo.erase(node_it);

                if (node == leaf)
                    // do not use head for computing dominators
                    continue;

                // computer intersection of all predecessors'dominators
                Nodes new_dom = {};
                for (auto succ : get_successors(node))
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

                new_dom.push_back(node);

                if (new_dom == postdominators[node])
                    continue;

                postdominators[node] = new_dom;
                for (auto pred : get_predecessors(node))
                    todo.push_back(pred);
            }

            return postdominators;
        }

        
        std::map<irblock_t, irblock_t> IRGraph::compute_immediate_dominators()
        {
            std::map<irblock_t, Nodes> tmp;
            std::map<irblock_t, irblock_t> idom;

            if (nodes.size() == 0)
                return idom;

            auto first_node = nodes[0];

            // compute the dominators
            tmp = compute_dominators(first_node);
            
            // remove itself from dominators
            for (auto item = tmp.begin(); item != tmp.end(); item++)
            {
                auto rem = std::find(item->second.begin(), item->second.end(), item->first);

                if (rem != item->second.end())
                    item->second.erase(rem);
            }

            for (auto n : nodes)
            {
                for (auto s : tmp[n])
                {
                    for (auto t : tmp[n])
                    {
                        if (t == s)
                            continue;
                        
                        if (std::find(tmp[s].begin(), tmp[s].end(), t) != tmp[s].end())
                        {
                            auto rem = std::find(tmp[n].begin(), tmp[n].end(), t);

                            if (rem != tmp[n].end())
                                tmp[n].erase(rem);
                        }
                    }
                }
            }

            for (auto n : nodes)
            {
                if (tmp[n].size() == 1)
                    idom[n] = tmp[n][0];
                else
                    idom[n] = nullptr; 
            }

            return idom;
        }

        std::map<irblock_t, std::set<irblock_t>> IRGraph::compute_dominance_frontier()
        {
            /*
            * Compute the immediate dominators from all the
            * nodes.
            */
            auto idoms = compute_immediate_dominators();
            std::map<irblock_t, std::set<irblock_t>> frontier;


            for (auto& idom : idoms)
            {
                if (predecessors.find(idom.first) == predecessors.end() || predecessors.at(idom.first).size() < 2)
                    continue;

                // check if the node has more than 1 predecessor
                // this node is a convergence node
                for (auto& runner : predecessors.at(idom.first))
                {
                    // check if the predecessor is in the
                    // map of immediate dominators nodes.
                    if (idoms.find(runner) == idoms.end())
                        continue;
                    
                    while (runner != idom.second)
                    {
                        frontier[runner].insert(idom.first);
                        runner = idoms.at(runner);
                    }
                }
                
            }

            return frontier;
        }
        
        irgraph_t IRGraph::copy()
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
        
        size_t IRGraph::get_number_of_successors(irblock_t node)
        {
            if (successors.find(node) == successors.end())
                return 0;
            return successors[node].size();
        }

        
        Nodes IRGraph::get_successors(irblock_t node)
        {
            if (successors.find(node) == successors.end())
                return {};
            return successors[node];
        }

        
        size_t IRGraph::get_number_of_predecessors(irblock_t node)
        {
            if (predecessors.find(node) == predecessors.end())
                return 0;
            return predecessors[node].size();
        }

        
        Nodes IRGraph::get_predecessors(irblock_t node)
        {
            if (predecessors.find(node) == predecessors.end())
                return {};
            return predecessors[node];
        }

        
        IRGraph::node_type_t IRGraph::get_type_of_node(irblock_t node)
        {
            if (get_number_of_successors(node) > 1)
                return BRANCH_NODE;
            else if (get_number_of_predecessors(node) > 1)
                return JOIN_NODE;
            else
                return REGULAR_NODE;
        }

        
        Nodes IRGraph::reachable_nodes_forward(irblock_t head)
        {
            Nodes todo;
            Nodes reachable;

            todo.push_back(head);

            while (!todo.empty())
            {
                // similar to python pop.
                auto node_it = todo.begin();
                auto node = *node_it;
                todo.erase(node_it);

                // node already in reachable
                if (std::find(reachable.begin(), reachable.end(), node) != reachable.end())
                    continue;

                reachable.push_back(node);

                for (auto next_node : get_successors(node))
                    todo.push_back(next_node);
            }

            return reachable;
        }

        
        Nodes IRGraph::reachable_nodes_backward(irblock_t leaf)
        {
            Nodes todo;
            Nodes reachable;

            todo.push_back(leaf);

            while (!todo.empty())
            {
                // similar to python pop.
                auto node_it = todo.begin();
                auto node = *node_it;
                todo.erase(node_it);

                // node already in reachable
                if (std::find(reachable.begin(), reachable.end(), node) != reachable.end())
                    continue;

                reachable.push_back(node);

                for (auto next_node : get_predecessors(node))
                    todo.push_back(next_node);
            }

            return reachable;
        }

        
        Nodes IRGraph::build_ebb(irblock_t r)
        {
            Nodes ebb;

            add_bbs(r, ebb);

            return ebb;
        }

        
        Nodes IRGraph::Deep_First_Search(irblock_t head)
        {
            Nodes todo;
            Nodes done;

            todo.push_back(head);

            while (!todo.empty())
            {
                // pop last element
                auto node_it = todo.end();
                node_it--;
                auto node = *node_it;
                todo.erase(node_it);

                if (std::find(done.begin(), done.end(), node) != done.end())
                    continue;

                done.push_back(node);

                for (auto succ : get_successors(node))
                    todo.push_back(succ);
            }

            return done;
        }

        
        Nodes IRGraph::Breadth_First_Search(irblock_t head)
        {
            Nodes todo;
            Nodes done;

            todo.push_back(head);

            while (!todo.empty())
            {
                // pop first element
                auto node_it = todo.begin();
                auto node = *node_it;
                todo.erase(node_it);

                if (std::find(done.begin(), done.end(), node) != done.end())
                    continue;

                done.push_back(node);

                for (auto succ : get_successors(node))
                    todo.push_back(succ);
            }

            return done;
        }

        void IRGraph::add_bbs(irblock_t r, Nodes ebb)
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

        
        void IRGraph::generate_dot_file(std::string name)
        {
            std::ofstream stream;

            stream.open(name + std::string(".dot")); // open dot file and write on it
            //std::stringstream stream;

            stream << "digraph \"" << name << "\"{\n";
            stream << "style=\"dashed\";\n";
            stream << "color=\"black\";\n";
            stream << "label=\"" << name << "\";\n";

            // fisrt of all print the nodes
            for (auto node : nodes)
            {
                stream << "\"" << node->get_name() << "\""
                       << " [shape=box, style=filled, fillcolor=lightgrey, label=\"";

                auto content = node->to_string();

                while (content.find("\n") != content.npos)
                    content.replace(content.find("\n"), 1, "\\l", 2);

                stream << content;

                stream << "\"];\n\n";
            }

            for (auto edge : edges)
            {
                auto bb1 = std::get<0>(edge);
                auto bb2 = std::get<1>(edge);

                stream << "\"" << bb1->get_name() << "\" -> "
                       << "\"" << bb2->get_name() << "\" [style=\"solid,bold\",color=black,weight=10,constraint=true];\n";
            }

            stream << "}";

            stream.close();
        }

        
        void IRGraph::generate_dominator_tree(std::string name)
        {
            IRGraph graph;

            auto idoms = this->compute_immediate_dominators();

            // add the nodes and the edges.
            for (auto idom : idoms)
            {
                graph.add_node(idom.first);

                if (idom.second)
                {
                    graph.add_node(idom.second);

                    graph.add_edge(idom.second, idom.first);
                }
            }

            graph.generate_dot_file(name);
        }

        const std::uint64_t IRGraph::get_cyclomatic_complexity()
        {
            if (cyclomatic_complexity != -1)
            {
                return cyclomatic_complexity;
            }

            auto logger = LOGGER::logger();

            // take a copy of nodes and edges
            auto & nodes_aux = nodes;
            auto & edges_aux = edges;
            
            auto E = edges_aux.size();
            auto N = nodes_aux.size();


            size_t P = 0;

            // Go through all the nodes to calculate those
            // which are exit nodes
            for (auto node : nodes_aux)
            {
                auto statements = node->get_statements();
                // check all instructions
                for (auto stmnt : statements)
                {
                    if (ret_ir(stmnt))
                    {
                        P += 1;
                        break;
                    }
                }
            }

            cyclomatic_complexity = E - N + P*2;

            logger->info("Calculated cyclomatic complexity: {}", cyclomatic_complexity);

            return cyclomatic_complexity;
        }

    } // namespace MJOLNIR
} // namespace KUNAI