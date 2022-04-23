#include "ir_graph.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        /**
         * IRGraph class
         */

        IRGraph::IRGraph() {}

        
        bool IRGraph::add_node(std::shared_ptr<IRBlock> node)
        {
            if (std::find(nodes.begin(), nodes.end(), node) != nodes.end())
                return false; // node already exists, return false.
            nodes.push_back(node);
            return true;
        }

        
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

        
        void IRGraph::add_uniq_edge(std::shared_ptr<IRBlock> src, std::shared_ptr<IRBlock> dst)
        {
            // if src is not in nodes, or if
            // dst is not in the successors from source
            if ((std::find(nodes.begin(), nodes.end(), src) == nodes.end()) ||
                (std::find(successors[src].begin(), successors[src].end(), dst) == successors[src].end()))
                add_edge(src, dst);
        }

        
        void IRGraph::add_block_to_sucessors(std::shared_ptr<IRBlock> node, std::shared_ptr<IRBlock> successor)
        {
            successors[node].push_back(successor);
        }

        
        void IRGraph::add_block_to_predecessors(std::shared_ptr<IRBlock> node, std::shared_ptr<IRBlock> predecessor)
        {
            predecessors[node].push_back(predecessor);
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

        
        void IRGraph::delete_block_from_sucessors(std::shared_ptr<IRBlock> node, std::shared_ptr<IRBlock> block)
        {
            auto node_it = successors.find(node);
            if (node_it != successors.end())
            {
                auto succ_it = std::find(node_it->second.begin(), node_it->second.end(), block);
                node_it->second.erase(succ_it);
            }
        }

        
        void IRGraph::delete_block_from_precessors(std::shared_ptr<IRBlock> node, std::shared_ptr<IRBlock> block)
        {
            auto node_it = predecessors.find(node);
            if (node_it != predecessors.end())
            {
                auto pred_it = std::find(node_it->second.begin(), node_it->second.end(), block);
                node_it->second.erase(pred_it);
            }
        }

        
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

        
        Nodes IRGraph::reachable_sons(std::shared_ptr<IRBlock> head)
        {
            return IRGraph::reachable_nodes_forward(head);
        }

        
        Nodes IRGraph::reachable_parents(std::shared_ptr<IRBlock> leaf)
        {
            return IRGraph::reachable_nodes_backward(leaf);
        }

        
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

        
        std::map<std::shared_ptr<IRBlock>, std::shared_ptr<IRBlock>> IRGraph::compute_immediate_dominators()
        {
            std::map<std::shared_ptr<IRBlock>, Nodes> tmp;
            std::map<std::shared_ptr<IRBlock>, std::shared_ptr<IRBlock>> idom;

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
        
        size_t IRGraph::get_number_of_successors(std::shared_ptr<IRBlock> node)
        {
            if (successors.find(node) == successors.end())
                return 0;
            return successors[node].size();
        }

        
        Nodes IRGraph::get_successors(std::shared_ptr<IRBlock> node)
        {
            if (successors.find(node) == successors.end())
                return {};
            return successors[node];
        }

        
        size_t IRGraph::get_number_of_predecessors(std::shared_ptr<IRBlock> node)
        {
            if (predecessors.find(node) == predecessors.end())
                return 0;
            return predecessors[node].size();
        }

        
        Nodes IRGraph::get_predecessors(std::shared_ptr<IRBlock> node)
        {
            if (predecessors.find(node) == predecessors.end())
                return {};
            return predecessors[node];
        }

        
        IRGraph::node_type_t IRGraph::get_type_of_node(std::shared_ptr<IRBlock> node)
        {
            if (get_number_of_successors(node) > 1)
                return BRANCH_NODE;
            else if (get_number_of_predecessors(node) > 1)
                return JOIN_NODE;
            else
                return REGULAR_NODE;
        }

        
        Nodes IRGraph::reachable_nodes_forward(std::shared_ptr<IRBlock> head)
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

        
        Nodes IRGraph::reachable_nodes_backward(std::shared_ptr<IRBlock> leaf)
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

        
        Nodes IRGraph::build_ebb(std::shared_ptr<IRBlock> r)
        {
            Nodes ebb;

            add_bbs(r, ebb);

            return ebb;
        }

        
        Nodes IRGraph::Deep_First_Search(std::shared_ptr<IRBlock> head)
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

        
        Nodes IRGraph::Breadth_First_Search(std::shared_ptr<IRBlock> head)
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
                    if (is_ret(stmnt))
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