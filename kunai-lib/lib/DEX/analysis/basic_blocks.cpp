//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file basic_blocks.cpp
// @brief Definitions of basic blocks from analysis.hpp

#include "Kunai/DEX/analysis/analysis.hpp"

using namespace KUNAI::DEX;

void BasicBlocks::remove_node(DVMBasicBlock *node)
{
    // with this we provide RAII
    // TODO: Remove unused variable
    std::unique_ptr<DVMBasicBlock> const node_ (node);

    if (std::find(nodes_.begin(), nodes_.end(), node) == nodes_.end())
        throw exceptions::AnalysisException("remove_mode: given node does not exist in graph");

    if (node->is_start_block() || node->is_end_block())
        throw exceptions::AnalysisException("remove_node: start or end blocks cannot be removed");

    auto node_type = get_node_type(node);

    if (node_type == JOIN_NODE) // len(predecessors) > 1
    {
        auto suc = *sucessors_[node].begin();

        // delete from predecessors of sucessor
        predecessors_[suc].erase(node);
        // remove the edge
        std::remove(edges_.begin(), edges_.end(), std::make_pair(node, suc));

        for (auto pred : predecessors_[node])
        {
            // delete the edge from predecessor to the node
            std::remove(edges_.begin(), edges_.end(), std::make_pair(pred, node));
            // delete from sucessors[pred] the node
            sucessors_[pred].erase(node);
        }

        for (auto pred : predecessors_[node])
        {
            // now add new one with successor
            edges_.push_back(std::make_pair(pred, suc));
            // add the predecessor of sucesspr
            predecessors_[suc]
                .insert(pred);
            // add the sucessor of pred
            sucessors_[pred].insert(suc);
        }
    }
    else if (node_type == BRANCH_NODE) // len(sucessors) > 1
    {
        auto pred = *predecessors_[node].begin();

        // delete from sucessors of pred
        sucessors_[pred].erase(node);
        // remove the edge
        remove(edges_.begin(), edges_.end(), std::make_pair(pred, node));

        // now disconnect the node from the sucessors
        for (auto suc : sucessors_[node])
        {
            // remove the edges node->suc
            std::remove(edges_.begin(), edges_.end(), std::make_pair(node, suc));
            // remove the node as predecessor of this sucessor
            predecessors_[suc].erase(node);
        }

        for (auto suc : sucessors_[node])
        {
            // add the edges
            edges_.push_back(std::make_pair(pred, suc));
            // add the predecessor of sucesspr
            predecessors_[suc].insert(pred);
            // add the sucessor of pred
            sucessors_[pred].insert(suc);
        }
    }
    else
    {
        DVMBasicBlock *pred, *suc;
        if (predecessors_[node].size() == 1)
        {
            pred = *predecessors_[node].begin();

            // delete from sucessors of pred
            sucessors_[pred].erase(node);
            // remove the edge
            remove(edges_.begin(), edges_.end(), std::make_pair(pred, node));
        }

        if (sucessors_[node].size() == 1)
        {
            auto suc = *sucessors_[node].begin();

            // delete from predecessors of sucessor
            predecessors_[suc].erase(node);
            // remove the edge
            std::remove(edges_.begin(), edges_.end(), std::make_pair(node, suc));
        }

        if (pred != nullptr && suc != nullptr)
        {
            edges_.push_back(std::make_pair(pred, suc));
            // add sucessor to pred
            sucessors_[pred].insert(suc);
            // add predecessor to suc
            predecessors_[suc].insert(pred);
        }
    }

    // now delete the node from the predecessors and sucessors
    predecessors_.erase(node);
    sucessors_.erase(node);

    // finally delete from vector
    std::remove(nodes_.begin(), nodes_.end(), node);
}

DVMBasicBlock *BasicBlocks::get_basic_block_by_idx(std::uint64_t idx)
{
    for (const auto node : nodes_)
    {
        if (node->is_start_block() || node->is_end_block())
            continue;
        if (idx >= node->get_first_address() && idx <= node->get_last_address())
            return node;
    }

    return nullptr;
}
