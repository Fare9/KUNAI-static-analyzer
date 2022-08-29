#include "KUNAI/mjolnIR/Analysis/single_block_optimizations.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        std::optional<irblock_t> nop_removal(irblock_t &block)
        {
            auto &stmnts = block->get_statements();
            bool changed = false;

            for (auto it = stmnts.begin(); it != stmnts.end(); it++)
            {
                auto instr = *it;
                if (nop_ir(instr))
                {
                    it = stmnts.erase(it);
                    changed = true;
                }
            }

            if (!changed)
                return std::nullopt;

            return block;
        }

    }
}