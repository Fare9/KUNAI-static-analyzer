#include "KUNAI/DEX/Analysis/dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {

        /**
         * ExceptionAnalysis class
         */

        ExceptionAnalysis::ExceptionAnalysis(exceptions_data exception, BasicBlocks *basic_blocks) : exception(exception)
        {
            for (auto &handler : exception.handler)
            {
                handler.basic_blocks.push_back(basic_blocks->get_basic_block_by_idx(handler.handler_start_addr));
            }
        }

        std::string ExceptionAnalysis::show_buff()
        {
            std::stringstream buff;

            buff << std::hex << exception.try_value_start_addr << ":" << std::hex << exception.try_value_end_addr << std::endl;

            for (auto &handler : exception.handler)
            {
                if (!handler.basic_blocks.size())
                    buff << "\t(" << std::hex << handler.handler_type << " -> " << std::hex << handler.handler_start_addr << ")" << std::endl;
                else
                {
                    auto bb = std::any_cast<KUNAI::DEX::dvmbasicblock_t>(handler.basic_blocks[0]);
                    buff << "\t(" << std::hex << handler.handler_type << " -> " << std::hex << handler.handler_start_addr << " " << std::hex << bb->get_start() << ")" << std::endl;
                }
            }

            return buff.str();
        }

        /**
         * Exception class
         */

        Exception::Exception() {}

        void Exception::add(std::vector<exceptions_data> &exceptions, BasicBlocks* basic_blocks)
        {
            for (auto &exception : exceptions)
            {
                this->exceptions.push_back(
                    std::make_unique<ExceptionAnalysis>(exception, basic_blocks));
            }
        }

        ExceptionAnalysis* Exception::get_exception(std::uint64_t start_addr, std::uint64_t end_addr) const
        {
            for (auto &exception : exceptions)
            {
                auto try_value_start_addr = exception->get_exception_data().try_value_start_addr;
                auto try_end_addr = exception->get_exception_data().try_value_end_addr;

                if ((try_value_start_addr >= start_addr) && (try_end_addr <= end_addr))
                    return exception.get();
                else if ((end_addr <= try_end_addr) && (start_addr >= try_value_start_addr))
                    return exception.get();
            }

            return nullptr;
        }
    } // namespace DEX
} // namespace KUNAI
