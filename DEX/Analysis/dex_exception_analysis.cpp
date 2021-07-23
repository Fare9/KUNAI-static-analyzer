#include "dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {
        ExceptionAnalysis::ExceptionAnalysis(exceptions_data exception, std::shared_ptr<BasicBlocks> basic_blocks)
        {
            this->exception = exception;

            for (auto it = exception.handler.begin(); it != exception.handler.end(); it++)
            {
                it->basic_blocks.push_back(basic_blocks->get_basic_block_by_idx(it->handler_start_addr));
            }
        }

        ExceptionAnalysis::~ExceptionAnalysis() {}

        std::string ExceptionAnalysis::show_buff()
        {
            std::stringstream buff;

            buff << std::hex << exception.try_value_start_addr << ":" << std::hex << exception.try_value_end_addr << std::endl;

            for (auto it = exception.handler.begin(); it != exception.handler.end(); it++)
            {
                if (it->basic_blocks.size() == 0)
                {
                    buff << "\t(" << std::hex << it->handler_type << " -> " << std::hex << it->handler_start_addr << ")" << std::endl;
                }
                else
                {
                    auto bb = std::any_cast<std::shared_ptr<KUNAI::DEX::DVMBasicBlock>>(it->basic_blocks[0]);
                    buff << "\t(" << std::hex << it->handler_type << " -> " << std::hex << it->handler_start_addr << " " << std::hex << bb->get_start() << ")" << std::endl;
                }
            }

            return buff.str();
        }

        exceptions_data ExceptionAnalysis::get()
        {
            return exception;
        }

        Exception::Exception() {}

        Exception::~Exception() {}

        void Exception::add(std::vector<exceptions_data> exceptions, std::shared_ptr<BasicBlocks> basic_blocks)
        {
            for (auto it = exceptions.begin(); it != exceptions.end(); it++)
            {
                this->exceptions.push_back(
                    std::make_shared<ExceptionAnalysis>(*it, basic_blocks));
            }
        }

        std::shared_ptr<ExceptionAnalysis> Exception::get_exception(std::uint64_t start_addr, std::uint64_t end_addr)
        {
            for (auto it = exceptions.begin(); it != exceptions.end(); it++)
            {
                if ( ((*it)->get().try_value_start_addr >= start_addr) && ((*it)->get().try_value_end_addr <= end_addr))
                    return *it;
                else if ( (end_addr <= (*it)->get().try_value_end_addr) && (start_addr >= (*it)->get().try_value_start_addr))
                    return *it;
            }

            return nullptr;
        }

        std::vector<std::shared_ptr<ExceptionAnalysis>> Exception::gets()
        {
            return exceptions;
        }

        
    } // namespace DEX
} // namespace KUNAI
