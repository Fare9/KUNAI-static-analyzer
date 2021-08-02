#include "dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {

        /**
         * ExceptionAnalysis class
         */

        /**
         * @brief Constructor of ExceptionAnalysis.
         * @param exception: exception_data structure which contains handle information.
         * @param basic_blocks: all the basic blocks where are all the instruction's basic blocks.
         * @return void
         */
        ExceptionAnalysis::ExceptionAnalysis(exceptions_data exception, std::shared_ptr<BasicBlocks> basic_blocks)
        {
            this->exception = exception;

            for (auto it = exception.handler.begin(); it != exception.handler.end(); it++)
            {
                it->basic_blocks.push_back(basic_blocks->get_basic_block_by_idx(it->handler_start_addr));
            }
        }

        /**
         * @brief ExceptionAnalysis destructor.
         * @return void
         */
        ExceptionAnalysis::~ExceptionAnalysis() {}

        /**
         * @brief Get a string with all the information from the ExceptionAnalysis object.
         * @return std::string
         */
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

        /**
         * @brief Get exception data structure.
         * @return exceptions_data
         */
        exceptions_data ExceptionAnalysis::get()
        {
            return exception;
        }

        /**
         * Exception class
         */

        /**
         * @brief Constructor of Exception class, this contains a vector of ExceptionAnalysis objects.
         * @return void
         */
        Exception::Exception() {}

        /**
         * @brief Destructor of Exception class.
         * @return void
         */
        Exception::~Exception() {}

        /**
         * @brief Add new ExceptionAnalysis for each exceptions_data receive and basic blocks.
         * @param exceptions: vector with exceptions_data structures.
         * @param basic_blocks: BasicBlocks object for the ExceptionAnalysis object.
         * @return void.
         */
        void Exception::add(std::vector<exceptions_data> exceptions, std::shared_ptr<BasicBlocks> basic_blocks)
        {
            for (auto it = exceptions.begin(); it != exceptions.end(); it++)
            {
                this->exceptions.push_back(
                    std::make_shared<ExceptionAnalysis>(*it, basic_blocks));
            }
        }

        /**
         * @brief Get a ExceptionAnalysis object get by the start and end address of the try handler.
         * @param start_addr: start try value address.
         * @param end_addr: end try value address.
         * @return std::shared_ptr<ExceptionAnalysis>
         */
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

        /**
         * @brief Get all the ExceptionAnalysis objects.
         * @return std::vector<std::shared_ptr<ExceptionAnalysis>>
         */
        std::vector<std::shared_ptr<ExceptionAnalysis>> Exception::gets()
        {
            return exceptions;
        }

        
    } // namespace DEX
} // namespace KUNAI
