#include "dex.hpp"

namespace KUNAI
{
    namespace DEX
    {

        /**
         * @brief Construct a new DEX::DEX object, this will just apply the parsing,
         *        it will also create the objects for the DalvikOpcodes and the DexDisassembler.
         *        the disassembly will not be applied in the first moment.
         * 
         * @param input_file 
         * @param file_size 
         */
        DEX::DEX(std::ifstream &input_file, std::uint64_t file_size)
        {
            try
            {
                this->dex_parser = std::make_shared<DexParser>();
                this->dex_parser->parse_dex_file(input_file, file_size);
                this->dex_parsing_correct = true;
                this->dalvik_opcodes = std::make_shared<DalvikOpcodes>(dex_parser);
                this->dex_disassembler = std::make_shared<DexDisassembler>(dex_parsing_correct, dex_parser, dalvik_opcodes);
            }
            catch (const std::exception &e)
            {
                std::cout << e.what();
                this->dex_parsing_correct = false;
            }
        }

        /**
         * @brief Destroy the DEX::DEX object
         * 
         */
        DEX::~DEX() {}

        /**
         * @brief Get the DexParser of the DEX file, this will contain all the headers.
         *        Parsing was already applied.
         * 
         * @return std::shared_ptr<DexParser> 
         */
        std::shared_ptr<DexParser> DEX::get_parser()
        {
            if (dex_parsing_correct)
                return dex_parser;
            return nullptr;
        }

        /**
         * @brief Get a DalvikOpcodes object, this is commonly used internally by
         *        disassembler and other classes for Dalvik information.
         * 
         * @return std::shared_ptr<DalvikOpcodes> 
         */
        std::shared_ptr<DalvikOpcodes> DEX::get_dalvik_opcode_object()
        {
            if (dalvik_opcodes)
                return dalvik_opcodes;
            return nullptr;
        }

        /**
         * @brief Get the DexDisassembler if you want to apply manually the
         *        disassembly of the instructions.
         * 
         * @return std::shared_ptr<DexDisassembler> 
         */
        std::shared_ptr<DexDisassembler> DEX::get_dex_disassembler()
        {
            if (dex_disassembler)
                return dex_disassembler;
            return nullptr;
        }

        /**
         * @brief This method returns the analysis object from the DEX
         *        this contains all the analyzed data, together with
         *        ClassAnalysis and MethodAnalysis objects. The instructions
         *        are already disassembled, and the methods are created with
         *        the CFG.
         * 
         * @return std::shared_ptr<Analysis> 
         */
        std::shared_ptr<Analysis> DEX::get_dex_analysis()
        {
            if (!dex_parsing_correct)
                return nullptr;
            
            dex_disassembler->disassembly_analysis();

            if (!dex_disassembler->get_disassembly_correct())
                return nullptr;

            dex_analysis = std::make_shared<Analysis>(dex_parser, dalvik_opcodes, dex_disassembler->get_instructions());

            return dex_analysis;
        }

        /**
         * @brief Get if parsing was correct or not.
         * 
         * @return true 
         * @return false 
         */
        bool DEX::get_parsing_correct()
        {
            return dex_parsing_correct;
        }

        /**
         * @brief Get the unique dex object object
         * 
         * @param input_file 
         * @param file_size 
         * @return std::unique_ptr<DEX> 
         */
        std::unique_ptr<DEX> get_unique_dex_object(std::ifstream &input_file, std::uint64_t file_size)
        {
            return std::make_unique<DEX>(input_file, file_size);
        }

        /**
         * @brief Get the shared dex object object
         * 
         * @param input_file 
         * @param file_size 
         * @return std::shared_ptr<DEX> 
         */
        std::shared_ptr<DEX> get_shared_dex_object(std::ifstream &input_file, std::uint64_t file_size)
        {
            return std::make_shared<DEX>(input_file, file_size);
        }
    }
}