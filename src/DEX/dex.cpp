#include "dex.hpp"

namespace KUNAI
{
    namespace DEX
    {
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

        analysis_t DEX::get_dex_analysis(bool create_xrefs)
        {
            if (!dex_parsing_correct)
                return nullptr;

            dex_disassembler->disassembly_analysis();

            if (!dex_disassembler->get_disassembly_correct())
                return nullptr;

            dex_analysis = std::make_shared<Analysis>(dex_parser, dalvik_opcodes, dex_disassembler->get_instructions(), create_xrefs);

            return dex_analysis;
        }

        std::unique_ptr<DEX> get_unique_dex_object(std::ifstream &input_file, std::uint64_t file_size)
        {
            return std::make_unique<DEX>(input_file, file_size);
        }

        dex_t get_shared_dex_object(std::ifstream &input_file, std::uint64_t file_size)
        {
            return std::make_shared<DEX>(input_file, file_size);
        }
    }
}