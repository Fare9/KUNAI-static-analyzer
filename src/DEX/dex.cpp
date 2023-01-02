#include "KUNAI/DEX/dex.hpp"

namespace KUNAI
{
    namespace DEX
    {
        DEX::DEX(std::ifstream &input_file, std::uint64_t file_size)
        {
            try
            {
                this->dex_parser = std::make_unique<DexParser>();
                this->dex_parser->parse_dex_file(input_file, file_size);
                this->dex_parsing_correct = true;
                this->dalvik_opcodes = std::make_unique<DalvikOpcodes>(dex_parser.get());
                this->dex_disassembler = std::make_unique<DexDisassembler>(dex_parsing_correct, dex_parser.get(), dalvik_opcodes.get());
            }
            catch (const std::exception &e)
            {
                std::cout << e.what();
                this->dex_parsing_correct = false;
            }
        }

        Analysis *DEX::get_dex_analysis(bool create_xrefs)
        {
            if (!dex_parsing_correct)
                return nullptr;

            dex_disassembler->disassembly_analysis();

            if (!dex_disassembler->get_disassembly_correct())
                return nullptr;

            dex_analysis = std::make_unique<Analysis>(dex_parser.get(), dalvik_opcodes.get(), dex_disassembler.get(), create_xrefs);

            return dex_analysis.get();
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