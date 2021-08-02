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

        DEX::~DEX() {}

        std::shared_ptr<DexParser> DEX::get_parser()
        {
            if (dex_parsing_correct)
                return dex_parser;
            return nullptr;
        }

        std::shared_ptr<DalvikOpcodes> DEX::get_dalvik_opcode_object()
        {
            if (dalvik_opcodes)
                return dalvik_opcodes;
            return nullptr;
        }

        std::shared_ptr<DexDisassembler> DEX::get_dex_disassembler()
        {
            if (dex_disassembler)
                return dex_disassembler;
            return nullptr;
        }

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

        bool DEX::get_parsing_correct()
        {
            return dex_parsing_correct;
        }

        std::unique_ptr<DEX> get_unique_dex_object(std::ifstream &input_file, std::uint64_t file_size)
        {
            return std::make_unique<DEX>(input_file, file_size);
        }

        std::shared_ptr<DEX> get_shared_dex_object(std::ifstream &input_file, std::uint64_t file_size)
        {
            return std::make_shared<DEX>(input_file, file_size);
        }
    }
}