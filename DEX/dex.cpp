#include "dex.hpp"

namespace KUNAI {
    namespace DEX {
        DEX::DEX(std::ifstream& input_file, std::uint64_t file_size)
        {
            try
            {
                dex_parser = std::make_unique<DexParser>();
                dex_parser->parse_dex_file(input_file, file_size);
                dex_parsing_correct = true;
            } 
            catch (const std::exception& e)
            {
                std::cout << e.what();
                dex_parsing_correct = false;
            }
        }

        std::shared_ptr<DexParser> DEX::get_parser()
        {
            if (dex_parsing_correct)
                return dex_parser;
            return nullptr;
        }
        
        DEX::~DEX(){}

    }
}