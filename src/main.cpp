/***
 * Main file from project.
 */

#include <iostream>
#include <memory>
#include "KUNAI/DEX/dex.hpp"

int
main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "USAGE: " << argv[0] << " <dex_file>" << std::endl;
        return 1;
    } 

    std::ifstream dex_file;

    dex_file.open(argv[1], std::ios::binary);

    auto fsize = dex_file.tellg();
    dex_file.seekg(0, std::ios::end);
    fsize = dex_file.tellg() - fsize;
    dex_file.seekg(0);

    std::unique_ptr<KUNAI::DEX::DEX> dex = std::make_unique<KUNAI::DEX::DEX>(dex_file, fsize);
    


    auto dex_disassembler = dex->get_dex_disassembler();

    if (dex->get_parsing_correct())
        std::cout << *dex->get_parser();

    auto dex_analysis = dex->get_dex_analysis(true);
    
    if (dex_disassembler->get_disassembly_correct())
        std::cout << *dex_disassembler;

    dex_analysis->create_xref();

    return 0;
}