/***
 * Disassembler file for DEX, here we will present a simple
 * way to do a disassembly of a DEX file, show both class and
 * method.
 */

#include <iostream>
#include <memory>
#include "dex.hpp"

int 
main(int argc, char**argv)
{
     if (argc != 2)
    {
        std::cout << "USAGE: " << argv[0] << " <dex_file>" << std::endl;
        return 1;
    }

    std::cout << "[+] DEX Disassembly, going to disassemble the binary '" << argv[1] << "'" << std::endl;

    std::ifstream dex_file;

    dex_file.open(argv[1], std::ios::binary);

    auto fsize = dex_file.tellg();
    dex_file.seekg(0, std::ios::end);
    fsize = dex_file.tellg() - fsize;
    dex_file.seekg(0);

    // Set the logging level in spdlog, we set to debug
    spdlog::set_level(spdlog::level::debug);

    // get a unique_ptr from a DEX object.
    // this will parse the DEX file, and nothing more.
    auto dex = KUNAI::DEX::get_unique_dex_object(dex_file, fsize);

    // check if parsing was correct.
    if (!dex->get_parsing_correct())
    {
        std::cerr << "[-] Parsing of DEX file was not correct." << std::endl;
        return 1;
    }

    // get the disassembler from the DEX object
    auto dex_disassembler = dex->get_dex_disassembler();

    // apply the disassembly
    dex_disassembler->disassembly_analysis();

    // check if it was correct
    if (!dex_disassembler->get_disassembly_correct())
    {
        std::cerr << "[-] Disassembly was incorrect, cannot show instructions." << std::endl;
        return 1;
    }
    
    // print the disassembly
    std::cout << *dex_disassembler;
}