#include <iostream>
#include <memory>
#include <chrono>
#include <spdlog/spdlog.h>

#include "KUNAI/mjolnIR/Lifters/lifter_android.hpp"
#include "KUNAI/DEX/dex.hpp"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "USAGE: " << argv[0] << " <dex_file>\n";
        return 1;
    }

    // Set the logging level in spdlog, we set to debug
    spdlog::set_level(spdlog::level::off);

    std::ifstream dex_file;

    dex_file.open(argv[1], std::ios::binary);

    auto fsize = dex_file.tellg();
    dex_file.seekg(0, std::ios::end);
    fsize = dex_file.tellg() - fsize;
    dex_file.seekg(0);

    //std::cout << "Measuring time complete analysis (parsing+disassembly+analysis)\n";

    auto start = std::chrono::high_resolution_clock::now();

    // get a unique_ptr from a DEX object.
    // this will parse the DEX file, and nothing more.
    auto dex = KUNAI::DEX::get_unique_dex_object(dex_file, fsize);

    // check if parsing was correct.
    if (!dex->get_parsing_correct())
    {
        std::cerr << "[-] Parsing of DEX file was not correct." << std::endl;
        return 1;
    }

    auto dex_analysis = dex->get_dex_analysis(true);

    // now create xrefs
    dex_analysis->create_xref();

    auto end = std::chrono::high_resolution_clock::now();

    auto lapse = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    //time in microseconds
    std::cout << lapse.count() << "\n";

    return 0;
}