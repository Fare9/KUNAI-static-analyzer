// compile me with: g++ -O3 -std=c++17 extract-dex-information.cpp -o extract-dex-information -lkunai
#include <iostream>

#include <KUNAI/DEX/dex.hpp>
#include <spdlog/spdlog.h>

int
main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "[-] USAGE: " << argv[0] << " <DEX>\n";
        return 1;
    }

    // set the level of spdlog to off
    // no messages
    spdlog::set_level(spdlog::level::off);

    std::ifstream dex_file;

    dex_file.open(argv[1], std::ios::binary);

    auto fsize = dex_file.tellg();
    dex_file.seekg(0, std::ios::end);
    fsize = dex_file.tellg() - fsize;
    dex_file.seekg(0);

    auto dex_object = KUNAI::DEX::get_unique_dex_object(dex_file, fsize);

    if (!dex_object->get_parsing_correct())
    {
        return 2;
    }

    auto dex_header = dex_object->get_parser();

    auto dex_strings = dex_header->get_strings();

    std::cout << "strings:" << dex_strings->get_number_of_strings() << ";";

    auto dex_types = dex_header->get_types();

    std::cout << "types:" << dex_types->get_number_of_types() << ";";

    auto dex_protos = dex_header->get_protos();

    std::cout << "protos:" << dex_protos->get_number_of_protos() << ";";

    auto dex_fields = dex_header->get_fields();

    std::cout << "fields:" << dex_fields->get_number_of_fields() << ";";

    auto dex_methods = dex_header->get_methods();

    std::cout << "methods:" << dex_methods->get_number_of_methods() << ";";

    auto dex_classes = dex_header->get_classes();

    std::cout << "classes:" << dex_classes->get_number_of_classes() << ";";

    auto disassembler_object = dex_object->get_dex_disassembler();

    disassembler_object->disassembly_analysis();

    if (!disassembler_object->get_disassembly_correct())
    {
        return 3;
    }

    auto & instructions = disassembler_object->get_instructions();

    size_t number_of_instructions_recovered = 0;

    for (const auto & key_instr : instructions)
    {
        number_of_instructions_recovered += key_instr.second.size();
    }

    std::cout << "instructions:" << number_of_instructions_recovered << ";";

    return 0;
}