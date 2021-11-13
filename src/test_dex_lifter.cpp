#include <iostream>
#include <memory>

#include "lifter_android.hpp"
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

    std::cout << *dex_disassembler;

    auto lifter_android = std::make_shared<KUNAI::LIFTER::LifterAndroid>();
    std::shared_ptr<KUNAI::MJOLNIR::IRBlock> bb = std::make_shared<KUNAI::MJOLNIR::IRBlock>();

    auto method_instructions = dex_disassembler->get_instructions();

    for (auto it = method_instructions.begin(); it != method_instructions.end(); it++)
    {
        auto class_def = std::get<0>(it->first);
        auto direct_method = std::get<1>(it->first);

        for (auto instr = it->second.begin(); instr != it->second.end(); instr++)
        {
            if (instr->second)
                lifter_android->lift_android_instruction(instr->second, bb);
        }
    }

    auto statements = bb->get_statements();

    std::cout << "[+] DEX Lifting in MjolnIR (KUNAI IR)" << std::endl;

    for (auto statement : statements)
    {
        std::cout << statement->to_string() << std::endl;
    }

    return 0;
}