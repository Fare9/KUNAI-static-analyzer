#include <iostream>
#include <memory>

#include "lifter_android.hpp"
#include "dex.hpp"

int main(int argc, char **argv)
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

    auto lifter_android = std::make_shared<KUNAI::LIFTER::LifterAndroid>();

    auto method_instructions = dex_disassembler->get_instructions();

    for (auto it = method_instructions.begin(); it != method_instructions.end(); it++)
    {
        auto class_def = std::get<0>(it->first);
        auto direct_method = std::get<1>(it->first);

        std::shared_ptr<KUNAI::MJOLNIR::IRBlock> bb = std::make_shared<KUNAI::MJOLNIR::IRBlock>();

        std::cout << "[+] Lifting method " << direct_method->full_name() << " to KUNAI's IR (MjolnIR)" << std::endl;

        std::cout << "[!] Disassembly: " << std::endl;

        for (auto instr = it->second.begin(); instr != it->second.end(); instr++)
        {
            std::cout << std::right << std::setfill('0') << std::setw(8) << std::hex << instr->first << "  ";
            instr->second->show_instruction();
            std::cout << std::endl;

            lifter_android->lift_android_instruction(instr->second, bb);
        }

        std::cout << std::endl;
        std::cout << "[!] Lifted: " << std::endl;

        auto statements = bb->get_statements();

        for (auto statement : statements)
        {
            std::cout << statement->to_string() << std::endl;
        }
        
        std::cout << std::endl;
    }

    /*
    auto s_ptr = statements.begin();
    std::shared_ptr<KUNAI::MJOLNIR::IRStmnt> s = (*s_ptr);
    auto s1 = std::dynamic_pointer_cast<KUNAI::MJOLNIR::IRAssign>(s);
    std::advance(s_ptr, 1);
    s = *s_ptr;
    auto s2 = std::dynamic_pointer_cast<KUNAI::MJOLNIR::IRAssign>(s);

    std::cout << "\n\n\n";
    std::cout << "Example comparison of sentence1[Dest] and sentence2[Src]" << std::endl;

    if (s1->get_destination()->equals(s2->get_source()))
    {
        std::cout << s1->to_string() << std::endl;
        std::cout << "And" << std::endl;
        std::cout << s2->to_string() << std::endl;
        std::cout << "Are connected by destination and source" << std::endl;
    }
    */
    return 0;
}