// compile me with: g++ -std=c++17 instruction-counter.cpp -o instruction-counter -lkunai
#include <iostream>

#include <KUNAI/DEX/dex.hpp>

int
main(int argc, char **argv)
{
    // check that 4 arguments were given
    if (argc != 4)
    {
        std::cerr << "[-] USAGE: " << argv[0] << " <dex_file> <class_name> <method_name>\n";
        return 1;
    }

    // watch info and error messages from Kunai
    spdlog::set_level(spdlog::level::err);

    auto class_name = std::string(argv[2]);
    auto method_name = std::string(argv[3]);

    std::ifstream dex_file;

    dex_file.open(argv[1], std::ios::binary);

    auto fsize = dex_file.tellg();
    dex_file.seekg(0, std::ios::end);
    fsize = dex_file.tellg() - fsize;
    dex_file.seekg(0);

    auto dex_object = KUNAI::DEX::get_unique_dex_object(dex_file, fsize);

    if (!dex_object->get_parsing_correct())
    {
        std::cerr << "Error analyzing " << argv[1] << ", maybe DEX file is not correct...\n";
        return 2;
    }

    // first of all obtain the disassembler object
    auto dex_disassembler = dex_object->get_dex_disassembler();

    // because obtaining the dex object does not apply
    // any analysis, we have to apply disassembly by
    // ourselves
    dex_disassembler->disassembly_analysis();

    // check that all the disassembly was correct
    if (!dex_disassembler->get_disassembly_correct())
    {
        std::cerr << "Error in the disassembly of " << argv[1] << ", maybe some method was incorrect...\n";
        return 3;
    }

    // instruction_map_t = std::map<std::tuple<classdef_t, encodedmethod_t>, std::map<std::uint64_t, instruction_t>>;
    // instruction_map_t is a map that contains as key a tuple with the class and method
    // and as value a map that contains ordered the address of the instruction
    // and the instruction.
    auto instruction_map = dex_disassembler->get_instructions();

    // let's create the variable that will hold all those instructions
    // that appear and how many times appear.
    std::unordered_map<std::uint32_t, size_t> instruction_counter;

    for (const auto& disassembly : instruction_map)
    {

        auto& classdef = std::get<0>(disassembly.first);
        auto& encoded_method = std::get<1>(disassembly.first);

        if (classdef == nullptr || encoded_method == nullptr)
            continue;
        
        if (!class_name.compare(classdef->get_class_idx()->get_name()) &&
            !method_name.compare(*encoded_method->get_method()->get_method_name()))
        {
            std::cout << "Disassembly of " << class_name << "->" << method_name << "\n";

            for (const auto& instructions : disassembly.second)
            {
                std::cout << std::right << std::setfill('0') << std::setw(8) << std::hex << instructions.first << "  ";
                instructions.second->show_instruction();
                std::cout << std::endl;


                auto& instr = instructions.second;
                auto op = instr->get_OP();

                if (instruction_counter.find(op) == instruction_counter.end())
                    instruction_counter[op] = 0;
                
                instruction_counter[op] += 1;
            }

            break;
        }
    }

    std::cout << "\n\nInstruction counter program\n";
    std::cout << "Number of different instructions present in the method: " << instruction_counter.size() << "\n";
    for (const auto & op_instr : instruction_counter)
    {
        auto op_num = op_instr.first;
        auto op_str = dex_object->get_dalvik_opcode_object()->get_instruction_name(op_num);
        
        std::cout << op_str << " [" << op_num << "] appears: " << op_instr.second << " times\n";
    }

    return 0;
}