// compile me with: g++ -std=c++17 instruction-counter.cpp -o instruction-counter -lkunai
#include <iostream>
#include <list>

#include <KUNAI/DEX/dex.hpp>

int main(int argc, char **argv)
{
    bool use_recursive = false;

    if (argc == 1 || (argc > 1 && !strcmp("-h", argv[1])))
    {
        std::cerr << "[-] USAGE: " << argv[0] << " <dex_file> <class_name> <method_name> [-r]\n";
        std::cerr << "\t-r: optional argument, use recursive disassembly algorithm\n";
        return 1;
    }

    // check that 4 arguments were given
    if (argc < 4)
    {
        std::cerr << "[-] USAGE: " << argv[0] << " <dex_file> <class_name> <method_name> [-r]\n";
        std::cerr << "\t-r: optional argument, use recursive disassembly algorithm\n";
        return 1;
    }

    // check if one argument more was given
    // check if it is correct
    if (argc > 4 && strcmp("-r", argv[4]))
    {
        std::cerr << "The option " << argv[4] << " is not recognized...\n\n\n";
        std::cerr << "[-] USAGE: " << argv[0] << " <dex_file> <class_name> <method_name> [-r]\n";
        std::cerr << "\t-r: optional argument, use recursive disassembly algorithm\n";
        return 1;
    }
    else if (argc > 4 && !strcmp("-r", argv[4]))
    {
        use_recursive = true;
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

    // if recursive disassembly is requested by the user
    // use this one!
    if (use_recursive)
        dex_disassembler->set_disassembler_type(KUNAI::DEX::RECURSIVE_TRAVERSAL_DISASSEMBLER);

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
    auto & instruction_map = dex_disassembler->get_instructions();

    // let's create the variable that will hold all those instructions
    // that appear and how many times appear.
    std::unordered_map<std::uint32_t, size_t> instruction_counter;

    for (const auto &disassembly : instruction_map)
    {

        auto &classdef = std::get<0>(disassembly.first);
        auto &encoded_method = std::get<1>(disassembly.first);

        if (classdef == nullptr || encoded_method == nullptr)
            continue;

        if (!class_name.compare(classdef->get_class_idx()->get_name()) &&
            !method_name.compare(*encoded_method->get_method()->get_method_name()))
        {
            std::cout << "Disassembly of " << class_name << "->" << method_name << "\n";

            for (const auto &instructions : disassembly.second)
            {
                std::cout << std::right << std::setfill('0') << std::setw(8) << std::hex << instructions.first << "  ";

                auto &raw_values = instructions.second->get_raw();

                if (raw_values.size() > 8)
                {
                    auto remaining = 8 - (raw_values.size() % 8);

                    size_t aux = 0;

                    for (auto value : raw_values)
                    {
                        std::cout << std::right << std::setfill('0') << std::setw(2) << std::hex << (std::uint32_t)value << " ";
                        aux++;
                        if (aux % 8 == 0)
                        {
                            std::cout << "\n"
                                      << "          ";
                        }
                    }

                    for (std::uint8_t i = 0; i < remaining; i++)
                        std::cout << "   ";
                }
                else
                {
                    for (auto value : raw_values)
                        std::cout << std::right << std::setfill('0') << std::setw(2) << std::hex << (std::uint32_t)value << " ";

                    for (std::uint8_t i = 0, remaining_size = 8 - raw_values.size(); i < remaining_size; i++)
                        std::cout << "   ";
                }

                instructions.second->show_instruction();
                std::cout << std::endl;

                auto &instr = instructions.second;
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
    for (const auto &op_instr : instruction_counter)
    {
        auto op_num = op_instr.first;
        auto op_str = dex_object->get_dalvik_opcode_object()->get_instruction_name(op_num);

        std::cout << op_str << " [" << op_num << "] appears: " << op_instr.second << " times\n";
    }

    return 0;
}