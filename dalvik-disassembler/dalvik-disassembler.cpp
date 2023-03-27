/// Compile clang++ -std=c++20 dalvik-disassembler.cpp -o dalvik-disassembler -lkunai
#include <iostream>

#include <Kunai/DEX/dex.hpp>

void show_help(char **argv)
{
    std::cerr << "[-] USAGE: " << argv[0] << " <dex_file> <class_name> <method_name> [-r]\n";
    std::cerr << "\t<dex_file>: dex file to disassembly\n";
    std::cerr << "\t<class_name>: name of the class to extract\n";
    std::cerr << "\t<method_name>: name of the method to extract\n";
    std::cerr << "\t[-r]: optional argument, use recursive disassembly algorithm\n";
    std::cerr << "\t[-b]: show the instructions as basic blocks\n";
    std::cerr << "\t[-p]: show a plot with the blocks in .dot format\n";
}

void show_instruction(KUNAI::DEX::Instruction *instr)
{
    std::cout << std::right << std::setfill('0') << std::setw(8) << std::hex << instr->get_address() << "  ";

    const auto &opcodes = instr->get_opcodes();

    if (opcodes.size() > 8)
    {
        auto remaining = 8 - (opcodes.size() % 8);

        size_t aux = 0;

        for (const auto opcode : opcodes)
        {
            std::cout << std::right << std::setfill('0') << std::setw(2) << std::hex << (std::uint32_t)opcode << " ";
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
        for (const auto opcode : opcodes)
            std::cout << std::right << std::setfill('0') << std::setw(2) << std::hex << (std::uint32_t)opcode << " ";

        for (std::uint8_t i = 0, remaining_size = 8 - opcodes.size(); i < remaining_size; ++i)
            std::cout << "   ";
    }

    std::cout << instr->print_instruction() << "\n";
}

int main(int argc, char **argv)
{
    /// use the recursive disassembler?
    bool use_recursive = false;
    bool show_blocks = false;
    bool show_plot = false;

    if (argc == 1 || (argc > 1 && !strcmp("-h", argv[1])))
    {
        show_help(argv);
        return 1;
    }

    // check that 4 arguments were given
    if (argc < 4)
    {
        show_help(argv);
        return 1;
    }

    // check if one argument more was given
    // check if it is correct
    if (argc > 4)
    {
        const std::vector<std::string_view> optional_args(argv + 4, argv + argc);

        for (const auto val : optional_args)
        {
            if (val == "-r")
                use_recursive = true;
            if (val == "-b")
                show_blocks = true;
            if (val == "-p")
                show_plot = true;
        }
    }

    /// watch info and error messages from Kunai
    spdlog::set_level(spdlog::level::err);

    /// class name and method name
    auto class_name = std::string(argv[2]);
    auto method_name = std::string(argv[3]);

    // now the dex file...
    auto dex_file = KUNAI::DEX::Dex::parse_dex_file(argv[1]);

    if (!dex_file->get_parsing_correct())
    {
        std::cerr << "Error analyzing " << argv[1] << ", maybe DEX file is not correct...\n";
        return 2;
    }

    // obtain the disassembler from the DEX object
    auto dex_disassembler = dex_file->get_dex_disassembler();

    // if recursive disassembly is requested
    // change the disassembly method
    if (use_recursive)
        dex_disassembler->set_disassembly_algorithm(KUNAI::DEX::DexDisassembler::disassembly_algorithm::RECURSIVE_TRAVERSAL_ALGORITHM);

    if (show_blocks || show_plot)
    {
        auto analysis = dex_file->get_analysis(false);

        const auto &methods = analysis->get_methods();

        for (const auto &method : methods)
        {
            if (method.second->external())
                continue;

            auto encoded_method = std::get<KUNAI::DEX::EncodedMethod *>(method.second->get_encoded_method());
            auto cls_ = reinterpret_cast<KUNAI::DEX::DVMClass *>(encoded_method->getMethodID()->get_class());

            if (cls_ == nullptr)
                continue;

            if (cls_->get_name() != class_name ||
                encoded_method->getMethodID()->get_name() != method_name)
                continue;

            if (show_blocks)
            {
                const auto &blocks = method.second->get_basic_blocks();

                for (const auto block : blocks.get_nodes())
                {
                    if (block->is_start_block())
                        std::cout << "BB-Start Block\n";
                    else if (block->is_end_block())
                        std::cout << "BB-End Block\n";
                    else if (block->is_try_block())
                        std::cout << "BB-" << block->get_first_address() << " (try block)"
                                  << "\n";
                    else if (block->is_catch_block())
                        std::cout << "BB-" << block->get_first_address() << " (catch block)"
                                  << "\n";
                    else
                        std::cout << "BB-" << block->get_first_address() << "\n";

                    for (auto &instr : block->get_instructions())
                        show_instruction(instr);
                }
            }
            else if (show_plot)
            {
                std::string file_name = class_name + "." + method_name + ".dot";
                method.second->dump_dot_file(file_name);
            }
        }
    }
    else
    {
        /// we have to apply the disassembly ourselves since
        /// the library only applies parsing
        dex_disassembler->disassembly_dex();

        if (!dex_disassembler->correct_disassembly())
        {
            std::cerr << "Error in the disassembly of " << argv[1] << ", maybe some method was incorrect...\n";
            return 3;
        }

        const auto &dex_instructions = dex_disassembler->get_dex_instructions();

        /// dex instructions contain the next pair:
        ///     - encoded_method : instructions from method
        /// we will check for each one if class name and method names
        /// are correct
        for (const auto &disassembly : dex_instructions)
        {
            auto encoded_method = disassembly.first;
            const auto &instrs = disassembly.second;

            auto cls_ = reinterpret_cast<KUNAI::DEX::DVMClass *>(encoded_method->getMethodID()->get_class());

            if (cls_ == nullptr)
                continue;

            if (cls_->get_name() != class_name ||
                encoded_method->getMethodID()->get_name() != method_name)
                continue;

            for (const auto &instr : instrs)
            {
                show_instruction(instr.get());
            }
        }
    }
}
