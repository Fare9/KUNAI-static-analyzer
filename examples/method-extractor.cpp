#include <iostream>
#include <cctype>
#include <algorithm>

#include <Kunai/DEX/dex.hpp>
#include <Kunai/DEX/DVM/dalvik_opcodes.hpp>

void show_help(const char * name)
{
    std::cerr << "Usage: " << name << " <dex_file> -m 'method name' [-p <prototype>] [-c 'LClassName']\n";
    std::cerr << "\t-m: specify the name of the method.\n";
    std::cerr << "\t-p: specify prototype for the method, example \"-p '(IIL)F\" (optional)\n";
    std::cerr << "\t-c: specify name of the class in Dalvik Format, example \"-c 'LSampleClass''\" (optional)\n";
}

int
main(int argc, char ** argv)
{
    std::string method_name;
    std::string prototype;
    std::string class_name;
    
    if (argc < 4)
    {
        show_help(argv[0]);
        return -1;
    }    
    
    const std::vector<std::string_view> args(argv + 2, argv + argc);
    
    for (size_t I = 0, E = args.size(); I < E; ++I)
    {
        auto get_command = [&]()
        {
            if (++I >= E)
                return std::string("");
            if (args[I].size() == 0 || args[I].at(0) == '-')
                return std::string("");
            return std::string(args[I]);
        };


        auto command = args[I];

        if (command == "-m")
        {
            method_name = get_command();

            if (method_name.empty())
            {
                show_help(argv[0]);
                return -2;
            }
        }
        else if (command == "-p")
        {
            prototype = get_command();

            if (prototype.empty())
            {
                show_help(argv[0]);
                return -2;
            }
        }
        else if (command == "-c")
        {
            class_name = get_command();

            if (class_name.empty())
            {
                show_help(argv[0]);
                return -2;
            }
        }
    }

    if (method_name.empty())
    {
        show_help(argv[0]);
        return -2;
    }

    // parse dex file
    auto dex_file = KUNAI::DEX::Dex::parse_dex_file(argv[1]);

    if (!dex_file->get_parsing_correct())
    {
        std::cerr << "Error analyzing " << argv[1] << ", maybe DEX file is not correct...\n";
        return -1;
    } 

    // get pointer to parser
    auto dex_parser = dex_file->get_parser();

    // get methods for checking them
    auto & dex_methods = dex_parser->get_methods();

    std::map<std::size_t, std::string> method_ids;

    std::size_t I = -1;

    std::ranges::for_each(dex_methods.methods(), [&](const KUNAI::DEX::methodid_t &method)
    {
        ++I;
        
        if (method->get_name() != method_name)
            return;
        if (!prototype.empty() && 
            !(method->get_proto()->get_shorty_idx() != prototype))
            return;
        if (!class_name.empty() &&
            !(method->get_class()->get_raw() != class_name))
            return;
        
        method_ids[I] = method->pretty_method();
    });

    std::cout << "id,full_name\n";

    for (auto & key_value : method_ids)
    {
        std::cout << key_value.first << "," << key_value.second << "\n";
    }
}