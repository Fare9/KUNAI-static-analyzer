#include <iostream>
#include <cctype>

#include <Kunai/DEX/dex.hpp>
#include <Kunai/DEX/DVM/dalvik_opcodes.hpp>

int main(int argc, char **argv){
    
    // check argument
    if (argc != 2)
    {
        std::cerr << "[-] USAGE: " << argv[0] << " <dex file>" << std::endl;
        return -1;
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

    const auto& dex_header = dex_parser->get_header_const();

    std::cout << "DEX Checksum: " << dex_header.get_dex_header_const().checksum << '\n';

    std::cout << "DEX Magic: ";
    for (auto val : dex_header.get_dex_header_const().magic)
        if (std::isprint(val))
            std::cout << val;
    std::cout << '\n';

    auto & classes = dex_parser->get_classes();

    for (auto & classdef : classes.get_classdefs())
    {
        auto class_data = classdef->get_class_idx();

        if (!class_data) continue;

        std::cout << "Object Type: " << class_data->print_type() 
            << ", Name: " << class_data->get_name() << '\n';

        const auto & source_file = classdef->get_source_file();

        if (!source_file.empty())
            std::cout << "Source file: " << source_file << '\n';

        std::cout << "Access Flags: " << KUNAI::DEX::DalvikOpcodes::get_access_flags_str(classdef->get_access_flags())
            << '\n';
        
        auto & class_data_item = classdef->get_class_data_item();

        std::cout << "Number of static fields: " << class_data_item.get_number_of_static_fields()
                  << "\nNumber of instance fields: " << class_data_item.get_number_of_instance_fields()
                  << "\nNumber of direct methods: " << class_data_item.get_number_of_direct_methods()
                  << "\nNumber of virtual methods: " << class_data_item.get_number_of_virtual_methods() << '\n';
        
        auto superclass = classdef->get_superclass();

        if (superclass)
            std::cout << "Object Type: " << superclass->print_type() << ", Name: " << superclass->get_name() << "\n";
    }
}