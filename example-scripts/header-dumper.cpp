// compile me with: g++ -std=c++17 header-dumper.cpp -o header-dumper -lkunai

#include <iostream>

#include <KUNAI/DEX/dex.hpp>
#include <spdlog/spdlog.h>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "[-] USAGE: " << argv[0] << " <dex file>" << std::endl;
        return -1;
    }

    // watch only info and error messages from Kunai
    spdlog::set_level(spdlog::level::info);

    auto logger = KUNAI::LOGGER::logger();

    logger->info("Starting the analysis of {}", argv[1]);

    std::ifstream dex_file;
    dex_file.open(argv[1], std::ios::binary);

    auto fsize = dex_file.tellg();
    dex_file.seekg(0, std::ios::end);
    fsize = dex_file.tellg() - fsize;
    dex_file.seekg(0);

    auto dex_object = KUNAI::DEX::get_unique_dex_object(dex_file, fsize);

    if (!dex_object->get_parsing_correct())
    {
        logger->error("Error analyzing {}, maybe DEX file is not correct...", argv[1]);
        return -1;
    }

    auto dex_parser = dex_object->get_parser();

    logger->info("Dex version number: {}", dex_parser->get_header_version());
    logger->info("Dex version string: {}", dex_parser->get_header_version_str());

    auto dex_header = dex_parser->get_header();

    logger->info("File size: {:d}, Checksum: 0x{:x}, Header size: {:d}", dex_header->get_dex_header().file_size, (std::uint32_t)dex_header->get_dex_header().checksum, dex_header->get_dex_header().header_size);

    logger->info("String ids size: {:d}, String ids offset: 0x{:x}", dex_header->get_dex_header().string_ids_size, dex_header->get_dex_header().string_ids_off);

    auto vector_class_defs = dex_parser->get_classes_def_item();

    logger->info("[!] ClassDefs");

    for (auto &class_def : vector_class_defs)
    {
        auto class_data = class_def->get_class_idx();

        if (!class_data)
            continue;

        logger->info("[+] ClassDef:");

        logger->info("Type of object: {}, name: {}", class_data->print_type(), class_data->get_name());

        auto source_file = class_def->get_source_file_idx();
        if (source_file)
            logger->info("Source file of the class: {}", *source_file);

        // enum that can be checked!
        logger->info("Access Flag: {}", class_def->get_access_flags());

        logger->info("Implemented interfaces: {:d}", class_def->get_number_of_interfaces());

        auto class_data_item = class_def->get_class_data();

        logger->info("Number of static fields: {}, number of instance fields: {}, number of direct methods: {}, number of virtual methods: {}",
                     class_data_item->get_number_of_static_fields(),
                     class_data_item->get_number_of_instance_fields(),
                     class_data_item->get_number_of_direct_methods(),
                     class_data_item->get_number_of_virtual_methods());

        auto superclass = class_def->get_superclass_idx();

        if (!superclass)
            continue;

        logger->info("[Superclass] Type of object: {}, name: {}", superclass->print_type(), superclass->get_name());
    }

    auto vector_method_ids = dex_parser->get_methods_id_item();

    logger->info("[!] MethodIds");

    for (auto &method_id : vector_method_ids)
    {
        auto prototype = method_id->get_method_prototype();
        auto method_class = method_id->get_method_class();
        auto class_name = std::string("");
        auto name = *method_id->get_method_name();
        
        if (method_class->get_type() == KUNAI::DEX::Type::CLASS)
            class_name = std::dynamic_pointer_cast<KUNAI::DEX::Class>(method_class)->get_name();
        else
            class_name = method_class->get_raw();

        logger->info("[+] MethodId:");
        logger->info("Method name: {}, method prototype: {}, class: {}", name, prototype->get_proto_str(), class_name);
    }

    auto strings = dex_parser->get_strings();

    logger->info("[!] Strings");
    
    for (size_t n_strings = strings->get_number_of_strings(), i = 0; i < n_strings; i++)
        logger->info("String[{}] = {}", i, *strings->get_string_from_order(i));

    return 0;
}