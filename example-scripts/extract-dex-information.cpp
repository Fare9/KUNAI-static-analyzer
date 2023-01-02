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

    // disassembly
    auto analysis_object = dex_object->get_dex_analysis(true);

    // create the xrefs
    analysis_object->create_xref();

    auto disassembler_object = dex_object->get_dex_disassembler();

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

    auto class_analysis = analysis_object->get_classes();

    std::cout << "classanalysis:" << class_analysis.size() << ";";

    size_t classes_xref_from = 0, 
           classes_xref_to = 0, 
           classes_xref_const_class = 0,
           class_xref_new_instance = 0;

    for (auto & class_ : class_analysis)
    {
        classes_xref_from += class_->get_xref_from().size();
        classes_xref_to += class_->get_xref_to().size();
        classes_xref_const_class += class_->get_xref_const_class().size();
        class_xref_new_instance += class_->get_xref_new_instance().size();
    }

    std::cout << "classes_xref_from:" << classes_xref_from << ";"
              << "classes_xref_to:" << classes_xref_to << ";"
              << "classes_xref_const_class:" << classes_xref_const_class << ";"
              << "class_xref_new_instance:" << class_xref_new_instance << ";";

    auto method_analysis = analysis_object->get_methods();

    size_t method_xref_const_class = 0, 
           method_xref_from = 0,
           method_xref_new_instance = 0,
           method_xref_read = 0,
           method_xref_to = 0,
           method_xref_write = 0;

    for (auto & method : method_analysis)
    {
        method_xref_const_class += method->get_xref_const_class().size();
        method_xref_from += method->get_xref_from().size();
        method_xref_new_instance += method->get_xref_new_instance().size();
        method_xref_read += method->get_xref_read().size();
        method_xref_to += method->get_xref_to().size();
        method_xref_write += method->get_xref_write().size();
    }

    std::cout << "methodanalysis:" << method_analysis.size() << ";";

    std::cout << "method_xref_const_class:" << method_xref_const_class << ";"
              << "method_xref_from:" << method_xref_from << ";"
              << "method_xref_new_instance:" << method_xref_new_instance << ";"
              << "method_xref_read:" << method_xref_read << ";"
              << "method_xref_to:" << method_xref_to << ";"
              << "method_xref_write:" << method_xref_write << ";";

    auto field_analysis = analysis_object->get_fields();

    size_t field_xref_read = 0,
           field_xref_write = 0;

    for (auto & field : field_analysis)
    {
        field_xref_read += field->get_xref_read().size();
        field_xref_write += field->get_xref_write().size();
    }

    std::cout << "fieldanalysis:" << field_analysis.size() << ";";

    std::cout << "field_xref_read:" << field_xref_read << ";"
              << "field_xref_write:" << field_xref_write << ";";

    auto string_analysis = analysis_object->get_strings();

    size_t str_xref_from = 0;

    for (auto str : string_analysis)
    {
        str_xref_from += str->get_xref_from().size();
    }

    std::cout << "stringanalysis:" << string_analysis.size() << ";";

    std::cout << "str_xref_from:" << str_xref_from;

    return 0;
}