#include <iostream>
#include <memory>
#include <cassert>
#include <algorithm>

#include <spdlog/spdlog.h>
// necessary to work with dex import the main DEX object
#include "KUNAI/DEX/dex.hpp"

int main()
{
    std::ifstream dex_file;

    dex_file.open("../../tests/test-assignment-arith-logic/Main.dex", std::ios::binary);

    auto fsize = dex_file.tellg();
    dex_file.seekg(0, std::ios::end);
    fsize = dex_file.tellg() - fsize;
    dex_file.seekg(0);

    spdlog::set_level(spdlog::level::debug);

    auto dex_object = KUNAI::DEX::get_unique_dex_object(dex_file, fsize);

    // check parsing is correct
    const auto parsing_correct = dex_object->get_parsing_correct();

    assert(parsing_correct == true);

    auto& dex_parser = dex_object->get_parser();

    // compare first the dex header
    auto& dex_header = dex_parser->get_header();

    std::uint8_t magic_correct[] = {'d','e','x','\n','0','3','5',0};
    std::uint8_t signature_correct[] = {0x7c,0xda,0x95,0x8e,0x95,0x7e,0x5f,0x4,0x6c,0x10,0x23,0xa,0x5c,0xf3,0x6e,0x66,0xf6,0xf9,0x84,0xa6};
    
    std::cout << "Testing DEX header\n";

    assert(std::equal(std::begin(dex_header->get_dex_header().magic), std::end(dex_header->get_dex_header().magic), std::begin(magic_correct)) == true);
    assert(dex_header->get_dex_header().checksum == 0x123ff4f7);
    assert(std::equal(std::begin(dex_header->get_dex_header().signature), std::end(dex_header->get_dex_header().signature), std::begin(signature_correct)));
    assert(dex_header->get_dex_header().file_size == 0x6e8);
    assert(dex_header->get_dex_header().header_size == 0x70);
    assert(dex_header->get_dex_header().endian_tag == 0x12345678);
    assert(dex_header->get_dex_header().link_size == 0);
    assert(dex_header->get_dex_header().link_off == 0);
    assert(dex_header->get_dex_header().map_off == 0x624);
    assert(dex_header->get_dex_header().string_ids_size == 0x27);
    assert(dex_header->get_dex_header().string_ids_off == 0x70);
    assert(dex_header->get_dex_header().type_ids_size == 0x0f);
    assert(dex_header->get_dex_header().type_ids_off == 0x10c);
    assert(dex_header->get_dex_header().proto_ids_size == 0x9);
    assert(dex_header->get_dex_header().proto_ids_off == 0x148);
    assert(dex_header->get_dex_header().field_ids_size == 0x4);
    assert(dex_header->get_dex_header().field_ids_off == 0x1b4);
    assert(dex_header->get_dex_header().method_ids_size == 0x0c);
    assert(dex_header->get_dex_header().method_ids_off == 0x1d4);
    assert(dex_header->get_dex_header().class_defs_size == 1);
    assert(dex_header->get_dex_header().class_defs_off == 0x234);
    assert(dex_header->get_dex_header().data_size == 0x494);
    assert(dex_header->get_dex_header().data_off == 0x254);

    std::cout << "Testing DEX Strings\n";
    auto &dex_strings = dex_parser->get_strings();
    std::vector<const char *> correct_strings = {"<clinit>","<init>","F","I","J","LMain;","Ldalvik/annotation/Throws;","Ljava/io/InputStream;","Ljava/io/PrintStream;","Ljava/lang/Exception;","Ljava/lang/Object;","Ljava/lang/String;","Ljava/lang/System;","Ljava/util/Scanner;","Main.java","Test case","V","VF","VI","VJ","VL","VZ","Z","[Ljava/lang/String;","c <= 0","c <= 20","c > 0","c > 20","close","field_boolean","field_int","in","main","nextInt","ojete de vaca","out","println","value","~~D8{\"compilation-mode\":\"debug\",\"min-api\":1,\"version\":\"1.5.13-q1\"}"};
    for (size_t i = 0, n_strings = dex_strings->get_number_of_strings(); i < n_strings; i++)
    {
        auto comparison = dex_strings->get_string_from_order(i)->compare(correct_strings[i]);
        assert(comparison == 0);
    }

    std::cout << "Testing DEX Types\n";
    auto &dex_types = dex_parser->get_types();
    std::vector<const char*> correct_types = {"F","I","J","LMain;","Ldalvik/annotation/Throws;","Ljava/io/InputStream;","Ljava/io/PrintStream;","Ljava/lang/Exception;","Ljava/lang/Object;","Ljava/lang/String;","Ljava/lang/System;","Ljava/util/Scanner;","V","Z","[Ljava/lang/String;"};

    for (size_t i = 0, n_types = dex_types->get_number_of_types(); i < n_types; i++)
    {
        auto comparison = dex_types->get_type_from_order(i)->get_raw().compare(correct_types[i]);
        assert(comparison == 0);
    }

    std::cout << "Testing Dex Protos\n";
    auto &dex_protos = dex_parser->get_protos();
    auto &protos_vector = dex_protos->get_protos();
    std::vector<const char*> correct_protos = {"()I","()V","(F)V","(I)V","(J)V","(Ljava/io/InputStream;)V","(Ljava/lang/String;)V","(Z)V","([Ljava/lang/String;)V"};
    for (size_t i = 0, n_protos = protos_vector.size(); i < n_protos; i++)
    {
        auto comparison = protos_vector.at(i)->get_proto_str().compare(correct_protos[i]);
        assert(comparison == 0);
    }

    std::cout << "Testing Dex Fields\n";
    auto &dex_fields = dex_parser->get_fields();
    auto &fields = dex_fields->get_fields();
    std::vector<const char*> correct_fields = {"Z LMain;->field_boolean","I LMain;->field_int","Ljava/io/InputStream; Ljava/lang/System;->in","Ljava/io/PrintStream; Ljava/lang/System;->out"};
    for (size_t i = 0, n_fields = fields.size(); i < n_fields; i++)
    {
        auto comparison = fields.at(i)->get_field_str().compare(correct_fields[i]);
        assert(comparison == 0);
    }

    std::cout << "Testing Dex Methods\n";
    auto &dex_methods = dex_parser->get_methods();
    std::vector<const char*> correct_methods = {"LMain;.<clinit>()V","LMain;.<init>()V","LMain;.main([Ljava/lang/String;)V","Ljava/io/PrintStream;.println(F)V","Ljava/io/PrintStream;.println(I)V","Ljava/io/PrintStream;.println(J)V","Ljava/io/PrintStream;.println(Ljava/lang/String;)V","Ljava/io/PrintStream;.println(Z)V","Ljava/lang/Object;.<init>()V","Ljava/util/Scanner;.<init>(Ljava/io/InputStream;)V","Ljava/util/Scanner;.close()V","Ljava/util/Scanner;.nextInt()I"};
    for (size_t i = 0, n_methods = dex_methods->get_number_of_methods(); i < n_methods; i++)
    {
        auto comparison = dex_methods->get_method_by_order(i)->get_method_str().compare(correct_methods[i]);
        assert(comparison == 0);
    }

    return 0;
}