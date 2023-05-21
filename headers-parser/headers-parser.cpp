#include <iostream>

#include <Kunai/DEX/dex.hpp>


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

    // HEADER
    auto& dex_header = dex_parser->get_header();
    std::cout << dex_header << "\n";

    // STRINGS
    auto& dex_strings = dex_parser->get_strings();
    std::cout << dex_strings << "\n";
    
    // TYPES
    auto& dex_types = dex_parser->get_types();
    std::cout << dex_types << "\n";
    
    // FIELDS
    auto& dex_fields = dex_parser->get_fields();
    std::cout << dex_fields << "\n";

    // CLASSES
    auto& dex_classes = dex_parser->get_classes();
    std::cout << dex_classes << "\n";

    // PROTOS 
    auto& dex_protos = dex_parser->get_protos();
    std::cout << dex_protos << "\n";

    // METHODS
    auto& dex_methods = dex_parser->get_methods();
    std::cout << dex_methods << "\n";
    
    // MAP LIST
    auto& dex_map_list = dex_parser->get_maplist();
    std::cout << dex_map_list << "\n";

    return 0;
}