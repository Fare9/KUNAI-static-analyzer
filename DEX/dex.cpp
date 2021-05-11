#include "dex.hpp"

namespace KUNAI {
    namespace DEX {
        DEX::DEX(std::ifstream& input_file, std::uint64_t file_size)
        {
            dex_header = std::make_shared<DexHeader>(input_file, file_size);
            dex_strings = std::make_shared<DexStrings>(input_file, file_size, dex_header->get_dex_header().string_ids_size, dex_header->get_dex_header().string_ids_off);
            dex_types = std::make_shared<DexTypes>(input_file, dex_header->get_dex_header().type_ids_size, dex_header->get_dex_header().type_ids_off, dex_strings);
            dex_protos = std::make_shared<DexProtos>(input_file, file_size, dex_header->get_dex_header().proto_ids_size, dex_header->get_dex_header().proto_ids_off, dex_strings, dex_types);
            dex_fields = std::make_shared<DexFields>(input_file, dex_header->get_dex_header().field_ids_size, dex_header->get_dex_header().field_ids_off, dex_strings, dex_types);
            dex_methods = std::make_shared<DexMethods>(input_file, dex_header->get_dex_header().method_ids_size, dex_header->get_dex_header().method_ids_off, dex_strings, dex_types, dex_protos);
            dex_classes = std::make_shared<DexClasses>(input_file, file_size, dex_header->get_dex_header().class_defs_size, dex_header->get_dex_header().class_defs_off, dex_strings, dex_types, dex_fields, dex_methods);
        }
        
        DEX::~DEX(){}


        std::ostream& operator<<(std::ostream& os, const DEX& entry)
        {
            if (entry.dex_header)
                os << *entry.dex_header;
            if (entry.dex_strings)
                os << *entry.dex_strings;
            if (entry.dex_types)
                os << *entry.dex_types;
            if (entry.dex_protos)
                os << *entry.dex_protos;
            if (entry.dex_fields)
                os << *entry.dex_fields;
            if (entry.dex_methods)
                os << *entry.dex_methods;
            if (entry.dex_classes)
                os << *entry.dex_classes;
            
            return os;
        }
    }
}