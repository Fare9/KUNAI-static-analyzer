#include "dex_parser.hpp"

namespace KUNAI
{
    namespace DEX
    {
        DexParser::DexParser() {}

        DexParser::~DexParser() {}

        void DexParser::parse_dex_file(std::ifstream &input_file, std::uint64_t file_size)
        {
            std::uint8_t header[8];

            if (file_size < sizeof(DexHeader::dexheader_t))
                throw exceptions::IncorrectDexFile("File is not a DEX file");

            if (!KUNAI::read_data_file<std::uint8_t[8]>(header, sizeof(std::uint8_t[8]), input_file))
                throw exceptions::ParserReadingException("Error parsing header");

            // quick check
            // big check should be done in detector module
            if (!memcmp(header, dex_magic_035, 8) &&
                !memcmp(header, dex_magic_037, 8) &&
                !memcmp(header, dex_magic_038, 8) &&
                !memcmp(header, dex_magic_039, 8))
                throw exceptions::IncorrectDexFile("Error dex magic not recognized");

            input_file.seekg(0);
            
            dex_header = std::make_shared<DexHeader>(input_file, file_size);
            dex_strings = std::make_shared<DexStrings>(input_file, file_size, dex_header->get_dex_header().string_ids_size, dex_header->get_dex_header().string_ids_off);
            dex_types = std::make_shared<DexTypes>(input_file, dex_header->get_dex_header().type_ids_size, dex_header->get_dex_header().type_ids_off, dex_strings);
            dex_protos = std::make_shared<DexProtos>(input_file, file_size, dex_header->get_dex_header().proto_ids_size, dex_header->get_dex_header().proto_ids_off, dex_strings, dex_types);
            dex_fields = std::make_shared<DexFields>(input_file, dex_header->get_dex_header().field_ids_size, dex_header->get_dex_header().field_ids_off, dex_strings, dex_types);
            dex_methods = std::make_shared<DexMethods>(input_file, dex_header->get_dex_header().method_ids_size, dex_header->get_dex_header().method_ids_off, dex_strings, dex_types, dex_protos);
            dex_classes = std::make_shared<DexClasses>(input_file, file_size, dex_header->get_dex_header().class_defs_size, dex_header->get_dex_header().class_defs_off, dex_strings, dex_types, dex_fields, dex_methods);
        }

        std::shared_ptr<DexHeader>  DexParser::get_header()
        {
            return dex_header;
        }

        std::shared_ptr<DexStrings> DexParser::get_strings()
        {
            return dex_strings;
        }

        std::shared_ptr<DexTypes>   DexParser::get_types()
        {
            return dex_types;
        }

        std::shared_ptr<DexProtos>  DexParser::get_protos()
        {
            return dex_protos;
        }

        std::shared_ptr<DexFields>  DexParser::get_fields()
        {
            return dex_fields;
        }

        std::shared_ptr<DexMethods> DexParser::get_methods()
        {
            return dex_methods;
        }

        std::shared_ptr<DexClasses> DexParser::get_classes()
        {
            return dex_classes;
        }

        std::ostream& operator<<(std::ostream& os, const DexParser& entry)
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