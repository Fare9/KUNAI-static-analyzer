#include "KUNAI/DEX/parser/dex_types.hpp"

namespace KUNAI
{
    namespace DEX
    {
        /***
         * Type class
         */
        Type::Type(type_e type, std::string raw) : type(type),
                                                   raw(raw)
        {
        }

        /***
         * Fundamental class
         */

        Fundamental::Fundamental(fundamental_e f_type,
                                 std::string name)
            : Type(FUNDAMENTAL, name),
              f_type(f_type),
              name(name)
        {
        }

        std::string Fundamental::print_fundamental_type()
        {
            switch (f_type)
            {
            case BOOLEAN:
                return "boolean";
            case BYTE:
                return "byte";
            case CHAR:
                return "char";
            case DOUBLE:
                return "double";
            case FLOAT:
                return "float";
            case INT:
                return "int";
            case LONG:
                return "long";
            case SHORT:
                return "short";
            case VOID:
                return "void";
            }
            return "";
        }

        /***
         * Class class
         */

        Class::Class(std::string name)
            : Type(CLASS, name),
              name(name)
        {
        }

        /***
         * Array class
         */

        Array::Array(std::string raw, size_t depth, type_t &array)
            : Type(ARRAY, raw),
                depth(depth),
                array(std::move(array))
        {
        }

        /***
         * Unknown class
         */

        Unknown::Unknown(type_e type, std::string raw)
            : Type(type, raw)
        {
        }

        /***
         * DexTypes class
         */

        DexTypes::DexTypes(std::ifstream &input_file,
                           std::uint32_t number_of_types,
                           std::uint32_t types_offsets,
                           DexStrings* dex_str) : number_of_types(number_of_types),
                                                    offset(types_offsets),
                                                    dex_str(dex_str)
        {
            if (!parse_types(input_file))
                throw exceptions::ParserReadingException("Error reading DEX types");
        }

        DexTypes::~DexTypes() = default;

        Type* DexTypes::get_type_by_id(std::uint32_t type_id)
        {
            return types[type_id].get();
        }

        Type* DexTypes::get_type_from_order(std::uint32_t pos)
        {
            size_t i = pos;

            for (auto & type : types)
            {
                if (!i--)
                    return type.second.get();
            }

            return nullptr;
        }

        /**
         * Private methods
         */

        type_t DexTypes::parse_type(std::string name)
        {
            auto logger = LOGGER::logger();

            type_t type;

            switch (name.at(0))
            {
            case 'Z':
                type = std::make_unique<Fundamental>(Fundamental::BOOLEAN, name);
                break;
            case 'B':
                type = std::make_unique<Fundamental>(Fundamental::BYTE, name);
                break;
            case 'C':
                type = std::make_unique<Fundamental>(Fundamental::CHAR, name);
                break;
            case 'D':
                type = std::make_unique<Fundamental>(Fundamental::DOUBLE, name);
                break;
            case 'F':
                type = std::make_unique<Fundamental>(Fundamental::FLOAT, name);
                break;
            case 'I':
                type = std::make_unique<Fundamental>(Fundamental::INT, name);
                break;
            case 'J':
                type = std::make_unique<Fundamental>(Fundamental::LONG, name);
                break;
            case 'S':
                type = std::make_unique<Fundamental>(Fundamental::SHORT, name);
                break;
            case 'V':
                type = std::make_unique<Fundamental>(Fundamental::VOID, name);
                break;
            case 'L':
                type = std::make_unique<Class>(name);
                break;
            case '[':
            {
                size_t depth = 0;
                std::string name_cpy = name;
                while(name_cpy.at(0) == '[')
                {
                    depth++;
                    name_cpy = name_cpy.substr(1, name_cpy.length() - 1);
                }
                type_t aux_type;
                aux_type = parse_type(name_cpy);
                type = std::make_unique<Array>(name, depth, aux_type);
                break;
            }
            default:
                type = std::make_unique<Unknown>(Type::UNKNOWN, name);
            }

            return type;
        }

        bool DexTypes::parse_types(std::ifstream &input_file)
        {
            auto logger = LOGGER::logger();

            auto current_offset = input_file.tellg();

            size_t i;
            std::uint32_t type_id;
            type_t type;

            // move to offset where are the string ids
            input_file.seekg(offset);

#ifdef DEBUG
            logger->debug("DexTypes start parsing types in offset {} with size {}", offset, number_of_types);
#endif

            for (i = 0; i < number_of_types; i++)
            {
                if (!KUNAI::read_data_file<std::uint32_t>(type_id, sizeof(std::uint32_t), input_file))
                    return false;

                if (type_id >= dex_str->get_number_of_strings())
                {
                    logger->error("Error reading types type_id out of string bound ({} >= {})", type_id, dex_str->get_number_of_strings());
                    throw exceptions::IncorrectStringId("Error reading types type_id out of string bound");
                }

                type = parse_type(*dex_str->get_string_from_order(type_id));

                types[type_id] = std::move(type);

#ifdef DEBUG
                logger->debug("parsed type number {}", i);
#endif
            }

            input_file.seekg(current_offset);

            logger->info("DexTypes parsing correct");

            return true;
        }

        std::ostream &operator<<(std::ostream &os, const DexTypes &entry)
        {
            size_t i = 0;
            os << std::hex;
            os << std::setw(30) << std::left << std::setfill(' ') << "=========== DEX Types ===========" << "\n";

            for (auto &type : entry.types)
                os << std::left << std::setfill(' ') << "Type (" << std::dec << i++ << std::hex << "): " << type.first << "-> \"" << type.second->get_raw() << "\"" << "\n";

            return os;
        }

        std::fstream &operator<<(std::fstream &fos, const DexTypes &entry)
        {
            std::stringstream stream;

            stream << std::hex;
            stream << "<types>" << "\n";

            for (auto &type : entry.types)
            {
                stream << "\t<type>" << "\n";
                stream << "\t\t<id>" << type.first << "</id>" << "\n";
                stream << "\t\t<value>" << type.second->get_raw() << "</value>" << "\n";
                stream << "\t</type>" << "\n";
            }

            stream << "</types>" << "\n";

            fos.write(stream.str().c_str(), stream.str().size());

            return fos;
        }
    }
}