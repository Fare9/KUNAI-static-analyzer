#include "KUNAI/DEX/parser/dex_fields.hpp"

namespace KUNAI
{
    namespace DEX
    {

        /***
         * FieldID class
         */

        FieldID::FieldID(std::uint16_t class_idx,
                         std::uint16_t type_idx,
                         std::uint32_t name_idx,
                         dexstrings_t &dex_strings,
                         dextypes_t &dex_types)
        {
            this->class_idx = std::make_pair(class_idx, dex_types->get_type_from_order(class_idx));
            this->type_idx = std::make_pair(type_idx, dex_types->get_type_from_order(type_idx));
            this->name_idx = std::make_pair(name_idx, dex_strings->get_string_from_order(name_idx));
        }

        type_t FieldID::get_class_idx()
        {
            return class_idx.second;
        }

        type_t FieldID::get_type_idx()
        {
            return type_idx.second;
        }

        std::string *FieldID::get_name_idx()
        {
            return name_idx.second;
        }

        std::string FieldID::get_field_str()
        {
            return type_idx.second->get_raw() + " " + class_idx.second->get_raw() + "->" + *(name_idx.second);
        }

        std::ostream &operator<<(std::ostream &os, const FieldID &entry)
        {
            os << entry.type_idx.second->get_raw() << " ";
            os << entry.class_idx.second->get_raw() << "->";
            os << *(entry.name_idx.second);

            os << "\n";
            return os;
        }

        /***
         * DexFields class
         */
        DexFields::DexFields(std::ifstream &input_file,
                             std::uint32_t number_of_fields,
                             std::uint32_t offset,
                             dexstrings_t &dex_strings,
                             dextypes_t &dex_types) : number_of_fields(number_of_fields),
                                                      offset(offset),
                                                      dex_strings(dex_strings),
                                                      dex_types(dex_types)
        {
            if (!parse_fields(input_file))
                throw exceptions::ParserReadingException("Error reading DEX fields");
        }

        fieldid_t DexFields::get_field_id_by_order(size_t pos)
        {
            if (pos >= field_ids.size())
                return nullptr;
            return field_ids[pos];
        }

        bool DexFields::parse_fields(std::ifstream &input_file)
        {
            auto logger = LOGGER::logger();

            fieldid_t field_id;
            auto current_offset = input_file.tellg();
            size_t i = 0;
            std::uint16_t class_idx, type_idx;
            std::uint32_t name_idx;

            // move to offset for analysis
            input_file.seekg(offset);

#ifdef DEBUG
            logger->debug("DexFields start parsing in offset {} and size {}", offset, number_of_fields);
#endif

            auto number_of_types = dex_types->get_number_of_types();
            auto number_of_strings = dex_strings->get_number_of_strings();

            for (i = 0; i < number_of_fields; i++)
            {
                if (!KUNAI::read_data_file<std::uint16_t>(class_idx, sizeof(std::uint16_t), input_file))
                    return false;

                if (class_idx >= number_of_types)
                {
                    logger->error("Error reading fields class_idx out of type bound ({} >= {}):", class_idx, number_of_types);
                    throw exceptions::IncorrectTypeId("Error reading fields class_idx out of type bound");
                }

                if (!KUNAI::read_data_file<std::uint16_t>(type_idx, sizeof(std::uint16_t), input_file))
                    return false;

                if (type_idx >= number_of_types)
                {
                    logger->error("Error reading fields type_idx out of type bound ({} >= {})", type_idx, number_of_types);
                    throw exceptions::IncorrectTypeId("Error reading fields type_idx out of type bound");
                }

                if (!KUNAI::read_data_file<std::uint32_t>(name_idx, sizeof(std::uint32_t), input_file))
                    return false;

                if (name_idx >= number_of_strings)
                {
                    logger->error("Error reading fields name_idx out of string bound ({} >= {})", name_idx, number_of_strings);
                    throw exceptions::IncorrectStringId("Error reading fields name_idx out of string bound");
                }

                field_id = std::make_shared<FieldID>(class_idx, type_idx, name_idx, dex_strings, dex_types);

                field_ids.push_back(field_id);

#ifdef DEBUG
                logger->debug("parsed field_id number {}", i);
#endif
            }

            input_file.seekg(current_offset);

            return true;
        }

        std::ostream &operator<<(std::ostream &os, const DexFields &entry)
        {
            size_t i = 0;
            os << std::hex;
            os << std::setw(30) << std::left << std::setfill(' ') << "=========== DEX Fields ===========" << "\n";

            for (auto field_id : entry.field_ids)
            {
                os << std::left << std::setfill(' ') << "Field (" << std::dec << i++ << std::hex << "): ";
                os << *field_id;
            }

            return os;
        }

        std::fstream &operator<<(std::fstream &fos, const DexFields &entry)
        {
            std::stringstream stream;
            stream << std::hex;
            stream << std::setw(30) << std::left << std::setfill(' ') << "<fields>" << "\n";
            for (auto field_id : entry.field_ids)
            {
                stream << "\t<field>" << "\n";
                stream << "\t\t<type>" << field_id->get_type_idx()->get_raw() << "</type>" << "\n";
                stream << "\t\t<class>" << field_id->get_class_idx()->get_raw() << "</class>" << "\n";
                stream << "\t\t<name>" << *field_id->get_name_idx() << "</name>" << "\n";
                stream << "\t</field>" << "\n";
            }
            stream << std::setw(30) << std::left << std::setfill(' ') << "</fields>" << "\n";

            fos.write(stream.str().c_str(), stream.str().size());

            return fos;
        }
    }
}