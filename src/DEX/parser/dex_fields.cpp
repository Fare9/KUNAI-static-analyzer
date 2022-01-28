#include "dex_fields.hpp"

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
                         std::shared_ptr<DexStrings>& dex_strings,
                         std::shared_ptr<DexTypes>& dex_types)
        {
            this->class_idx[class_idx] = dex_types->get_type_from_order(class_idx);
            this->type_idx[type_idx] = dex_types->get_type_from_order(type_idx);
            this->name_idx[name_idx] = dex_strings->get_string_from_order(name_idx);
        }

        Type *FieldID::get_class_idx()
        {
            if (class_idx.empty())
                return nullptr;
            return class_idx.begin()->second;
        }

        Type *FieldID::get_type_idx()
        {
            if (type_idx.empty())
                return nullptr;
            return type_idx.begin()->second;
        }

        std::string *FieldID::get_name_idx()
        {
            if (name_idx.empty())
                return nullptr;
            return name_idx.begin()->second;
        }

        std::ostream &operator<<(std::ostream &os, const FieldID &entry)
        {
            if (entry.type_idx.size() > 0)
            {
                os << entry.type_idx.begin()->second->get_raw();
                os << " ";
            }
            if (entry.class_idx.size() > 0)
            {
                os << entry.class_idx.begin()->second->get_raw();
                os << "->";
            }
            if (entry.name_idx.size() > 0)
                os << *entry.name_idx.begin()->second;
            os << std::endl;
            return os;
        }

        /***
         * DexFields class
         */
        DexFields::DexFields(std::ifstream &input_file,
                             std::uint32_t number_of_fields,
                             std::uint32_t offset,
                             std::shared_ptr<DexStrings>& dex_strings,
                             std::shared_ptr<DexTypes>& dex_types) : number_of_fields(number_of_fields),
                                                                    offset(offset),
                                                                    dex_strings(dex_strings),
                                                                    dex_types(dex_types)
        {
            if (!parse_fields(input_file))
                throw exceptions::ParserReadingException("Error reading DEX fields");
        }

        DexFields::~DexFields()
        {
            if (!field_ids.empty())
            {
                for (size_t i = 0; i < field_ids.size(); i++)
                    delete field_ids[i];
                field_ids.clear();
            }
        }

        FieldID *DexFields::get_field_id_by_order(size_t pos)
        {
            if (pos >= field_ids.size())
                return nullptr;
            return field_ids[pos];
        }

        bool DexFields::parse_fields(std::ifstream &input_file)
        {
            auto logger = LOGGER::logger();

            FieldID *field_id;
            auto current_offset = input_file.tellg();
            size_t i = 0;
            std::uint16_t class_idx, type_idx;
            std::uint32_t name_idx;

            // move to offset for analysis
            input_file.seekg(offset);

            #ifdef DEBUG
            logger->debug("DexFields start parsing in offset {} and size {}", offset, number_of_fields);
            #endif

            for (i = 0; i < number_of_fields; i++)
            {
                if (!KUNAI::read_data_file<std::uint16_t>(class_idx, sizeof(std::uint16_t), input_file))
                    return false;

                if (class_idx >= dex_types->get_number_of_types())
                {
                    logger->error("Error reading fields class_idx out of type bound ({} >= {}):", class_idx, dex_types->get_number_of_types());
                    throw exceptions::IncorrectTypeId("Error reading fields class_idx out of type bound");
                }

                if (!KUNAI::read_data_file<std::uint16_t>(type_idx, sizeof(std::uint16_t), input_file))
                    return false;

                if (type_idx >= dex_types->get_number_of_types())
                {
                    logger->error("Error reading fields type_idx out of type bound ({} >= {})", type_idx, dex_types->get_number_of_types());
                    throw exceptions::IncorrectTypeId("Error reading fields type_idx out of type bound");
                }

                if (!KUNAI::read_data_file<std::uint32_t>(name_idx, sizeof(std::uint32_t), input_file))
                    return false;

                if (name_idx >= dex_strings->get_number_of_strings())
                {
                    logger->error("Error reading fields name_idx out of string bound ({} >= {})", name_idx, dex_strings->get_number_of_strings());
                    throw exceptions::IncorrectStringId("Error reading fields name_idx out of string bound");
                }

                field_id = new FieldID(class_idx, type_idx, name_idx, dex_strings, dex_types);

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
            os << std::setw(30) << std::left << std::setfill(' ') << "=========== DEX Fields ===========" << std::endl;
            for (auto it = entry.field_ids.begin(); it != entry.field_ids.end(); it++)
            {
                FieldID *field_id = *it;
                os << std::left << std::setfill(' ') << "Field (" << std::dec << i++ << std::hex << "): ";
                os << *field_id;
            }

            return os;
        }

        std::fstream &operator<<(std::fstream &fos, const DexFields &entry)
        {
            std::stringstream stream;
            stream << std::hex;
            stream << std::setw(30) << std::left << std::setfill(' ') << "<fields>" << std::endl;
            for (auto it = entry.field_ids.begin(); it != entry.field_ids.end(); it++)
            {
                FieldID *field_id = *it;
                stream << "\t<field>" << std::endl;
                stream << "\t\t<type>" << field_id->get_type_idx()->get_raw() << "</type>" << std::endl;
                stream << "\t\t<class>" << field_id->get_class_idx()->get_raw() << "</class>" << std::endl;
                stream << "\t\t<name>" << *field_id->get_name_idx() << "</name>" << std::endl;
                stream << "\t</field>" << std::endl;
            }
            stream << std::setw(30) << std::left << std::setfill(' ') << "</fields>" << std::endl;

            fos.write(stream.str().c_str(), stream.str().size());

            return fos;
        }
    }
}