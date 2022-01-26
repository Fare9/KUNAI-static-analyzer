#include "dex_methods.hpp"

namespace KUNAI
{
    namespace DEX
    {
        /***
         * Class MethodID
         */

        MethodID::MethodID(std::uint16_t class_idx,
                           std::uint16_t proto_idx,
                           std::uint32_t name_idx,
                           std::shared_ptr<DexStrings> dex_strings,
                           std::shared_ptr<DexTypes> dex_types,
                           std::shared_ptr<DexProtos> dex_protos)
        {
            this->class_idx[class_idx] = dex_types->get_type_from_order(class_idx);
            this->proto_idx[proto_idx] = dex_protos->get_proto_by_order(proto_idx);
            this->name_idx[name_idx] = dex_strings->get_string_from_order(name_idx);
        }

        std::ostream &operator<<(std::ostream &os, const MethodID &entry)
        {
            os << entry.class_idx.begin()->second->get_raw() << "." << *entry.name_idx.begin()->second;
            os << "(";
            for (size_t j = 0; j < entry.proto_idx.begin()->second->get_number_of_parameters(); j++)
            {
                os << entry.proto_idx.begin()->second->get_parameter_type_by_order(j)->get_raw();
                if (j != entry.proto_idx.begin()->second->get_number_of_parameters() - 1)
                    os << ", ";
            }
            os << ")" << entry.proto_idx.begin()->second->get_return_idx()->get_raw();
            os << std::endl;
            return os;
        }

        /***
         * Class DexMethods
         */
        DexMethods::DexMethods(std::ifstream &input_file,
                               std::uint32_t number_of_methods,
                               std::uint32_t offset,
                               std::shared_ptr<DexStrings> dex_strings,
                               std::shared_ptr<DexTypes> dex_types,
                               std::shared_ptr<DexProtos> dex_protos) : number_of_methods(number_of_methods),
                                                                        offset(offset),
                                                                        dex_strings(dex_strings),
                                                                        dex_types(dex_types),
                                                                        dex_protos(dex_protos)
        {
            if (!parse_methods(input_file))
                throw exceptions::ParserReadingException("Error reading DEX methods");
        }

        DexMethods::~DexMethods()
        {
            if (!method_ids.empty())
            {
                for (size_t i = 0; i < method_ids.size(); i++)
                    delete method_ids[i];
                method_ids.clear();
            }
        }

        MethodID *DexMethods::get_method_by_order(size_t pos)
        {
            if (pos >= method_ids.size())
                return nullptr;
            return method_ids[pos];
        }

        bool DexMethods::parse_methods(std::ifstream &input_file)
        {
            auto logger = LOGGER::logger();

            MethodID *method_id;
            auto current_offset = input_file.tellg();
            size_t i = 0;
            std::uint16_t class_idx = 0, proto_idx = 0;
            std::uint32_t name_idx = 0;

            // set to current offset
            input_file.seekg(offset);

            #ifdef DEBUG
            logger->debug("DexMethods parsing methods in offset {} and size {}", offset, number_of_methods);
            #endif

            for (i = 0; i < number_of_methods; i++)
            {
                if (!KUNAI::read_data_file<std::uint16_t>(class_idx, sizeof(std::uint16_t), input_file))
                    return false;

                if (class_idx >= dex_types->get_number_of_types())
                {
                    logger->error("Error reading methods class_idx out of type bound ({} >= {})", class_idx, dex_types->get_number_of_types());
                    throw exceptions::IncorrectTypeId("Error reading methods class_idx out of type bound");
                }

                if (!KUNAI::read_data_file<std::uint16_t>(proto_idx, sizeof(std::uint16_t), input_file))
                    return false;

                if (proto_idx >= dex_protos->get_number_of_protos())
                {
                    logger->error("Error reading methods proto_idx out of proto bound ({} >= {})", proto_idx, dex_protos->get_number_of_protos());
                    throw exceptions::IncorrectProtoId("Error reading methods proto_idx out of proto bound");
                }

                if (!KUNAI::read_data_file<std::uint32_t>(name_idx, sizeof(std::uint32_t), input_file))
                    return false;

                if (name_idx >= dex_strings->get_number_of_strings())
                {
                    logger->error("Error reading methods name_idx out of string bound ({} >= {})", name_idx, dex_strings->get_number_of_strings());
                    throw exceptions::IncorrectStringId("Error reading methods name_idx out of string bound");
                }

                method_id = new MethodID(class_idx, proto_idx, name_idx, dex_strings, dex_types, dex_protos);

                method_ids.push_back(method_id);

                #ifdef DEBUG
                logger->debug("Added method number {}", i);
                #endif
            }

            // set to previous offset
            input_file.seekg(current_offset);

            logger->info("DexMethods parsing correct");

            return true;
        }

        std::ostream &operator<<(std::ostream &os, const DexMethods &entry)
        {
            size_t i = 0;
            os << std::hex;
            os << std::setw(30) << std::left << std::setfill(' ') << "=========== DEX Methods ===========" << std::endl;
            for (auto it = entry.method_ids.begin(); it != entry.method_ids.end(); it++)
            {
                auto method_id = *it;
                os << "Method (" << i++ << "): ";
                os << *method_id;
            }

            return os;
        }
    }
}