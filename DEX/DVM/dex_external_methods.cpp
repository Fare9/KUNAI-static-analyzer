#include "dex_external_methods.hpp"

namespace KUNAI {
    namespace DEX {
        ExternalMethod::ExternalMethod(std::string class_idx, std::vector<std::string> proto_idx, std::string name_idx)
        {
            this->class_idx = class_idx;
            this->proto_idx = proto_idx;
            this->name_idx = name_idx;
        }

        ExternalMethod::~ExternalMethod() {}

        std::string ExternalMethod::get_name()
        {
            return name_idx;
        }

        std::string ExternalMethod::get_class_name()
        {
            return class_idx;
        }

        std::string ExternalMethod::get_descriptor()
        {
            std::string method_proto = "";

            if (proto_idx.size() == 1)
            {
                return "()" + proto_idx[0];
            }

            method_proto = "(";

            for (size_t i = 0; i < proto_idx.size()-1; i++)
            {
                method_proto += proto_idx[i];
                if (i != proto_idx.size()-2)
                    method_proto += ", ";
            }

            method_proto += ")" + proto_idx[proto_idx.size()-1];

            return method_proto;
        }

        /**
         * @brief return classname + name + proto separated by spaces.
         * @return std::string with full name.
         */
        std::string ExternalMethod::full_name()
        {
            return class_idx + " " + name_idx + " " + get_descriptor();
        }

        /**
         * @brief return classname + name + proto in a way useful to look up in permission maps.
         * @return std::string with full name for permission check.
         */
        std::string ExternalMethod::permission_api_name()
        {
            return class_idx + "-" + name_idx + "-" + get_descriptor();
        }

        std::ostream& operator<<(std::ostream& os, const ExternalMethod& entry)
        {
            std::string method_proto = "";

            if (entry.proto_idx.size() == 1)
            {
                os  << "()" << entry.proto_idx[0];
                return os;
            }

            method_proto = "(";

            for (size_t i = 0; i < entry.proto_idx.size()-1; i++)
            {
                method_proto += entry.proto_idx[i];
                if (i != entry.proto_idx.size()-2)
                    method_proto += ", ";
            }

            method_proto += ")" + entry.proto_idx[entry.proto_idx.size()-1];

            os << entry.class_idx << "->" << entry.name_idx << method_proto;

            return os;
        }

    }
}