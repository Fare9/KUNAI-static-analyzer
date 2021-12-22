#include "dex_external_methods.hpp"

namespace KUNAI
{
    namespace DEX
    {
        ExternalMethod::ExternalMethod(std::string class_idx,
                                       std::string name_idx,
                                       std::string proto_idx) : ParentMethod(ParentMethod::EXTERNAL_METHOD_T),
                                                                class_idx(class_idx),
                                                                name_idx(name_idx),
                                                                proto_idx(proto_idx)
        {
        }

        std::ostream &operator<<(std::ostream &os, const ExternalMethod &entry)
        {
            std::string method_proto = "";

            if (entry.proto_idx.size() == 1)
            {
                os << "()" << entry.proto_idx[0];
                return os;
            }

            method_proto = entry.proto_idx;

            os << entry.class_idx << "->" << entry.name_idx << method_proto;

            return os;
        }

    }
}