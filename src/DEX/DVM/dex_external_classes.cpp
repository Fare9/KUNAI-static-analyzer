#include "KUNAI/DEX/DVM/dex_external_classes.hpp"

namespace KUNAI {
    namespace DEX {

            ExternalClass::ExternalClass(std::string name) :
                name (name)
            {
            }

            ExternalClass::~ExternalClass()
            {
                if (!methods.empty())
                    methods.clear();
            }

            externalmethod_t ExternalClass::get_method_by_pos(std::uint64_t pos)
            {
                if (pos >= methods.size())
                    return nullptr;
                return methods[pos];
            }

    }
}