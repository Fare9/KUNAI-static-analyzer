#include "dex_external_classes.hpp"

namespace KUNAI {
    namespace DEX {

            ExternalClass::ExternalClass(std::string name) :
                ParentClass(ParentClass::EXTERNAL_CLASS_T),
                name (name)
            {
            }

            ExternalClass::~ExternalClass()
            {
                if (!methods.empty())
                    methods.clear();
            }

            std::shared_ptr<ExternalMethod> ExternalClass::get_method_by_pos(std::uint64_t pos)
            {
                if (pos >= methods.size())
                    return nullptr;
                return methods[pos];
            }

    }
}