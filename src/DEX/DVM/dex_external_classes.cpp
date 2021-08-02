#include "dex_external_classes.hpp"

namespace KUNAI {
    namespace DEX {

            ExternalClass::ExternalClass(std::string name)
            {
                this->name = name;
            }

            ExternalClass::~ExternalClass()
            {
                if (!methods.empty())
                    methods.clear();
            }

            std::string ExternalClass::get_name()
            {
                return name;
            }

            std::uint64_t ExternalClass::get_number_of_methods()
            {
                return methods.size();
            }

            std::shared_ptr<ExternalMethod> ExternalClass::get_method_by_pos(std::uint64_t pos)
            {
                if (pos >= methods.size())
                    return nullptr;
                return methods[pos];
            }

            void ExternalClass::add_method(std::shared_ptr<ExternalMethod> method)
            {
                methods.push_back(method);
            }

    }
}