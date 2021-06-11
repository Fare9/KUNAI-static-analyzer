/**
 * @file dex_external_classes.hpp
 * @author @Farenain
 * 
 * @brief Android external class used to create specific
 *        object in the analysis of the code, for calls
 *        to external classes.
 */

#ifndef DEX_EXTERNAL_CLASSES_HPP
#define DEX_EXTERNAL_CLASSES_HPP

#include <iostream>
#include <vector>
#include <memory>

#include "dex_external_methods.hpp"

namespace KUNAI {
    namespace DEX {

        class ExternalClass
        {
        public:
            ExternalClass(std::string name);
            ~ExternalClass();

            std::string get_name();
            std::uint64_t get_number_of_methods();
            std::shared_ptr<ExternalMethod> get_method_by_pos(std::uint64_t pos);
            void add_method(std::shared_ptr<ExternalMethod> method);

        private:
            std::string name;
            std::vector<std::shared_ptr<ExternalMethod>> methods;
        };
    }
}

#endif