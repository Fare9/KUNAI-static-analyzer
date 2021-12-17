/**
 * @file dex_external_methods.hpp
 * @author @Farenain
 * 
 * @brief Android external methods, used
 *        in exteran classes, used to fill
 *        those calls to methods not declared
 *        in the app.
 */

#ifndef DEX_EXTERNAL_METHODS_HPP
#define DEX_EXTERNAL_METHODS_HPP

#include <iostream>
#include <vector>

#include "dex_parents.hpp"
#include "dex_dvm_types.hpp"

namespace KUNAI {
    namespace DEX {

        class ExternalMethod : public ParentMethod
        {
        public:
            ExternalMethod(std::string class_idx, std::string name_idx, std::string proto_idx);
            ~ExternalMethod();

            std::string get_name();
            std::string get_class_name();
            std::string get_descriptor();
            std::string full_name();
            std::string permission_api_name();
            
            DVMTypes::ACCESS_FLAGS get_access_flags();

            friend std::ostream& operator<<(std::ostream& os, const ExternalMethod& entry);
        private:
            std::string class_idx;
            std::string name_idx;
            DVMTypes::ACCESS_FLAGS access_flags;
            std::string proto_idx;
        };

    }
}

#endif