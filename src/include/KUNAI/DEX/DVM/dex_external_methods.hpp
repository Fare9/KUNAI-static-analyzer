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

#include "KUNAI/DEX/DVM/dex_dvm_types.hpp"

namespace KUNAI
{
    namespace DEX
    {

        class ExternalMethod;

        using externalmethod_t = std::shared_ptr<ExternalMethod>;

        class ExternalMethod
        {
        public:
            /**
             * @brief Construct a new External Method object
             * 
             * @param class_idx 
             * @param name_idx 
             * @param proto_idx 
             */
            ExternalMethod(std::string class_idx, std::string name_idx, std::string proto_idx);
            
            /**
             * @brief Destroy the External Method object
             */
            ~ExternalMethod() = default;

            /**
             * @brief Get the method name
             * 
             * @return std::string 
             */
            std::string& get_name()
            {
                return name_idx;
            }

            /**
             * @brief Get the method class name
             * 
             * @return std::string 
             */
            std::string& get_class_name()
            {
                return class_idx;
            }

            /**
             * @brief Get the method descriptor
             * 
             * @return std::string 
             */
            std::string& get_descriptor()
            {
                return proto_idx;
            }

            /**
             * @brief return classname + name + proto separated by spaces.
             * @return std::string with full name.
             */
            std::string full_name()
            {
                return class_idx + " " + name_idx + " " + proto_idx;
            }

            /**
             * @brief return classname + name + proto in a way useful to look up in permission maps.
             * @return std::string with full name for permission check.
             */
            std::string permission_api_name()
            {
                return class_idx + "-" + name_idx + "-" + proto_idx;
            }

            /**
             * @brief Get the access flags of the method, in this case NONE.
             * 
             * @return DVMTypes::ACCESS_FLAGS 
             */
            DVMTypes::ACCESS_FLAGS get_access_flags()
            {
                return DVMTypes::ACCESS_FLAGS::NONE;
            }

            friend std::ostream &operator<<(std::ostream &os, const ExternalMethod &entry);

        private:
            std::string class_idx;
            std::string name_idx;
            DVMTypes::ACCESS_FLAGS access_flags;
            std::string proto_idx;
        };

    }
}

#endif