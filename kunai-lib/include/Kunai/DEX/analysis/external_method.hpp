//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file external_method.hpp
// @brief This external method will be created for the analysis in those cases
// the method is in another dex or is not in the apk.

#ifndef KUNAI_DEX_ANALYSIS_EXTERNAL_METHOD_HPP
#define KUNAI_DEX_ANALYSIS_EXTERNAL_METHOD_HPP

#include "Kunai/DEX/DVM/dvm_types.hpp"

#include <iostream>


namespace KUNAI
{
namespace DEX
{
    class ExternalMethod
    {
        /// @brief name of the class
        std::string& class_idx;

        /// @brief name of the method
        std::string& name_idx;

        /// @brief prototype of the method
        std::string& proto_idx;

        /// @brief name that joins class+method+proto
        mutable std::string pretty_name;

        /// @brief access flags of the method
        TYPES::access_flags access_flags = TYPES::access_flags::NONE;

    public:
        ExternalMethod(std::string& class_idx, std::string& name_idx, std::string& proto_idx)
            : class_idx(class_idx), name_idx(name_idx), proto_idx(proto_idx)
        {}

        /// @brief Return the name of the class where the method is
        /// @return name of the class
        std::string& get_class_idx() const
        {
            return class_idx;
        }

        /// @brief Get the name of the external method
        /// @return name of the method
        std::string& get_name_idx() const
        {
            return name_idx;
        }

        /// @brief Get the prototype of the external method
        /// @return prototype of the method
        std::string& get_proto_idx() const
        {
            return proto_idx;
        }

        /// @brief Get a pretty printed version of the name
        /// that includes class name, name of the method
        /// and the prototype.
        /// @return pretty printed version of the name
        std::string& pretty_method_name() const
        {
            if (pretty_name.empty())
                pretty_name = class_idx+"->"+name_idx+proto_idx;
            return pretty_name;
        }

        /// @brief Get the access flags from the method
        /// @return NONE access flags
        TYPES::access_flags get_access_flags() const
        {
            return access_flags;
        }
    };
} // namespace DEX
} // namespace KUNAI


#endif