//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file external_class.hpp
// @brief A class for managing external classes that does not exist in
// current DEX or current apk file

#ifndef KUNAI_DEX_ANALYSIS_EXTERNAL_CLASS_HPP
#define KUNAI_DEX_ANALYSIS_EXTERNAL_CLASS_HPP

#include "Kunai/DEX/analysis/external_method.hpp"
#include <iostream>
#include <vector>
#include <memory>

namespace KUNAI
{
namespace DEX
{
    class ExternalClass
    {
        /// @brief name of the external class
        std::string& name;
        /// @brief Vector with all the external methods from the current class
        std::vector<ExternalMethod*> methods;
    public:
        ExternalClass(std::string& name) : name(name)
        {}

        /// @brief Get the name of the external class
        /// @return name of the class
        std::string& get_name()
        {
            return name;
        }

        /// @brief Get a constant reference to the methods of the class
        /// @return constant reference to methods
        const std::vector<ExternalMethod*>& get_methods() const
        {
            return methods;
        }

        /// @brief Get a reference to the methods of the class
        /// @return reference to methods
        std::vector<ExternalMethod*>& get_methods()
        {
            return methods;
        }

        /// @brief Add an external method to the list of methods
        /// @param method new method of the class
        void add_external_method(ExternalMethod* method)
        {
            methods.push_back(method);
        }
    };
} // namespace DEX
} // namespace KUNAI


#endif