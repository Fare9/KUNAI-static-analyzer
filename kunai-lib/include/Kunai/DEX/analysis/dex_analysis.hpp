//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file dex_analysis.hpp
// @brief This file offer all the analysis funcionality in just one class
// we will use all the utilities from analysis.hpp

#ifndef KUNAI_DEX_ANALYSIS_DEX_ANALYSIS_HPP
#define KUNAI_DEX_ANALYSIS_DEX_ANALYSIS_HPP

#include "Kunai/DEX/analysis/analysis.hpp"
#include "Kunai/DEX/DVM/dex_disassembler.hpp"

namespace KUNAI
{
namespace DEX
{
    class Analysis
    {
        /// @brief all the dex parsers from the analysis
        std::vector<Parser*> parsers;

        /// @brief map with all the classes, we use as key
        /// the class name
        std::unordered_map<std::string,
            std::unique_ptr<ClassAnalysis>> classes;

        /// @brief Map with external classes, created on
        /// demands with classes that do not belong to the
        /// apk
        std::unordered_map<std::string,
            std::unique_ptr<ExternalClass>> external_classes;
        
        /// @brief Map with std::string as key the StringAnalysis
        /// value
        std::unordered_map<std::string,
            std::unique_ptr<StringAnalysis>> strings;
        
        /// @brief Map with the methods with the method full name
        /// as key
        std::unordered_map<std::string,
            std::unique_ptr<MethodAnalysis>> methods;

        /// @brief Map of the external methods with the full name
        /// as key
        std::unordered_map<std::string,
            std::unique_ptr<ExternalMethod>> external_methods;

        /// @brief MethodAnalysis stored by a hash
        std::unordered_map<std::string, MethodAnalysis*> methods_hashes;
        
        /// @brief map hash for quickly getting the methods (maybe will be deleted)
        std::unordered_map<std::string,
            MethodAnalysis*> method_hashes;

        std::vector<FieldAnalysis*> all_fields;
        
        /// @brief Pointer to a disassembler for obtaining the instructions
        DexDisassembler * disassembler;

        /// @brief are the xrefs already created?
        bool created_xrefs = false;

        /// @brief Internal method for creating the xref for `current_class`
        /// There are four steps involved in getting the xrefs:
        ///     * xrefs for class instantiation and static class usage.
        ///     * xrefs for method calls
        ///     * xrefs for string usage
        ///     * xrefs field manipuation
        /// All the information is stored in the Analysis objects.
        /// It might be quite slow as all instructions are parsed.
        /// @param current_class class to create the xrefs.
        void _create_xrefs(ClassDef * current_class);

        /// @brief Get a method by its hash, return the MethodAnalysis object
        /// in case it doesn't exists, create an ExternalMethod
        /// @param class_name name of method's class
        /// @param method_name name of the method
        /// @param method_descriptor prototype of the method
        /// @return a MethodAnalysis pointer
        MethodAnalysis* _resolve_method(std::string & class_name, 
            std::string & method_name, std::string & method_descriptor);

    public:
        Analysis(Parser * parser, DexDisassembler * disassembler, bool create_xrefs) : 
            created_xrefs(!create_xrefs), disassembler(disassembler)
        {
            if (parser)
                add(parser);
        }

        /// @brief Add all the classes and methods from a parser
        /// to the analysis class.
        /// @param parser parser to extract the information
        void add(Parser * parser);

        /// @brief Create class, method, string and field cross references
        /// if you are using multiple DEX files, this function must
        /// be called when all DEX files are added.
        /// If you call the function after every DEX file, it will only
        /// work for the first time.
        /// ADD ALL DEX FIRST
        void create_xrefs();

        /// @brief Get a ClassAnalysis object by the class name
        /// @param class_name name of the class to retrieve
        /// @return pointer to ClassAnalysis*
        ClassAnalysis* get_class_analysis(std::string &class_name)
        {
            auto it = classes.find(class_name);

            if (it != classes.end())
                return it->second.get();
            
            return nullptr;
        }

        /// @brief Get a constant reference to the classes
        /// @return constant reference to map with classes
        const std::unordered_map<std::string, std::unique_ptr<ClassAnalysis>>&
        get_classes() const
        {
            return classes;
        }

        /// @brief Get a reference to the classes
        /// @return reference to map with classes
        std::unordered_map<std::string, std::unique_ptr<ClassAnalysis>>&
        get_classes()
        {
            return classes;
        }

        /// @brief Get a constant reference to external classes
        /// @return constant reference to external classes
        const std::unordered_map<std::string, std::unique_ptr<ExternalClass>>&
        get_external_classes() const
        {
            return external_classes;
        }

        /// @brief Get a reference to external classes
        /// @return reference to external classes
        std::unordered_map<std::string, std::unique_ptr<ExternalClass>>&
        get_external_classes()
        {
            return external_classes;
        }
    
        /// @brief Get a MethodAnalysis pointer given an Encoded or External Method
        /// @param method method to retrieve
        /// @return MethodAnalysis from the given method
        MethodAnalysis * get_method(std::variant<EncodedMethod*, ExternalMethod*> method)
        {
            std::string method_key;

            if (method.index() == 0)
                method_key = std::get<EncodedMethod*>(method)->getMethodID()->pretty_method();
            else
                method_key = std::get<ExternalMethod*>(method)->pretty_method_name();
            
            auto it = methods.find(method_key);

            if (it == methods.end())
                return nullptr;
            
            return it->second.get();
        }

        /// @brief Obtain a method anaylsis by different values
        /// @param class_name class name of the method
        /// @param method_name name of the method
        /// @param method_descriptor prototype descriptor of the method
        /// @return pointer to MethodAnalysis or nullptr
        MethodAnalysis* get_method_analysis_by_name(std::string &class_name, 
            std::string &method_name, 
            std::string &method_descriptor)
        {
            std::string m_hash = class_name+method_name+method_descriptor;

            auto it = method_hashes.find(m_hash);

            if (it == method_hashes.end())
                return nullptr;

            return it->second;
        }

        /// @brief Obtain a MethodID by different values
        /// @param class_name class name of the method
        /// @param method_name name of the method
        /// @param method_descriptor prototype descriptor of the method
        /// @return pointer to MethodID or nullptr
        MethodID* get_method_id_by_name(std::string &class_name, 
            std::string &method_name, 
            std::string &method_descriptor)
        {
            auto m_a = get_method_analysis_by_name(class_name, method_name, method_descriptor);

            if (m_a && (!m_a->external()))
                return std::get<EncodedMethod*>(m_a->get_encoded_method())->getMethodID();
            
            return nullptr;
        }

        /// @brief Return a constant reference to the method analysis
        /// @return constant reference to map with MethodAnalysis
        const std::unordered_map<std::string, std::unique_ptr<MethodAnalysis>>&
        get_methods() const
        {
            return methods;
        }

        /// @brief Return a reference to the method analysis
        /// @return reference to map woth MethodAnalysis
        std::unordered_map<std::string, std::unique_ptr<MethodAnalysis>>&
        get_methods()
        {
            return methods;
        }

        /// @brief Return a constant reference to the ExternalMethods
        /// @return constant reference to map with ExternalMethod
        const std::unordered_map<std::string, std::unique_ptr<ExternalMethod>>&
        get_external_methods() const
        {
            return external_methods;
        }

        /// @brief Return a reference to the ExternalMethods
        /// @return reference to map with ExternalMethod
        std::unordered_map<std::string, std::unique_ptr<ExternalMethod>>&
        get_external_methods()
        {
            return external_methods;
        }

        /// @brief Get a field given an encoded field
        /// @param field field to obtain the FieldAnalysis
        /// @return FieldAnalysis object
        FieldAnalysis* get_field_analysis(EncodedField* field)
        {
            auto class_analysis = get_class_analysis(reinterpret_cast<DVMClass*>(field->get_field()->get_class())->get_name());

            if (class_analysis)
                return class_analysis->get_field_analysis(field);
            
            return nullptr;
        }

        /// @brief Get all the fields from all the classes
        /// @return reference to vector with all the fields
        std::vector<FieldAnalysis*>& get_fields();

        /// @brief Get a constant reference to the StringAnalysis objects
        /// @return constant reference to StringAnalysis map
        const std::unordered_map<std::string,
            std::unique_ptr<StringAnalysis>>& get_string_analysis() const
        {
            return strings;
        }

        /// @brief Get a reference to the StringAnalysis map
        /// @return reference to StringAnalysis map
        std::unordered_map<std::string,
            std::unique_ptr<StringAnalysis>>& get_string_analysis()
        {
            return strings;
        }

        /// @brief Find classes by name with regular expression,
        /// the method returns a list of ClassAnalysis object that
        /// match the regex.
        /// @param name regex of name class to find
        /// @param no_external want external classes too?
        /// @return vector with all ClassAnalysis objects
        std::vector<ClassAnalysis*> find_classes(std::string& name, bool no_external);

        /// @brief Find MethodAnalysis object by name with regular expression.
        /// This time is necessary to specify more values for the method.
        /// @param class_name name of the class to retrieve
        /// @param method_name name of the method to retrieve
        /// @param descriptor descriptor of this method
        /// @param accessflags 
        /// @param no_external 
        /// @return 
        std::vector<MethodAnalysis*> find_methods(std::string& class_name, 
            std::string& method_name, 
            std::string& descriptor, 
            std::string& accessflags,
            bool no_external);

        /// @brief Find the strings that match a provided regular expression
        /// @param str regular expression to find as string
        /// @return vector of StringAnalysis objects
        std::vector<StringAnalysis*> find_strings(std::string& str);

        /// @brief Find FieldAnalysis objects using regular expressions
        /// find those that are in classes.
        /// @param class_name name of the class where field is
        /// @param field_name name of the field
        /// @param field_type type of the field
        /// @param accessflags access flags of the field
        /// @return vector with all the fields that match the regex
        std::vector<FieldAnalysis*> find_fields(std::string& class_name, 
            std::string& field_name, 
            std::string& field_type,
            std::string& accessflags);
    };
} // namespace DEX
} // namespace KUNAI


#endif