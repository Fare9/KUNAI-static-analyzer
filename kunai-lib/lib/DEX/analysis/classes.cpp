//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file classes.cpp
// @brief Definitions of classes from analysis.hpp

#include "Kunai/DEX/analysis/analysis.hpp"

using namespace KUNAI::DEX;


std::string& ClassAnalysis::extends() const
{
    if (!extends_.empty())
        return extends_;

    if (is_external)
        extends_ = "Ljava/lang/Object;";
    else
        extends_ = std::get<ClassDef*>(class_def)->get_superclass()->get_name();
    
    return extends_;
}

std::string& ClassAnalysis::name() const
{
    if (!name_.empty())
        return name_;
    
    if (is_external)
        name_ = std::string(std::get<ExternalClass*>(class_def)->get_name());
    else
        name_ = std::string(std::get<ClassDef*>(class_def)->get_class_idx()->get_name());
    
    return name_;
}

void ClassAnalysis::add_method(MethodAnalysis* method_analysis)
{
    std::string method_key;

    method_key = method_analysis->get_full_name();

    methods[method_key] = method_analysis;

    if (is_external)
        std::get<ExternalClass*>(class_def)->add_external_method(std::get<ExternalMethod*>(method_analysis->get_encoded_method()));
}


std::vector<DVMClass*>& ClassAnalysis::implements()
{
    if (is_external)
        throw exceptions::AnalysisException("implements: external class is not supported for implemented interfaces");

    return std::get<ClassDef*>(class_def)->get_interfaces();
}

MethodAnalysis * ClassAnalysis::get_method_analysis(std::variant<EncodedMethod *, ExternalMethod *> method)
{
    std::string method_key;

    if (method.index() == 0)
        method_key = std::get<EncodedMethod*>(method)->getMethodID()->pretty_method();
    else
        method_key = std::get<ExternalMethod*>(method)->pretty_method_name();
    
    if (methods.find(method_key) == methods.end())
        return nullptr;

    return methods[method_key];
}

FieldAnalysis *ClassAnalysis::get_field_analysis(EncodedField *field)
{
    if (fields.find(field) == fields.end())
        return nullptr;

    return fields[field].get();
}

