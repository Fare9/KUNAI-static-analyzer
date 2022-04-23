#include "dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {

        ClassAnalysis::ClassAnalysis(std::variant<classdef_t, externalclass_t> class_def) : class_def(class_def)
        {
            is_external = class_def.index() == 0 ? false : true;
        }

        void ClassAnalysis::add_method(methodanalysis_t method_analysis)
        {
            std::string method_key;

            if (method_analysis->external())
                method_key = std::get<externalmethod_t>(method_analysis->get_method())->full_name();
            else
                method_key = std::get<encodedmethod_t>(method_analysis->get_method())->full_name();

            methods[method_key] = method_analysis;

            if (is_external)
                std::get<externalclass_t>(class_def)->add_method(std::get<externalmethod_t>(method_analysis->get_method()));
        }

        std::vector<Class *> ClassAnalysis::implements()
        {
            std::vector<Class *> implemented_interfaces;

            if (is_external)
                return implemented_interfaces;

            auto class_def = std::get<classdef_t>(this->class_def);

            for (size_t i = 0, n_of_interfaces = class_def->get_number_of_interfaces(); i < n_of_interfaces ; i++)
                implemented_interfaces.push_back(class_def->get_interface_by_pos(i));

            return implemented_interfaces;
        }

        std::string ClassAnalysis::extends()
        {
            if (is_external)
                return "Ljava/lang/Object;";

            return std::get<classdef_t>(this->class_def)->get_superclass_idx()->get_name();
        }

        std::string ClassAnalysis::name()
        {
            if (is_external)
            {
                return std::get<externalclass_t>(this->class_def)->get_name();
            }
            else
            {
                return std::get<classdef_t>(this->class_def)->get_class_idx()->get_name();
            }
        }

        bool ClassAnalysis::is_android_api()
        {
            if (!is_external)
                return false;

            std::string class_name = this->name();

            for (auto known_api : known_apis)
            {
                if (class_name.find(known_api) == 0)
                    return true;
            }

            return false;
        }

        std::vector<methodanalysis_t> ClassAnalysis::get_methods()
        {
            std::vector<methodanalysis_t> methods_list;

            for (auto it = methods.begin(); it != methods.end(); it++)
            {
                methods_list.push_back(it->second);
            }

            return methods_list;
        }

        std::vector<fieldanalysis_t> ClassAnalysis::get_fields()
        {
            std::vector<fieldanalysis_t> field_list;

            for (auto it = fields.begin(); it != fields.end(); it++)
            {
                field_list.push_back(it->second);
            }

            return field_list;
        }

        methodanalysis_t ClassAnalysis::get_method_analysis(std::variant<encodedmethod_t, externalmethod_t> method)
        {
            std::string method_key;

            if (method.index() == 0)
                method_key = std::get<encodedmethod_t>(method)->full_name();
            else
                method_key = std::get<externalmethod_t>(method)->full_name();

            if (methods.find(method_key) != methods.end())
            {
                return methods[method_key];
            }
            else
            {
                return nullptr;
            }
        }

        fieldanalysis_t ClassAnalysis::get_field_analysis(encodedfield_t field)
        {
            if (fields.find(field) != fields.end())
            {
                return fields[field];
            }
            else
            {
                return nullptr;
            }
        }

        void ClassAnalysis::add_field_xref_read(methodanalysis_t method,
                                                classanalysis_t classobj,
                                                encodedfield_t field,
                                                std::uint64_t off)
        {
            if (fields.find(field) == fields.end())
                fields[field] = std::make_shared<FieldAnalysis>(field);
            fields[field]->add_xref_read(classobj, method, off);
        }

        void ClassAnalysis::add_field_xref_write(methodanalysis_t method,
                                                 classanalysis_t classobj,
                                                 encodedfield_t field,
                                                 std::uint64_t off)
        {
            if (fields.find(field) == fields.end())
                fields[field] = std::make_shared<FieldAnalysis>(field);
            fields[field]->add_xref_write(classobj, method, off);
        }

        void ClassAnalysis::add_method_xref_to(methodanalysis_t method1,
                                               classanalysis_t classobj,
                                               methodanalysis_t method2,
                                               std::uint64_t off)
        {
            std::string method_key;

            if (method1->external())
                method_key = std::get<externalmethod_t>(method1->get_method())->full_name();
            else
                method_key = std::get<encodedmethod_t>(method1->get_method())->full_name();

            if (methods.find(method_key) == methods.end())
                this->add_method(method1);
            methods[method_key]->add_xref_to(classobj, method2, off);
        }

        void ClassAnalysis::add_method_xref_from(methodanalysis_t method1,
                                                 classanalysis_t classobj,
                                                 methodanalysis_t method2,
                                                 std::uint64_t off)
        {
            std::string method_key;

            if (method1->external())
                method_key = std::get<externalmethod_t>(method1->get_method())->full_name();
            else
                method_key = std::get<encodedmethod_t>(method1->get_method())->full_name();

            if (methods.find(method_key) == methods.end())
                this->add_method(method1);
            methods[method_key]->add_xref_from(classobj, method2, off);
        }

        void ClassAnalysis::add_xref_to(DVMTypes::REF_TYPE ref_kind,
                                        classanalysis_t classobj,
                                        methodanalysis_t methodobj,
                                        std::uint64_t offset)
        {
            xrefto[classobj].insert({ref_kind, methodobj, offset});
        }

        void ClassAnalysis::add_xref_from(DVMTypes::REF_TYPE ref_kind,
                                          classanalysis_t classobj,
                                          methodanalysis_t methodobj,
                                          std::uint64_t offset)
        {
            xreffrom[classobj].insert({ref_kind, methodobj, offset});
        }

        void ClassAnalysis::add_xref_new_instance(methodanalysis_t methodobj, std::uint64_t offset)
        {
            xrefnewinstance.insert({methodobj, offset});
        }

        void ClassAnalysis::add_xref_const_class(methodanalysis_t methodobj, std::uint64_t offset)
        {
            xrefconstclass.insert({methodobj, offset});
        }

    }
}