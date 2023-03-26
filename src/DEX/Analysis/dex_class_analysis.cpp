#include "KUNAI/DEX/Analysis/dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {

        ClassAnalysis::ClassAnalysis(std::variant<ClassDef *, ExternalClass *> class_def) : class_def(class_def)
        {
            is_external = class_def.index() == 0 ? false : true;
        }

        void ClassAnalysis::add_method(MethodAnalysis *method_analysis)
        {
            std::string method_key;

            if (method_analysis->external())
                method_key = std::get<ExternalMethod *>(method_analysis->get_method())->full_name();
            else
                method_key = std::get<EncodedMethod *>(method_analysis->get_method())->full_name();

            methods[method_key] = method_analysis;

            if (is_external)
                std::get<ExternalClass *>(class_def)->add_method(std::get<ExternalMethod *>(method_analysis->get_method()));
        }

        std::vector<Class *> ClassAnalysis::implements()
        {
            std::vector<Class *> implemented_interfaces;

            if (is_external)
                return implemented_interfaces;

            auto class_def = std::get<ClassDef *>(this->class_def);

            for (size_t i = 0, n_of_interfaces = class_def->get_number_of_interfaces(); i < n_of_interfaces; i++)
                implemented_interfaces.push_back(class_def->get_interface_by_pos(i));

            return implemented_interfaces;
        }

        std::string ClassAnalysis::extends()
        {
            if (is_external)
                return "Ljava/lang/Object;";

            return std::get<ClassDef *>(this->class_def)->get_superclass_idx()->get_name();
        }

        std::string ClassAnalysis::name()
        {
            if (is_external)
            {
                return std::get<ExternalClass *>(this->class_def)->get_name();
            }
            else
            {
                return std::get<ClassDef *>(this->class_def)->get_class_idx()->get_name();
            }
        }

        bool ClassAnalysis::is_android_api()
        {
            if (!is_external)
                return false;

            std::string const class_name = this->name();

            for (auto known_api : known_apis)
            {
                if (class_name.find(known_api) == 0)
                    return true;
            }

            return false;
        }

        std::vector<MethodAnalysis *> ClassAnalysis::get_methods()
        {
            std::vector<MethodAnalysis *> methods_list;

            for (auto it = methods.begin(); it != methods.end(); it++)
            {
                methods_list.push_back(it->second);
            }

            return methods_list;
        }

        std::vector<FieldAnalysis *> ClassAnalysis::get_fields()
        {
            std::vector<FieldAnalysis *> field_list;

            for (auto it = fields.begin(); it != fields.end(); it++)
            {
                field_list.push_back(it->second.get());
            }

            return field_list;
        }

        MethodAnalysis *ClassAnalysis::get_method_analysis(std::variant<EncodedMethod *, ExternalMethod *> method)
        {
            std::string method_key;

            if (method.index() == 0)
                method_key = std::get<EncodedMethod *>(method)->full_name();
            else
                method_key = std::get<ExternalMethod *>(method)->full_name();

            if (methods.find(method_key) != methods.end())
            {
                return methods[method_key];
            }
            else
            {
                return nullptr;
            }
        }

        FieldAnalysis *ClassAnalysis::get_field_analysis(EncodedField *field)
        {
            if (fields.find(field) != fields.end())
            {
                return fields[field].get();
            }
            else
            {
                return nullptr;
            }
        }

        void ClassAnalysis::add_field_xref_read(MethodAnalysis *method,
                                                ClassAnalysis *classobj,
                                                EncodedField *field,
                                                std::uint64_t off)
        {
            if (fields.find(field) == fields.end())
                fields[field] = std::make_unique<FieldAnalysis>(field);
            fields[field]->add_xref_read(classobj, method, off);
        }

        void ClassAnalysis::add_field_xref_write(MethodAnalysis *method,
                                                 ClassAnalysis *classobj,
                                                 EncodedField *field,
                                                 std::uint64_t off)
        {
            if (fields.find(field) == fields.end())
                fields[field] = std::make_unique<FieldAnalysis>(field);
            fields[field]->add_xref_write(classobj, method, off);
        }

        void ClassAnalysis::add_method_xref_to(MethodAnalysis *method1,
                                               ClassAnalysis *classobj,
                                               MethodAnalysis *method2,
                                               std::uint64_t off)
        {
            std::string method_key;

            if (method1->external())
                method_key = std::get<ExternalMethod*>(method1->get_method())->full_name();
            else
                method_key = std::get<EncodedMethod*>(method1->get_method())->full_name();

            if (methods.find(method_key) == methods.end())
                this->add_method(method1);
            methods[method_key]->add_xref_to(classobj, method2, off);
        }

        void ClassAnalysis::add_method_xref_from(MethodAnalysis *method1,
                                                 ClassAnalysis *classobj,
                                                 MethodAnalysis *method2,
                                                 std::uint64_t off)
        {
            std::string method_key;

            if (method1->external())
                method_key = std::get<ExternalMethod*>(method1->get_method())->full_name();
            else
                method_key = std::get<EncodedMethod*>(method1->get_method())->full_name();

            if (methods.find(method_key) == methods.end())
                this->add_method(method1);
            methods[method_key]->add_xref_from(classobj, method2, off);
        }

        void ClassAnalysis::add_xref_to(DVMTypes::REF_TYPE ref_kind,
                                        ClassAnalysis *classobj,
                                        MethodAnalysis *methodobj,
                                        std::uint64_t offset)
        {
            xrefto[classobj].insert({ref_kind, methodobj, offset});
        }

        void ClassAnalysis::add_xref_from(DVMTypes::REF_TYPE ref_kind,
                                          ClassAnalysis *classobj,
                                          MethodAnalysis *methodobj,
                                          std::uint64_t offset)
        {
            xreffrom[classobj].insert({ref_kind, methodobj, offset});
        }

        void ClassAnalysis::add_xref_new_instance(MethodAnalysis* methodobj, std::uint64_t offset)
        {
            xrefnewinstance.insert({methodobj, offset});
        }

        void ClassAnalysis::add_xref_const_class(MethodAnalysis* methodobj, std::uint64_t offset)
        {
            xrefconstclass.insert({methodobj, offset});
        }

    }
}