#include "dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {
        /**
         * @brief constructor of ClassAnalysis class.
         * @param class_def: std::any that can be a shared_ptr of ClassDef or of ExternalClass.
         * @return void
         */
        ClassAnalysis::ClassAnalysis(std::any class_def)
        {
            if (class_def.type() == typeid(std::shared_ptr<ClassDef>))
                this->is_external = false;
            else
                this->is_external = true;

            this->class_def = class_def;
        }

        /**
         * @brief destructor of ClassAnalysis class.
         * @return void.
         */
        ClassAnalysis::~ClassAnalysis() {}

        /**
         * @brief return if the analyzed class is external or not.
         * @return bool
         */
        bool ClassAnalysis::is_class_external()
        {
            return is_external;
        }

        /**
         * @brief add a method to the class, in case of being external
         *        add the external method object to the class.
         * @param method_analysis: std::shared_ptr<MethodAnalysis> object part of the class.
         * @return void.
         */
        void ClassAnalysis::add_method(std::shared_ptr<MethodAnalysis> method_analysis)
        {
            std::string method_key;

            if (method_analysis->external())
                method_key = std::any_cast<std::shared_ptr<ExternalMethod>>(method_analysis->get_method())->full_name();
            else
                method_key = std::any_cast<std::shared_ptr<EncodedMethod>>(method_analysis->get_method())->full_name();

            methods[method_key] = method_analysis;

            if (is_external)
                std::any_cast<std::shared_ptr<ExternalClass>>(class_def)->add_method(std::any_cast<std::shared_ptr<ExternalMethod>>(method_analysis->get_method()));
        }

        /**
         * @brief get a list of interfaces that the class implements, in the case of ExternalClass, none.
         * @return std::vector<Class *>
         */
        std::vector<Class *> ClassAnalysis::implements()
        {
            std::vector<Class *> implemented_interfaces;

            if (is_external)
                return implemented_interfaces;

            auto class_def = std::any_cast<std::shared_ptr<ClassDef>>(this->class_def);

            for (size_t i = 0; class_def->get_number_of_interfaces(); i++)
                implemented_interfaces.push_back(class_def->get_interface_by_pos(i));

            return implemented_interfaces;
        }

        /**
         * @brief get the name of the class from which current class extends.
         * @return std::string
         */
        std::string ClassAnalysis::extends()
        {
            if (is_external)
                return "Ljava/lang/Object;";

            auto class_def = std::any_cast<std::shared_ptr<ClassDef>>(this->class_def);

            return class_def->get_superclass_idx()->get_name();
        }

        /**
         * @brief get the name of the class.
         * @return std::string
         */
        std::string ClassAnalysis::name()
        {
            if (is_external)
            {
                auto class_def = std::any_cast<std::shared_ptr<ExternalClass>>(this->class_def);
                return class_def->get_name();
            }
            else
            {
                auto class_def = std::any_cast<std::shared_ptr<ClassDef>>(this->class_def);
                return class_def->get_class_idx()->get_name();
            }
        }

        /**
         * @brief check if the current class is an Android API.
         * @return bool
         */
        bool ClassAnalysis::is_android_api()
        {
            if (!is_external)
                return false;

            std::string class_name = this->name();

            for (auto it = known_apis.begin(); it != known_apis.end(); it++)
            {
                if (class_name.find(*it) == 0)
                    return true;
            }

            return false;
        }

        /**
         * @brief return a vector with all the MethodAnalysis objects.
         * @return std::vector<std::shared_ptr<MethodAnalysis>>
         */
        std::vector<std::shared_ptr<MethodAnalysis>> ClassAnalysis::get_methods()
        {
            std::vector<std::shared_ptr<MethodAnalysis>> methods_list;

            for (auto it = methods.begin(); it != methods.end(); it++)
            {
                methods_list.push_back(it->second);
            }

            return methods_list;
        }

        /**
         * @brief return a vector with all the FieldAnalysis objects.
         * @return std::vector<std::shared_ptr<FieldAnalysis>>
         */
        std::vector<std::shared_ptr<FieldAnalysis>> ClassAnalysis::get_fields()
        {
            std::vector<std::shared_ptr<FieldAnalysis>> field_list;

            for (auto it = fields.begin(); it != fields.end(); it++)
            {
                field_list.push_back(it->second);
            }

            return field_list;
        }

        /**
         * @brief Get number of stored methods.
         * @return size_t
         */
        size_t ClassAnalysis::get_nb_methods()
        {
            return methods.size();
        }

        /**
         * @brief Get one of the MethodAnalysis object by given EncodedMethod or ExternalMethod.
         * @param method: std::any (`std::shared_ptr<EncodedMethod>' or `std::shared_ptr<ExternalMethod>') to look for.
         * @return std::shared_ptr<MethodAnalysis>
         */
        std::shared_ptr<MethodAnalysis> ClassAnalysis::get_method_analysis(std::any method)
        {
            std::string method_key;

            if (method.type() == typeid(std::shared_ptr<EncodedMethod>))
                method_key = std::any_cast<std::shared_ptr<EncodedMethod>>(method)->full_name();
            else
                method_key = std::any_cast<std::shared_ptr<ExternalMethod>>(method)->full_name();

            if (methods.find(method_key) != methods.end())
            {
                return methods[method_key];
            }
            else
            {
                return nullptr;
            }
        }

        /**
         * @brief Get one of the FieldAnalysis object by given std::shared_ptr<EncodedField>
         * @param field: std::shared_ptr<EncodedField> object to look for.
         * @return std::shared_ptr<FieldAnalysis>
         */
        std::shared_ptr<FieldAnalysis> ClassAnalysis::get_field_analysis(std::shared_ptr<EncodedField> field)
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

        /**
         * @brief Create a std::shared_ptr<FieldAnalysis> object and add a read xref.
         * @param method: std::shared_ptr<MethodAnalysis> to add in xref.
         * @param classobj: std::shared_ptr<ClassAnalysis> to add in xref.
         * @param field: std::shared_ptr<EncodedField> object to create FieldAnalysis object.
         * @param off: std::uint64_t to add in xref.
         * @return void
         */
        void ClassAnalysis::add_field_xref_read(std::shared_ptr<MethodAnalysis> method,
                                                std::shared_ptr<ClassAnalysis> classobj,
                                                std::shared_ptr<EncodedField> field,
                                                std::uint64_t off)
        {
            if (fields.find(field) == fields.end())
                fields[field] = std::make_shared<FieldAnalysis>(field);
            fields[field]->add_xref_read(classobj, method, off);
        }

        /**
         * @brief Create a std::shared_ptr<FieldAnalysis> object and add a write xref.
         * @param method: std::shared_ptr<MethodAnalysis> to add in xref.
         * @param classobj: std::shared_ptr<ClassAnalysis> to add in xref.
         * @param field: std::shared_ptr<EncodedField> object to create FieldAnalysis object.
         * @param off: std::uint64_t to add in xref.
         * @return void
         */
        void ClassAnalysis::add_field_xref_write(std::shared_ptr<MethodAnalysis> method,
                                                 std::shared_ptr<ClassAnalysis> classobj,
                                                 std::shared_ptr<EncodedField> field,
                                                 std::uint64_t off)
        {
            if (fields.find(field) == fields.end())
                fields[field] = std::make_shared<FieldAnalysis>(field);
            fields[field]->add_xref_write(classobj, method, off);
        }

        /**
         * @brief Create a std::shared<MethodAnalysis> object and add a to xref.
         * @param method1: std::shared_ptr<MethodAnalysis> method to add to class and add xref to.
         * @param classobj: std::shared_ptr<ClassAnalysis> class to add to xref.
         * @param method2: std::shared_ptr<MethodAnalysis> method to add to xref.
         * @param off: std::uint64_t offset to add to xref.
         * @return void
         */
        void ClassAnalysis::add_method_xref_to(std::shared_ptr<MethodAnalysis> method1,
                                               std::shared_ptr<ClassAnalysis> classobj,
                                               std::shared_ptr<MethodAnalysis> method2,
                                               std::uint64_t off)
        {
            std::string method_key;

            if (method1->external())
                method_key = std::any_cast<std::shared_ptr<ExternalMethod>>(method1->get_method())->full_name();
            else
                method_key = std::any_cast<std::shared_ptr<EncodedMethod>>(method1->get_method())->full_name();

            if (methods.find(method_key) == methods.end())
                this->add_method(method1);
            methods[method_key]->add_xref_to(classobj, method2, off);
        }

        /**
         * @brief Create a std::shared<MethodAnalysis> object and add a from xref.
         * @param method1: std::shared_ptr<MethodAnalysis> method to add to class and add xref from.
         * @param classobj: std::shared_ptr<ClassAnalysis> class to add to xref.
         * @param method2: std::shared_ptr<MethodAnalysis> method to add to xref.
         * @param off: std::uint64_t offset to add to xref.
         * @return void
         */
        void ClassAnalysis::add_method_xref_from(std::shared_ptr<MethodAnalysis> method1,
                                                 std::shared_ptr<ClassAnalysis> classobj,
                                                 std::shared_ptr<MethodAnalysis> method2,
                                                 std::uint64_t off)
        {
            std::string method_key;

            if (method1->external())
                method_key = std::any_cast<std::shared_ptr<ExternalMethod>>(method1->get_method())->full_name();
            else
                method_key = std::any_cast<std::shared_ptr<EncodedMethod>>(method1->get_method())->full_name();

            if (methods.find(method_key) == methods.end())
                this->add_method(method1);
            methods[method_key]->add_xref_from(classobj, method2, off);
        }

        /**
         * @brief Create a cross reference to another class.
         *        XrefTo means, that the current class calls another class.
         *        The current class should also be contained in another class' XrefFrom list.     
         * @param ref_kind: DVMTypes::REF_TYPE type of call done to the other class.
         * @param classobj: std::shared_ptr<ClassAnalysis>
         * @param methodobj: std::shared_ptr<MethodAnalysis> methods from which other class is called.
         * @param offset: std::uint64_t offset where the call is done.
         */
        void ClassAnalysis::add_xref_to(DVMTypes::REF_TYPE ref_kind,
                                        std::shared_ptr<ClassAnalysis> classobj,
                                        std::shared_ptr<MethodAnalysis> methodobj,
                                        std::uint64_t offset)
        {
            xrefto[classobj].insert({ref_kind, methodobj, offset});
        }

        /**
         * @brief Create a cross reference from this class.
         *        XrefFrom means, that current class is called by another class.
         * @param ref_kind: DVMTypes::REF_TYPE type of call done by another class.
         * @param classobj: std::shared_ptr<ClassAnalysis> class called.
         * @param methodobj: std::shared_ptr<MethodAnalysis> method from class that calls the other.
         * @param offset: std::uint64_t offset where the call is done.
         */
        void ClassAnalysis::add_xref_from(DVMTypes::REF_TYPE ref_kind,
                                          std::shared_ptr<ClassAnalysis> classobj,
                                          std::shared_ptr<MethodAnalysis> methodobj,
                                          std::uint64_t offset)
        {
            xreffrom[classobj].insert({ref_kind, methodobj, offset});
        }

        /**
         * @brief return all the references to other classes called by this class.
         * @return std::map<std::shared_ptr<ClassAnalysis>, std::set<std::tuple<DVMTypes::REF_TYPE, std::shared_ptr<MethodAnalysis>, std::uint64_t>>>
         */
        std::map<std::shared_ptr<ClassAnalysis>, std::set<std::tuple<DVMTypes::REF_TYPE, std::shared_ptr<MethodAnalysis>, std::uint64_t>>>
        ClassAnalysis::get_xref_to()
        {
            return xrefto;
        }

        /**
         * @brief return all the classes that call this class.
         * @return std::map<std::shared_ptr<ClassAnalysis>, std::set<std::tuple<DVMTypes::REF_TYPE, std::shared_ptr<MethodAnalysis>, std::uint64_t>>>
         */
        std::map<std::shared_ptr<ClassAnalysis>, std::set<std::tuple<DVMTypes::REF_TYPE, std::shared_ptr<MethodAnalysis>, std::uint64_t>>>
        ClassAnalysis::get_xref_from()
        {
            return xreffrom;
        }

        /**
         * @brief Add a new reference to a method where the class is instantiated.
         * @param methodobj: std::shared_ptr<MethodAnalysis> method where class is instantiated.
         * @param offset: offset where class is instantiated.
         * @return void
         */
        void ClassAnalysis::add_xref_new_instance(std::shared_ptr<MethodAnalysis> methodobj, std::uint64_t offset)
        {
            xrefnewinstance.insert({methodobj, offset});
        }

        /**
         * @brief Return all the references where the call is instantiated.
         * @return std::set<std::tuple<std::shared_ptr<MethodAnalysis>, std::uint64_t>>
         */
        std::set<std::tuple<std::shared_ptr<MethodAnalysis>, std::uint64_t>> ClassAnalysis::get_xref_new_instance()
        {
            return xrefnewinstance;
        }

        /**
         * @brief Add a crossreference to a method referencing this classtype.
         * @param methodobj: std::shared_ptr<MethodAnalysis> method where class is referenced.
         * @param offset: offset where class is referenced.
         * @return void
         */
        void ClassAnalysis::add_xref_const_class(std::shared_ptr<MethodAnalysis> methodobj, std::uint64_t offset)
        {
            xrefconstclass.insert({methodobj, offset});
        }

        /**
         * @brief Return all the methods where this class is referenced.
         * @return std::set<std::tuple<std::shared_ptr<MethodAnalysis>, std::uint64_t>>
         */
        std::set<std::tuple<std::shared_ptr<MethodAnalysis>, std::uint64_t>> ClassAnalysis::get_xref_const_class()
        {
            return xrefconstclass;
        }

    }
}