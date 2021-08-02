/***
 * @file dex_analysis.hpp
 * @author @Farenain
 * 
 * @brief File for analysis classes so we avoid
 *        interdependencies, here we will have
 *        ClassAnalysis, MethodAnalysis, FileAnalysis,
 *        etc.
 * 
 * Based on Analysis classes from Androguard.
 */

#ifndef DEX_ANALYSIS_HPP
#define DEX_ANALYSIS_HPP

#include <any>
#include <iostream>
#include <vector>
#include <tuple>
#include <set>
#include <algorithm>
#include <regex>

#include "dex_classes.hpp"
#include "dex_external_classes.hpp"
#include "dex_encoded.hpp"
#include "dex_external_methods.hpp"
#include "dex_fields.hpp"
#include "dex_instructions.hpp"

namespace KUNAI
{
    namespace DEX
    {
        class StringAnalysis;
        class BasicBlocks;
        class DVMBasicBlock;
        class MethodAnalysis;
        class ExceptionAnalysis;
        class Exception;
        class FieldAnalysis;
        class ClassAnalysis;
        class Analysis;

        /**
         * Analysis definition
         * 
         * @brief Class used to manage all the information from the analysis
         *        it contains the xrefs between Classes, Methods, Fields and
         *        Strings. Also it creates the BasicBlocks from the code.
         *        XREFs are created for:
         *        * classes (ClassAnalysis object)
         *        * methods (MethodAnalysis object)
         *        * strings (StringAnalysis object)
         *        * fields (FieldAnalysis object)         
         * 
         */
        class Analysis
        {
        public:
            Analysis(std::shared_ptr<DexParser> dex_parser, std::shared_ptr<DalvikOpcodes> dalvik_opcodes, 
                std::map<std::tuple<std::shared_ptr<ClassDef>, std::shared_ptr<EncodedMethod>>, std::map<std::uint64_t, std::shared_ptr<Instruction>>> instructions);
            ~Analysis();

            void add(std::shared_ptr<DexParser> dex_parser);
            void create_xref();

            // ClassAnalysis methods
            bool is_class_present(std::string class_name);
            std::shared_ptr<ClassAnalysis> get_class_analysis(std::string class_name);
            std::vector<std::shared_ptr<ClassAnalysis>> get_classes();
            std::vector<std::shared_ptr<ClassAnalysis>> get_external_classes();
            std::vector<std::shared_ptr<ClassAnalysis>> get_internal_classes();

            // MethodAnalysis methods
            std::shared_ptr<MethodAnalysis> get_method(std::any method);
            MethodID *get_method_by_name(std::string class_name, std::string method_name, std::string method_descriptor);
            std::shared_ptr<MethodAnalysis> get_method_analysis_by_name(std::string class_name, std::string method_name, std::string method_descriptor);
            std::vector<std::shared_ptr<MethodAnalysis>> get_methods();

            // FieldAnalysis methods
            std::shared_ptr<FieldAnalysis> get_field_analysis(std::shared_ptr<EncodedField> field);
            std::vector<std::shared_ptr<FieldAnalysis>> get_fields();

            // StringAnalysis methods
            std::map<std::string, std::shared_ptr<StringAnalysis>> get_strings_analysis();
            std::vector<std::shared_ptr<StringAnalysis>> get_strings();

            // class analysis by regular expression
            std::vector<std::shared_ptr<ClassAnalysis>> find_classes(std::string name, bool no_external);
            std::vector<std::shared_ptr<MethodAnalysis>> find_methods(std::string class_name, std::string method_name, std::string descriptor, std::string accessflags, bool no_external);
            std::vector<std::shared_ptr<StringAnalysis>> find_strings(std::string string);
            std::vector<std::shared_ptr<FieldAnalysis>> find_fields(std::string class_name, std::string field_name, std::string field_type, std::string accessflags);

        private:
            void _create_xref(std::shared_ptr<KUNAI::DEX::ClassDef> current_class);
            std::shared_ptr<MethodAnalysis> _resolve_method(std::string class_name, std::string method_name, std::string method_descriptor);

            std::vector<std::shared_ptr<DexParser>> dex_parsers;

            /**
             * @brief map with std::string of class name
             * as key and std::shared_ptr<ClassAnalysis> as value.
             */
            std::map<std::string, std::shared_ptr<ClassAnalysis>> classes;

            /**
             * @brief map with std::string as key and std::shared_ptr<StringAnalysis>
             * as value
             */
            std::map<std::string, std::shared_ptr<StringAnalysis>> strings;

            /**
             * @brief map with std::shared_ptr<EncodedMethod> or std::shared_ptr<ExternalMethod>
             * as keys and std::shared_ptr<MethodAnalysis> as value.
             */
            std::map<std::string, std::shared_ptr<MethodAnalysis>> methods;

            /**
             * @brief map hash for quickly getting the Method
             */
            std::map<std::tuple<std::string, std::string, std::string>, std::shared_ptr<MethodAnalysis>> method_hashes;

            std::shared_ptr<DalvikOpcodes> dalvik_opcodes;
            std::map<std::tuple<std::shared_ptr<ClassDef>, std::shared_ptr<EncodedMethod>>,
                     std::map<std::uint64_t, std::shared_ptr<Instruction>>> instructions;

            // check if xrefs were already created or not
            bool created_xrefs;
        };

        /**
         * ClassAnalysis definition
         */
        class ClassAnalysis
        {
        public:
            // any must be restricted to
            // ClassDef or ExternalClass
            ClassAnalysis(std::any class_def);

            ~ClassAnalysis();

            std::any get_class_definition();
            bool is_class_external();

            void add_method(std::shared_ptr<MethodAnalysis> method_analysis);

            std::vector<Class *> implements();
            std::string extends();
            std::string name();

            bool is_android_api();
            std::vector<std::shared_ptr<MethodAnalysis>> get_methods();
            std::vector<std::shared_ptr<FieldAnalysis>> get_fields();
            size_t get_nb_methods();
            std::shared_ptr<MethodAnalysis> get_method_analysis(std::any method);
            std::shared_ptr<FieldAnalysis> get_field_analysis(std::shared_ptr<EncodedField> field);

            void add_field_xref_read(std::shared_ptr<MethodAnalysis> method,
                                     std::shared_ptr<ClassAnalysis> classobj,
                                     std::shared_ptr<EncodedField> field,
                                     std::uint64_t off);
            void add_field_xref_write(std::shared_ptr<MethodAnalysis> method,
                                      std::shared_ptr<ClassAnalysis> classobj,
                                      std::shared_ptr<EncodedField> field,
                                      std::uint64_t off);

            void add_method_xref_to(std::shared_ptr<MethodAnalysis> method1,
                                    std::shared_ptr<ClassAnalysis> classobj,
                                    std::shared_ptr<MethodAnalysis> method2,
                                    std::uint64_t off);
            void add_method_xref_from(std::shared_ptr<MethodAnalysis> method1,
                                      std::shared_ptr<ClassAnalysis> classobj,
                                      std::shared_ptr<MethodAnalysis> method2,
                                      std::uint64_t off);

            void add_xref_to(DVMTypes::REF_TYPE ref_kind,
                             std::shared_ptr<ClassAnalysis> classobj,
                             std::shared_ptr<MethodAnalysis> methodobj,
                             std::uint64_t offset);
            void add_xref_from(DVMTypes::REF_TYPE ref_kind,
                               std::shared_ptr<ClassAnalysis> classobj,
                               std::shared_ptr<MethodAnalysis> methodobj,
                               std::uint64_t offset);

            std::map<std::shared_ptr<ClassAnalysis>,
                     std::set<std::tuple<DVMTypes::REF_TYPE,
                                         std::shared_ptr<MethodAnalysis>,
                                         std::uint64_t>>>
            get_xref_to();
            std::map<std::shared_ptr<ClassAnalysis>,
                     std::set<std::tuple<DVMTypes::REF_TYPE,
                                         std::shared_ptr<MethodAnalysis>,
                                         std::uint64_t>>>
            get_xref_from();

            void add_xref_new_instance(std::shared_ptr<MethodAnalysis> methodobj, std::uint64_t offset);
            std::set<std::tuple<std::shared_ptr<MethodAnalysis>, std::uint64_t>> get_xref_new_instance();
            void add_xref_const_class(std::shared_ptr<MethodAnalysis> methodobj, std::uint64_t offset);
            std::set<std::tuple<std::shared_ptr<MethodAnalysis>, std::uint64_t>> get_xref_const_class();

        private:
            // ClassDef or ExternalClass object
            std::any class_def;

            bool is_external;

            // map with method analysis for DEX analysis
            std::map<std::string, std::shared_ptr<MethodAnalysis>> methods;
            std::map<std::shared_ptr<EncodedField>, std::shared_ptr<FieldAnalysis>> fields;

            std::map<std::shared_ptr<ClassAnalysis>, std::set<std::tuple<DVMTypes::REF_TYPE, std::shared_ptr<MethodAnalysis>, std::uint64_t>>> xrefto;
            std::map<std::shared_ptr<ClassAnalysis>, std::set<std::tuple<DVMTypes::REF_TYPE, std::shared_ptr<MethodAnalysis>, std::uint64_t>>> xreffrom;

            std::set<std::tuple<std::shared_ptr<MethodAnalysis>, std::uint64_t>> xrefnewinstance;
            std::set<std::tuple<std::shared_ptr<MethodAnalysis>, std::uint64_t>> xrefconstclass;

            std::vector<std::string> known_apis{
                "Landroid/", "Lcom/android/internal/util", "Ldalvik/", "Ljava/", "Ljavax/", "Lorg/apache/",
                "Lorg/json/", "Lorg/w3c/dom/", "Lorg/xml/sax", "Lorg/xmlpull/v1/", "Ljunit/", "Landroidx/"};
        };

        /**
         * DVMBasicBlock definition
         */
        class DVMBasicBlock : std::enable_shared_from_this<DVMBasicBlock>
        {
        public:
            DVMBasicBlock(std::uint64_t start,
                          std::shared_ptr<DalvikOpcodes> dalvik_opcodes,
                          std::shared_ptr<BasicBlocks> context,
                          std::shared_ptr<EncodedMethod> method,
                          std::map<std::uint64_t, std::shared_ptr<Instruction>> instructions);

            std::uint64_t get_start();
            std::uint64_t get_end();

            std::vector<std::shared_ptr<Instruction>> get_instructions();
            std::shared_ptr<Instruction> get_last();
            std::vector<std::tuple<std::uint64_t, std::uint64_t, std::shared_ptr<DVMBasicBlock>>> get_next();
            std::vector<std::tuple<std::uint64_t, std::uint64_t, std::shared_ptr<DVMBasicBlock>>> get_prev();

            void set_parent(std::tuple<std::uint64_t, std::uint64_t, std::shared_ptr<DVMBasicBlock>> bb);
            void set_child();
            void set_child(std::vector<int64_t> values);
            std::uint64_t get_last_length();
            std::uint64_t get_nb_instructions();
            void push(std::shared_ptr<Instruction> instr);

            std::shared_ptr<Instruction> get_special_instruction(std::uint64_t idx);

            std::shared_ptr<ExceptionAnalysis> get_exception_analysis();
            void set_exception_analysis(std::shared_ptr<ExceptionAnalysis> exception_analysis);

        private:
            std::uint64_t start, end; // first and final idx from the basic block
            std::shared_ptr<DalvikOpcodes> dalvik_opcodes;
            std::shared_ptr<BasicBlocks> context;
            std::shared_ptr<EncodedMethod> method;
            std::map<std::uint64_t, std::shared_ptr<Instruction>> instructions;
            std::map<std::uint64_t, std::shared_ptr<Instruction>> special_instructions;
            std::uint64_t last_length;
            std::uint64_t nb_instructions;
            std::string name;
            std::shared_ptr<ExceptionAnalysis> exception_analysis;

            std::vector<std::tuple<std::uint64_t, std::uint64_t, std::shared_ptr<DVMBasicBlock>>> parents;
            std::vector<std::tuple<std::uint64_t, std::uint64_t, std::shared_ptr<DVMBasicBlock>>> childs;
        };

        /**
         * BasicBlocks definition
         */
        class BasicBlocks
        {
        public:
            BasicBlocks();
            ~BasicBlocks();

            void push_basic_block(std::shared_ptr<DVMBasicBlock> basic_block);
            std::shared_ptr<DVMBasicBlock> pop_basic_block();
            std::shared_ptr<DVMBasicBlock> get_basic_block_by_idx(std::uint64_t idx);
            size_t get_number_of_basic_blocks();
            std::vector<std::shared_ptr<DVMBasicBlock>> get_basic_blocks();

        private:
            std::vector<std::shared_ptr<DVMBasicBlock>> basic_blocks;
        };

        /**
         * MethodAnalysis definition
         */
        class MethodAnalysis
        {
        public:
            MethodAnalysis(std::any method_encoded, std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::map<std::uint64_t, std::shared_ptr<Instruction>> instructions);
            ~MethodAnalysis();

            bool external();
            bool is_android_api();

            std::any get_method();
            std::string name();
            std::string descriptor();
            std::string access();
            std::string class_name();
            std::string full_name();

            void add_xref_read(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<FieldAnalysis> field_object, std::uint64_t offset);
            void add_xref_write(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<FieldAnalysis> field_object, std::uint64_t offset);
            std::any get_xref_read(bool withoffset);
            std::any get_xref_write(bool withoffset);

            void add_xref_to(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset);
            void add_xref_from(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset);
            std::any get_xref_to(bool withoffset);
            std::any get_xref_from(bool withoffset);

            void add_xref_new_instance(std::shared_ptr<ClassAnalysis> class_object, std::uint64_t offset);
            void add_xref_const_class(std::shared_ptr<ClassAnalysis> class_object, std::uint64_t offset);
            std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::uint64_t>> get_xref_new_instance();
            std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::uint64_t>> get_xref_const_class();

            std::map<std::uint64_t, std::shared_ptr<Instruction>> get_instructions();
            std::shared_ptr<BasicBlocks> get_basic_blocks();
            std::shared_ptr<Exception> get_exceptions();

        private:
            bool is_external;
            std::any method_encoded;
            std::shared_ptr<DalvikOpcodes> dalvik_opcodes;
            std::map<std::uint64_t, std::shared_ptr<Instruction>> instructions;
            std::shared_ptr<BasicBlocks> basic_blocks;
            std::shared_ptr<Exception> exceptions;

            void create_basic_block();

            std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<FieldAnalysis>, std::uint64_t>> xrefread;
            std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<FieldAnalysis>, std::uint64_t>> xrefwrite;

            std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>> xrefto;
            std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>> xreffrom;

            std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::uint64_t>> xrefnewinstance;
            std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::uint64_t>> xrefconstclass;

            std::vector<std::string> known_apis{
                "Landroid/", "Lcom/android/internal/util", "Ldalvik/", "Ljava/", "Ljavax/", "Lorg/apache/",
                "Lorg/json/", "Lorg/w3c/dom/", "Lorg/xml/sax", "Lorg/xmlpull/v1/", "Ljunit/", "Landroidx/"};
        };

        /**
         * FieldAnalysis definition
         */
        class FieldAnalysis
        {
        public:
            FieldAnalysis(std::shared_ptr<EncodedField> field);
            ~FieldAnalysis();

            std::string name();
            void add_xref_read(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset);
            void add_xref_write(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset);
            std::any get_xref_read(bool withoffset);
            std::any get_xref_write(bool withoffset);
            std::shared_ptr<EncodedField> get_field();

            friend std::ostream &operator<<(std::ostream &os, const FieldAnalysis &entry);

        private:
            std::shared_ptr<EncodedField> field;
            std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>> xrefread;
            std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>> xrefwrite;
        };

        /**
         * StringAnalysis definition
         */
        class StringAnalysis
        {
        public:
            StringAnalysis(std::string *value);
            ~StringAnalysis();

            void add_xref_from(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset);
            std::any get_xref_from(bool withoffset);

            void set_value(std::string *value);
            std::string *get_value();

            bool is_overwritten();

        private:
            std::string *value;
            std::string *orig_value;
            std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>> xreffrom;
        };

        /**
         * ExceptionAnalysis definition
         */
        class ExceptionAnalysis
        {
        public:
            ExceptionAnalysis(exceptions_data exception, std::shared_ptr<BasicBlocks> basic_blocks);
            ~ExceptionAnalysis();

            std::string show_buff();
            exceptions_data get();

        private:
            exceptions_data exception;
            std::shared_ptr<BasicBlocks> basic_blocks;
        };

        /**
         * Exception definition
         */
        class Exception
        {
        public:
            Exception();
            ~Exception();

            void add(std::vector<exceptions_data> exceptions, std::shared_ptr<BasicBlocks> basic_blocks);
            std::shared_ptr<ExceptionAnalysis> get_exception(std::uint64_t start_addr, std::uint64_t end_addr);
            std::vector<std::shared_ptr<ExceptionAnalysis>> gets();

        private:
            std::vector<std::shared_ptr<ExceptionAnalysis>> exceptions;
        };
    } // namespace DEX
} // namespace KUNAI

#endif