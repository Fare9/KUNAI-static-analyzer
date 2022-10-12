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
#include <variant>
#include <iostream>
#include <vector>
#include <tuple>
#include <set>
#include <algorithm>
#include <regex>

#include "KUNAI/DEX/parser/dex_classes.hpp"
#include "KUNAI/DEX/DVM/dex_external_classes.hpp"
#include "KUNAI/DEX/parser/dex_encoded.hpp"
#include "KUNAI/DEX/DVM/dex_external_methods.hpp"
#include "KUNAI/DEX/parser/dex_fields.hpp"
#include "KUNAI/DEX/DVM/dex_instructions.hpp"
#include "KUNAI/DEX/DVM/dex_disassembler.hpp"

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
        using analysis_t = std::shared_ptr<Analysis>;
        using classanalysis_t = std::shared_ptr<ClassAnalysis>;
        using dvmbasicblock_t = std::shared_ptr<DVMBasicBlock>;
        using basicblocks_t = std::shared_ptr<BasicBlocks>;
        using methodanalysis_t = std::shared_ptr<MethodAnalysis>;
        using fieldanalysis_t = std::shared_ptr<FieldAnalysis>;
        using stringanalysis_t = std::shared_ptr<StringAnalysis>;
        using exceptionanalysis_t = std::shared_ptr<ExceptionAnalysis>;
        using exception_t = std::shared_ptr<Exception>;

        using xref_method_idx = std::set<std::tuple<methodanalysis_t, std::uint64_t>>;
        using class_xref = std::map<classanalysis_t, std::set<std::tuple<DVMTypes::REF_TYPE, methodanalysis_t, std::uint64_t>>>;

        class Analysis
        {
        public:
            /**
             * @brief Analysis class constructor, just accept an initial DexParser object
             *        to start the analysis.
             * @param dex_parser: dexparser_t parser to initialize the analysis.
             * @param dalvik_opcodes: dalvikopcodes_t used to initialize some objects.
             * @param instructions: instruction_map_t instructions to initialize methods.
             * @param create_xrefs: create or not the xrefs for the analysis, not necessary of you do not want cross references, which takes long time.
             * @return void
             */
            Analysis(dexparser_t dex_parser, dalvikopcodes_t dalvik_opcodes,
                     instruction_map_t instructions, bool create_xrefs);

            /**
             * @brief Analysis class destructor, nothing really interesting in here.
             * @return void
             */
            ~Analysis() = default;

            /**
             * @brief Analysis method to add a new dex_parser we will have to analyze
             *        classes, methods, fields and strings from the parser as new objects
             *        will be added.
             * @param dex_parser: dexparser_t new parser object to add.
             * @return void
             */
            void add(dexparser_t dex_parser);

            /**
             * @brief Create Class, Method, String and Field cross references
             *        If you are using multiple DEX files, this function must
             *        be called when all DEX files are added.
             *        If you call the function after every DEX file, it will only
             *        work for the first time.
             *        So ADD ALL THE DEX FIRST.
             * @return void
             */
            void create_xref();

            // ClassAnalysis methods

            /**
             * @brief Check if a class is present in the Analysis object
             * @param class_name: std::string class name to search.
             * @return bool
             */
            bool is_class_present(std::string& class_name);

            /**
             * @brief get a ClassAnalysis object by a class_name.
             * @param class_name: std::string class name to retrieve its ClassAnalysis object.
             * @return classanalysis_t
             */
            classanalysis_t get_class_analysis(std::string& class_name);

            /**
             * @brief Get all the ClassAnalysis objects in a vector.
             * @return std::vector<classanalysis_t>
             */
            std::vector<classanalysis_t> get_classes();

            /**
             * @brief Get all the external classes in a vector.
             * @return std::vector<classanalysis_t>
             */
            std::vector<classanalysis_t> get_external_classes();

            /**
             * @brief Get all the internal classes in a vector.
             * @return std::vector<classanalysis_t>
             */
            std::vector<classanalysis_t> get_internal_classes();

            // MethodAnalysis methods

            /**
             * @brief Get MethodAnalysis object by giving an EncodedMethod or
             *        ExternalMethod object.
             * @param method: std::variant<encodedmethod_t, externalmethod_t>
             * @return methodanalysis_t
             */
            methodanalysis_t get_method(std::variant<encodedmethod_t, externalmethod_t> method);

            /**
             * @brief Get MethodID from internal methods by class name, method name and
             *        method descriptor.
             * @param class_name: std::string class name of the method.
             * @param method_name: std::string method name.
             * @param method_descriptor: std::string method descriptor (parameters and return value).
             * @return methodid_t
             */
            methodid_t get_method_by_name(std::string& class_name, std::string& method_name, std::string& method_descriptor);

            /**
             * @brief Get a method analysis given class name, method name and method descriptor.
             * @param class_name: std::string class name of the method.
             * @param method_name: std::string method name.
             * @param method_descriptor: std::string method descriptor (parameters and return value).
             * @return methodanalysis_t
             */
            methodanalysis_t get_method_analysis_by_name(std::string& class_name, std::string& method_name, std::string& method_descriptor);

            /**
             * @brief Get all the MethodAnalysis object in a vector.
             * @return std::vector<methodanalysis_t>
             */
            std::vector<methodanalysis_t> get_methods();

            /**
             * @brief Get the methods with hashes object.
             * 
             * @return const std::unordered_map<std::string, methodanalysis_t>& 
             */
            const std::unordered_map<std::string, methodanalysis_t>& get_methods_with_hashes() const
            {
                return method_hashes;
            }

            // FieldAnalysis methods

            /**
             * @brief Get a FieldAnalysis given a field object.
             * @param field: encodedfield_t field to look for its FieldAnalysis.
             * @return fieldanalysis_t
             */
            fieldanalysis_t get_field_analysis(encodedfield_t field);

            /**
             * @brief Get all the FieldAnalysis objects in a vector.
             * @return std::vector<fieldanalysis_t>
             */
            std::vector<fieldanalysis_t> get_fields();

            // StringAnalysis methods

            /**
             * @brief Get the map of std::string, StringAnalysis objects.
             * @return std::map<std::string, stringanalysis_t>
             */
            const std::map<std::string, stringanalysis_t> &get_strings_analysis() const
            {
                return strings;
            }

            /**
             * @brief Get a vector of all the StringAnalysis objects.
             * @return std::vector<stringanalysis_t>
             */
            std::vector<stringanalysis_t> get_strings();

            // class analysis by regular expression

            /**
             * @brief Find classes by name with regular expression,
             *        the method returns a list of ClassAnalysis object
             *        that match the regex.
             * @brief name: std::string name with the regex to search.
             * @brief no_external: bool want external classes?
             * @return std::vector<classanalysis_t>
             */
            std::vector<classanalysis_t> find_classes(std::string name, bool no_external);

            /**
             * @brief Find MethodAnalysis by name with regular expression,
             *        return a list of MethodAnalysis objects by different
             *        regular expressions. This time is necessary to specify
             *        more values specific from the method.
             * @param class_name: std::string name of the class where the method is.
             * @param method_name: std::string name of the method.
             * @param descriptor: std::string descriptor of method prototype.
             * @param accessflags: std::string access flags from the method.
             * @param no_external: bool want external classes?
             * @return std::vector<methodanalysis_t>
             */
            std::vector<methodanalysis_t> find_methods(std::string class_name, std::string method_name, std::string descriptor, std::string accessflags, bool no_external);

            /**
             * @brief Find StringAnalysis objects using regular expressions.
             * @param string: std::string regex to look for.
             * @return std::vector<stringanalysis_t>
             */
            std::vector<stringanalysis_t> find_strings(std::string string);

            /**
             * @brief Find FieldAnalysis objects using regular expression,
             *        find those in classes.
             * @param class_name: std::string class where the field is.
             * @param field_name: std::string name of the field.
             * @param field_type: std::string type of the field.
             * @param accessflags: accessflags from the field.
             * @return std::vector<fieldanalysis_t>
             */
            std::vector<fieldanalysis_t> find_fields(std::string class_name, std::string field_name, std::string field_type, std::string accessflags);

        private:
            /**
             * @brief Internal method for creating the xref for `current_class'. There are four steps
             *        involved in getting the xrefs:
             *          - xrefs for class instantiation and static class usage.
             *          - xrefs for method calls
             *          - xrefs for string usage
             *          - xrefs field manipulation
             *        All the information is stored in the Analysis objects.
             *        It might be quite slow as all instructions are parsed.
             * @param current_class: KUNAI::DEX::classdef_t class to create the xrefs.
             * @return void
             */
            void _create_xref(KUNAI::DEX::classdef_t &current_class);

            /**
             * @brief get a method by its hash, return the MethodAnalysis object.
             * @param class_name: std::string name of the method's class.
             * @param method_name: std::string name of the method.
             * @param method_descriptor: std::string descriptor (proto) of the method.
             * @return methodanalysis_t
             */
            methodanalysis_t _resolve_method(std::string class_name, std::string method_name, std::string method_descriptor);

            /**
             * @brief DEXParser objects
             */
            std::vector<dexparser_t> dex_parsers;

            /**
             * @brief map with std::string of class name
             * as key and classanalysis_t as value.
             */
            std::unordered_map<std::string, classanalysis_t> classes;

            /**
             * @brief map with std::string as key and stringanalysis_t
             * as value
             */
            std::map<std::string, stringanalysis_t> strings;

            /**
             * @brief map with encodedmethod_t or externalmethod_t
             * as keys and methodanalysis_t as value.
             */
            std::map<std::string, methodanalysis_t> methods;

            /**
             * @brief map hash for quickly getting the Method
             */
            std::unordered_map<std::string, methodanalysis_t> method_hashes;

            dalvikopcodes_t dalvik_opcodes;
            std::map<std::tuple<classdef_t, encodedmethod_t>,
                     std::map<std::uint64_t, instruction_t>>
                instructions;

            // check if xrefs were already created or not
            bool created_xrefs;
        };

        /**
         * ClassAnalysis definition
         */
        class ClassAnalysis
        {
        public:
            /**
             * @brief constructor of ClassAnalysis class.
             * @param class_def: std::variant<classdef_t, externalclass_t>
             * @return void
             */
            ClassAnalysis(std::variant<classdef_t, externalclass_t> class_def);

            /**
             * @brief destructor of ClassAnalysis class.
             * @return void.
             */
            ~ClassAnalysis() = default;

            /**
             * @brief Get the class definition object.
             *
             * @return std::variant<classdef_t, externalclass_t>
             */
            std::variant<classdef_t, externalclass_t> get_class_definition()
            {
                return class_def;
            }

            /**
             * @brief return if the analyzed class is external or not.
             * @return bool
             */
            bool is_class_external()
            {
                return is_external;
            }

            /**
             * @brief add a method to the class, in case of being external
             *        add the external method object to the class.
             * @param method_analysis: methodanalysis_t object part of the class.
             * @return void.
             */
            void add_method(methodanalysis_t method_analysis);

            /**
             * @brief get a list of interfaces that the class implements, in the case of ExternalClass, none.
             * @return std::vector<class_t>
             */
            std::vector<class_t> implements();

            /**
             * @brief get the name of the class from which current class extends.
             * @return std::string
             */
            std::string extends();

            /**
             * @brief get the name of the class.
             * @return std::string
             */
            std::string name();

            /**
             * @brief check if the current class is an Android API.
             * @return bool
             */
            bool is_android_api();

            /**
             * @brief return a vector with all the MethodAnalysis objects.
             * @return std::vector<methodanalysis_t>
             */
            std::vector<methodanalysis_t> get_methods();

            /**
             * @brief return a vector with all the FieldAnalysis objects.
             * @return std::vector<fieldanalysis_t>
             */
            std::vector<fieldanalysis_t> get_fields();

            /**
             * @brief Get number of stored methods.
             * @return size_t
             */
            size_t get_nb_methods()
            {
                return methods.size();
            }

            /**
             * @brief Get one of the MethodAnalysis object by given EncodedMethod or ExternalMethod.
             *
             * @param method Method to get as a parent method.
             * @return methodanalysis_t
             */
            methodanalysis_t get_method_analysis(std::variant<encodedmethod_t, externalmethod_t> method);

            /**
             * @brief Get one of the FieldAnalysis object by given encodedfield_t
             * @param field: encodedfield_t object to look for.
             * @return fieldanalysis_t
             */
            fieldanalysis_t get_field_analysis(encodedfield_t field);

            /**
             * @brief Create a fieldanalysis_t object and add a read xref.
             * @param method: methodanalysis_t to add in xref.
             * @param classobj: classanalysis_t to add in xref.
             * @param field: encodedfield_t object to create FieldAnalysis object.
             * @param off: std::uint64_t to add in xref.
             * @return void
             */
            void add_field_xref_read(methodanalysis_t method,
                                     classanalysis_t classobj,
                                     encodedfield_t field,
                                     std::uint64_t off);

            /**
             * @brief Create a fieldanalysis_t object and add a write xref.
             * @param method: methodanalysis_t to add in xref.
             * @param classobj: classanalysis_t to add in xref.
             * @param field: encodedfield_t object to create FieldAnalysis object.
             * @param off: std::uint64_t to add in xref.
             * @return void
             */
            void add_field_xref_write(methodanalysis_t method,
                                      classanalysis_t classobj,
                                      encodedfield_t field,
                                      std::uint64_t off);

            /**
             * @brief Create a std::shared<MethodAnalysis> object and add a to xref.
             * @param method1: methodanalysis_t method to add to class and add xref to.
             * @param classobj: classanalysis_t class to add to xref.
             * @param method2: methodanalysis_t method to add to xref.
             * @param off: std::uint64_t offset to add to xref.
             * @return void
             */
            void add_method_xref_to(methodanalysis_t method1,
                                    classanalysis_t classobj,
                                    methodanalysis_t method2,
                                    std::uint64_t off);

            /**
             * @brief Create a std::shared<MethodAnalysis> object and add a from xref.
             * @param method1: methodanalysis_t method to add to class and add xref from.
             * @param classobj: classanalysis_t class to add to xref.
             * @param method2: methodanalysis_t method to add to xref.
             * @param off: std::uint64_t offset to add to xref.
             * @return void
             */
            void add_method_xref_from(methodanalysis_t method1,
                                      classanalysis_t classobj,
                                      methodanalysis_t method2,
                                      std::uint64_t off);

            /**
             * @brief Create a cross reference to another class.
             *        XrefTo means, that the current class calls another class.
             *        The current class should also be contained in another class' XrefFrom list.
             * @param ref_kind: DVMTypes::REF_TYPE type of call done to the other class.
             * @param classobj: classanalysis_t
             * @param methodobj: methodanalysis_t methods from which other class is called.
             * @param offset: std::uint64_t offset where the call is done.
             */
            void add_xref_to(DVMTypes::REF_TYPE ref_kind,
                             classanalysis_t classobj,
                             methodanalysis_t methodobj,
                             std::uint64_t offset);

            /**
             * @brief Create a cross reference from this class.
             *        XrefFrom means, that current class is called by another class.
             * @param ref_kind: DVMTypes::REF_TYPE type of call done by another class.
             * @param classobj: classanalysis_t class called.
             * @param methodobj: methodanalysis_t method from class that calls the other.
             * @param offset: std::uint64_t offset where the call is done.
             */
            void add_xref_from(DVMTypes::REF_TYPE ref_kind,
                               classanalysis_t classobj,
                               methodanalysis_t methodobj,
                               std::uint64_t offset);

            /**
             * @brief return all the references to other classes called by this class.
             * @return class_xref
             */
            const class_xref &
            get_xref_to() const
            {
                return xrefto;
            }

            /**
             * @brief return all the classes that call this class.
             * @return class_xref
             */
            const class_xref &
            get_xref_from() const
            {
                return xreffrom;
            }

            /**
             * @brief Add a new reference to a method where the class is instantiated.
             * @param methodobj: methodanalysis_t method where class is instantiated.
             * @param offset: offset where class is instantiated.
             * @return void
             */
            void add_xref_new_instance(methodanalysis_t methodobj, std::uint64_t offset);

            /**
             * @brief Return all the references where the call is instantiated.
             * @return xref_method_idx
             */
            const xref_method_idx &get_xref_new_instance() const
            {
                return xrefnewinstance;
            }

            /**
             * @brief Add a crossreference to a method referencing this classtype.
             * @param methodobj: methodanalysis_t method where class is referenced.
             * @param offset: offset where class is referenced.
             * @return void
             */
            void add_xref_const_class(methodanalysis_t methodobj, std::uint64_t offset);

            /**
             * @brief Return all the methods where this class is referenced.
             * @return xref_method_idx
             */
            const xref_method_idx &get_xref_const_class() const
            {
                return xrefconstclass;
            }

        private:
            // ClassDef or ExternalClass object
            std::variant<classdef_t, externalclass_t> class_def;

            bool is_external;

            // map with method analysis for DEX analysis
            std::map<std::string, methodanalysis_t> methods;
            std::map<encodedfield_t, fieldanalysis_t> fields;

            class_xref xrefto;
            class_xref xreffrom;

            xref_method_idx xrefnewinstance;
            xref_method_idx xrefconstclass;

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
            /**
             * @brief Construct a new DVMBasicBlock object
             *
             * @param start
             * @param dalvik_opcodes
             * @param context
             * @param method
             * @param instructions
             */
            DVMBasicBlock(std::uint64_t start,
                          dalvikopcodes_t dalvik_opcodes,
                          basicblocks_t context,
                          encodedmethod_t method,
                          std::map<std::uint64_t, instruction_t> &instructions);

            /**
             * @brief Destroy the DVMBasicBlock object
             */
            ~DVMBasicBlock() = default;

            /**
             * @brief get start idx from the current basic block.
             * @return std::uint64_t
             */
            std::uint64_t get_start() const { return start; }

            /**
             * @brief get end idx from the current basic block.
             * @return std::uint64_t
             */
            std::uint64_t get_end() const { return end; }

            /**
             * @brief return all the instructions from current basic block.
             * @return std::vector<instruction_t>
             */
            std::vector<instruction_t> get_instructions();

            /**
             * @brief return the last instruction from the basic block.
             * @return instruction_t
             */
            instruction_t get_last();

            /**
             * @brief return all the child basic blocks.
             * @return std::vector<std::tuple<std::uint64_t, std::uint64_t, DVMBasicBlock*>>
             */
            const std::vector<std::tuple<std::uint64_t, std::uint64_t, DVMBasicBlock*>> &get_next() const
            {
                return childs;
            }

            /**
             * @brief return all the parent basic blocks.
             * @return std::vector<std::tuple<std::uint64_t, std::uint64_t, DVMBasicBlock*>>
             */
            const std::vector<std::tuple<std::uint64_t, std::uint64_t, DVMBasicBlock*>> &get_prev() const
            {
                return parents;
            }

            /**
             * @brief push a basic block into the vector of parent basic blocks.
             * @param bb: std::vector<std::tuple<std::uint64_t, std::uint64_t, dvmbasicblock_t>> to push in vector.
             * @return void
             */
            void set_parent(std::tuple<std::uint64_t, std::uint64_t, DVMBasicBlock*>);

            /**
             * @brief set a children basic block, if no argument is given, this is taken from context.
             * @return void
             */
            void set_child();

            /**
             * @brief set a set of children basic blocks.
             * @param values: ids from context of basic blocks to push into vector.
             * @return void
             */
            void set_child(const std::vector<int64_t> &values);

            /**
             * @brief return last length of DVMBasicBlock.
             * @return std::uint64_t
             */
            std::uint64_t get_last_length()
            {
                return last_length;
            }

            /**
             * @brief return the number of instructions of the DVMBasicBlock.
             * @return std::uint64_t
             */
            std::uint64_t get_nb_instructions()
            {
                return nb_instructions;
            }

            /**
             * @brief Calculate new values with an instruction and push in case is a special instruction.
             * @param instr: instruction_t object to increase diferent values and insert into special instructions.
             * @return void
             */
            void push(instruction_t instr);

            /**
             * @brief get one of the special instructions.
             * @param idx: std::uint64_t with index of the special instruction.
             * @return instruction_t
             */
            instruction_t get_special_instruction(std::uint64_t idx);

            /**
             * @brief return an exception analysis object.
             * @return exceptionanalysis_t
             */
            exceptionanalysis_t get_exception_analysis()
            {
                return exception_analysis;
            }

            /**
             * @brief set exception analysis object
             * @param exception_analysis: exceptionanalysis_t object.
             * @return void
             */
            void set_exception_analysis(exceptionanalysis_t exception_analysis)
            {
                this->exception_analysis = exception_analysis;
            }

        private:
            std::uint64_t start, end; // first and final idx from the basic block
            dalvikopcodes_t dalvik_opcodes;
            basicblocks_t context;
            encodedmethod_t method;
            std::map<std::uint64_t, instruction_t> instructions;
            std::map<std::uint64_t, instruction_t> special_instructions;
            std::uint64_t last_length;
            std::uint64_t nb_instructions;
            std::string name;
            exceptionanalysis_t exception_analysis;

            std::vector<std::tuple<std::uint64_t, std::uint64_t, DVMBasicBlock*>> parents;
            std::vector<std::tuple<std::uint64_t, std::uint64_t, DVMBasicBlock*>> childs;
        };

        /**
         * BasicBlocks definition
         */
        class BasicBlocks
        {
        public:
            /**
             * @brief Construct a new Basic Blocks object
             */
            BasicBlocks();

            /**
             * @brief Destroy the Basic Blocks object
             */
            ~BasicBlocks();

            /**
             * @brief push a given DVMBasicBlock into the vector.
             * @param basic_block: DVMBasicBlock object.
             * @return void
             */
            void push_basic_block(dvmbasicblock_t basic_block);

            /**
             * @brief pop the last basic block from the vector, pop operation remove it from the vector.
             * @return dvmbasicblock_t
             */
            dvmbasicblock_t pop_basic_block();

            /**
             * @brief get one basic block by the idx of the instruction.
             * @param idx: index of the instruction to retrieve its basic block.
             * @return dvmbasicblock_t
             */
            dvmbasicblock_t get_basic_block_by_idx(std::uint64_t idx);

            /**
             * @brief get the numbers of basic blocks.
             * @return size_t
             */
            size_t get_number_of_basic_blocks()
            {
                return basic_blocks.size();
            }

            /**
             * @brief get all the basic blocks.
             * @return std::vector<dvmbasicblock_t>
             */
            const std::vector<dvmbasicblock_t> &get_basic_blocks() const
            {
                return basic_blocks;
            }

        private:
            std::vector<dvmbasicblock_t> basic_blocks;
        };

        /**
         * MethodAnalysis definition
         */
        class MethodAnalysis
        {
        public:
            /**
             * @brief Constructor of MethodAnalysis it will initialize
             * various variables.
             * @param method_encoded: std::variant<encodedmethod_t, externalmethod_t>
             * @param dalvik_opcodes: dalvikopcodes_t object.
             * @param instructions: std::map<std::uint64_t, instruction_t> all the DEX instructions.
             * @return void.
             */
            MethodAnalysis(std::variant<encodedmethod_t, externalmethod_t> method_encoded, dalvikopcodes_t dalvik_opcodes, std::map<std::uint64_t, instruction_t> instructions);

            /**
             * @brief MethodAnalysis destructor.
             * @return void.
             */
            ~MethodAnalysis() = default;

            /**
             * @brief return if the method is instance of externalmethod_t
             * @return bool
             */
            bool external()
            {
                return is_external;
            }

            /**
             * @brief check if current method is an Android API.
             * @return bool
             */
            bool is_android_api();

            /**
             * @brief Return method_encoded object, this can
             * be of different types EncodedMethod or ExternalMethod
             * must check which one it is.
             * @return std::variant<encodedmethod_t, externalmethod_t>
             */
            std::variant<encodedmethod_t, externalmethod_t> get_method()
            {
                return method_encoded;
            }

            /**
             * @brief return the method name.
             * @return std::string
             */
            std::string name();

            /**
             * @brief return method prototype (descriptor)
             * @return std::string
             */
            std::string descriptor();

            /**
             * @brief return access as string.
             * @return std::string
             */
            std::string access();

            /**
             * @brief get the class name from the method.
             * @return std::string
             */
            std::string class_name();

            /**
             * @brief get whole name with class name, method name and descriptor.
             * @return std::string
             */
            std::string full_name();

            /**
             * @brief Insert a new xref of method reading.
             * @param class_object: ClassAnalysis where the method is read.
             * @param field_object: FieldAnalysis maybe where the method is read from... Dunno
             * @param offset: offset of the instruction where the method is read.
             * @return void
             */
            void add_xref_read(classanalysis_t class_object, fieldanalysis_t field_object, std::uint64_t offset);

            /**
             * @brief Insert a new xref of method written.
             * @param class_object: ClassAnalysis where the method is written.
             * @param field_object: FieldAnalysis maybe where the method is written... Dunno
             * @param offset: offset of the instruction where the method is written.
             * @return void
             */
            void add_xref_write(classanalysis_t class_object, fieldanalysis_t field_object, std::uint64_t offset);

            /**
             * @brief Return all the xref where method is read, with or without offset.
             *
             * @return std::vector<std::tuple<KUNAI::DEX::classanalysis_t, KUNAI::DEX::fieldanalysis_t, uint64_t>>
             */
            const std::vector<std::tuple<KUNAI::DEX::classanalysis_t, KUNAI::DEX::fieldanalysis_t, uint64_t>> &get_xref_read() const
            {
                return xrefread;
            }

            /**
             * @brief Return all the xref where method is written
             *
             * @return const std::vector<std::tuple<KUNAI::DEX::classanalysis_t, KUNAI::DEX::fieldanalysis_t, uint64_t>>&
             */
            const std::vector<std::tuple<KUNAI::DEX::classanalysis_t, KUNAI::DEX::fieldanalysis_t, uint64_t>> &get_xref_write() const
            {
                return xrefwrite;
            }

            /**
             * @brief Add a reference to a method called by this method.
             * @param class_object: classanalysis_t class of the method called.
             * @param method_object: methodanalysis_t method called from current method.
             * @param offset: std::uint64_t offset where call is done.
             * @return void
             */
            void add_xref_to(classanalysis_t class_object, methodanalysis_t method_object, std::uint64_t offset);

            /**
             * @brief Add a reference of a method that calls current method.
             * @param class_object: classanalysis_t class of the method that calls current method.
             * @param method_object: methodanalysis_t method that calls current method.
             * @param offset: std::uint64_t offset where call is done.
             * @return void
             */
            void add_xref_from(classanalysis_t class_object, methodanalysis_t method_object, std::uint64_t offset);

            /**
             * @brief get the methods where current method is called, with or without offset.
             *
             * @return const std::vector<std::tuple<KUNAI::DEX::classanalysis_t, KUNAI::DEX::methodanalysis_t, uint64_t>>&
             */
            const std::vector<std::tuple<KUNAI::DEX::classanalysis_t, KUNAI::DEX::methodanalysis_t, uint64_t>> &get_xref_to() const
            {
                return xrefto;
            }

            /**
             * @brief get the methods called by current method, with or without offset.
             *
             * @return const std::vector<std::tuple<KUNAI::DEX::classanalysis_t, KUNAI::DEX::methodanalysis_t, uint64_t>>&
             */
            const std::vector<std::tuple<KUNAI::DEX::classanalysis_t, KUNAI::DEX::methodanalysis_t, uint64_t>> &get_xref_from() const
            {
                return xreffrom;
            }

            /**
             * @brief Add a cross reference to another class that is instanced within this method.
             * @param class_object: classanalysis_t class_object instanced class.
             * @param offset: std::uint64_t offset of the method
             * @return void
             */
            void add_xref_new_instance(classanalysis_t class_object, std::uint64_t offset);

            /**
             * @brief Add a cross reference to another classtype.
             * @param class_object: classanalysis_t
             * @param offset: std::uint64_t
             * @return void
             */
            void add_xref_const_class(classanalysis_t class_object, std::uint64_t offset);

            /**
             * @brief return the cross references of classes instanced by this method.
             * @return std::vector<std::tuple<classanalysis_t, std::uint64_t>>
             */
            const std::vector<std::tuple<classanalysis_t, std::uint64_t>> &get_xref_new_instance() const
            {
                return xrefnewinstance;
            }

            /**
             * @brief return all the cross references of another classtype.
             * @return std::vector<std::tuple<classanalysis_t, std::uint64_t>>
             */
            const std::vector<std::tuple<classanalysis_t, std::uint64_t>> &get_xref_const_class() const
            {
                return xrefconstclass;
            }

            /**
             * @brief get the instructions from the method.
             * @return std::map<std::uint64_t, instruction_t>
             */
            const std::map<std::uint64_t, instruction_t> &get_instructions() const
            {
                return instructions;
            }

            /**
             * @brief get the basic blocks with the DVMBasicBlocks with the instructions.
             * @return basicblocks_t
             */
            basicblocks_t &get_basic_blocks()
            {
                return basic_blocks;
            }

            /**
             * @brief Get all the exceptions from the method.
             * @return exception_t
             */
            exception_t &get_exceptions()
            {
                return exceptions;
            }

        private:
            /**
             * @brief method to create basic blocks for the method.
             * @return void
             */
            void create_basic_block();

            bool is_external;
            std::variant<encodedmethod_t, externalmethod_t> method_encoded;
            dalvikopcodes_t dalvik_opcodes;
            std::map<std::uint64_t, instruction_t> instructions;
            basicblocks_t basic_blocks;
            exception_t exceptions;

            std::vector<std::tuple<classanalysis_t, fieldanalysis_t, std::uint64_t>> xrefread;
            std::vector<std::tuple<classanalysis_t, fieldanalysis_t, std::uint64_t>> xrefwrite;

            std::vector<std::tuple<classanalysis_t, methodanalysis_t, std::uint64_t>> xrefto;
            std::vector<std::tuple<classanalysis_t, methodanalysis_t, std::uint64_t>> xreffrom;

            std::vector<std::tuple<classanalysis_t, std::uint64_t>> xrefnewinstance;
            std::vector<std::tuple<classanalysis_t, std::uint64_t>> xrefconstclass;

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
            /**
             * @brief Constructor of FieldAnalysis class.
             * @param field: FieldID pointer base of the class.
             * @return void
             */
            FieldAnalysis(encodedfield_t field);

            /**
             * @brief Destructor of FieldAnalysis class.
             * @return void
             */
            ~FieldAnalysis() = default;

            /**
             * @brief retrieve name of Field.
             * @return std::string
             */
            std::string &name()
            {
                return *field->get_field()->get_name_idx();
            }

            /**
             * @brief Add a xref where this field is read.
             * @param class_object: class where this field is read.
             * @param method_object: method where this field is read.
             * @param offset: instruction offset where field is read.
             * @return void
             */
            void add_xref_read(classanalysis_t class_object, methodanalysis_t method_object, std::uint64_t offset);

            /**
             * @brief Add a xref where this field is written.
             * @param class_object: class where this field is written.
             * @param method_object: method where this field is written.
             * @param offset: instruction offset where field is written.
             * @return void
             */
            void add_xref_write(classanalysis_t class_object, methodanalysis_t method_object, std::uint64_t offset);

            /**
             * @brief Get all the read cross references from the field.
             *
             * @return const std::vector<std::tuple<classanalysis_t, methodanalysis_t, std::uint64_t>>&
             */
            const std::vector<std::tuple<classanalysis_t, methodanalysis_t, std::uint64_t>> &get_xref_read() const
            {
                return xrefread;
            }

            /**
             * @brief Get all the write cross references from the field.
             *
             * @return const std::vector<std::tuple<classanalysis_t, methodanalysis_t, std::uint64_t>>&
             */
            const std::vector<std::tuple<classanalysis_t, methodanalysis_t, std::uint64_t>> &get_xref_write() const
            {
                return xrefwrite;
            }

            /**
             * @brief return the FieldID pointer.
             * @return FieldID*
             */
            encodedfield_t &get_field()
            {
                return field;
            }

            friend std::ostream &operator<<(std::ostream &os, const FieldAnalysis &entry);

        private:
            encodedfield_t field;
            std::vector<std::tuple<classanalysis_t, methodanalysis_t, std::uint64_t>> xrefread;
            std::vector<std::tuple<classanalysis_t, methodanalysis_t, std::uint64_t>> xrefwrite;
        };

        /**
         * StringAnalysis definition
         */
        class StringAnalysis
        {
        public:
            /**
             * @brief StringAnalysis constructor, set the value and original value.
             * @param value: std::string* pointer to one of the program strings.
             * @return void
             */
            StringAnalysis(std::string *value);

            /**
             * @brief StringAnalysis destructor, clear vector.
             * @return void
             */
            ~StringAnalysis();

            /**
             * @brief Add a reference of a method that reads current string.
             * @param class_object: classanalysis_t class of the method that reads current string.
             * @param method_object: methodanalysis_t method that reads current string.
             * @param offset: std::uint64_t offset where call is done.
             * @return void
             */
            void add_xref_from(classanalysis_t class_object, methodanalysis_t method_object, std::uint64_t offset);

            /**
             * @brief Get the read xref from the string
             *
             * @return const std::vector<std::tuple<classanalysis_t, methodanalysis_t, std::uint64_t>>&
             */
            const std::vector<std::tuple<classanalysis_t, methodanalysis_t, std::uint64_t>> &get_xref_from() const
            {
                return xreffrom;
            }

            /**
             * @brief set the value from StringAnalysis.
             * @param value: std::string* new value to put in object.
             * @return void
             */
            void set_value(std::string *value)
            {
                this->value = value;
            }

            /**
             * @brief get the value from the object.
             * @return std::string*
             */
            std::string *get_value()
            {
                return value;
            }

            /**
             * @brief get if the value has been overwritten or not.
             * @return bool
             */
            bool is_overwritten()
            {
                return value == orig_value;
            }

        private:
            std::string *value;
            std::string *orig_value;
            std::vector<std::tuple<classanalysis_t, methodanalysis_t, std::uint64_t>> xreffrom;
        };

        /**
         * ExceptionAnalysis definition
         */
        class ExceptionAnalysis
        {
        public:
            /**
             * @brief Constructor of ExceptionAnalysis.
             * @param exception: exception_data structure which contains handle information.
             * @param basic_blocks: all the basic blocks where are all the instruction's basic blocks.
             * @return void
             */
            ExceptionAnalysis(exceptions_data exception, basicblocks_t basic_blocks);

            /**
             * @brief ExceptionAnalysis destructor.
             * @return void
             */
            ~ExceptionAnalysis() = default;

            /**
             * @brief Get a string with all the information from the ExceptionAnalysis object.
             * @return std::string
             */
            std::string show_buff();

            /**
             * @brief Get exception data structure.
             * @return exceptions_data
             */
            exceptions_data &get()
            {
                return exception;
            }

        private:
            exceptions_data exception;
            basicblocks_t basic_blocks;
        };

        /**
         * Exception definition
         */
        class Exception
        {
        public:
            /**
             * @brief Constructor of Exception class, this contains a vector of ExceptionAnalysis objects.
             * @return void
             */
            Exception();

            /**
             * @brief Destructor of Exception class.
             * @return void
             */
            ~Exception() = default;

            /**
             * @brief Add new ExceptionAnalysis for each exceptions_data receive and basic blocks.
             * @param exceptions: vector with exceptions_data structures.
             * @param basic_blocks: BasicBlocks object for the ExceptionAnalysis object.
             * @return void.
             */
            void add(std::vector<exceptions_data> exceptions, basicblocks_t basic_blocks);

            /**
             * @brief Get a ExceptionAnalysis object get by the start and end address of the try handler.
             * @param start_addr: start try value address.
             * @param end_addr: end try value address.
             * @return exceptionanalysis_t
             */
            exceptionanalysis_t get_exception(std::uint64_t start_addr, std::uint64_t end_addr);

            /**
             * @brief Get all the ExceptionAnalysis objects.
             * @return std::vector<exceptionanalysis_t>
             */
            const std::vector<exceptionanalysis_t> &gets() const
            {
                return exceptions;
            }

        private:
            std::vector<exceptionanalysis_t> exceptions;
        };
    } // namespace DEX
} // namespace KUNAI

#endif