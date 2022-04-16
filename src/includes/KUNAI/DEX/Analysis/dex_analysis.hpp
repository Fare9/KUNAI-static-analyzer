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

#include "dex_parents.hpp"

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

        typedef std::set<std::tuple<std::shared_ptr<MethodAnalysis>, std::uint64_t>> xref_method_idx;
        typedef std::map<std::shared_ptr<ClassAnalysis>, std::set<std::tuple<DVMTypes::REF_TYPE, std::shared_ptr<MethodAnalysis>, std::uint64_t>>> class_xref;

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
            /**
             * @brief Analysis class constructor, just accept an initial DexParser object
             *        to start the analysis.
             * @param dex_parser: std::shared_ptr<DexParser> parser to initialize the analysis.
             * @param dalvik_opcodes: std::shared_ptr<DalvikOpcodes> used to initialize some objects.
             * @param instructions: std::map<std::tuple<std::shared_ptr<ClassDef>, std::shared_ptr<EncodedMethod>>, std::map<std::uint64_t, std::shared_ptr<Instruction>>> instructions to initialize methods.
             * @return void
             */
            Analysis(std::shared_ptr<DexParser> dex_parser, std::shared_ptr<DalvikOpcodes> dalvik_opcodes,
                     std::map<std::tuple<std::shared_ptr<ClassDef>, std::shared_ptr<EncodedMethod>>, std::map<std::uint64_t, std::shared_ptr<Instruction>>> instructions);

            /**
             * @brief Analysis class destructor, nothing really interesting in here.
             * @return void
             */
            ~Analysis() = default;

            /**
             * @brief Analysis method to add a new dex_parser we will have to analyze
             *        classes, methods, fields and strings from the parser as new objects
             *        will be added.
             * @param dex_parser: std::shared_ptr<DexParser> new parser object to add.
             * @return void
             */
            void add(std::shared_ptr<DexParser> dex_parser);

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
            bool is_class_present(std::string class_name);

            /**
             * @brief get a ClassAnalysis object by a class_name.
             * @param class_name: std::string class name to retrieve its ClassAnalysis object.
             * @return std::shared_ptr<ClassAnalysis>
             */
            std::shared_ptr<ClassAnalysis> get_class_analysis(std::string class_name);

            /**
             * @brief Get all the ClassAnalysis objects in a vector.
             * @return std::vector<std::shared_ptr<ClassAnalysis>>
             */
            std::vector<std::shared_ptr<ClassAnalysis>> get_classes();

            /**
             * @brief Get all the external classes in a vector.
             * @return std::vector<std::shared_ptr<ClassAnalysis>>
             */
            std::vector<std::shared_ptr<ClassAnalysis>> get_external_classes();

            /**
             * @brief Get all the internal classes in a vector.
             * @return std::vector<std::shared_ptr<ClassAnalysis>>
             */
            std::vector<std::shared_ptr<ClassAnalysis>> get_internal_classes();

            // MethodAnalysis methods

            /**
             * @brief Get MethodAnalysis object by giving an EncodedMethod or
             *        ExternalMethod object.
             * @param method: std::shared_ptr<ParentMethod>
             * @return std::shared_ptr<MethodAnalysis>
             */
            std::shared_ptr<MethodAnalysis> get_method(std::shared_ptr<ParentMethod> method);

            /**
             * @brief Get MethodID from internal methods by class name, method name and
             *        method descriptor.
             * @param class_name: std::string class name of the method.
             * @param method_name: std::string method name.
             * @param method_descriptor: std::string method descriptor (parameters and return value).
             * @return MethodID*
             */
            MethodID *get_method_by_name(std::string class_name, std::string method_name, std::string method_descriptor);

            /**
             * @brief Get a method analysis given class name, method name and method descriptor.
             * @param class_name: std::string class name of the method.
             * @param method_name: std::string method name.
             * @param method_descriptor: std::string method descriptor (parameters and return value).
             * @return std::shared_ptr<MethodAnalysis>
             */
            std::shared_ptr<MethodAnalysis> get_method_analysis_by_name(std::string class_name, std::string method_name, std::string method_descriptor);

            /**
             * @brief Get all the MethodAnalysis object in a vector.
             * @return std::vector<std::shared_ptr<MethodAnalysis>>
             */
            std::vector<std::shared_ptr<MethodAnalysis>> get_methods();

            // FieldAnalysis methods

            /**
             * @brief Get a FieldAnalysis given a field object.
             * @param field: std::shared_ptr<EncodedField> field to look for its FieldAnalysis.
             * @return std::shared_ptr<FieldAnalysis>
             */
            std::shared_ptr<FieldAnalysis> get_field_analysis(std::shared_ptr<EncodedField> field);

            /**
             * @brief Get all the FieldAnalysis objects in a vector.
             * @return std::vector<std::shared_ptr<FieldAnalysis>>
             */
            std::vector<std::shared_ptr<FieldAnalysis>> get_fields();

            // StringAnalysis methods

            /**
             * @brief Get the map of std::string, StringAnalysis objects.
             * @return std::map<std::string, std::shared_ptr<StringAnalysis>>
             */
            const std::map<std::string, std::shared_ptr<StringAnalysis>> &get_strings_analysis() const
            {
                return strings;
            }

            /**
             * @brief Get a vector of all the StringAnalysis objects.
             * @return std::vector<std::shared_ptr<StringAnalysis>>
             */
            std::vector<std::shared_ptr<StringAnalysis>> get_strings();

            // class analysis by regular expression

            /**
             * @brief Find classes by name with regular expression,
             *        the method returns a list of ClassAnalysis object
             *        that match the regex.
             * @brief name: std::string name with the regex to search.
             * @brief no_external: bool want external classes?
             * @return std::vector<std::shared_ptr<ClassAnalysis>>
             */
            std::vector<std::shared_ptr<ClassAnalysis>> find_classes(std::string name, bool no_external);

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
             * @return std::vector<std::shared_ptr<MethodAnalysis>>
             */
            std::vector<std::shared_ptr<MethodAnalysis>> find_methods(std::string class_name, std::string method_name, std::string descriptor, std::string accessflags, bool no_external);

            /**
             * @brief Find StringAnalysis objects using regular expressions.
             * @param string: std::string regex to look for.
             * @return std::vector<std::shared_ptr<StringAnalysis>>
             */
            std::vector<std::shared_ptr<StringAnalysis>> find_strings(std::string string);

            /**
             * @brief Find FieldAnalysis objects using regular expression,
             *        find those in classes.
             * @param class_name: std::string class where the field is.
             * @param field_name: std::string name of the field.
             * @param field_type: std::string type of the field.
             * @param accessflags: accessflags from the field.
             * @return std::vector<std::shared_ptr<FieldAnalysis>>
             */
            std::vector<std::shared_ptr<FieldAnalysis>> find_fields(std::string class_name, std::string field_name, std::string field_type, std::string accessflags);

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
             * @param current_class: std::shared_ptr<KUNAI::DEX::ClassDef> class to create the xrefs.
             * @return void
             */
            void _create_xref(std::shared_ptr<KUNAI::DEX::ClassDef> current_class);

            /**
             * @brief get a method by its hash, return the MethodAnalysis object.
             * @param class_name: std::string name of the method's class.
             * @param method_name: std::string name of the method.
             * @param method_descriptor: std::string descriptor (proto) of the method.
             * @return std::shared_ptr<MethodAnalysis>
             */
            std::shared_ptr<MethodAnalysis> _resolve_method(std::string class_name, std::string method_name, std::string method_descriptor);

            /**
             * @brief DEXParser objects
             */
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
                     std::map<std::uint64_t, std::shared_ptr<Instruction>>>
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
             * @param class_def: std::shared_ptr<ParentClass> that can be a shared_ptr of ClassDef or of ExternalClass.
             * @return void
             */
            ClassAnalysis(std::shared_ptr<ParentClass> class_def);

            /**
             * @brief destructor of ClassAnalysis class.
             * @return void.
             */
            ~ClassAnalysis() = default;

            /**
             * @brief Get the class definition object.
             *
             * @return std::shared_ptr<ParentClass>
             */
            std::shared_ptr<ParentClass> get_class_definition()
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
             * @param method_analysis: std::shared_ptr<MethodAnalysis> object part of the class.
             * @return void.
             */
            void add_method(std::shared_ptr<MethodAnalysis> method_analysis);

            /**
             * @brief get a list of interfaces that the class implements, in the case of ExternalClass, none.
             * @return std::vector<Class *>
             */
            std::vector<Class *> implements();

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
             * @return std::vector<std::shared_ptr<MethodAnalysis>>
             */
            std::vector<std::shared_ptr<MethodAnalysis>> get_methods();

            /**
             * @brief return a vector with all the FieldAnalysis objects.
             * @return std::vector<std::shared_ptr<FieldAnalysis>>
             */
            std::vector<std::shared_ptr<FieldAnalysis>> get_fields();

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
             * @return std::shared_ptr<MethodAnalysis>
             */
            std::shared_ptr<MethodAnalysis> get_method_analysis(std::shared_ptr<ParentMethod> method);

            /**
             * @brief Get one of the FieldAnalysis object by given std::shared_ptr<EncodedField>
             * @param field: std::shared_ptr<EncodedField> object to look for.
             * @return std::shared_ptr<FieldAnalysis>
             */
            std::shared_ptr<FieldAnalysis> get_field_analysis(std::shared_ptr<EncodedField> field);

            /**
             * @brief Create a std::shared_ptr<FieldAnalysis> object and add a read xref.
             * @param method: std::shared_ptr<MethodAnalysis> to add in xref.
             * @param classobj: std::shared_ptr<ClassAnalysis> to add in xref.
             * @param field: std::shared_ptr<EncodedField> object to create FieldAnalysis object.
             * @param off: std::uint64_t to add in xref.
             * @return void
             */
            void add_field_xref_read(std::shared_ptr<MethodAnalysis> method,
                                     std::shared_ptr<ClassAnalysis> classobj,
                                     std::shared_ptr<EncodedField> field,
                                     std::uint64_t off);

            /**
             * @brief Create a std::shared_ptr<FieldAnalysis> object and add a write xref.
             * @param method: std::shared_ptr<MethodAnalysis> to add in xref.
             * @param classobj: std::shared_ptr<ClassAnalysis> to add in xref.
             * @param field: std::shared_ptr<EncodedField> object to create FieldAnalysis object.
             * @param off: std::uint64_t to add in xref.
             * @return void
             */
            void add_field_xref_write(std::shared_ptr<MethodAnalysis> method,
                                      std::shared_ptr<ClassAnalysis> classobj,
                                      std::shared_ptr<EncodedField> field,
                                      std::uint64_t off);

            /**
             * @brief Create a std::shared<MethodAnalysis> object and add a to xref.
             * @param method1: std::shared_ptr<MethodAnalysis> method to add to class and add xref to.
             * @param classobj: std::shared_ptr<ClassAnalysis> class to add to xref.
             * @param method2: std::shared_ptr<MethodAnalysis> method to add to xref.
             * @param off: std::uint64_t offset to add to xref.
             * @return void
             */
            void add_method_xref_to(std::shared_ptr<MethodAnalysis> method1,
                                    std::shared_ptr<ClassAnalysis> classobj,
                                    std::shared_ptr<MethodAnalysis> method2,
                                    std::uint64_t off);

            /**
             * @brief Create a std::shared<MethodAnalysis> object and add a from xref.
             * @param method1: std::shared_ptr<MethodAnalysis> method to add to class and add xref from.
             * @param classobj: std::shared_ptr<ClassAnalysis> class to add to xref.
             * @param method2: std::shared_ptr<MethodAnalysis> method to add to xref.
             * @param off: std::uint64_t offset to add to xref.
             * @return void
             */
            void add_method_xref_from(std::shared_ptr<MethodAnalysis> method1,
                                      std::shared_ptr<ClassAnalysis> classobj,
                                      std::shared_ptr<MethodAnalysis> method2,
                                      std::uint64_t off);

            /**
             * @brief Create a cross reference to another class.
             *        XrefTo means, that the current class calls another class.
             *        The current class should also be contained in another class' XrefFrom list.
             * @param ref_kind: DVMTypes::REF_TYPE type of call done to the other class.
             * @param classobj: std::shared_ptr<ClassAnalysis>
             * @param methodobj: std::shared_ptr<MethodAnalysis> methods from which other class is called.
             * @param offset: std::uint64_t offset where the call is done.
             */
            void add_xref_to(DVMTypes::REF_TYPE ref_kind,
                             std::shared_ptr<ClassAnalysis> classobj,
                             std::shared_ptr<MethodAnalysis> methodobj,
                             std::uint64_t offset);

            /**
             * @brief Create a cross reference from this class.
             *        XrefFrom means, that current class is called by another class.
             * @param ref_kind: DVMTypes::REF_TYPE type of call done by another class.
             * @param classobj: std::shared_ptr<ClassAnalysis> class called.
             * @param methodobj: std::shared_ptr<MethodAnalysis> method from class that calls the other.
             * @param offset: std::uint64_t offset where the call is done.
             */
            void add_xref_from(DVMTypes::REF_TYPE ref_kind,
                               std::shared_ptr<ClassAnalysis> classobj,
                               std::shared_ptr<MethodAnalysis> methodobj,
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
             * @param methodobj: std::shared_ptr<MethodAnalysis> method where class is instantiated.
             * @param offset: offset where class is instantiated.
             * @return void
             */
            void add_xref_new_instance(std::shared_ptr<MethodAnalysis> methodobj, std::uint64_t offset);

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
             * @param methodobj: std::shared_ptr<MethodAnalysis> method where class is referenced.
             * @param offset: offset where class is referenced.
             * @return void
             */
            void add_xref_const_class(std::shared_ptr<MethodAnalysis> methodobj, std::uint64_t offset);

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
            std::shared_ptr<ParentClass> class_def;

            bool is_external;

            // map with method analysis for DEX analysis
            std::map<std::string, std::shared_ptr<MethodAnalysis>> methods;
            std::map<std::shared_ptr<EncodedField>, std::shared_ptr<FieldAnalysis>> fields;

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
                          std::shared_ptr<DalvikOpcodes> dalvik_opcodes,
                          std::shared_ptr<BasicBlocks> context,
                          std::shared_ptr<EncodedMethod> method,
                          std::map<std::uint64_t, std::shared_ptr<Instruction>> &instructions);

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
             * @return std::vector<std::shared_ptr<Instruction>>
             */
            std::vector<std::shared_ptr<Instruction>> get_instructions();

            /**
             * @brief return the last instruction from the basic block.
             * @return std::shared_ptr<Instruction>
             */
            std::shared_ptr<Instruction> get_last();

            /**
             * @brief return all the child basic blocks.
             * @return std::vector<std::tuple<std::uint64_t, std::uint64_t, std::shared_ptr<DVMBasicBlock>>>
             */
            const std::vector<std::tuple<std::uint64_t, std::uint64_t, std::shared_ptr<DVMBasicBlock>>> &get_next() const
            {
                return childs;
            }

            /**
             * @brief return all the parent basic blocks.
             * @return std::vector<std::tuple<std::uint64_t, std::uint64_t, DVMBasicBlock*>>
             */
            const std::vector<std::tuple<std::uint64_t, std::uint64_t, DVMBasicBlock *>> &get_prev() const
            {
                return parents;
            }

            /**
             * @brief push a basic block into the vector of parent basic blocks.
             * @param bb: std::vector<std::tuple<std::uint64_t, std::uint64_t, DVMBasicBlock*>> to push in vector.
             * @return void
             */
            void set_parent(std::tuple<std::uint64_t, std::uint64_t, DVMBasicBlock *>);

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
             * @param instr: std::shared_ptr<Instruction> object to increase diferent values and insert into special instructions.
             * @return void
             */
            void push(std::shared_ptr<Instruction> instr);

            /**
             * @brief get one of the special instructions.
             * @param idx: std::uint64_t with index of the special instruction.
             * @return std::shared_ptr<Instruction>
             */
            std::shared_ptr<Instruction> get_special_instruction(std::uint64_t idx);

            /**
             * @brief return an exception analysis object.
             * @return std::shared_ptr<ExceptionAnalysis>
             */
            std::shared_ptr<ExceptionAnalysis> get_exception_analysis()
            {
                return exception_analysis;
            }

            /**
             * @brief set exception analysis object
             * @param exception_analysis: std::shared_ptr<ExceptionAnalysis> object.
             * @return void
             */
            void set_exception_analysis(std::shared_ptr<ExceptionAnalysis> exception_analysis)
            {
                this->exception_analysis = exception_analysis;
            }

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

            std::vector<std::tuple<std::uint64_t, std::uint64_t, DVMBasicBlock *>> parents;
            std::vector<std::tuple<std::uint64_t, std::uint64_t, std::shared_ptr<DVMBasicBlock>>> childs;
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
            void push_basic_block(std::shared_ptr<DVMBasicBlock> basic_block);

            /**
             * @brief pop the last basic block from the vector, pop operation remove it from the vector.
             * @return std::shared_ptr<DVMBasicBlock>
             */
            std::shared_ptr<DVMBasicBlock> pop_basic_block();

            /**
             * @brief get one basic block by the idx of the instruction.
             * @param idx: index of the instruction to retrieve its basic block.
             * @return std::shared_ptr<DVMBasicBlock>
             */
            std::shared_ptr<DVMBasicBlock> get_basic_block_by_idx(std::uint64_t idx);

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
             * @return std::vector<std::shared_ptr<DVMBasicBlock>>
             */
            const std::vector<std::shared_ptr<DVMBasicBlock>> &get_basic_blocks() const
            {
                return basic_blocks;
            }

        private:
            std::vector<std::shared_ptr<DVMBasicBlock>> basic_blocks;
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
             * @param method_encoded: std::shared_ptr<ParentMethod>
             * @param dalvik_opcodes: std::shared_ptr<DalvikOpcodes> object.
             * @param instructions: std::map<std::uint64_t, std::shared_ptr<Instruction>> all the DEX instructions.
             * @return void.
             */
            MethodAnalysis(std::shared_ptr<ParentMethod> method_encoded, std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::map<std::uint64_t, std::shared_ptr<Instruction>> instructions);

            /**
             * @brief MethodAnalysis destructor.
             * @return void.
             */
            ~MethodAnalysis() = default;

            /**
             * @brief return if the method is instance of std::shared_ptr<ExternalMethod>
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
             * @return std::shared_ptr<ParentMethod>
             */
            std::shared_ptr<ParentMethod> get_method()
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
            void add_xref_read(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<FieldAnalysis> field_object, std::uint64_t offset);

            /**
             * @brief Insert a new xref of method written.
             * @param class_object: ClassAnalysis where the method is written.
             * @param field_object: FieldAnalysis maybe where the method is written... Dunno
             * @param offset: offset of the instruction where the method is written.
             * @return void
             */
            void add_xref_write(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<FieldAnalysis> field_object, std::uint64_t offset);

            /**
             * @brief Return all the xref where method is read, with or without offset.
             *
             * @return std::vector<std::tuple<std::shared_ptr<KUNAI::DEX::ClassAnalysis>, std::shared_ptr<KUNAI::DEX::FieldAnalysis>, uint64_t>>
             */
            const std::vector<std::tuple<std::shared_ptr<KUNAI::DEX::ClassAnalysis>, std::shared_ptr<KUNAI::DEX::FieldAnalysis>, uint64_t>> &get_xref_read() const
            {
                return xrefread;
            }

            /**
             * @brief Return all the xref where method is written
             *
             * @return const std::vector<std::tuple<std::shared_ptr<KUNAI::DEX::ClassAnalysis>, std::shared_ptr<KUNAI::DEX::FieldAnalysis>, uint64_t>>&
             */
            const std::vector<std::tuple<std::shared_ptr<KUNAI::DEX::ClassAnalysis>, std::shared_ptr<KUNAI::DEX::FieldAnalysis>, uint64_t>> &get_xref_write() const
            {
                return xrefwrite;
            }

            /**
             * @brief Add a reference to a method called by this method.
             * @param class_object: std::shared_ptr<ClassAnalysis> class of the method called.
             * @param method_object: std::shared_ptr<MethodAnalysis> method called from current method.
             * @param offset: std::uint64_t offset where call is done.
             * @return void
             */
            void add_xref_to(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset);

            /**
             * @brief Add a reference of a method that calls current method.
             * @param class_object: std::shared_ptr<ClassAnalysis> class of the method that calls current method.
             * @param method_object: std::shared_ptr<MethodAnalysis> method that calls current method.
             * @param offset: std::uint64_t offset where call is done.
             * @return void
             */
            void add_xref_from(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset);

            /**
             * @brief get the methods where current method is called, with or without offset.
             *
             * @return const std::vector<std::tuple<std::shared_ptr<KUNAI::DEX::ClassAnalysis>, std::shared_ptr<KUNAI::DEX::MethodAnalysis>, uint64_t>>&
             */
            const std::vector<std::tuple<std::shared_ptr<KUNAI::DEX::ClassAnalysis>, std::shared_ptr<KUNAI::DEX::MethodAnalysis>, uint64_t>> &get_xref_to() const
            {
                return xrefto;
            }

            /**
             * @brief get the methods called by current method, with or without offset.
             *
             * @return const std::vector<std::tuple<std::shared_ptr<KUNAI::DEX::ClassAnalysis>, std::shared_ptr<KUNAI::DEX::MethodAnalysis>, uint64_t>>&
             */
            const std::vector<std::tuple<std::shared_ptr<KUNAI::DEX::ClassAnalysis>, std::shared_ptr<KUNAI::DEX::MethodAnalysis>, uint64_t>> &get_xref_from() const
            {
                return xreffrom;
            }

            /**
             * @brief Add a cross reference to another class that is instanced within this method.
             * @param class_object: std::shared_ptr<ClassAnalysis> class_object instanced class.
             * @param offset: std::uint64_t offset of the method
             * @return void
             */
            void add_xref_new_instance(std::shared_ptr<ClassAnalysis> class_object, std::uint64_t offset);

            /**
             * @brief Add a cross reference to another classtype.
             * @param class_object: std::shared_ptr<ClassAnalysis>
             * @param offset: std::uint64_t
             * @return void
             */
            void add_xref_const_class(std::shared_ptr<ClassAnalysis> class_object, std::uint64_t offset);

            /**
             * @brief return the cross references of classes instanced by this method.
             * @return std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::uint64_t>>
             */
            const std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::uint64_t>> &get_xref_new_instance() const
            {
                return xrefnewinstance;
            }

            /**
             * @brief return all the cross references of another classtype.
             * @return std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::uint64_t>>
             */
            const std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::uint64_t>> &get_xref_const_class() const
            {
                return xrefconstclass;
            }

            /**
             * @brief get the instructions from the method.
             * @return std::map<std::uint64_t, std::shared_ptr<Instruction>>
             */
            const std::map<std::uint64_t, std::shared_ptr<Instruction>> &get_instructions() const
            {
                return instructions;
            }

            /**
             * @brief get the basic blocks with the DVMBasicBlocks with the instructions.
             * @return std::shared_ptr<BasicBlocks>
             */
            std::shared_ptr<BasicBlocks>& get_basic_blocks()
            {
                return basic_blocks;
            }

            /**
             * @brief Get all the exceptions from the method.
             * @return std::shared_ptr<Exception>
             */
            std::shared_ptr<Exception>& get_exceptions()
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
            std::shared_ptr<ParentMethod> method_encoded;
            std::shared_ptr<DalvikOpcodes> dalvik_opcodes;
            std::map<std::uint64_t, std::shared_ptr<Instruction>> instructions;
            std::shared_ptr<BasicBlocks> basic_blocks;
            std::shared_ptr<Exception> exceptions;

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
            /**
             * @brief Constructor of FieldAnalysis class.
             * @param field: FieldID pointer base of the class.
             * @return void
             */
            FieldAnalysis(std::shared_ptr<EncodedField> field);

            /**
             * @brief Destructor of FieldAnalysis class.
             * @return void
             */
            ~FieldAnalysis() = default;

            /**
             * @brief retrieve name of Field.
             * @return std::string
             */
            std::string& name()
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
            void add_xref_read(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset);

            /**
             * @brief Add a xref where this field is written.
             * @param class_object: class where this field is written.
             * @param method_object: method where this field is written.
             * @param offset: instruction offset where field is written.
             * @return void
             */
            void add_xref_write(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset);

            /**
             * @brief Get all the read cross references from the field.
             *
             * @return const std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>>&
             */
            const std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>> &get_xref_read() const
            {
                return xrefread;
            }

            /**
             * @brief Get all the write cross references from the field.
             *
             * @return const std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>>&
             */
            const std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>> &get_xref_write() const
            {
                return xrefwrite;
            }

            /**
             * @brief return the FieldID pointer.
             * @return FieldID*
             */
            std::shared_ptr<EncodedField>& get_field()
            {
                return field;
            }

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
             * @param class_object: std::shared_ptr<ClassAnalysis> class of the method that reads current string.
             * @param method_object: std::shared_ptr<MethodAnalysis> method that reads current string.
             * @param offset: std::uint64_t offset where call is done.
             * @return void
             */
            void add_xref_from(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset);

            /**
             * @brief Get the read xref from the string
             *
             * @return const std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>>&
             */
            const std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>> &get_xref_from() const
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
            std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>> xreffrom;
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
            ExceptionAnalysis(exceptions_data exception, std::shared_ptr<BasicBlocks> basic_blocks);

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
            exceptions_data& get()
            {
                return exception;
            }

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
            void add(std::vector<exceptions_data> exceptions, std::shared_ptr<BasicBlocks> basic_blocks);

            /**
             * @brief Get a ExceptionAnalysis object get by the start and end address of the try handler.
             * @param start_addr: start try value address.
             * @param end_addr: end try value address.
             * @return std::shared_ptr<ExceptionAnalysis>
             */
            std::shared_ptr<ExceptionAnalysis> get_exception(std::uint64_t start_addr, std::uint64_t end_addr);

            /**
             * @brief Get all the ExceptionAnalysis objects.
             * @return std::vector<std::shared_ptr<ExceptionAnalysis>>
             */
            const std::vector<std::shared_ptr<ExceptionAnalysis>> &gets() const
            {
                return exceptions;
            }

        private:
            std::vector<std::shared_ptr<ExceptionAnalysis>> exceptions;
        };
    } // namespace DEX
} // namespace KUNAI

#endif