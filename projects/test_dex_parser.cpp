#include <iostream>
#include <memory>

#include <spdlog/spdlog.h>
// necessary to work with dex import the main DEX object
#include "KUNAI/DEX/dex.hpp"


int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "USAGE: " << argv[0] << " <dex_file>" << std::endl;
        return 1;
    }

    /**
     * This code should be done inside of a file preprocessing
     * module from the tool, but it's useful enough for testing.
     */
    std::ifstream dex_file;

    dex_file.open(argv[1], std::ios::binary);

    auto fsize = dex_file.tellg();
    dex_file.seekg(0, std::ios::end);
    fsize = dex_file.tellg() - fsize;
    dex_file.seekg(0);

    /**
     * For starting working with DEX we will need to create a base
     * DEX object, this object will contain the parser, the disassembler
     * and the analysis object. For performance improvement, DEX
     * object will just parse the file, it will not disassembly neither
     * will analyze the DEX, that's something you can decide to do.
     */

    // Set the logging level in spdlog, we set to debug
    spdlog::set_level(spdlog::level::debug);

    // `DEX' class is inside of namespace `DEX' inside of namespace `KUNAI'
    // you have a couple of methods in KUNAI::DEX one to return a dex unique_ptr
    // and the other to return a dex shared_ptr. As we have both methods we can
    // assign it with "auto" from C++
    auto dex_object = KUNAI::DEX::get_unique_dex_object(dex_file, fsize); 

    // okay, now we have a parsed dex file, and a `dex' object
    /**
     * The `dex' object can give us different objects:
     *     - a parser object (used here): useful for getting information about the DEX file.
     *     - a dalvik opcodes object: used internally to get information about the opcodes.
     *     - a disassembler to get the disassembled dex.
     *     - (ToDo) an analysis object to get the different analysis from DEX.
     */

    // let's going to take the parser and work with it.
    // we can check in any case if the parsing was correct
    // so in any other case reported and leave.

    if (!dex_object->get_parsing_correct())
    {
        std::cerr << "[!] Error parsing dex object, exiting program..." << std::endl;
        return 1;
    }
    
    auto dex_parser = dex_object->get_parser();

    /**
     * The dex parser object contains different objects that represent each
     * part of dex file, so we have the next:
     *  - DexHeader: header from DEX file, it contains a structure with different
     *    values like offsets or sizes, you know, that kind of "WHAT'S UP MY GLIP GLOPS!"
     *  - DexStrings: all the strings from the DEX file, strings are nuts and bolts from
     *    DEX everything are strings, even the strings are here as strings.
     *  - DexTypes: commonly the code has different types, from the basic types from the
     *    language (int, char, double, those things).
     *  - DexProtos: prototypes from methods and fields.
     *  - DexFields: global fields declared in Java code, are composed of class name, type
     *    and name of the field.
     *  - DexMethods: All the methods in the dex file, from each one, a MethodID can be
     *    obtained and get more information about the methods.
     *  - DexClasses: All the classes from the dex file, different objects can be accessed
     *    from here that represent all the encoded method and encoded fields from the class.
     */

    // check dex header version
    std::cout << "Dex header version number: " << dex_parser->get_header_version() << std::endl;

    // or directly as string
    std::cout << dex_parser->get_header_version_str() << std::endl;

    // do we want to print all the ClassDef?
    auto & class_defs = dex_parser->get_classes_def_item();

    std::cout << "[ClassDef] classes: " << std::endl;
    
    for (auto & class_def : class_defs)
    {
        if (class_def->get_class_idx() == nullptr)
            continue;
        std::cout << "\tClass name: " << class_def->get_class_idx()->get_name() << std::endl;
        std::cout << "\tSource file name: ";
        if (class_def->get_source_file_idx())
            std::cout << *class_def->get_source_file_idx() << std::endl;
        else
            std::cout << "None" << std::endl;
        std::cout << std::endl;
    }

    // Do we want to print all the MethodIDs?
    auto & method_ids = dex_parser->get_methods_id_item();

    std::cout << "[MethodId] methods: " << std::endl;
    for (auto & method : method_ids)
    {
        std::cout << "\tMethod name: " << *method->get_method_name() << std::endl;
        /**
         * get_method_class returns a Type* that can be
         * enum type_t {
                FUNDAMENTAL,
                CLASS,
                ARRAY,
                UNKNOWN
            };
        * this time as it's returned by a method, and its is
        * class we can cast it to KUNAI::DEX::Class*
        */
        std::cout << "\tMethod class: " << method->get_method_class()->get_raw() << std::endl;
        std::cout << std::endl;
    }

    /**
     * Let's going to explore the header...
     * the header is just a basic structure inside of DexHeader.
     */
    auto dex_header = dex_parser->get_header();

    std::cout << "File size: " << dex_header->get_dex_header().file_size << std::endl;
    
    std::cout << "Magic: ";
    
    for (size_t i = 0; i < 8; i++)
        std::cout << static_cast<std::uint32_t>(dex_header->get_dex_header().magic[i]) << " ";
    std::cout << std::endl;

    std::cout << "Signature: ";
    for (size_t i = 0; i < 20; i++)
        std::cout << static_cast<std::uint32_t>(dex_header->get_dex_header().signature[i]) << " ";
    std::cout << std::endl;

    /**
     * Strings can be obtained by position, by offset, or a vector
     * with all the strings, if you get it by offset or by order
     * you receive a String pointer (*) so use it properly. I recommend
     * to use it by position because by offset is used internally.
     */
    auto strings = dex_parser->get_strings();
    std::cout << "Strings: " << std::endl;
    for (size_t i = 0; i < strings->get_number_of_strings(); i++)
        std::cout << i << " " << *strings->get_string_from_order(i) << std::endl;
    std::cout << std::endl;

    /**
     * Types are something interesting, because DexTypes always return
     * a base class "Type", but some classes are derived from it:
     * Fundamental (a fundamental type like boolean, byte, char, double
     * float, int, long, short or void), Class (which represent any kind
     * of object), Array which is a class to hold a vector of different types
     * or Unknown not recognized type.
     */
    auto types = dex_parser->get_types();

    std::cout << "Types: " << std::endl;

    for (size_t i = 0; i < types->get_number_of_types(); i++)
    {
        auto type = types->get_type_from_order(i);

        // we can ask which type it is to do the correct cast
        switch (type->get_type())
        {
        case KUNAI::DEX::Type::FUNDAMENTAL:
            type = reinterpret_cast<KUNAI::DEX::Fundamental*>(type);
            break;
        case KUNAI::DEX::Type::CLASS:
            type = reinterpret_cast<KUNAI::DEX::Class*>(type);
            break;
        case KUNAI::DEX::Type::ARRAY:
            type = reinterpret_cast<KUNAI::DEX::Array*>(type);
            break;
        case KUNAI::DEX::Type::UNKNOWN:
            type = reinterpret_cast<KUNAI::DEX::Unknown*>(type);
            break;
        }

        // all of those has the method "get_raw" to print the raw type.
        std::cout << "Type raw: " << type->get_raw() << std::endl;
    }

    auto & methods = dex_parser->get_methods_id_item();

    std::cout << "Methods: " << std::endl;
    for (size_t i = 0; i < methods.size(); i++)
    {
        auto & method = methods[i];

        std::cout << "Class: " << method->get_method_class()->get_raw() << std::endl;
        std::cout << "Method: " << *method->get_method_name() << std::endl;
        std::cout << "Description: " << method->get_method_prototype()->get_proto_str() << std::endl;
        std::cout << std::endl;

    }

    /**
     * Do you just want to show a pretty print of the whole header?
     * In that case this C++ overloaded operator is for you, you can
     * use the dex_parser directly to be printed.
     */
    /*
    std::cout << "Print of all the header: " << std::endl;
    std::cout << *dex_parser;
    std::cout << std::endl;
    */
}

