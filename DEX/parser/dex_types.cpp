#include "dex_types.hpp"

namespace KUNAI
{
    namespace DEX
    {
        /***
         * Type class
         */

        /**
         * @brief Constructor of Type base class.
         * @param type: Type of object from Type::type_t enum.
         * @param raw: raw string with the name.
         */
        Type::Type(type_t type, std::string raw)
        {
            this->type = type;
            this->raw = raw;
        }

        /**
         * @brief Type destructor.
         */
        Type::~Type() {}

        /**
         * @brief get raw string from the type object.
         * @return raw: string from the type.
         */
        std::string Type::get_raw()
        {
            return this->raw;
        }

        /***
         * Fundamental class
         */
        /**
         * @brief Constructor of derived Fundamental class.
         * @param f_type: Fundamental::fundamental_t enum value, type of fundamental.
         * @param name: std::string with name of fundamental type.
         */
        Fundamental::Fundamental(fundamental_t f_type,
                                 std::string name)
            : Type(FUNDAMENTAL, name)
        {
            this->f_type = f_type;
            this->name = name;
        }

        /**
         * @brief Fundamental Destructor
         */
        Fundamental::~Fundamental() {}

        /**
         * @brief return Type::type_t of object, in this case FUNDAMENTAL.
         * @return Type::type_t FUNDAMENTAL.
         */
        Fundamental::type_t Fundamental::get_type()
        {
            return FUNDAMENTAL;
        }

        /**
         * @brief return std::string with the FUndamental type.
         * @return std::string "Fundamental"
         */
        std::string Fundamental::print_type()
        {
            return "Fundamental";
        }

        /**
         * @brief Get fundamental type from enum Fundamental::fundamental_t.
         * @return Fundamental::fundamental_t value.
         */
        Fundamental::fundamental_t Fundamental::get_fundamental_type()
        {
            return this->f_type;
        }

        /**
         * @brief return the type of Fundamental of the object.
         * @return std::string with string representation of Fundamental::fundamental_t value.
         */
        std::string Fundamental::print_fundamental_type()
        {
            switch (this->f_type)
            {
            case BOOLEAN:
                return "boolean";
            case BYTE:
                return "byte";
            case CHAR:
                return "char";
            case DOUBLE:
                return "double";
            case FLOAT:
                return "float";
            case INT:
                return "int";
            case LONG:
                return "long";
            case SHORT:
                return "short";
            case VOID:
                return "void";
            }
            return "";
        }

        /**
         * @brief get name of fundamental type.
         * @return std::string with name of fundamental type.
         */
        std::string Fundamental::get_name()
        {
            return this->name;
        }
        /***
         * Class class
         */
        /**
         * @brief Constructor of derived Class class.
         * @param name: std::string with name of Class type.
         */
        Class::Class(std::string name)
            : Type(CLASS, name)
        {
            this->name = name;
        }

        /**
         * @brief Destructor of derived Class class.
         */
        Class::~Class() {}

        /**
         * @brief get Type::type_t enum value for Class object.
         * @return Type::type_t with value CLASS.
         */
        Class::type_t Class::get_type()
        {
            return CLASS;
        }

        /**
         * @brief get Type::type_t enum value as string.
         * @return Type::type_t as string.
         */
        std::string Class::print_type()
        {
            return "Class";
        }

        /**
         * @brief get name of Class type.
         * @return std::string with name of Class type.
         */
        std::string Class::get_name()
        {
            return this->name;
        }

        /***
         * Array class
         */

        /**
         * @brief Constructor of derived Array class.
         * @param array: a std::vector with types inside of the array.
         * @param name: std::string with name of Array type.
         */
        Array::Array(std::vector<Type *> array,
                     std::string raw)
            : Type(ARRAY, raw)
        {
            this->array = array;
        }

        /**
         * @brief Destructor of Array class, destroy all the values inside of the array.
         */
        Array::~Array()
        {
            if (!array.empty())
                delete array[0];
            array.clear();
        }

        /**
         * @brief get type from Type::type_t enum.
         * @return return ARRAY from Type::type_t.
         */
        Array::type_t Array::get_type()
        {
            return ARRAY;
        }

        /**
         * @brief get type from Type::type_t enum as string.
         * @return ARRAY from Type::type_t as string.
         */
        std::string Array::print_type()
        {
            return "Array";
        }

        /**
         * @brief get the vector of the array of types.
         * @return std::vector<Type *> with all the types in the array.
         */
        std::vector<Type *> Array::get_array()
        {
            return array;
        }

        /***
         * Unknown class
         */
        /**
         * @brief Constructor of unrecognized type.
         * @param type: for parent class.
         * @param raw: for parent class.
         */
        Unknown::Unknown(type_t type, std::string raw)
            : Type(type, raw)
        {
        }

        /**
         * @brief Destructor of unrecognized type.
         */
        Unknown::~Unknown() {}

        /**
         * @brief return Type::type_t of UNKNOWN.
         * @return Type::type_t UNKNOWN.
         */
        Unknown::type_t Unknown::get_type()
        {
            return UNKNOWN;
        }

        /**
         * @brief return Type::type_t of UNKNOWN as std::string.
         * @return Type::type_t UNKNOWN as std::string.
         */
        std::string Unknown::print_type()
        {
            return "Unknown";
        }

        /***
         * DexTypes class
         */
        
        /**
         * @brief Constructor of DexTypes class used to parse types from DEX.
         * @param input_file: file where to read the types.
         * @param number_of_types: number of types to parse.
         * @param types_offsets: offset from the file where the types are.
         * @param dex_str: strings from the DEX.
         */
        DexTypes::DexTypes(std::ifstream &input_file,
                           std::uint32_t number_of_types,
                           std::uint32_t types_offsets,
                           std::shared_ptr<DexStrings> dex_str)
        {
            this->number_of_types = number_of_types;
            this->offset = types_offsets;
            this->dex_str = dex_str;

            if (!parse_types(input_file))
                throw exceptions::ParserReadingException("Error reading DEX types");
        }

        /**
         * @brief Destructor of DexTypes class, clear all the Types.
         */
        DexTypes::~DexTypes()
        {
            if (!types.empty())
                types.clear();
        }

        /**
         * @brief Get a type by a given type id as it appears in DEX.
         * @param type_id: type to retrieve by its id.
         * @return Type* object.
         */
        Type *DexTypes::get_type_by_id(std::uint32_t type_id)
        {
            return types[type_id];
        }

        /**
         * @brief Get a type by a given position from the map.
         * @param pos: position of the Type in the map.
         * @return Type* object.
         */
        Type *DexTypes::get_type_from_order(std::uint32_t pos)
        {
            size_t i = pos;

            for (auto it = types.begin(); it != types.end(); it++)
            {
                if (i-- == 0)
                    return it->second;
            }

            return nullptr;
        }

        /**
         * @brief Return the number of types in map.
         * @return number of type objects.
         */
        std::uint32_t DexTypes::get_number_of_types()
        {
            return number_of_types;
        }

        /**
         * @brief Return the offset where Types start.
         * @return offset where Types are in DEX file.
         */
        std::uint32_t DexTypes::get_offset()
        {
            return offset;
        }

        /**
         * Private methods
         */
        /**
         * @brief Method to create a Type given a raw name.
         * @param name: raw name of the type.
         * @return Type* object created.
         */
        Type *DexTypes::parse_type(std::string name)
        {
            Type *type;
            if (name.length() == 1)
            {
                if (name == "Z")
                    type = new Fundamental(Fundamental::BOOLEAN, name);
                else if (name == "B")
                    type = new Fundamental(Fundamental::BYTE, name);
                else if (name == "C")
                    type = new Fundamental(Fundamental::CHAR, name);
                else if (name == "D")
                    type = new Fundamental(Fundamental::DOUBLE, name);
                else if (name == "F")
                    type = new Fundamental(Fundamental::FLOAT, name);
                else if (name == "I")
                    type = new Fundamental(Fundamental::INT, name);
                else if (name == "J")
                    type = new Fundamental(Fundamental::LONG, name);
                else if (name == "S")
                    type = new Fundamental(Fundamental::SHORT, name);
                else if (name == "V")
                    type = new Fundamental(Fundamental::VOID, name);
            }
            else if (name[0] == 'L')
                type = new Class(name);
            else if (name[0] == '[')
            {
                std::vector<Type *> aux_vec;
                Type *aux_type;
                aux_type = parse_type(name.substr(1, name.length() - 1));
                aux_vec.push_back(aux_type);
                type = new Array(aux_vec, name);
            }
            else
                type = new Unknown(Type::UNKNOWN, name);

            return type;
        }

        /**
         * @brief parser method for DEX types. It will make use of method parse_type.
         * @param input_file: file where to read the types.
         * @return true if everything was okay, false in other way.
         */
        bool DexTypes::parse_types(std::ifstream &input_file)
        {
            auto current_offset = input_file.tellg();
            size_t i;
            std::uint32_t type_id;
            Type *type;

            // move to offset where are the string ids
            input_file.seekg(offset);

            for (i = 0; i < number_of_types; i++)
            {
                if (!KUNAI::read_data_file<std::uint32_t>(type_id, sizeof(std::uint32_t), input_file))
                    return false;

                if (type_id >= dex_str->get_number_of_strings())
                    throw exceptions::IncorrectStringId("Error reading types type_id out of string bound");

                type = this->parse_type(*dex_str->get_string_from_order(type_id));

                types.insert(std::pair<std::uint32_t, Type *>(type_id, type));
            }

            input_file.seekg(current_offset);
            return true;
        }

        std::ostream &operator<<(std::ostream &os, const DexTypes &entry)
        {
            size_t i = 0;
            os << std::hex;
            os << std::setw(30) << std::left << std::setfill(' ') << "=========== DEX Types ===========" << std::endl;
            for (auto it = entry.types.begin(); it != entry.types.end(); it++)
            {
                os << std::left << std::setfill(' ') << "Type (" << std::dec << i++ << std::hex << "): " << it->first << "-> \"" << it->second->get_raw() << "\"" << std::endl;
            }

            return os;
        }

        std::fstream &operator<<(std::fstream &fos, const DexTypes &entry)
        {
            std::stringstream stream;

            stream << std::hex;
            stream << "<types>" << std::endl;
            for (auto it = entry.types.begin(); it != entry.types.end(); it++)
            {
                stream << "\t<type>" << std::endl;
                stream << "\t\t<id>" << it->first << "</id>" << std::endl;
                stream << "\t\t<value>" << it->second->get_raw() << "</value>" << std::endl;
                stream << "\t</type>" << std::endl;
            }
            stream << "</types>" << std::endl;

            fos.write(stream.str().c_str(), stream.str().size());

            return fos;
        }
    }
}