#include "dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {
        
        /**
         * @brief StringAnalysis constructor, set the value and original value.
         * @param value: std::string* pointer to one of the program strings.
         * @return void
         */
        StringAnalysis::StringAnalysis(std::string *value)
        {
            this->value = value;
            this->orig_value = value;        
        }

        /**
         * @brief StringAnalysis destructor, clear vector.
         * @return void
         */
        StringAnalysis::~StringAnalysis()
        {
            if (!xreffrom.empty())
                xreffrom.clear();
        }

        /**
         * @brief Add a reference of a method that reads current string.
         * @param class_object: std::shared_ptr<ClassAnalysis> class of the method that reads current string.
         * @param method_object: std::shared_ptr<MethodAnalysis> method that reads current string.
         * @param offset: std::uint64_t offset where call is done.
         * @return void
         */
        void StringAnalysis::add_xref_from(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset)
        {
            xreffrom.push_back({class_object, method_object, offset});
        }

        /**
         * @brief get the methods where the string is read, with or without offset.
         * @param withoffset: bool return offsets or not.
         * @return if withoffset == true `std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>>'
         *         if withoffset == false `std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>>>'
         */
        std::any StringAnalysis::get_xref_from(bool withoffset)
        {
            std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>>> xreffrom_no_offset;

            if (withoffset)
                return xreffrom;
            else
            {
                for (auto it = xreffrom.begin(); it != xreffrom.end(); it++)
                {
                    xreffrom_no_offset.push_back({std::get<0>(*it), std::get<1>(*it)});
                }

                return xreffrom_no_offset;
            }
        }

        /**
         * @brief set the value from StringAnalysis.
         * @param value: std::string* new value to put in object.
         * @return void
         */
        void StringAnalysis::set_value(std::string *value)
        {
            this->value = value;
        }

        /**
         * @brief get the value from the object.
         * @return std::string*
         */
        std::string* StringAnalysis::get_value()
        {
            return value;
        }

        /**
         * @brief get if the value has been overwritten or not.
         * @return bool
         */
        bool StringAnalysis::is_overwritten()
        {
            return value == orig_value;
        }
    } // namespace DEX
} // namespace KUNAI
