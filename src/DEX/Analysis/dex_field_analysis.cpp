#include "dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {
        /**
         * FieldAnalysis methods
         */

        /**
         * @brief Constructor of FieldAnalysis class.
         * @param field: FieldID pointer base of the class.
         * @return void
         */
        FieldAnalysis::FieldAnalysis(std::shared_ptr<EncodedField> field)
        {
            this->field = field;
        }

        /**
         * @brief Destructor of FieldAnalysis class.
         * @return void
         */
        FieldAnalysis::~FieldAnalysis() {}

        /**
         * @brief retrieve name of Field.
         * @return std::string
         */
        std::string FieldAnalysis::name()
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
        void FieldAnalysis::add_xref_read(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset)
        {
            xrefread.push_back({class_object, method_object, offset});
        }

        /**
         * @brief Add a xref where this field is written.
         * @param class_object: class where this field is written.
         * @param method_object: method where this field is written.
         * @param offset: instruction offset where field is written.
         * @return void
         */
        void FieldAnalysis::add_xref_write(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset)
        {
            xrefwrite.push_back({class_object, method_object, offset});
        }

        /**
         * @brief Get all the read cross references from the field.
         * 
         * @return const std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>>& 
         */
        const std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>>& FieldAnalysis::get_xref_read()
        {
            return xrefread;
        }

        /**
         * @brief Get all the write cross references from the field.
         * 
         * @return const std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>>& 
         */
        const std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>>& FieldAnalysis::get_xref_write()
        {
            return xrefwrite;
        }

        /**
         * @brief return the FieldID pointer.
         * @return FieldID*
         */
         std::shared_ptr<EncodedField> FieldAnalysis::get_field()
        {
            return field;
        }

    }
}