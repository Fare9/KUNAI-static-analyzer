/**
 * @file dex_parents.hpp
 * @author @Farenain
 * @brief ParentClass & ParentMethod class used to avoid the use of std::any 
 *        in ClassAnalysis & MethodAnalysis classes
 * 
 */

#ifndef DEX_PARENTS_HPP
#define DEX_PARENTS_HPP

namespace KUNAI
{
    namespace DEX
    {
        class ParentClass
        {
        public:
            enum type_t
            {
                INTERNAL_CLASS_T,
                EXTERNAL_CLASS_T,
            };
            ParentClass(type_t type) { this->type = type; }

            bool is_external() { return type == EXTERNAL_CLASS_T; }
            bool is_internal() { return type == INTERNAL_CLASS_T; }

            virtual ~ParentClass() = default;
        private:
            type_t type;
        };

        class ParentMethod
        {
        public:
            enum type_t
            {
                INTERNAL_METHOD_T,
                EXTERNAL_METHOD_T
            };

            ParentMethod(type_t type){ this->type = type; }

            bool is_external() { return type == EXTERNAL_METHOD_T; }
            bool is_internal() { return type == INTERNAL_METHOD_T; }

            virtual ~ParentMethod() = default;
        private:
            type_t type;
        };
    }
}

#endif