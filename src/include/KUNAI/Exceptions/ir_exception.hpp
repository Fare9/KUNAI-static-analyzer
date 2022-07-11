#ifndef IR_EXCEPTION_HPP
#define IR_EXCEPTION_HPP

#include <iostream>

namespace exceptions {

    class IRException : public std::exception
    {
    public:
        IRException(const std::string& msg) : _msg(msg) {}

        virtual const char* what() const noexcept override
        {
            return _msg.c_str();
        }
        
    private:
        std::string _msg;
    };

}

#endif