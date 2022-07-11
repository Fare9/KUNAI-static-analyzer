#ifndef LIFTER_EXCEPTION_HPP
#define LIFTER_EXCEPTION_HPP

#include <iostream>

namespace exceptions {

    class LifterException : public std::exception
    {
    public:
        LifterException(const std::string& msg) : _msg(msg) {}

        virtual const char* what() const noexcept override
        {
            return _msg.c_str();
        }
        
    private:
        std::string _msg;
    };

}

#endif