#ifndef APKUNZIP_EXCEPTION_HPP
#define APKUNZIP_EXCEPTION_HPP

#include <iostream>

namespace exceptions {
    class ApkUnzipException : public std::exception
    {
    public:
        ApkUnzipException(const std::string& msg) : _msg(msg) {}

        virtual const char* what() const noexcept override
        {
            return _msg.c_str();
        }
    
    private:
        std::string _msg;
    };
}

#endif