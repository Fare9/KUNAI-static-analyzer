#ifndef OUTOFBOUND_EXCEPTION_HPP
#define OUTOFBOUND_EXCEPTION_HPP

#include <iostream>

namespace exceptions {

		class OutOfBoundException : public std::exception
		{
		public:
			OutOfBoundException(const std::string& msg) : _msg(msg) {}

			virtual const char* what() const noexcept override
			{
				return _msg.c_str();
			}

		private:
			std::string _msg;
		};
}

#endif