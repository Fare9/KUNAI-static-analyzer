#ifndef INCORRECTVALUE_EXCEPTION_HPP
#define INCORRECTVALUE_EXCEPTION_HPP

#include <iostream>

namespace exceptions {

		class IncorrectValue : public std::exception
		{
		public:
			IncorrectValue(const std::string& msg) : _msg(msg) {}

			virtual const char* what() const noexcept override
			{
				return _msg.c_str();
			}

		private:
			std::string _msg;
		};
}

#endif