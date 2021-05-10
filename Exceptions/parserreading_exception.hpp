#ifndef PARSERREADING_EXCEPTION_HPP
#define PARSERREADING_EXCEPTION_HPP

#include <iostream>

namespace exceptions {

		class ParserReadingException : public std::exception
		{
		public:
			ParserReadingException(const std::string& msg) : _msg(msg) {}

			virtual const char* what() const noexcept override
			{
				return _msg.c_str();
			}

		private:
			std::string _msg;
		};
}

#endif