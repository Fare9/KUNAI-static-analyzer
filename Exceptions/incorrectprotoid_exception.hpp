#ifndef INCORRECTPROTOID_EXCEPTION_HPP
#define INCORRECTPROTOID_EXCEPTION_HPP

#include <iostream>

namespace exceptions {

		class IncorrectProtoId : public std::exception
		{
		public:
			IncorrectProtoId(const std::string& msg) : _msg(msg) {}

			virtual const char* what() const noexcept override
			{
				return _msg.c_str();
			}

		private:
			std::string _msg;
		};
}

#endif