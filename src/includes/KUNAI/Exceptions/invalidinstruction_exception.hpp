#ifndef INVALIDINSTRUCTION_EXCEPTION_HPP
#define INVALIDINSTRUCTION_EXCEPTION_HPP

#include <iostream>

namespace exceptions {

		class InvalidInstruction : public std::exception
		{
		public:
			InvalidInstruction(const std::string& msg) : _msg(msg) {}

			virtual const char* what() const noexcept override
			{
				return _msg.c_str();
			}

		private:
			std::string _msg;
		};
}

#endif