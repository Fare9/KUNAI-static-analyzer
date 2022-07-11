#ifndef DISASSEMBLER_EXCEPTION_HPP
#define DISASSEMBLER_EXCEPTION_HPP

#include <iostream>

namespace exceptions {

		class DisassemblerException : public std::exception
		{
		public:
			DisassemblerException(const std::string& msg) : _msg(msg) {}

			virtual const char* what() const noexcept override
			{
				return _msg.c_str();
			}

		private:
			std::string _msg;
		};
}

#endif