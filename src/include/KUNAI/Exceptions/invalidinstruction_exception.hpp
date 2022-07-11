#ifndef INVALIDINSTRUCTION_EXCEPTION_HPP
#define INVALIDINSTRUCTION_EXCEPTION_HPP

#include <iostream>

namespace exceptions {

		class InvalidInstruction : public std::exception
		{
		public:
			InvalidInstruction(const std::string& msg, std::uint64_t instr_size) : _msg(msg), _instr_size(instr_size) {}

			virtual const char* what() const noexcept override
			{
				return _msg.c_str();
			}

			virtual const std::uint64_t size() const noexcept
			{
				return _instr_size;
			}

		private:
			std::string _msg;
			std::uint64_t _instr_size;
		};
}

#endif