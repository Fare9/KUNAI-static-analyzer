#ifndef INCORRECTDEXFILE_EXCEPTION_HPP
#define INCORRECTDEXFILE_EXCEPTION_HPP

#include <iostream>

namespace exceptions {

		class IncorrectDexFile : public std::exception
		{
		public:
			IncorrectDexFile(const std::string& msg) : _msg(msg) {}

			virtual const char* what() const noexcept override
			{
				return _msg.c_str();
			}

		private:
			std::string _msg;
		};
}

#endif