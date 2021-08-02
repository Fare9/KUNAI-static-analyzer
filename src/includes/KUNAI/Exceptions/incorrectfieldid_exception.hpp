#ifndef INCORRECTFIELDID_EXCEPTION_HPP
#define INCORRECTFIELDID_EXCEPTION_HPP

#include <iostream>

namespace exceptions {

		class IncorrectFieldId : public std::exception
		{
		public:
			IncorrectFieldId(const std::string& msg) : _msg(msg) {}

			virtual const char* what() const noexcept override
			{
				return _msg.c_str();
			}

		private:
			std::string _msg;
		};
}

#endif