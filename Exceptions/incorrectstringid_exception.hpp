#ifndef INCORRECTSTRINGID_EXCEPTION_HPP
#define INCORRECTSTRINGID_EXCEPTION_HPP

#include <iostream>

namespace exceptions {

		class IncorrectStringId : public std::exception
		{
		public:
			IncorrectStringId(const std::string& msg) : _msg(msg) {}

			virtual const char* what() const noexcept override
			{
				return _msg.c_str();
			}

		private:
			std::string _msg;
		};
}

#endif