#ifndef INCORRECTMETHODID_EXCEPTION_HPP
#define INCORRECTMETHODID_EXCEPTION_HPP

#include <iostream>

namespace exceptions {

		class IncorrectMethodId : public std::exception
		{
		public:
			IncorrectMethodId(const std::string& msg) : _msg(msg) {}

			virtual const char* what() const noexcept override
			{
				return _msg.c_str();
			}

		private:
			std::string _msg;
		};
}

#endif