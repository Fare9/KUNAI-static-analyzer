#ifndef INCORRECTTYPEID_EXCEPTION_HPP
#define INCORRECTTYPEID_EXCEPTION_HPP

#include <iostream>

namespace exceptions {

		class IncorrectTypeId : public std::exception
		{
		public:
			IncorrectTypeId(const std::string& msg) : _msg(msg) {}

			virtual const char* what() const noexcept override
			{
				return _msg.c_str();
			}

		private:
			std::string _msg;
		};
}

#endif