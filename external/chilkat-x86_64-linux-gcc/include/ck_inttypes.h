

#ifndef CK_INT_TYPES_H
#define CK_INT_TYPES_H
	
#if defined(__MINGW32__)
#include <inttypes.h>

#elif (defined(BORLAND) || defined(BCCXE2) || defined(BCCXE3) || defined(BCCXE4)) && !defined(BCC55)
#include  <stdint.h>

#elif defined(WINAPI_FAMILY) && WINAPI_FAMILY == WINAPI_FAMILY_PHONE_APP
#include <cstdint>

#elif defined(CK_WINDOWS) || defined(WIN32) || defined(_WIN32) || defined(_WINDOWS) || defined(WINCE) || defined(__BORLANDC__) || defined(__BCPLUSPLUS__)

#if !defined(_STDINT)
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned __int64 uint64_t;
#endif

#else


// All others...
#include <inttypes.h>

#if defined(CHILKATSWIG) && (defined(CK_LINUX) || defined(MACOSX) || defined(CK_SOLARIS))

#define _INT64_TYPEDEF_DEFINED	
typedef int64_t __int64;	
typedef uint64_t ulong64;	


#elif !defined(CHILKATSWIG)
		
#ifndef _INT64_TYPEDEF_DEFINED
#define _INT64_TYPEDEF_DEFINED	
typedef int64_t __int64;	
typedef uint64_t ulong64;	
#endif

#endif // CHILKATSWIG

#endif

#endif  // CK_INT_TYPES_H

