// CkLogW.h: interface for the CkLogW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkLogW_H
#define _CkLogW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkWideCharBase.h"

class CkByteData;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkLogW
class CK_VISIBLE_PUBLIC CkLogW  : public CkWideCharBase
{
	

	private:
	
	// Don't allow assignment or copying these objects.
	CkLogW(const CkLogW &);
	CkLogW &operator=(const CkLogW &);

    public:
	CkLogW(void);
	virtual ~CkLogW(void);

	

	static CkLogW *createNew(void);
	

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------


	// ----------------------
	// Methods
	// ----------------------
	// Clears the log. The initialTag is the initial top-level context tag for the new log.
	void Clear(const wchar_t *initialTag);

	// Enters a new context labelled with the given tag. Must be paired with a matching
	// call to LeaveContext.
	void EnterContext(const wchar_t *tag);

	// Leaves the current context. A context that is entered and exited without any
	// logging within the context is automatically removed from the log. (To say it
	// another way: Empty contexts are automaticallly removed from the log upon leaving
	// the context.)
	void LeaveContext(void);

	// Adds a tagged message to the log (i.e. a name/value pair).
	void LogData(const wchar_t *tag, const wchar_t *message);

	// Logs binary data in base64 format.
	void LogDataBase64(const wchar_t *tag, CkByteData &data);

	// Logs binary data in base64 format.
	void LogDataBase64_2(const wchar_t *tag, const void *pByteData, unsigned long szByteData);

	// Logs binary data in hex format.
	void LogDataHex(const wchar_t *tag, CkByteData &data);

	// Logs binary data in hex format.
	void LogDataHex2(const wchar_t *tag, const void *pByteData, unsigned long szByteData);

	// Logs a string, but only up to the 1st maxNumChars characters of the string.
	void LogDataMax(const wchar_t *tag, const wchar_t *message, int maxNumChars);

	// Logs the current date/time in RFC822 format. If gmt is true, then the GMT/UTC
	// time is logged. Otherwise it is the local time.
	void LogDateTime(const wchar_t *tag, bool gmt);

	// Logs an error within the current context.
	void LogError(const wchar_t *message);

	// Logs the hash of binary data in hex format. The tag can be SHA1, SHA256,
	// SHA384, SHA512, or MD5.
	void LogHash2(const wchar_t *tag, const wchar_t *hashAlg, const void *pByteData, unsigned long szByteData);

	// Logs an informational message within the current context.
	void LogInfo(const wchar_t *message);

	// Logs an integer.
	void LogInt(const wchar_t *tag, int value);

	// Logs a 64-bit integer.
	void LogInt64(const wchar_t *tag, __int64 value);

	// Logs the current time in HH:MM:SS:mmm format.
	void LogTimestamp(const wchar_t *tag);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
