// CkLog.h: interface for the CkLog class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkLog_H
#define _CkLog_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkByteData;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkLog
class CK_VISIBLE_PUBLIC CkLog  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkLog(const CkLog &);
	CkLog &operator=(const CkLog &);

    public:
	CkLog(void);
	virtual ~CkLog(void);

	static CkLog *createNew(void);
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
	void Clear(const char *initialTag);


	// Enters a new context labelled with the given tag. Must be paired with a matching
	// call to LeaveContext.
	void EnterContext(const char *tag);


	// Leaves the current context. A context that is entered and exited without any
	// logging within the context is automatically removed from the log. (To say it
	// another way: Empty contexts are automaticallly removed from the log upon leaving
	// the context.)
	void LeaveContext(void);


	// Adds a tagged message to the log (i.e. a name/value pair).
	void LogData(const char *tag, const char *message);


	// Logs binary data in base64 format.
	void LogDataBase64(const char *tag, CkByteData &data);


	// Logs binary data in base64 format.
	void LogDataBase64_2(const char *tag, const void *pByteData, unsigned long szByteData);


	// Logs binary data in hex format.
	void LogDataHex(const char *tag, CkByteData &data);


	// Logs binary data in hex format.
	void LogDataHex2(const char *tag, const void *pByteData, unsigned long szByteData);


	// Logs a string, but only up to the 1st maxNumChars characters of the string.
	void LogDataMax(const char *tag, const char *message, int maxNumChars);


	// Logs the current date/time in RFC822 format. If gmt is true, then the GMT/UTC
	// time is logged. Otherwise it is the local time.
	void LogDateTime(const char *tag, bool gmt);


	// Logs an error within the current context.
	void LogError(const char *message);


	// Logs the hash of binary data in hex format. The tag can be SHA1, SHA256,
	// SHA384, SHA512, or MD5.
	void LogHash2(const char *tag, const char *hashAlg, const void *pByteData, unsigned long szByteData);


	// Logs an informational message within the current context.
	void LogInfo(const char *message);


	// Logs an integer.
	void LogInt(const char *tag, int value);


	// Logs a 64-bit integer.
	void LogInt64(const char *tag, __int64 value);


	// Logs the current time in HH:MM:SS:mmm format.
	void LogTimestamp(const char *tag);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
