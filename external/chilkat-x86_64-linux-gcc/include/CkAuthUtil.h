// CkAuthUtil.h: interface for the CkAuthUtil class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkAuthUtil_H
#define _CkAuthUtil_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"




#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkAuthUtil
class CK_VISIBLE_PUBLIC CkAuthUtil  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkAuthUtil(const CkAuthUtil &);
	CkAuthUtil &operator=(const CkAuthUtil &);

    public:
	CkAuthUtil(void);
	virtual ~CkAuthUtil(void);

	static CkAuthUtil *createNew(void);
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
	// Generates a Walmart authentication signature for Walmart REST API calls. Returns
	// a JSON string that contains both the WM_SEC_AUTH_SIGNATURE and WM_SEC.TIMESTAMP.
	bool WalmartSignature(const char *requestUrl, const char *consumerId, const char *privateKey, const char *requestMethod, CkString &outStr);

	// Generates a Walmart authentication signature for Walmart REST API calls. Returns
	// a JSON string that contains both the WM_SEC_AUTH_SIGNATURE and WM_SEC.TIMESTAMP.
	const char *walmartSignature(const char *requestUrl, const char *consumerId, const char *privateKey, const char *requestMethod);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
