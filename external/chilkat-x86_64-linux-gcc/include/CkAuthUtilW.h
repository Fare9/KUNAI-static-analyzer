// CkAuthUtilW.h: interface for the CkAuthUtilW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkAuthUtilW_H
#define _CkAuthUtilW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkWideCharBase.h"




#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkAuthUtilW
class CK_VISIBLE_PUBLIC CkAuthUtilW  : public CkWideCharBase
{
	

	private:
	
	// Don't allow assignment or copying these objects.
	CkAuthUtilW(const CkAuthUtilW &);
	CkAuthUtilW &operator=(const CkAuthUtilW &);

    public:
	CkAuthUtilW(void);
	virtual ~CkAuthUtilW(void);

	

	static CkAuthUtilW *createNew(void);
	

	
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
	bool WalmartSignature(const wchar_t *requestUrl, const wchar_t *consumerId, const wchar_t *privateKey, const wchar_t *requestMethod, CkString &outStr);
	// Generates a Walmart authentication signature for Walmart REST API calls. Returns
	// a JSON string that contains both the WM_SEC_AUTH_SIGNATURE and WM_SEC.TIMESTAMP.
	const wchar_t *walmartSignature(const wchar_t *requestUrl, const wchar_t *consumerId, const wchar_t *privateKey, const wchar_t *requestMethod);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
