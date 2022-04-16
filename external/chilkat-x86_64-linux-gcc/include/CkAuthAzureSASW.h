// CkAuthAzureSASW.h: interface for the CkAuthAzureSASW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkAuthAzureSASW_H
#define _CkAuthAzureSASW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkWideCharBase.h"




#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkAuthAzureSASW
class CK_VISIBLE_PUBLIC CkAuthAzureSASW  : public CkWideCharBase
{
	

	private:
	
	// Don't allow assignment or copying these objects.
	CkAuthAzureSASW(const CkAuthAzureSASW &);
	CkAuthAzureSASW &operator=(const CkAuthAzureSASW &);

    public:
	CkAuthAzureSASW(void);
	virtual ~CkAuthAzureSASW(void);

	

	static CkAuthAzureSASW *createNew(void);
	

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// This is the signing key (access key) that must be kept private. It is a base64
	// string such as
	// "abdTvCZFFoWUyre6erlNN+IOb9qhXgDsyhrxmZvpmxqFDwpl9oD0X9Fy0hIQa6L5UohznRLmkCtUYySO
	// 4Y2eaw=="
	void get_AccessKey(CkString &str);
	// This is the signing key (access key) that must be kept private. It is a base64
	// string such as
	// "abdTvCZFFoWUyre6erlNN+IOb9qhXgDsyhrxmZvpmxqFDwpl9oD0X9Fy0hIQa6L5UohznRLmkCtUYySO
	// 4Y2eaw=="
	const wchar_t *accessKey(void);
	// This is the signing key (access key) that must be kept private. It is a base64
	// string such as
	// "abdTvCZFFoWUyre6erlNN+IOb9qhXgDsyhrxmZvpmxqFDwpl9oD0X9Fy0hIQa6L5UohznRLmkCtUYySO
	// 4Y2eaw=="
	void put_AccessKey(const wchar_t *newVal);

	// Defines the format of the string to sign.
	// 
	// The format is specified as a comma-separated list of names. For example:
	// 
	// signedpermissions,signedstart,signedexpiry,canonicalizedresource,signedidentifier,signedIP,signedProtocol,signedversion,rscc,rscd,rsce,rscl,rsct
	// This will result in an actual string-to-sign that is composed of the values for
	// each name separated by newline (LF) chars. For example:
	// signedpermissions + "\n" +  
	// signedstart + "\n" +  
	// signedexpiry + "\n" +  
	// canonicalizedresource + "\n" +  
	// signedidentifier + "\n" +  
	// signedIP + "\n" +  
	// signedProtocol + "\n" +  
	// signedversion + "\n" +  
	// rscc + "\n" +  
	// rscd + "\n" +  
	// rsce + "\n" +  
	// rscl + "\n" +  
	// rsct
	void get_StringToSign(CkString &str);
	// Defines the format of the string to sign.
	// 
	// The format is specified as a comma-separated list of names. For example:
	// 
	// signedpermissions,signedstart,signedexpiry,canonicalizedresource,signedidentifier,signedIP,signedProtocol,signedversion,rscc,rscd,rsce,rscl,rsct
	// This will result in an actual string-to-sign that is composed of the values for
	// each name separated by newline (LF) chars. For example:
	// signedpermissions + "\n" +  
	// signedstart + "\n" +  
	// signedexpiry + "\n" +  
	// canonicalizedresource + "\n" +  
	// signedidentifier + "\n" +  
	// signedIP + "\n" +  
	// signedProtocol + "\n" +  
	// signedversion + "\n" +  
	// rscc + "\n" +  
	// rscd + "\n" +  
	// rsce + "\n" +  
	// rscl + "\n" +  
	// rsct
	const wchar_t *stringToSign(void);
	// Defines the format of the string to sign.
	// 
	// The format is specified as a comma-separated list of names. For example:
	// 
	// signedpermissions,signedstart,signedexpiry,canonicalizedresource,signedidentifier,signedIP,signedProtocol,signedversion,rscc,rscd,rsce,rscl,rsct
	// This will result in an actual string-to-sign that is composed of the values for
	// each name separated by newline (LF) chars. For example:
	// signedpermissions + "\n" +  
	// signedstart + "\n" +  
	// signedexpiry + "\n" +  
	// canonicalizedresource + "\n" +  
	// signedidentifier + "\n" +  
	// signedIP + "\n" +  
	// signedProtocol + "\n" +  
	// signedversion + "\n" +  
	// rscc + "\n" +  
	// rscd + "\n" +  
	// rsce + "\n" +  
	// rscl + "\n" +  
	// rsct
	void put_StringToSign(const wchar_t *newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Clears all params set by the methods SetNonTokenParam and SetTokenParam.
	void Clear(void);

	// Generates and returns the SAS token based on the defined StringToSign and
	// params.
	bool GenerateToken(CkString &outStr);
	// Generates and returns the SAS token based on the defined StringToSign and
	// params.
	const wchar_t *generateToken(void);

	// Adds a non-token parameter name/value. This is a value that is included in the
	// string to sign, but is NOT included as a parameter in the Authorization header.
	bool SetNonTokenParam(const wchar_t *name, const wchar_t *value);

	// Adds a token parameter name/value. This is a value that is included in the
	// string to sign, and is also included as a parameter in the Authorization header.
	bool SetTokenParam(const wchar_t *name, const wchar_t *authParamName, const wchar_t *value);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
