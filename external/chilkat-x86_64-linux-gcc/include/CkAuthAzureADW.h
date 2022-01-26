// CkAuthAzureADW.h: interface for the CkAuthAzureADW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkAuthAzureADW_H
#define _CkAuthAzureADW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacksW.h"

class CkTaskW;
class CkSocketW;
class CkBaseProgressW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkAuthAzureADW
class CK_VISIBLE_PUBLIC CkAuthAzureADW  : public CkClassWithCallbacksW
{
	private:
	bool m_cbOwned;

	private:
	
	// Don't allow assignment or copying these objects.
	CkAuthAzureADW(const CkAuthAzureADW &);
	CkAuthAzureADW &operator=(const CkAuthAzureADW &);

    public:
	CkAuthAzureADW(void);
	virtual ~CkAuthAzureADW(void);

	

	static CkAuthAzureADW *createNew(void);
	

	CkAuthAzureADW(bool bCallbackOwned);
	static CkAuthAzureADW *createNew(bool bCallbackOwned);

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	CkBaseProgressW *get_EventCallbackObject(void) const;
	void put_EventCallbackObject(CkBaseProgressW *progress);


	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The access token to be used in Azure AD REST API requests. This property is set
	// on a successful call to ObtainAccessToken.
	void get_AccessToken(CkString &str);
	// The access token to be used in Azure AD REST API requests. This property is set
	// on a successful call to ObtainAccessToken.
	const wchar_t *accessToken(void);
	// The access token to be used in Azure AD REST API requests. This property is set
	// on a successful call to ObtainAccessToken.
	void put_AccessToken(const wchar_t *newVal);

	// Specifies the Azure AD client id of the calling web service. To find the calling
	// application's client ID, in the Azure Management Portal, click Active Directory,
	// click the directory, click the application, and then click Configure.
	void get_ClientId(CkString &str);
	// Specifies the Azure AD client id of the calling web service. To find the calling
	// application's client ID, in the Azure Management Portal, click Active Directory,
	// click the directory, click the application, and then click Configure.
	const wchar_t *clientId(void);
	// Specifies the Azure AD client id of the calling web service. To find the calling
	// application's client ID, in the Azure Management Portal, click Active Directory,
	// click the directory, click the application, and then click Configure.
	void put_ClientId(const wchar_t *newVal);

	// A key registered for the calling web service in Azure AD. To create a key, in
	// the Azure Management Portal, click Active Directory, click the directory, click
	// the application, and then click Configure.
	void get_ClientSecret(CkString &str);
	// A key registered for the calling web service in Azure AD. To create a key, in
	// the Azure Management Portal, click Active Directory, click the directory, click
	// the application, and then click Configure.
	const wchar_t *clientSecret(void);
	// A key registered for the calling web service in Azure AD. To create a key, in
	// the Azure Management Portal, click Active Directory, click the directory, click
	// the application, and then click Configure.
	void put_ClientSecret(const wchar_t *newVal);

	// If the access token is valid, contains the number of seconds remaining until it
	// expires. A value of 0 indicates an invalid or expired access token.
	int get_NumSecondsRemaining(void);

	// The App ID URI of the receiving web service. To find the App ID URI, in the
	// Azure Management Portal, click Active Directory, click the directory, click the
	// application, and then click Configure.
	void get_Resource(CkString &str);
	// The App ID URI of the receiving web service. To find the App ID URI, in the
	// Azure Management Portal, click Active Directory, click the directory, click the
	// application, and then click Configure.
	const wchar_t *resource(void);
	// The App ID URI of the receiving web service. To find the App ID URI, in the
	// Azure Management Portal, click Active Directory, click the directory, click the
	// application, and then click Configure.
	void put_Resource(const wchar_t *newVal);

	// Your Azure account tenant ID. (If you don't know what it is, Google "how to find
	// my azure tenant id" for help.)
	void get_TenantId(CkString &str);
	// Your Azure account tenant ID. (If you don't know what it is, Google "how to find
	// my azure tenant id" for help.)
	const wchar_t *tenantId(void);
	// Your Azure account tenant ID. (If you don't know what it is, Google "how to find
	// my azure tenant id" for help.)
	void put_TenantId(const wchar_t *newVal);

	// true if the AccessToken property contains a valid non-expired access token
	// obtained via the call to ObtainAccessToken.
	bool get_Valid(void);



	// ----------------------
	// Methods
	// ----------------------
	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTaskW &task);

	// Sends the HTTP request to fetch the access token. When this method completes
	// successfully, the access token is available in the AccessToken property. The
	// connection is an existing connection to login.microsoftonline.com.
	bool ObtainAccessToken(CkSocketW &connection);

	// Creates an asynchronous task to call the ObtainAccessToken method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ObtainAccessTokenAsync(CkSocketW &connection);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
