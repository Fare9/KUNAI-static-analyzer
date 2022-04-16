// CkAuthAzureAD.h: interface for the CkAuthAzureAD class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkAuthAzureAD_H
#define _CkAuthAzureAD_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacks.h"

class CkTask;
class CkSocket;
class CkBaseProgress;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkAuthAzureAD
class CK_VISIBLE_PUBLIC CkAuthAzureAD  : public CkClassWithCallbacks
{
    private:

	// Don't allow assignment or copying these objects.
	CkAuthAzureAD(const CkAuthAzureAD &);
	CkAuthAzureAD &operator=(const CkAuthAzureAD &);

    public:
	CkAuthAzureAD(void);
	virtual ~CkAuthAzureAD(void);

	static CkAuthAzureAD *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	CkBaseProgress *get_EventCallbackObject(void) const;
	void put_EventCallbackObject(CkBaseProgress *progress);


	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The access token to be used in Azure AD REST API requests. This property is set
	// on a successful call to ObtainAccessToken.
	void get_AccessToken(CkString &str);
	// The access token to be used in Azure AD REST API requests. This property is set
	// on a successful call to ObtainAccessToken.
	const char *accessToken(void);
	// The access token to be used in Azure AD REST API requests. This property is set
	// on a successful call to ObtainAccessToken.
	void put_AccessToken(const char *newVal);

	// Specifies the Azure AD client id of the calling web service. To find the calling
	// application's client ID, in the Azure Management Portal, click Active Directory,
	// click the directory, click the application, and then click Configure.
	void get_ClientId(CkString &str);
	// Specifies the Azure AD client id of the calling web service. To find the calling
	// application's client ID, in the Azure Management Portal, click Active Directory,
	// click the directory, click the application, and then click Configure.
	const char *clientId(void);
	// Specifies the Azure AD client id of the calling web service. To find the calling
	// application's client ID, in the Azure Management Portal, click Active Directory,
	// click the directory, click the application, and then click Configure.
	void put_ClientId(const char *newVal);

	// A key registered for the calling web service in Azure AD. To create a key, in
	// the Azure Management Portal, click Active Directory, click the directory, click
	// the application, and then click Configure.
	void get_ClientSecret(CkString &str);
	// A key registered for the calling web service in Azure AD. To create a key, in
	// the Azure Management Portal, click Active Directory, click the directory, click
	// the application, and then click Configure.
	const char *clientSecret(void);
	// A key registered for the calling web service in Azure AD. To create a key, in
	// the Azure Management Portal, click Active Directory, click the directory, click
	// the application, and then click Configure.
	void put_ClientSecret(const char *newVal);

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
	const char *resource(void);
	// The App ID URI of the receiving web service. To find the App ID URI, in the
	// Azure Management Portal, click Active Directory, click the directory, click the
	// application, and then click Configure.
	void put_Resource(const char *newVal);

	// Your Azure account tenant ID. (If you don't know what it is, Google "how to find
	// my azure tenant id" for help.)
	void get_TenantId(CkString &str);
	// Your Azure account tenant ID. (If you don't know what it is, Google "how to find
	// my azure tenant id" for help.)
	const char *tenantId(void);
	// Your Azure account tenant ID. (If you don't know what it is, Google "how to find
	// my azure tenant id" for help.)
	void put_TenantId(const char *newVal);

	// true if the AccessToken property contains a valid non-expired access token
	// obtained via the call to ObtainAccessToken.
	bool get_Valid(void);



	// ----------------------
	// Methods
	// ----------------------
	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTask &task);


	// Sends the HTTP request to fetch the access token. When this method completes
	// successfully, the access token is available in the AccessToken property. The
	// connection is an existing connection to login.microsoftonline.com.
	bool ObtainAccessToken(CkSocket &connection);

	// Sends the HTTP request to fetch the access token. When this method completes
	// successfully, the access token is available in the AccessToken property. The
	// connection is an existing connection to login.microsoftonline.com.
	CkTask *ObtainAccessTokenAsync(CkSocket &connection);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
