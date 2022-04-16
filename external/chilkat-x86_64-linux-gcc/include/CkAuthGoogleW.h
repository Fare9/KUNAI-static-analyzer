// CkAuthGoogleW.h: interface for the CkAuthGoogleW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkAuthGoogleW_H
#define _CkAuthGoogleW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacksW.h"

class CkPfxW;
class CkTaskW;
class CkSocketW;
class CkBaseProgressW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkAuthGoogleW
class CK_VISIBLE_PUBLIC CkAuthGoogleW  : public CkClassWithCallbacksW
{
	private:
	bool m_cbOwned;

	private:
	
	// Don't allow assignment or copying these objects.
	CkAuthGoogleW(const CkAuthGoogleW &);
	CkAuthGoogleW &operator=(const CkAuthGoogleW &);

    public:
	CkAuthGoogleW(void);
	virtual ~CkAuthGoogleW(void);

	

	static CkAuthGoogleW *createNew(void);
	

	CkAuthGoogleW(bool bCallbackOwned);
	static CkAuthGoogleW *createNew(bool bCallbackOwned);

	
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
	// The access token to be used in Google API requests. This property is set on a
	// successful call to ObtainAccessToken.
	// Important: This class is used for authenticating calls to the Google Cloud Platform API and Google Apps API using a service account.. 
	// For 3-legged OAuth2, where a browser must be used to interactively get permission from the Google account owner, use the Chilkat OAuth2 class/object.
	void get_AccessToken(CkString &str);
	// The access token to be used in Google API requests. This property is set on a
	// successful call to ObtainAccessToken.
	// Important: This class is used for authenticating calls to the Google Cloud Platform API and Google Apps API using a service account.. 
	// For 3-legged OAuth2, where a browser must be used to interactively get permission from the Google account owner, use the Chilkat OAuth2 class/object.
	const wchar_t *accessToken(void);
	// The access token to be used in Google API requests. This property is set on a
	// successful call to ObtainAccessToken.
	// Important: This class is used for authenticating calls to the Google Cloud Platform API and Google Apps API using a service account.. 
	// For 3-legged OAuth2, where a browser must be used to interactively get permission from the Google account owner, use the Chilkat OAuth2 class/object.
	void put_AccessToken(const wchar_t *newVal);

	// The client email address of the service account. If a JSON key is used, then the
	// client_email should already be specified within the JSON key, and this property
	// is unused. This property must be set if using a P12 key.
	void get_EmailAddress(CkString &str);
	// The client email address of the service account. If a JSON key is used, then the
	// client_email should already be specified within the JSON key, and this property
	// is unused. This property must be set if using a P12 key.
	const wchar_t *emailAddress(void);
	// The client email address of the service account. If a JSON key is used, then the
	// client_email should already be specified within the JSON key, and this property
	// is unused. This property must be set if using a P12 key.
	void put_EmailAddress(const wchar_t *newVal);

	// The expiration time, in seconds, of the access token to be requested. The
	// maximum value is 1 hour (3600 seconds). The default value is 3600.
	int get_ExpireNumSeconds(void);
	// The expiration time, in seconds, of the access token to be requested. The
	// maximum value is 1 hour (3600 seconds). The default value is 3600.
	void put_ExpireNumSeconds(int newVal);

	// This property can be set to override the default current date/time value for the
	// "iat" claim of the JWT. It can be set to a value indicating the number of
	// seconds from 1970-01-01T00:00:00Z UTC.
	// 
	// The default value is 0, which indicates to use the iat value for the current
	// system date/time. Unless explicitly needed, always leave this property at the
	// default value.
	// 
	int get_Iat(void);
	// This property can be set to override the default current date/time value for the
	// "iat" claim of the JWT. It can be set to a value indicating the number of
	// seconds from 1970-01-01T00:00:00Z UTC.
	// 
	// The default value is 0, which indicates to use the iat value for the current
	// system date/time. Unless explicitly needed, always leave this property at the
	// default value.
	// 
	void put_Iat(int newVal);

	// The JSON key for obtaining an access token. An application must set either the
	// P12 or JSON private key, but not both.
	void get_JsonKey(CkString &str);
	// The JSON key for obtaining an access token. An application must set either the
	// P12 or JSON private key, but not both.
	const wchar_t *jsonKey(void);
	// The JSON key for obtaining an access token. An application must set either the
	// P12 or JSON private key, but not both.
	void put_JsonKey(const wchar_t *newVal);

	// If the access token is valid, contains the number of seconds remaining until it
	// expires. A value of 0 indicates an invalid or expired access token.
	int get_NumSecondsRemaining(void);

	// A space-delimited list of the permissions that the application requests.
	void get_Scope(CkString &str);
	// A space-delimited list of the permissions that the application requests.
	const wchar_t *scope(void);
	// A space-delimited list of the permissions that the application requests.
	void put_Scope(const wchar_t *newVal);

	// The email address of the user for which the application is requesting delegated
	// access.
	void get_SubEmailAddress(CkString &str);
	// The email address of the user for which the application is requesting delegated
	// access.
	const wchar_t *subEmailAddress(void);
	// The email address of the user for which the application is requesting delegated
	// access.
	void put_SubEmailAddress(const wchar_t *newVal);

	// true if the AccessToken property contains a valid non-expired access token
	// obtained via the call to ObtainAccessToken.
	bool get_Valid(void);



	// ----------------------
	// Methods
	// ----------------------
	// Returns the private key in a PFX (P12) object. This is only possible if the
	// private key was previously set by calling SetP12.
	// The caller is responsible for deleting the object returned by this method.
	CkPfxW *GetP12(void);

	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTaskW &task);

	// Sends the HTTP request to fetch the access token. When this method completes
	// successfully, the access token is available in the AccessToken property. The
	// connection is an existing connection to www.googleapis.com.
	// 
	// Important: Make sure your computer's date/time is accurately set to the current
	// date/time, otherwise you'll get a 400 response status code with this error:
	// "Invalid JWT: Token must be a short-lived token (60 minutes) and in a reasonable
	// timeframe. Check your iat and exp values and use a clock with skew to account
	// for clock differences between systems.".
	// 
	bool ObtainAccessToken(CkSocketW &connection);

	// Creates an asynchronous task to call the ObtainAccessToken method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ObtainAccessTokenAsync(CkSocketW &connection);

	// Sets the P12 private key to be used for obtaining an access token. An
	// application must set either the P12 or JSON private key, but not both.
	bool SetP12(CkPfxW &key);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
