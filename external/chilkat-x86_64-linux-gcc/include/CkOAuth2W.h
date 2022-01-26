// CkOAuth2W.h: interface for the CkOAuth2W class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkOAuth2W_H
#define _CkOAuth2W_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacksW.h"

class CkTaskW;
class CkSocketW;
class CkBaseProgressW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkOAuth2W
class CK_VISIBLE_PUBLIC CkOAuth2W  : public CkClassWithCallbacksW
{
	private:
	bool m_cbOwned;

	private:
	
	// Don't allow assignment or copying these objects.
	CkOAuth2W(const CkOAuth2W &);
	CkOAuth2W &operator=(const CkOAuth2W &);

    public:
	CkOAuth2W(void);
	virtual ~CkOAuth2W(void);

	

	static CkOAuth2W *createNew(void);
	

	CkOAuth2W(bool bCallbackOwned);
	static CkOAuth2W *createNew(bool bCallbackOwned);

	
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
	// When the OAuth2 three-legged authorization has successfully completed in the
	// background thread, this property contains the access_token.
	// 
	// For example, a successful Google API JSON response looks like this:
	//  {
	//              "access_token": "ya29.Ci9ZA-Z0Q7vtnch8xxxxxxxxxxxxxxgDVOOV97-IBvTt958xxxxxx1sasw",
	//              "token_type": "Bearer",
	// 
	//             "expires_in": 3600,
	// 
	//             "refresh_token": "1/fYjEVR-3Oq9xxxxxxxxxxxxxxLzPtlNOeQ"
	// }
	// 
	void get_AccessToken(CkString &str);
	// When the OAuth2 three-legged authorization has successfully completed in the
	// background thread, this property contains the access_token.
	// 
	// For example, a successful Google API JSON response looks like this:
	//  {
	//              "access_token": "ya29.Ci9ZA-Z0Q7vtnch8xxxxxxxxxxxxxxgDVOOV97-IBvTt958xxxxxx1sasw",
	//              "token_type": "Bearer",
	// 
	//             "expires_in": 3600,
	// 
	//             "refresh_token": "1/fYjEVR-3Oq9xxxxxxxxxxxxxxLzPtlNOeQ"
	// }
	// 
	const wchar_t *accessToken(void);
	// When the OAuth2 three-legged authorization has successfully completed in the
	// background thread, this property contains the access_token.
	// 
	// For example, a successful Google API JSON response looks like this:
	//  {
	//              "access_token": "ya29.Ci9ZA-Z0Q7vtnch8xxxxxxxxxxxxxxgDVOOV97-IBvTt958xxxxxx1sasw",
	//              "token_type": "Bearer",
	// 
	//             "expires_in": 3600,
	// 
	//             "refresh_token": "1/fYjEVR-3Oq9xxxxxxxxxxxxxxLzPtlNOeQ"
	// }
	// 
	void put_AccessToken(const wchar_t *newVal);

	// When the OAuth2 three-legged authorization has completed in the background
	// thread, this property contains the response that contains the access_token, the
	// optional refresh_token, and any other information included in the final
	// response. If the authorization was denied, then this contains the error
	// response.
	// 
	// For example, a successful JSON response for a Google API looks like this:
	//  {
	//              "access_token": "ya29.Ci9ZA-Z0Q7vtnch8xxxxxxxxxxxxxxgDVOOV97-IBvTt958xxxxxx1sasw",
	//              "token_type": "Bearer",
	// 
	//             "expires_in": 3600,
	// 
	//             "refresh_token": "1/fYjEVR-3Oq9xxxxxxxxxxxxxxLzPtlNOeQ"
	// }
	// 
	// Note: Not all responses are JSON. A successful Facebook response is plain text
	// and looks like this:
	// access_token=EAAZALuOC1wAwBAKH6FKnxOkjfEP ... UBZBhYD5hSVBETBx6AZD&expires=5134653
	// 
	void get_AccessTokenResponse(CkString &str);
	// When the OAuth2 three-legged authorization has completed in the background
	// thread, this property contains the response that contains the access_token, the
	// optional refresh_token, and any other information included in the final
	// response. If the authorization was denied, then this contains the error
	// response.
	// 
	// For example, a successful JSON response for a Google API looks like this:
	//  {
	//              "access_token": "ya29.Ci9ZA-Z0Q7vtnch8xxxxxxxxxxxxxxgDVOOV97-IBvTt958xxxxxx1sasw",
	//              "token_type": "Bearer",
	// 
	//             "expires_in": 3600,
	// 
	//             "refresh_token": "1/fYjEVR-3Oq9xxxxxxxxxxxxxxLzPtlNOeQ"
	// }
	// 
	// Note: Not all responses are JSON. A successful Facebook response is plain text
	// and looks like this:
	// access_token=EAAZALuOC1wAwBAKH6FKnxOkjfEP ... UBZBhYD5hSVBETBx6AZD&expires=5134653
	// 
	const wchar_t *accessTokenResponse(void);

	// Some OAuth2 services, such as QuickBooks, do not allow for
	// "http://localhost:port" callback URLs. When this is the case, a desktop app
	// cannot pop up a browser and expect to get the final redirect callback. The
	// workaround is to set this property to a URI on your web server, which sends a
	// response to redirect back to "http://localhost:3017". Thus the callback becomes
	// a double redirect, which ends at localhost:port, and thus completes the circuit.
	// 
	// If the OAuth2 service allows for "http://localhost:port" callback URLs, then
	// leave this property empty.
	// 
	// As an example, one could set this property to
	// "https://www.yourdomain.com/OAuth2.php", where the PHP source contains the
	// following:
	// 
	void get_AppCallbackUrl(CkString &str);
	// Some OAuth2 services, such as QuickBooks, do not allow for
	// "http://localhost:port" callback URLs. When this is the case, a desktop app
	// cannot pop up a browser and expect to get the final redirect callback. The
	// workaround is to set this property to a URI on your web server, which sends a
	// response to redirect back to "http://localhost:3017". Thus the callback becomes
	// a double redirect, which ends at localhost:port, and thus completes the circuit.
	// 
	// If the OAuth2 service allows for "http://localhost:port" callback URLs, then
	// leave this property empty.
	// 
	// As an example, one could set this property to
	// "https://www.yourdomain.com/OAuth2.php", where the PHP source contains the
	// following:
	// 
	const wchar_t *appCallbackUrl(void);
	// Some OAuth2 services, such as QuickBooks, do not allow for
	// "http://localhost:port" callback URLs. When this is the case, a desktop app
	// cannot pop up a browser and expect to get the final redirect callback. The
	// workaround is to set this property to a URI on your web server, which sends a
	// response to redirect back to "http://localhost:3017". Thus the callback becomes
	// a double redirect, which ends at localhost:port, and thus completes the circuit.
	// 
	// If the OAuth2 service allows for "http://localhost:port" callback URLs, then
	// leave this property empty.
	// 
	// As an example, one could set this property to
	// "https://www.yourdomain.com/OAuth2.php", where the PHP source contains the
	// following:
	// 
	void put_AppCallbackUrl(const wchar_t *newVal);

	// Indicates the current progress of the OAuth2 three-legged authorization flow.
	// Possible values are:
	// 
	// 0: Idle. No OAuth2 has yet been attempted.
	// 1: Waiting for Redirect. The OAuth2 background thread is waiting to receive the
	// redirect HTTP request from the browser.
	// 2: Waiting for Final Response. The OAuth2 background thread is waiting for the
	// final access token response.
	// 3: Completed with Success. The OAuth2 flow has completed, the background thread
	// exited, and the successful JSON response is available in AccessTokenResponse
	// property.
	// 4: Completed with Access Denied. The OAuth2 flow has completed, the background
	// thread exited, and the error JSON is available in AccessTokenResponse property.
	// 5: Failed Prior to Completion. The OAuth2 flow failed to complete, the
	// background thread exited, and the error information is available in the
	// FailureInfo property.
	// 
	int get_AuthFlowState(void);

	// The URL used to obtain an authorization grant. For example, the Google APIs
	// authorization endpoint is "https://accounts.google.com/o/oauth2/v2/auth". (In
	// three-legged OAuth2, this is the very first point of contact that begins the
	// OAuth2 authentication flow.)
	void get_AuthorizationEndpoint(CkString &str);
	// The URL used to obtain an authorization grant. For example, the Google APIs
	// authorization endpoint is "https://accounts.google.com/o/oauth2/v2/auth". (In
	// three-legged OAuth2, this is the very first point of contact that begins the
	// OAuth2 authentication flow.)
	const wchar_t *authorizationEndpoint(void);
	// The URL used to obtain an authorization grant. For example, the Google APIs
	// authorization endpoint is "https://accounts.google.com/o/oauth2/v2/auth". (In
	// three-legged OAuth2, this is the very first point of contact that begins the
	// OAuth2 authentication flow.)
	void put_AuthorizationEndpoint(const wchar_t *newVal);

	// The "client_id" that identifies the application.
	// 
	// For example, if creating an app to use a Google API, one would create a client
	// ID by:
	//     Logging into the Google API Console (https://console.developers.google.com).
	//     Navigate to "Credentials".
	//     Click on "Create Credentials"
	//     Choose "OAuth client ID"
	//     Select the "Other" application type.
	//     Name your app and click "Create", and a client_id and client_secret will be
	//     generated.
	// Other API's, such as Facebook, should have something similar for generating a
	// client ID and client secret.
	// 
	void get_ClientId(CkString &str);
	// The "client_id" that identifies the application.
	// 
	// For example, if creating an app to use a Google API, one would create a client
	// ID by:
	//     Logging into the Google API Console (https://console.developers.google.com).
	//     Navigate to "Credentials".
	//     Click on "Create Credentials"
	//     Choose "OAuth client ID"
	//     Select the "Other" application type.
	//     Name your app and click "Create", and a client_id and client_secret will be
	//     generated.
	// Other API's, such as Facebook, should have something similar for generating a
	// client ID and client secret.
	// 
	const wchar_t *clientId(void);
	// The "client_id" that identifies the application.
	// 
	// For example, if creating an app to use a Google API, one would create a client
	// ID by:
	//     Logging into the Google API Console (https://console.developers.google.com).
	//     Navigate to "Credentials".
	//     Click on "Create Credentials"
	//     Choose "OAuth client ID"
	//     Select the "Other" application type.
	//     Name your app and click "Create", and a client_id and client_secret will be
	//     generated.
	// Other API's, such as Facebook, should have something similar for generating a
	// client ID and client secret.
	// 
	void put_ClientId(const wchar_t *newVal);

	// The "client_secret" for the application. Application credentials (i.e. what
	// identifies the application) consist of a client_id and client_secret. See the
	// ClientId property for more information.
	// 
	// Is the Client Secret Really a Secret?
	// 
	// This deserves some explanation. For a web-based application (where the code is
	// on the web server) and the user interacts with the application in a browser,
	// then YES, the client secret MUST be kept secret at all times. One does not want
	// to be interacting with a site that claims to be "Application XYZ" but is
	// actually an impersonator. But the Chilkat OAuth2 class is for desktop
	// applications and scripts (i.e. things that run on the local computer, not in a
	// browser).
	// 
	// Consider Mozilla Thunderbird. It is an application installed on your computer.
	// Thunderbird uses OAuth2 authentication for GMail accounts in the same way as
	// this OAuth2 API. When you add a GMail account and need to authenticate for the
	// 1st time, you'll get a popup window (a browser) where you interactively grant
	// authorization to Thunderbird. You implicitly know the Thunderbird application is
	// running because you started it. There can be no impersonation unless your
	// computer has already been hacked and when you thought you started Thunderbird,
	// you actually started some rogue app. But if you already started some rogue app,
	// then all has already been lost.
	// 
	// It is essentially impossible for desktop applications to embed a secret key
	// (such as the client secret) and assure confidentiality (i.e. that the key cannot
	// be obtained by some hacker. An application can hide the secret, and can make it
	// difficult to access, but in the end the secret cannot be assumed to be safe.
	// Therefore, the client_secret, for desktop (installed) applications is not
	// actually secret. One should still take care to shroud the client secret to some
	// extent, but know that whatever is done cannot be deemed secure. But this is OK.
	// The reason it is OK is that implicitly, when a person starts an application
	// (such as Thunderbird), the identity of the application is known. If a fake
	// Thunderbird was started, then all has already been lost. The security of the
	// system is in preventing the fake/rogue applications in the 1st place. If that
	// security has already been breached, then nothing else really matters.
	// 
	void get_ClientSecret(CkString &str);
	// The "client_secret" for the application. Application credentials (i.e. what
	// identifies the application) consist of a client_id and client_secret. See the
	// ClientId property for more information.
	// 
	// Is the Client Secret Really a Secret?
	// 
	// This deserves some explanation. For a web-based application (where the code is
	// on the web server) and the user interacts with the application in a browser,
	// then YES, the client secret MUST be kept secret at all times. One does not want
	// to be interacting with a site that claims to be "Application XYZ" but is
	// actually an impersonator. But the Chilkat OAuth2 class is for desktop
	// applications and scripts (i.e. things that run on the local computer, not in a
	// browser).
	// 
	// Consider Mozilla Thunderbird. It is an application installed on your computer.
	// Thunderbird uses OAuth2 authentication for GMail accounts in the same way as
	// this OAuth2 API. When you add a GMail account and need to authenticate for the
	// 1st time, you'll get a popup window (a browser) where you interactively grant
	// authorization to Thunderbird. You implicitly know the Thunderbird application is
	// running because you started it. There can be no impersonation unless your
	// computer has already been hacked and when you thought you started Thunderbird,
	// you actually started some rogue app. But if you already started some rogue app,
	// then all has already been lost.
	// 
	// It is essentially impossible for desktop applications to embed a secret key
	// (such as the client secret) and assure confidentiality (i.e. that the key cannot
	// be obtained by some hacker. An application can hide the secret, and can make it
	// difficult to access, but in the end the secret cannot be assumed to be safe.
	// Therefore, the client_secret, for desktop (installed) applications is not
	// actually secret. One should still take care to shroud the client secret to some
	// extent, but know that whatever is done cannot be deemed secure. But this is OK.
	// The reason it is OK is that implicitly, when a person starts an application
	// (such as Thunderbird), the identity of the application is known. If a fake
	// Thunderbird was started, then all has already been lost. The security of the
	// system is in preventing the fake/rogue applications in the 1st place. If that
	// security has already been breached, then nothing else really matters.
	// 
	const wchar_t *clientSecret(void);
	// The "client_secret" for the application. Application credentials (i.e. what
	// identifies the application) consist of a client_id and client_secret. See the
	// ClientId property for more information.
	// 
	// Is the Client Secret Really a Secret?
	// 
	// This deserves some explanation. For a web-based application (where the code is
	// on the web server) and the user interacts with the application in a browser,
	// then YES, the client secret MUST be kept secret at all times. One does not want
	// to be interacting with a site that claims to be "Application XYZ" but is
	// actually an impersonator. But the Chilkat OAuth2 class is for desktop
	// applications and scripts (i.e. things that run on the local computer, not in a
	// browser).
	// 
	// Consider Mozilla Thunderbird. It is an application installed on your computer.
	// Thunderbird uses OAuth2 authentication for GMail accounts in the same way as
	// this OAuth2 API. When you add a GMail account and need to authenticate for the
	// 1st time, you'll get a popup window (a browser) where you interactively grant
	// authorization to Thunderbird. You implicitly know the Thunderbird application is
	// running because you started it. There can be no impersonation unless your
	// computer has already been hacked and when you thought you started Thunderbird,
	// you actually started some rogue app. But if you already started some rogue app,
	// then all has already been lost.
	// 
	// It is essentially impossible for desktop applications to embed a secret key
	// (such as the client secret) and assure confidentiality (i.e. that the key cannot
	// be obtained by some hacker. An application can hide the secret, and can make it
	// difficult to access, but in the end the secret cannot be assumed to be safe.
	// Therefore, the client_secret, for desktop (installed) applications is not
	// actually secret. One should still take care to shroud the client secret to some
	// extent, but know that whatever is done cannot be deemed secure. But this is OK.
	// The reason it is OK is that implicitly, when a person starts an application
	// (such as Thunderbird), the identity of the application is known. If a fake
	// Thunderbird was started, then all has already been lost. The security of the
	// system is in preventing the fake/rogue applications in the 1st place. If that
	// security has already been breached, then nothing else really matters.
	// 
	void put_ClientSecret(const wchar_t *newVal);

	// Optional. Set this to true to send a code_challenge (as per RFC 7636) with the
	// authorization request. The default value is false.
	bool get_CodeChallenge(void);
	// Optional. Set this to true to send a code_challenge (as per RFC 7636) with the
	// authorization request. The default value is false.
	void put_CodeChallenge(bool newVal);

	// Optional. Only applies when the CodeChallenge property is set to true.
	// Possible values are "plain" or "S256". The default is "S256".
	void get_CodeChallengeMethod(CkString &str);
	// Optional. Only applies when the CodeChallenge property is set to true.
	// Possible values are "plain" or "S256". The default is "S256".
	const wchar_t *codeChallengeMethod(void);
	// Optional. Only applies when the CodeChallenge property is set to true.
	// Possible values are "plain" or "S256". The default is "S256".
	void put_CodeChallengeMethod(const wchar_t *newVal);

	// If the OAuth2 three-legged authorization failed prior to completion (the
	// AuthFlowState = 5), then information about the failure is contained in this
	// property. This property is automatically cleared when OAuth2 authorization
	// starts (i.e. when StartAuth is called).
	void get_FailureInfo(CkString &str);
	// If the OAuth2 three-legged authorization failed prior to completion (the
	// AuthFlowState = 5), then information about the failure is contained in this
	// property. This property is automatically cleared when OAuth2 authorization
	// starts (i.e. when StartAuth is called).
	const wchar_t *failureInfo(void);

	// Optional. Set this to true to send a nonce with the authorization request. The
	// default value is false.
	bool get_IncludeNonce(void);
	// Optional. Set this to true to send a nonce with the authorization request. The
	// default value is false.
	void put_IncludeNonce(bool newVal);

	// The port number to listen for the redirect URI request sent by the browser. If
	// set to 0, then a random unused port is used. The default value of this property
	// is 0.
	// 
	// In most cases, using a random unused port is the best choice. In some OAuth2
	// situations, such as with Facebook, a specific port number must be chosen. This
	// is due to the fact that Facebook requires an APP to have a Site URL, which must
	// exactly match the redirect_uri used in OAuth2 authorization. For example, the
	// Facebook Site URL might be "http://localhost:3017/" if port 3017 is the listen
	// port.
	// 
	int get_ListenPort(void);
	// The port number to listen for the redirect URI request sent by the browser. If
	// set to 0, then a random unused port is used. The default value of this property
	// is 0.
	// 
	// In most cases, using a random unused port is the best choice. In some OAuth2
	// situations, such as with Facebook, a specific port number must be chosen. This
	// is due to the fact that Facebook requires an APP to have a Site URL, which must
	// exactly match the redirect_uri used in OAuth2 authorization. For example, the
	// Facebook Site URL might be "http://localhost:3017/" if port 3017 is the listen
	// port.
	// 
	void put_ListenPort(int newVal);

	// If set, then an unused port will be chosen in the range from the ListenPort
	// property to this property. Some OAuth2 services, such as Google, require that
	// callback URL's, including port numbers, be selected in advance. This feature
	// allows for a range of callback URL's to be specified to cope with the
	// possibility that another application on the same computer might be using a
	// particular port.
	// 
	// For example, a Google ClientID might be configured with a set of authorized
	// callback URI's such as:
	//     http://localhost:55110/
	//     http://localhost:55112/
	//     http://localhost:55113/
	//     http://localhost:55114/
	//     http://localhost:55115/
	//     http://localhost:55116/
	//     http://localhost:55117/
	// 
	// In which case the ListenPort property would be set to 55110, and this property
	// would be set to 55117.
	// 
	int get_ListenPortRangeEnd(void);
	// If set, then an unused port will be chosen in the range from the ListenPort
	// property to this property. Some OAuth2 services, such as Google, require that
	// callback URL's, including port numbers, be selected in advance. This feature
	// allows for a range of callback URL's to be specified to cope with the
	// possibility that another application on the same computer might be using a
	// particular port.
	// 
	// For example, a Google ClientID might be configured with a set of authorized
	// callback URI's such as:
	//     http://localhost:55110/
	//     http://localhost:55112/
	//     http://localhost:55113/
	//     http://localhost:55114/
	//     http://localhost:55115/
	//     http://localhost:55116/
	//     http://localhost:55117/
	// 
	// In which case the ListenPort property would be set to 55110, and this property
	// would be set to 55117.
	// 
	void put_ListenPortRangeEnd(int newVal);

	// Defaults to "localhost". This should typically remain at the default value. It
	// is the loopback domain or IP address used for the redirect_uri. For example,
	// "http://localhost:2012/". (assuming 2012 was used or randomly chosen as the
	// listen port number) If the desired redirect_uri is to be
	// "http://127.0.0.1:2012/", then set this property equal to "127.0.0.1".
	void get_LocalHost(CkString &str);
	// Defaults to "localhost". This should typically remain at the default value. It
	// is the loopback domain or IP address used for the redirect_uri. For example,
	// "http://localhost:2012/". (assuming 2012 was used or randomly chosen as the
	// listen port number) If the desired redirect_uri is to be
	// "http://127.0.0.1:2012/", then set this property equal to "127.0.0.1".
	const wchar_t *localHost(void);
	// Defaults to "localhost". This should typically remain at the default value. It
	// is the loopback domain or IP address used for the redirect_uri. For example,
	// "http://localhost:2012/". (assuming 2012 was used or randomly chosen as the
	// listen port number) If the desired redirect_uri is to be
	// "http://127.0.0.1:2012/", then set this property equal to "127.0.0.1".
	void put_LocalHost(const wchar_t *newVal);

	// Defines the length of the nonce in bytes. The nonce is only included if the
	// IncludeNonce property = true. (The length of the nonce in characters will be
	// twice the length in bytes, because the nonce is a hex string.)
	// 
	// The default nonce length is 4 bytes.
	// 
	int get_NonceLength(void);
	// Defines the length of the nonce in bytes. The nonce is only included if the
	// IncludeNonce property = true. (The length of the nonce in characters will be
	// twice the length in bytes, because the nonce is a hex string.)
	// 
	// The default nonce length is 4 bytes.
	// 
	void put_NonceLength(int newVal);

	// This property contains the HTML returned to the browser when access is allowed
	// by the end-user. The default value is HTML that contains a META refresh to
	// https://www.chilkatsoft.com/oauth2_allowed.html. Your application should set
	// this property to display whatever HTML is desired when access is granted.
	// 
	// The default value of this property is:
	// 
	// <html>
	//   <head><meta http-equiv='refresh' content='0;url=https://www.chilkatsoft.com/oauth2_allowed.html'></head>
	//   <body>Thank you for allowing access.</body>
	// </html>
	// 
	// You may wish to change the refresh URL to a web page on your company website.
	// Alternatively, you can provide simple HTML that does not redirect anywhere but
	// displays whatever information you desire.
	// 
	void get_RedirectAllowHtml(CkString &str);
	// This property contains the HTML returned to the browser when access is allowed
	// by the end-user. The default value is HTML that contains a META refresh to
	// https://www.chilkatsoft.com/oauth2_allowed.html. Your application should set
	// this property to display whatever HTML is desired when access is granted.
	// 
	// The default value of this property is:
	// 
	// <html>
	//   <head><meta http-equiv='refresh' content='0;url=https://www.chilkatsoft.com/oauth2_allowed.html'></head>
	//   <body>Thank you for allowing access.</body>
	// </html>
	// 
	// You may wish to change the refresh URL to a web page on your company website.
	// Alternatively, you can provide simple HTML that does not redirect anywhere but
	// displays whatever information you desire.
	// 
	const wchar_t *redirectAllowHtml(void);
	// This property contains the HTML returned to the browser when access is allowed
	// by the end-user. The default value is HTML that contains a META refresh to
	// https://www.chilkatsoft.com/oauth2_allowed.html. Your application should set
	// this property to display whatever HTML is desired when access is granted.
	// 
	// The default value of this property is:
	// 
	// <html>
	//   <head><meta http-equiv='refresh' content='0;url=https://www.chilkatsoft.com/oauth2_allowed.html'></head>
	//   <body>Thank you for allowing access.</body>
	// </html>
	// 
	// You may wish to change the refresh URL to a web page on your company website.
	// Alternatively, you can provide simple HTML that does not redirect anywhere but
	// displays whatever information you desire.
	// 
	void put_RedirectAllowHtml(const wchar_t *newVal);

	// The HTML returned to the browser when access is denied by the end-user. The
	// default value is HTMl that contains a META refresh to
	// https://www.chilkatsoft.com/oauth2_denied.html. Your application should set this
	// property to display whatever HTML is desired when access is denied.
	// 
	// The default value of this property is:
	// 
	// <html>
	//   <head><meta http-equiv='refresh' content='0;url=https://www.chilkatsoft.com/oauth2_denied.html'></head>
	//   <body>The app will not have access.</body>
	// </html>
	// 
	// You may wish to change the refresh URL to a web page on your company website.
	// Alternatively, you can provide simple HTML that does not redirect anywhere but
	// displays whatever information you desire.
	// 
	void get_RedirectDenyHtml(CkString &str);
	// The HTML returned to the browser when access is denied by the end-user. The
	// default value is HTMl that contains a META refresh to
	// https://www.chilkatsoft.com/oauth2_denied.html. Your application should set this
	// property to display whatever HTML is desired when access is denied.
	// 
	// The default value of this property is:
	// 
	// <html>
	//   <head><meta http-equiv='refresh' content='0;url=https://www.chilkatsoft.com/oauth2_denied.html'></head>
	//   <body>The app will not have access.</body>
	// </html>
	// 
	// You may wish to change the refresh URL to a web page on your company website.
	// Alternatively, you can provide simple HTML that does not redirect anywhere but
	// displays whatever information you desire.
	// 
	const wchar_t *redirectDenyHtml(void);
	// The HTML returned to the browser when access is denied by the end-user. The
	// default value is HTMl that contains a META refresh to
	// https://www.chilkatsoft.com/oauth2_denied.html. Your application should set this
	// property to display whatever HTML is desired when access is denied.
	// 
	// The default value of this property is:
	// 
	// <html>
	//   <head><meta http-equiv='refresh' content='0;url=https://www.chilkatsoft.com/oauth2_denied.html'></head>
	//   <body>The app will not have access.</body>
	// </html>
	// 
	// You may wish to change the refresh URL to a web page on your company website.
	// Alternatively, you can provide simple HTML that does not redirect anywhere but
	// displays whatever information you desire.
	// 
	void put_RedirectDenyHtml(const wchar_t *newVal);

	// When the OAuth2 three-legged authorization has successfully completed in the
	// background thread, this property contains the refresh_token, if present.
	// 
	// For example, a successful Google API JSON response looks like this:
	//  {
	//              "access_token": "ya29.Ci9ZA-Z0Q7vtnch8xxxxxxxxxxxxxxgDVOOV97-IBvTt958xxxxxx1sasw",
	//              "token_type": "Bearer",
	// 
	//             "expires_in": 3600,
	// 
	//             "refresh_token": "1/fYjEVR-3Oq9xxxxxxxxxxxxxxLzPtlNOeQ"
	// }
	// 
	void get_RefreshToken(CkString &str);
	// When the OAuth2 three-legged authorization has successfully completed in the
	// background thread, this property contains the refresh_token, if present.
	// 
	// For example, a successful Google API JSON response looks like this:
	//  {
	//              "access_token": "ya29.Ci9ZA-Z0Q7vtnch8xxxxxxxxxxxxxxgDVOOV97-IBvTt958xxxxxx1sasw",
	//              "token_type": "Bearer",
	// 
	//             "expires_in": 3600,
	// 
	//             "refresh_token": "1/fYjEVR-3Oq9xxxxxxxxxxxxxxLzPtlNOeQ"
	// }
	// 
	const wchar_t *refreshToken(void);
	// When the OAuth2 three-legged authorization has successfully completed in the
	// background thread, this property contains the refresh_token, if present.
	// 
	// For example, a successful Google API JSON response looks like this:
	//  {
	//              "access_token": "ya29.Ci9ZA-Z0Q7vtnch8xxxxxxxxxxxxxxgDVOOV97-IBvTt958xxxxxx1sasw",
	//              "token_type": "Bearer",
	// 
	//             "expires_in": 3600,
	// 
	//             "refresh_token": "1/fYjEVR-3Oq9xxxxxxxxxxxxxxLzPtlNOeQ"
	// }
	// 
	void put_RefreshToken(const wchar_t *newVal);

	// This is an optional setting that defines the "resource" query parameter. For
	// example, to call the Microsoft Graph API, set this property value to
	// "https://graph.microsoft.com/". The Microsoft Dynamics CRM OAuth authentication
	// also requires the Resource property.
	void get_Resource(CkString &str);
	// This is an optional setting that defines the "resource" query parameter. For
	// example, to call the Microsoft Graph API, set this property value to
	// "https://graph.microsoft.com/". The Microsoft Dynamics CRM OAuth authentication
	// also requires the Resource property.
	const wchar_t *resource(void);
	// This is an optional setting that defines the "resource" query parameter. For
	// example, to call the Microsoft Graph API, set this property value to
	// "https://graph.microsoft.com/". The Microsoft Dynamics CRM OAuth authentication
	// also requires the Resource property.
	void put_Resource(const wchar_t *newVal);

	// Can be set to "form_post" to include a "response_mode=form_post" in the
	// authorization request. The default value is the empty string to omit the
	// "response_mode" query param.
	void get_ResponseMode(CkString &str);
	// Can be set to "form_post" to include a "response_mode=form_post" in the
	// authorization request. The default value is the empty string to omit the
	// "response_mode" query param.
	const wchar_t *responseMode(void);
	// Can be set to "form_post" to include a "response_mode=form_post" in the
	// authorization request. The default value is the empty string to omit the
	// "response_mode" query param.
	void put_ResponseMode(const wchar_t *newVal);

	// The default value is "code". Can be set to "id_token+code" for cases where
	// "response_type=id_token+code" is required in the authorization request.
	void get_ResponseType(CkString &str);
	// The default value is "code". Can be set to "id_token+code" for cases where
	// "response_type=id_token+code" is required in the authorization request.
	const wchar_t *responseType(void);
	// The default value is "code". Can be set to "id_token+code" for cases where
	// "response_type=id_token+code" is required in the authorization request.
	void put_ResponseType(const wchar_t *newVal);

	// This is an optional setting that defines the scope of access. For example,
	// Google API scopes are listed here:
	// https://developers.google.com/identity/protocols/googlescopes
	// 
	// For example, if wishing to grant OAuth2 authorization for Google Drive, one
	// would set this property to "https://www.googleapis.com/auth/drive".
	// 
	void get_Scope(CkString &str);
	// This is an optional setting that defines the scope of access. For example,
	// Google API scopes are listed here:
	// https://developers.google.com/identity/protocols/googlescopes
	// 
	// For example, if wishing to grant OAuth2 authorization for Google Drive, one
	// would set this property to "https://www.googleapis.com/auth/drive".
	// 
	const wchar_t *scope(void);
	// This is an optional setting that defines the scope of access. For example,
	// Google API scopes are listed here:
	// https://developers.google.com/identity/protocols/googlescopes
	// 
	// For example, if wishing to grant OAuth2 authorization for Google Drive, one
	// would set this property to "https://www.googleapis.com/auth/drive".
	// 
	void put_Scope(const wchar_t *newVal);

	// The URL for exchanging an authorization grant for an access token. For example,
	// the Google APIs token endpoint is "https://www.googleapis.com/oauth2/v4/token".
	// (In three-legged OAuth2, this is the very last point of contact that ends the
	// OAuth2 authentication flow.)
	void get_TokenEndpoint(CkString &str);
	// The URL for exchanging an authorization grant for an access token. For example,
	// the Google APIs token endpoint is "https://www.googleapis.com/oauth2/v4/token".
	// (In three-legged OAuth2, this is the very last point of contact that ends the
	// OAuth2 authentication flow.)
	const wchar_t *tokenEndpoint(void);
	// The URL for exchanging an authorization grant for an access token. For example,
	// the Google APIs token endpoint is "https://www.googleapis.com/oauth2/v4/token".
	// (In three-legged OAuth2, this is the very last point of contact that ends the
	// OAuth2 authentication flow.)
	void put_TokenEndpoint(const wchar_t *newVal);

	// When the OAuth2 three-legged authorization has successfully completed in the
	// background thread, this property contains the token_type, if present.
	// 
	// A successful Google API JSON response looks like this:
	//  {
	//              "access_token": "ya29.Ci9ZA-Z0Q7vtnch8xxxxxxxxxxxxxxgDVOOV97-IBvTt958xxxxxx1sasw",
	//              "token_type": "Bearer",
	// 
	//             "expires_in": 3600,
	// 
	//             "refresh_token": "1/fYjEVR-3Oq9xxxxxxxxxxxxxxLzPtlNOeQ"
	// }
	// 
	// Note: Some responses may not included a "token_type" param. In that case, this
	// property will remain empty.
	// 
	void get_TokenType(CkString &str);
	// When the OAuth2 three-legged authorization has successfully completed in the
	// background thread, this property contains the token_type, if present.
	// 
	// A successful Google API JSON response looks like this:
	//  {
	//              "access_token": "ya29.Ci9ZA-Z0Q7vtnch8xxxxxxxxxxxxxxgDVOOV97-IBvTt958xxxxxx1sasw",
	//              "token_type": "Bearer",
	// 
	//             "expires_in": 3600,
	// 
	//             "refresh_token": "1/fYjEVR-3Oq9xxxxxxxxxxxxxxLzPtlNOeQ"
	// }
	// 
	// Note: Some responses may not included a "token_type" param. In that case, this
	// property will remain empty.
	// 
	const wchar_t *tokenType(void);
	// When the OAuth2 three-legged authorization has successfully completed in the
	// background thread, this property contains the token_type, if present.
	// 
	// A successful Google API JSON response looks like this:
	//  {
	//              "access_token": "ya29.Ci9ZA-Z0Q7vtnch8xxxxxxxxxxxxxxgDVOOV97-IBvTt958xxxxxx1sasw",
	//              "token_type": "Bearer",
	// 
	//             "expires_in": 3600,
	// 
	//             "refresh_token": "1/fYjEVR-3Oq9xxxxxxxxxxxxxxLzPtlNOeQ"
	// }
	// 
	// Note: Some responses may not included a "token_type" param. In that case, this
	// property will remain empty.
	// 
	void put_TokenType(const wchar_t *newVal);

	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string, and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "NO_OAUTH2_SCOPE" - Do not includethe "scope" parameter when exchanging the
	//     authorization code for an access token.
	// 
	void get_UncommonOptions(CkString &str);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string, and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "NO_OAUTH2_SCOPE" - Do not includethe "scope" parameter when exchanging the
	//     authorization code for an access token.
	// 
	const wchar_t *uncommonOptions(void);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string, and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "NO_OAUTH2_SCOPE" - Do not includethe "scope" parameter when exchanging the
	//     authorization code for an access token.
	// 
	void put_UncommonOptions(const wchar_t *newVal);

	// If set to true, then the internal POST (on the background thread) that
	// exchanges the code for an access token will send the client_id/client_secret in
	// an "Authorization Basic ..." header where the client_id is the login and the
	// client_secret is the password.
	// 
	// Some services, such as fitbit.com, require the client_id/client_secret to be
	// passed in this way.
	// 
	// The default value of this property is false, which causes the
	// client_id/client_secret to be sent as query params.
	// 
	bool get_UseBasicAuth(void);
	// If set to true, then the internal POST (on the background thread) that
	// exchanges the code for an access token will send the client_id/client_secret in
	// an "Authorization Basic ..." header where the client_id is the login and the
	// client_secret is the password.
	// 
	// Some services, such as fitbit.com, require the client_id/client_secret to be
	// passed in this way.
	// 
	// The default value of this property is false, which causes the
	// client_id/client_secret to be sent as query params.
	// 
	void put_UseBasicAuth(bool newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Adds an additional custom query param (name=value) to the URL that is returned
	// by the StartAuth method. This method exists to satisfy OAuth installations that
	// require non-standard/custom query parms. This method can be called multiple
	// times, once per additional query parm to be added.
	bool AddAuthQueryParam(const wchar_t *name, const wchar_t *value);

	// Adds an additional custom query param (name=value) to the request that occurs
	// (internally) to exchange the authorization code for a token. This method exists
	// to satisfy OAuth installations that require non-standard/custom query parms.
	// This method can be called multiple times, once per additional query parm to be
	// added.
	bool AddTokenQueryParam(const wchar_t *name, const wchar_t *value);

	// Cancels an OAuth2 authorization flow that is in progress.
	bool Cancel(void);

	// Some OAuth2 providers can provide additional parameters in the redirect request
	// sent to the local listener (i.e. the Chilkat background thread). One such case
	// is for QuickBooks, It contains a realmId parameter such as the following:
	// http://localhost:55568/?state=xxxxxxxxxxxx&code=xxxxxxxxxxxx&realmId=1234567890
	// 
	// After the OAuth2 authentication is completed, an application can call this
	// method to get any of the parameter values. For example, to get the realmId
	// value, pass "realmId" in paramName.
	// 
	bool GetRedirectRequestParam(const wchar_t *paramName, CkString &outStr);
	// Some OAuth2 providers can provide additional parameters in the redirect request
	// sent to the local listener (i.e. the Chilkat background thread). One such case
	// is for QuickBooks, It contains a realmId parameter such as the following:
	// http://localhost:55568/?state=xxxxxxxxxxxx&code=xxxxxxxxxxxx&realmId=1234567890
	// 
	// After the OAuth2 authentication is completed, an application can call this
	// method to get any of the parameter values. For example, to get the realmId
	// value, pass "realmId" in paramName.
	// 
	const wchar_t *getRedirectRequestParam(const wchar_t *paramName);
	// Some OAuth2 providers can provide additional parameters in the redirect request
	// sent to the local listener (i.e. the Chilkat background thread). One such case
	// is for QuickBooks, It contains a realmId parameter such as the following:
	// http://localhost:55568/?state=xxxxxxxxxxxx&code=xxxxxxxxxxxx&realmId=1234567890
	// 
	// After the OAuth2 authentication is completed, an application can call this
	// method to get any of the parameter values. For example, to get the realmId
	// value, pass "realmId" in paramName.
	// 
	const wchar_t *redirectRequestParam(const wchar_t *paramName);

	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTaskW &task);

	// Monitors an already started OAuth2 authorization flow and returns when it is
	// finished.
	// 
	// Note: It rarely makes sense to call this method. If this programming language
	// supports callbacks, then MonitorAsync is a better choice. (See the Oauth2
	// project repositories at https://github.com/chilkatsoft for samples.) If a
	// programming language does not have callbacks, a better choice is to periodically
	// check the AuthFlowState property for a value >= 3. If there is no response from
	// the browser, the background thread (that is waiting on the browser) can be
	// cancelled by calling the Cancel method.
	// 
	bool Monitor(void);

	// Creates an asynchronous task to call the Monitor method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *MonitorAsync(void);

	// Sends a refresh request to the token endpoint to obtain a new access token.
	// After a successful refresh request, the AccessToken and RefreshToken properties
	// will be updated with new values.
	// 
	// Note: This method can only be called if the ClientId, ClientSecret, RefreshToken
	// and TokenEndpoint properties contain valid values.
	// 
	bool RefreshAccessToken(void);

	// Creates an asynchronous task to call the RefreshAccessToken method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *RefreshAccessTokenAsync(void);

	// Provides for the ability to add HTTP request headers for the request sent by the
	// RefreshAccesToken method. For example, if the "Accept: application/json" header
	// needs to be sent, then add it by calling this method with name = "Accept" and
	// value = "application/json".
	// 
	// Multiple headers may be added by calling this method once for each. To remove a
	// header, call this method with name equal to the header name, and with an empty
	// string for value.
	// 
	bool SetRefreshHeader(const wchar_t *name, const wchar_t *value);

	// Creates an asynchronous task to call the SetRefreshHeader method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SetRefreshHeaderAsync(const wchar_t *name, const wchar_t *value);

	// Convenience method to force the calling thread to sleep for a number of
	// milliseconds.
	void SleepMs(int millisec);

	// Initiates the three-legged OAuth2 flow. The various properties, such as
	// ClientId, ClientSecret, Scope, CodeChallenge, AuthorizationEndpoint, and
	// TokenEndpoint, should be set prior to calling this method.
	// 
	// This method does two things:
	//     Forms and returns a URL that is to be loaded in a browser.
	//     Starts a background thread that listens on a randomly selected unused port
	//     to receive the redirect request from the browser. The receiving of the request
	//     from the browser, and the sending of the HTTP request to complete the
	//     three-legged OAuth2 flow is done entirely in the background thread. The
	//     application controls this behavior by setting the various properties beforehand.
	// The return value is the URL to be loaded (navigated to) in a popup or embedded
	// browser.
	// 
	// Note: It's best not to call StartAuth if a previous call to StartAuth is in a
	// non-completed state. However, starting in v9.5.0.76, if a background thread from
	// a previous call to StartAuth is still running, it will be automatically
	// canceled. However,rather than relying on this automatic behavior, your
	// application should explicity Cancel the previous StartAuth before calling again.
	// 
	bool StartAuth(CkString &outStr);
	// Initiates the three-legged OAuth2 flow. The various properties, such as
	// ClientId, ClientSecret, Scope, CodeChallenge, AuthorizationEndpoint, and
	// TokenEndpoint, should be set prior to calling this method.
	// 
	// This method does two things:
	//     Forms and returns a URL that is to be loaded in a browser.
	//     Starts a background thread that listens on a randomly selected unused port
	//     to receive the redirect request from the browser. The receiving of the request
	//     from the browser, and the sending of the HTTP request to complete the
	//     three-legged OAuth2 flow is done entirely in the background thread. The
	//     application controls this behavior by setting the various properties beforehand.
	// The return value is the URL to be loaded (navigated to) in a popup or embedded
	// browser.
	// 
	// Note: It's best not to call StartAuth if a previous call to StartAuth is in a
	// non-completed state. However, starting in v9.5.0.76, if a background thread from
	// a previous call to StartAuth is still running, it will be automatically
	// canceled. However,rather than relying on this automatic behavior, your
	// application should explicity Cancel the previous StartAuth before calling again.
	// 
	const wchar_t *startAuth(void);

	// Calling this method is optional, and is only required if a proxy (HTTP or
	// SOCKS), an SSH tunnel, or if special connection related socket options need to
	// be used. When UseConnection is not called, the connection to the token endpoint
	// is a direct connection using TLS (or not) based on the TokenEndpoint. (If the
	// TokenEndpoint begins with "https://", then TLS is used.)
	// 
	// This method sets the socket object to be used for sending the requests to the
	// token endpoint in the background thread. The sock can be an already-connected
	// socket, or a socket object that is not yet connected. In some cases, such as for
	// a connection through an SSH tunnel, the sock must already be connected. In other
	// cases, an unconnected sock can be provided because the purpose for providing the
	// socket object is to specify settings such as for HTTP or SOCKS proxies.
	// 
	bool UseConnection(CkSocketW &sock);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
