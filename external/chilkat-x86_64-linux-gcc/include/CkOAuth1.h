// CkOAuth1.h: interface for the CkOAuth1 class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkOAuth1_H
#define _CkOAuth1_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkPrivateKey;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkOAuth1
class CK_VISIBLE_PUBLIC CkOAuth1  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkOAuth1(const CkOAuth1 &);
	CkOAuth1 &operator=(const CkOAuth1 &);

    public:
	CkOAuth1(void);
	virtual ~CkOAuth1(void);

	static CkOAuth1 *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The authorization header. This is what would be included in the Authorization
	// HTTP request header if it is going to be used as the means for providing the
	// OAuth1 authorization information.
	void get_AuthorizationHeader(CkString &str);
	// The authorization header. This is what would be included in the Authorization
	// HTTP request header if it is going to be used as the means for providing the
	// OAuth1 authorization information.
	const char *authorizationHeader(void);

	// This is the exact string that was signed. For example, if the signature method
	// is HMAC-SHA1, the BaseString is the exact string that passed to the HMAC-SHA1.
	// An application does not set the BaseString property. The BaseString is exposed
	// as a property to allow for debugging and to see the exact string that is signed.
	void get_BaseString(CkString &str);
	// This is the exact string that was signed. For example, if the signature method
	// is HMAC-SHA1, the BaseString is the exact string that passed to the HMAC-SHA1.
	// An application does not set the BaseString property. The BaseString is exposed
	// as a property to allow for debugging and to see the exact string that is signed.
	const char *baseString(void);

	// The consumer key.
	void get_ConsumerKey(CkString &str);
	// The consumer key.
	const char *consumerKey(void);
	// The consumer key.
	void put_ConsumerKey(const char *newVal);

	// The consumer secret.
	void get_ConsumerSecret(CkString &str);
	// The consumer secret.
	const char *consumerSecret(void);
	// The consumer secret.
	void put_ConsumerSecret(const char *newVal);

	// The URL encoded representation of the Signature property
	void get_EncodedSignature(CkString &str);
	// The URL encoded representation of the Signature property
	const char *encodedSignature(void);

	// The URL that includes the OAuth1 query params.
	void get_GeneratedUrl(CkString &str);
	// The URL that includes the OAuth1 query params.
	const char *generatedUrl(void);

	// The exact HMAC key used to sign the BaseString. An application does not directly
	// set the HmacKey. The HmacKey property is read-only and is provided for debugging
	// to see the exact HMAC key used to sign the BaseString. The HMAC key is composed
	// from the consumer secret (if it exists) and the token secret (if it exists).
	void get_HmacKey(CkString &str);
	// The exact HMAC key used to sign the BaseString. An application does not directly
	// set the HmacKey. The HmacKey property is read-only and is provided for debugging
	// to see the exact HMAC key used to sign the BaseString. The HMAC key is composed
	// from the consumer secret (if it exists) and the token secret (if it exists).
	const char *hmacKey(void);

	// The nonce.
	void get_Nonce(CkString &str);
	// The nonce.
	const char *nonce(void);
	// The nonce.
	void put_Nonce(const char *newVal);

	// The HTTP method, such as "GET", "POST", "PUT", "DELETE", or "HEAD". Defaults to
	// "GET".
	void get_OauthMethod(CkString &str);
	// The HTTP method, such as "GET", "POST", "PUT", "DELETE", or "HEAD". Defaults to
	// "GET".
	const char *oauthMethod(void);
	// The HTTP method, such as "GET", "POST", "PUT", "DELETE", or "HEAD". Defaults to
	// "GET".
	void put_OauthMethod(const char *newVal);

	// The OAuth URL, such as "http://echo.lab.madgex.com/echo.ashx". See
	// http://bettiolo.github.io/oauth-reference-page/ to compare Chilkat results with
	// another tool's results.
	// 
	// Note: The OAuthUrl should not include query parameters. For example, do not set
	// the OAuthUrl equal
	// tohttps://rest.sandbox.netsuite.com/app/site/hosting/restlet.nl?script=165&deploy
	// =1 Instead, set OAuthUrl equal
	// tohttps://rest.sandbox.netsuite.com/app/site/hosting/restlet.nl and then
	// subsequently call AddParam for each query parameter.
	// 
	void get_OauthUrl(CkString &str);
	// The OAuth URL, such as "http://echo.lab.madgex.com/echo.ashx". See
	// http://bettiolo.github.io/oauth-reference-page/ to compare Chilkat results with
	// another tool's results.
	// 
	// Note: The OAuthUrl should not include query parameters. For example, do not set
	// the OAuthUrl equal
	// tohttps://rest.sandbox.netsuite.com/app/site/hosting/restlet.nl?script=165&deploy
	// =1 Instead, set OAuthUrl equal
	// tohttps://rest.sandbox.netsuite.com/app/site/hosting/restlet.nl and then
	// subsequently call AddParam for each query parameter.
	// 
	const char *oauthUrl(void);
	// The OAuth URL, such as "http://echo.lab.madgex.com/echo.ashx". See
	// http://bettiolo.github.io/oauth-reference-page/ to compare Chilkat results with
	// another tool's results.
	// 
	// Note: The OAuthUrl should not include query parameters. For example, do not set
	// the OAuthUrl equal
	// tohttps://rest.sandbox.netsuite.com/app/site/hosting/restlet.nl?script=165&deploy
	// =1 Instead, set OAuthUrl equal
	// tohttps://rest.sandbox.netsuite.com/app/site/hosting/restlet.nl and then
	// subsequently call AddParam for each query parameter.
	// 
	void put_OauthUrl(const char *newVal);

	// The oauth_version. Defaults to "1.0". May be set to the empty string to exclude.
	void get_OauthVersion(CkString &str);
	// The oauth_version. Defaults to "1.0". May be set to the empty string to exclude.
	const char *oauthVersion(void);
	// The oauth_version. Defaults to "1.0". May be set to the empty string to exclude.
	void put_OauthVersion(const char *newVal);

	// Contains the normalized set of request parameters that are signed. This is a
	// read-only property made available for debugging purposes.
	void get_QueryString(CkString &str);
	// Contains the normalized set of request parameters that are signed. This is a
	// read-only property made available for debugging purposes.
	const char *queryString(void);

	// The realm (optional).
	void get_Realm(CkString &str);
	// The realm (optional).
	const char *realm(void);
	// The realm (optional).
	void put_Realm(const char *newVal);

	// The generated base64 signature.
	void get_Signature(CkString &str);
	// The generated base64 signature.
	const char *signature(void);

	// The signature method. Defaults to "HMAC-SHA1". Other possible choices are
	// "HMAC-SHA256", "RSA-SHA1", and "RSA-SHA2".
	void get_SignatureMethod(CkString &str);
	// The signature method. Defaults to "HMAC-SHA1". Other possible choices are
	// "HMAC-SHA256", "RSA-SHA1", and "RSA-SHA2".
	const char *signatureMethod(void);
	// The signature method. Defaults to "HMAC-SHA1". Other possible choices are
	// "HMAC-SHA256", "RSA-SHA1", and "RSA-SHA2".
	void put_SignatureMethod(const char *newVal);

	// The timestamp, such as "1441632569".
	void get_Timestamp(CkString &str);
	// The timestamp, such as "1441632569".
	const char *timestamp(void);
	// The timestamp, such as "1441632569".
	void put_Timestamp(const char *newVal);

	// The token.
	void get_Token(CkString &str);
	// The token.
	const char *token(void);
	// The token.
	void put_Token(const char *newVal);

	// The token secret
	void get_TokenSecret(CkString &str);
	// The token secret
	const char *tokenSecret(void);
	// The token secret
	void put_TokenSecret(const char *newVal);

	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "INCLUDE_REALM" - Introduced in v9.5.0.85. Include the Realm in the
	//     signature calculation and outputs.
	// 
	void get_UncommonOptions(CkString &str);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "INCLUDE_REALM" - Introduced in v9.5.0.85. Include the Realm in the
	//     signature calculation and outputs.
	// 
	const char *uncommonOptions(void);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "INCLUDE_REALM" - Introduced in v9.5.0.85. Include the Realm in the
	//     signature calculation and outputs.
	// 
	void put_UncommonOptions(const char *newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Adds an extra name/value parameter to the OAuth1 signature.
	bool AddParam(const char *name, const char *value);


	// Generate the signature based on the property settings. Input properties are
	// OauthVersion, OauthMethod, Url, ConsumerKey, ConsumerSecret, Token, TokenSecret,
	// Nonce, and Timestamp. Properties set by this method include: BaseString,
	// Signature, HmacKey, EncodedSignature, QueryString, GeneratedUrl,
	// andAuthorizationHeader.
	bool Generate(void);


	// Generates a random nonce numBytes in length and sets the Nonce property to the hex
	// encoded value.
	bool GenNonce(int numBytes);


	// Sets the Timestamp property to the current date/time.
	bool GenTimestamp(void);


	// Removes a name/value parameter from the OAuth1 signature.
	bool RemoveParam(const char *name);


	// Sets the RSA key to be used when the SignatureMethod is set to "RSA-SHA1" or
	// "RSA-SHA2".
	bool SetRsaKey(CkPrivateKey &privKey);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
