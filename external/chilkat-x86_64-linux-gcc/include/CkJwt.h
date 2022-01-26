// CkJwt.h: interface for the CkJwt class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkJwt_H
#define _CkJwt_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkPrivateKey;
class CkPublicKey;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkJwt
class CK_VISIBLE_PUBLIC CkJwt  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkJwt(const CkJwt &);
	CkJwt &operator=(const CkJwt &);

    public:
	CkJwt(void);
	virtual ~CkJwt(void);

	static CkJwt *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// If true, the JSON passed to CreateJwt and CreateJwtPk will be compacted to
	// remove unnecessary whitespace. This will result in the smallest possible JWT.
	// The default value is true.
	bool get_AutoCompact(void);
	// If true, the JSON passed to CreateJwt and CreateJwtPk will be compacted to
	// remove unnecessary whitespace. This will result in the smallest possible JWT.
	// The default value is true.
	void put_AutoCompact(bool newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Creates a JWT. The header is the JOSE JSON header. It can be the full JOSE JSON,
	// or it can be a shorthand string such as "HS256", "HS384", or "HS512", in which
	// case the standard JOSE header for the given algorithm will be used.
	// 
	// The payload is the JSON payload that contains the claims. The password is the secret.
	// Given that the secret is a shared passwod string, this method should only be
	// called for creating JWT's where the JOSE header's "alg" is HS256, HS384, or
	// HS512. For RS256, RS384, RS512, ES256, ES384, and ES512, call CreateJwtPk
	// instead.
	// 
	// When successful, this method returns a JWT with the format xxxxx.yyyyy.zzzzz,
	// where xxxxx is the base64url encoded JOSE header, yyyyy is the base64url encoded
	// payload, and zzzzz is the base64url signature.
	// 
	bool CreateJwt(const char *header, const char *payload, const char *password, CkString &outStr);

	// Creates a JWT. The header is the JOSE JSON header. It can be the full JOSE JSON,
	// or it can be a shorthand string such as "HS256", "HS384", or "HS512", in which
	// case the standard JOSE header for the given algorithm will be used.
	// 
	// The payload is the JSON payload that contains the claims. The password is the secret.
	// Given that the secret is a shared passwod string, this method should only be
	// called for creating JWT's where the JOSE header's "alg" is HS256, HS384, or
	// HS512. For RS256, RS384, RS512, ES256, ES384, and ES512, call CreateJwtPk
	// instead.
	// 
	// When successful, this method returns a JWT with the format xxxxx.yyyyy.zzzzz,
	// where xxxxx is the base64url encoded JOSE header, yyyyy is the base64url encoded
	// payload, and zzzzz is the base64url signature.
	// 
	const char *createJwt(const char *header, const char *payload, const char *password);

	// Creates a JWT using an RSA or ECC private key. The header is the JOSE JSON header.
	// It can be the full JOSE JSON, or it can be a shorthand string such as "RS256",
	// "RS384", "RS512", "ES256", "ES384", or "ES512", in which case the standard JOSE
	// header for the given algorithm will be used.
	// 
	// The payload is the JSON payload that contains the claims. The key is the private
	// key. This method should only be called for creating JWT's where the JOSE
	// header's "alg" is RS256, RS384, RS512, ES256, ES384, and ES512. If the secret is
	// a shared password string, then call CreateJwt instead.
	// 
	// When successful, this method returns a JWT with the format xxxxx.yyyyy.zzzzz,
	// where xxxxx is the base64url encoded JOSE header, yyyyy is the base64url encoded
	// payload, and zzzzz is the base64url signature.
	// 
	bool CreateJwtPk(const char *header, const char *payload, CkPrivateKey &key, CkString &outStr);

	// Creates a JWT using an RSA or ECC private key. The header is the JOSE JSON header.
	// It can be the full JOSE JSON, or it can be a shorthand string such as "RS256",
	// "RS384", "RS512", "ES256", "ES384", or "ES512", in which case the standard JOSE
	// header for the given algorithm will be used.
	// 
	// The payload is the JSON payload that contains the claims. The key is the private
	// key. This method should only be called for creating JWT's where the JOSE
	// header's "alg" is RS256, RS384, RS512, ES256, ES384, and ES512. If the secret is
	// a shared password string, then call CreateJwt instead.
	// 
	// When successful, this method returns a JWT with the format xxxxx.yyyyy.zzzzz,
	// where xxxxx is the base64url encoded JOSE header, yyyyy is the base64url encoded
	// payload, and zzzzz is the base64url signature.
	// 
	const char *createJwtPk(const char *header, const char *payload, CkPrivateKey &key);

	// Generates a JSON numeric value representing the number of seconds from
	// 1970-01-01T00:00:00Z UTC until the specified UTC date/time, ignoring leap
	// seconds. The date/time generated is equal to the current system time plus the
	// number of seconds specified by numSecOffset. The numSecOffset can be negative.
	int GenNumericDate(int numSecOffset);


	// Decodes the first part of a JWT (the "xxxxx" part of the "xxxxx.yyyyy.zzzzz"
	// JWT) and returns the JSON string. This is the JOSE header of the JWT.
	bool GetHeader(const char *token, CkString &outStr);

	// Decodes the first part of a JWT (the "xxxxx" part of the "xxxxx.yyyyy.zzzzz"
	// JWT) and returns the JSON string. This is the JOSE header of the JWT.
	const char *getHeader(const char *token);
	// Decodes the first part of a JWT (the "xxxxx" part of the "xxxxx.yyyyy.zzzzz"
	// JWT) and returns the JSON string. This is the JOSE header of the JWT.
	const char *header(const char *token);


	// Decodes the second part of a JWT (the "yyyyy" part of the "xxxxx.yyyyy.zzzzz"
	// JWT) and returns the JSON string. This is the claims payload of the JWT.
	bool GetPayload(const char *token, CkString &outStr);

	// Decodes the second part of a JWT (the "yyyyy" part of the "xxxxx.yyyyy.zzzzz"
	// JWT) and returns the JSON string. This is the claims payload of the JWT.
	const char *getPayload(const char *token);
	// Decodes the second part of a JWT (the "yyyyy" part of the "xxxxx.yyyyy.zzzzz"
	// JWT) and returns the JSON string. This is the claims payload of the JWT.
	const char *payload(const char *token);


	// Verifies the "exp" and/or "nbf" claims and returns true if the current system
	// date/time is within range. Returns false if the current system date/time is
	// outside the allowed range of time. The leeway may be set to a non-zero number of
	// seconds to allow for some small leeway (usually no more than a few minutes) to
	// account for clock skew.
	bool IsTimeValid(const char *jwt, int leeway);


	// Verifies a JWT that requires a shared password string for verification. The token
	// should be a JWT with the format xxxxx.yyyyy.zzzzz. This method should only be
	// called for JWT's using the HS256, HS384, or HS512 algorithms. The VerifyJwtPk
	// method should be called for verifying JWT's requiring an RSA or ECC key.
	// 
	// Returns true if the signature was verified. Returns false if the signature
	// was not successfully verified.
	// 
	// Note: This method will return false if the "alg" in the JOSE header is
	// anything other than the algorithms specifically for HMAC, namely "hs256,
	// "hs384", and "hs512". For example, if the "alg" is "none", then this method
	// immediately returns a failed status.
	// 
	// Further Explanation: This method calculates the signature using the password
	// provided by the application, and compares it against the signature found in the
	// JWT. If the signatures are equal, then the password is correct, and true is
	// returned.
	// 
	bool VerifyJwt(const char *token, const char *password);


	// Verifies a JWT that requires an RSA or ECC public key for verification. The token
	// should be a JWT with the format xxxxx.yyyyy.zzzzz. This method should only be
	// called for JWT's using the RS256, RS384, RS512, ES256, ES384, or ES512
	// algorithms.
	// 
	// Returns true if the signature was verified. Returns false if the signature
	// was not successfully verified.
	// 
	// Note: This method will return false if the "alg" in the JOSE header is
	// anything other than the algorithms specifically for RSA and ECC. For example, if
	// the "alg" is "none", then this method immediately returns a failed status.
	// 
	// Further Explanation: This method calculates the signature using the key
	// provided by the application, and compares it against the signature found in the
	// JWT. If the signatures are equal, then the key corresponds to the private key
	// used to sign, and true is returned.
	// 
	bool VerifyJwtPk(const char *token, CkPublicKey &key);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
