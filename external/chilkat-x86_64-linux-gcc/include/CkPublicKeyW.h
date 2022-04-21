// CkPublicKeyW.h: interface for the CkPublicKeyW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkPublicKeyW_H
#define _CkPublicKeyW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkWideCharBase.h"

class CkByteData;
class CkBinDataW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkPublicKeyW
class CK_VISIBLE_PUBLIC CkPublicKeyW  : public CkWideCharBase
{
	

	private:
	
	// Don't allow assignment or copying these objects.
	CkPublicKeyW(const CkPublicKeyW &);
	CkPublicKeyW &operator=(const CkPublicKeyW &);

    public:
	CkPublicKeyW(void);
	virtual ~CkPublicKeyW(void);

	

	static CkPublicKeyW *createNew(void);
	

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// Indicates whether this object is empty or holds a public key.
	bool get_Empty(void);

	// Gets the size (in bits) of the public key. For example: 1024, 2048, etc.
	int get_KeySize(void);

	// The type of public key. Can be "empty", "rsa", "dsa", or "ecc".
	void get_KeyType(CkString &str);
	// The type of public key. Can be "empty", "rsa", "dsa", or "ecc".
	const wchar_t *keyType(void);



	// ----------------------
	// Methods
	// ----------------------
	// Returns the public key in binary DER format. If the key type (such as RSA)
	// supports both PKCS1 and PKCS8 formats, then preferPkcs1 determine which format is
	// returned.
	bool GetDer(bool preferPkcs1, CkByteData &outBytes);

	// Returns the public key in DER format as an encoded string (such as base64 or
	// hex). If the key type (such as RSA) supports both PKCS1 and PKCS8 formats, then
	// preferPkcs1 determine which format is returned. The encoding specifies the encoding, which
	// is typically "base64".
	bool GetEncoded(bool preferPkcs1, const wchar_t *encoding, CkString &outStr);
	// Returns the public key in DER format as an encoded string (such as base64 or
	// hex). If the key type (such as RSA) supports both PKCS1 and PKCS8 formats, then
	// preferPkcs1 determine which format is returned. The encoding specifies the encoding, which
	// is typically "base64".
	const wchar_t *getEncoded(bool preferPkcs1, const wchar_t *encoding);
	// Returns the public key in DER format as an encoded string (such as base64 or
	// hex). If the key type (such as RSA) supports both PKCS1 and PKCS8 formats, then
	// preferPkcs1 determine which format is returned. The encoding specifies the encoding, which
	// is typically "base64".
	const wchar_t *encoded(bool preferPkcs1, const wchar_t *encoding);

	// Gets the public key in JWK (JSON Web Key) format.
	// 
	// RSA public keys have this JWK format:
	//          {"kty":"RSA",
	//           "n": "0vx7agoebGcQSuuPiLJXZptN9 ... U8awapJzKnqDKgw",
	//           "e":"AQAB"}
	// 
	// ECC public keys have this JWK format:
	//          {"kty":"EC",
	//           "crv":"P-256",
	//           "x":"MKBCTNIcKUSDii11ySs3526iDZ8AiTo7Tu6KPAqv7D4",
	//           "y":"4Etl6SRW2YiLUrN5vfvVHuhp7x8PxltmWWlbbM4IFyM"}
	// 
	// Ed25519 public keys (added in v9.5.0.83) have this JWK format:
	//          {"kty":"OKP",
	//           "crv":"Ed25519",
	//           "x": "SE2Kne5xt51z1eciMH2T2ftDQp96Gl6FhY6zSQujiP0"}
	// 
	bool GetJwk(CkString &outStr);
	// Gets the public key in JWK (JSON Web Key) format.
	// 
	// RSA public keys have this JWK format:
	//          {"kty":"RSA",
	//           "n": "0vx7agoebGcQSuuPiLJXZptN9 ... U8awapJzKnqDKgw",
	//           "e":"AQAB"}
	// 
	// ECC public keys have this JWK format:
	//          {"kty":"EC",
	//           "crv":"P-256",
	//           "x":"MKBCTNIcKUSDii11ySs3526iDZ8AiTo7Tu6KPAqv7D4",
	//           "y":"4Etl6SRW2YiLUrN5vfvVHuhp7x8PxltmWWlbbM4IFyM"}
	// 
	// Ed25519 public keys (added in v9.5.0.83) have this JWK format:
	//          {"kty":"OKP",
	//           "crv":"Ed25519",
	//           "x": "SE2Kne5xt51z1eciMH2T2ftDQp96Gl6FhY6zSQujiP0"}
	// 
	const wchar_t *getJwk(void);
	// Gets the public key in JWK (JSON Web Key) format.
	// 
	// RSA public keys have this JWK format:
	//          {"kty":"RSA",
	//           "n": "0vx7agoebGcQSuuPiLJXZptN9 ... U8awapJzKnqDKgw",
	//           "e":"AQAB"}
	// 
	// ECC public keys have this JWK format:
	//          {"kty":"EC",
	//           "crv":"P-256",
	//           "x":"MKBCTNIcKUSDii11ySs3526iDZ8AiTo7Tu6KPAqv7D4",
	//           "y":"4Etl6SRW2YiLUrN5vfvVHuhp7x8PxltmWWlbbM4IFyM"}
	// 
	// Ed25519 public keys (added in v9.5.0.83) have this JWK format:
	//          {"kty":"OKP",
	//           "crv":"Ed25519",
	//           "x": "SE2Kne5xt51z1eciMH2T2ftDQp96Gl6FhY6zSQujiP0"}
	// 
	const wchar_t *jwk(void);

	// Returns the JWK thumbprint for the public key. This is the thumbprint of the
	// JSON Web Key (JWK) as per RFC 7638.
	bool GetJwkThumbprint(const wchar_t *hashAlg, CkString &outStr);
	// Returns the JWK thumbprint for the public key. This is the thumbprint of the
	// JSON Web Key (JWK) as per RFC 7638.
	const wchar_t *getJwkThumbprint(const wchar_t *hashAlg);
	// Returns the JWK thumbprint for the public key. This is the thumbprint of the
	// JSON Web Key (JWK) as per RFC 7638.
	const wchar_t *jwkThumbprint(const wchar_t *hashAlg);

	// This method is deprecated. Applications should call GetDer with preference for
	// PKCS8 instead.
	// 
	// Gets the public key in PKCS8 DER format.
	// 
	bool GetOpenSslDer(CkByteData &outData);

	// This method is deprecated. Applications should call GetPem with preference for
	// PKCS8 instead.
	// 
	// Gets the public key in PKCS8 PEM format.
	// 
	bool GetOpenSslPem(CkString &outStr);
	// This method is deprecated. Applications should call GetPem with preference for
	// PKCS8 instead.
	// 
	// Gets the public key in PKCS8 PEM format.
	// 
	const wchar_t *getOpenSslPem(void);
	// This method is deprecated. Applications should call GetPem with preference for
	// PKCS8 instead.
	// 
	// Gets the public key in PKCS8 PEM format.
	// 
	const wchar_t *openSslPem(void);

	// Returns the public key in PEM format. If the key type (such as RSA) supports
	// both PKCS1 and PKCS8 formats, then preferPkcs1 determine which format is returned.
	bool GetPem(bool preferPkcs1, CkString &outStr);
	// Returns the public key in PEM format. If the key type (such as RSA) supports
	// both PKCS1 and PKCS8 formats, then preferPkcs1 determine which format is returned.
	const wchar_t *getPem(bool preferPkcs1);
	// Returns the public key in PEM format. If the key type (such as RSA) supports
	// both PKCS1 and PKCS8 formats, then preferPkcs1 determine which format is returned.
	const wchar_t *pem(bool preferPkcs1);

	// This method is deprecated. Applications should call GetEncoded with preference
	// for PKCS1 instead.
	// 
	// Gets the public key in PKCS1 format and returns in an encoded string, as
	// specified by the encoding argument.
	// 
	bool GetPkcs1ENC(const wchar_t *encoding, CkString &outStr);
	// This method is deprecated. Applications should call GetEncoded with preference
	// for PKCS1 instead.
	// 
	// Gets the public key in PKCS1 format and returns in an encoded string, as
	// specified by the encoding argument.
	// 
	const wchar_t *getPkcs1ENC(const wchar_t *encoding);
	// This method is deprecated. Applications should call GetEncoded with preference
	// for PKCS1 instead.
	// 
	// Gets the public key in PKCS1 format and returns in an encoded string, as
	// specified by the encoding argument.
	// 
	const wchar_t *pkcs1ENC(const wchar_t *encoding);

	// This method is deprecated. Applications should call GetEncoded with preference
	// for PKCS8 instead.
	// 
	// Gets the public key in PKCS8 format and returns in an encoded string, as
	// specified by the encoding argument.
	// 
	bool GetPkcs8ENC(const wchar_t *encoding, CkString &outStr);
	// This method is deprecated. Applications should call GetEncoded with preference
	// for PKCS8 instead.
	// 
	// Gets the public key in PKCS8 format and returns in an encoded string, as
	// specified by the encoding argument.
	// 
	const wchar_t *getPkcs8ENC(const wchar_t *encoding);
	// This method is deprecated. Applications should call GetEncoded with preference
	// for PKCS8 instead.
	// 
	// Gets the public key in PKCS8 format and returns in an encoded string, as
	// specified by the encoding argument.
	// 
	const wchar_t *pkcs8ENC(const wchar_t *encoding);

	// This method is deprecated. Applications should call GetDer with preference for
	// PKCS1 instead.
	// 
	// Gets the public key in PKCS1 DER format.
	// 
	bool GetRsaDer(CkByteData &outData);

	// Gets the public key in XML format. The format depends on the key type. The key
	// parts indicated by "..." are base64 encoded.
	// 
	// RSA public keys have this XML format:
	//   ...  ...
	// 
	// DSA public keys have this XML format:
	// 
	// ...
	// 
	// .........
	// 
	// ECC public keys have this XML format:
	// ...
	// 
	bool GetXml(CkString &outStr);
	// Gets the public key in XML format. The format depends on the key type. The key
	// parts indicated by "..." are base64 encoded.
	// 
	// RSA public keys have this XML format:
	//   ...  ...
	// 
	// DSA public keys have this XML format:
	// 
	// ...
	// 
	// .........
	// 
	// ECC public keys have this XML format:
	// ...
	// 
	const wchar_t *getXml(void);
	// Gets the public key in XML format. The format depends on the key type. The key
	// parts indicated by "..." are base64 encoded.
	// 
	// RSA public keys have this XML format:
	//   ...  ...
	// 
	// DSA public keys have this XML format:
	// 
	// ...
	// 
	// .........
	// 
	// ECC public keys have this XML format:
	// ...
	// 
	const wchar_t *xml(void);

	// Loads a public key from base64-encoded DER (can be PKCS1 or PKCS8).
	bool LoadBase64(const wchar_t *keyStr);

	// Loads a public key from any binary or string format where the data is contained
	// in bd. Chilkat automatically recognizes the format and key type (RSA, EC, DSA,
	// Ed25519, ..)
	bool LoadBd(CkBinDataW &bd);

	// Loads an ECDSA public key directly from Qx, Qy values specified as a hex
	// strings. The curveName can be one of the following:
	//     secp256r1
	//     secp384r1
	//     secp521r1
	//     secp256k1
	//     secp192r1
	//     secp224r1
	//     brainpoolp256r1
	//     brainpoolp160r1
	//     brainpoolp192r1
	//     brainpoolp224r1
	//     brainpoolp320r1
	//     brainpoolp384r1
	//     brainpoolp512r1
	// 
	// Note: ECDSA public keys of various formats, such as PKCS8, PKCS1, JWK, XML,
	// binary DER, PEM, etc., can be loaded using the LoadBd, LoadFromBinary,
	// LoadFromFile, and LoadFromString methods.
	// 
	bool LoadEcdsa(const wchar_t *curveName, const wchar_t *Qx, const wchar_t *Qy);

	// Loads the public key object from a 32-byte ed25519 key specified as a hex
	// string.
	bool LoadEd25519(const wchar_t *pubKey);

	// Loads a public key from binary DER. Auto-recognizes both PKCS1 and PKCS8
	// formats.
	bool LoadFromBinary(CkByteData &keyBytes);

	// Loads a public key from a file. The file can be in any string or binary format
	// such as binary DER (PKCS1 or PKCS8), PEM, XML, or encoded binary DER (such as
	// base64 encoded binary DER). The format of the contents of the file is
	// auto-recognized.
	// 
	// Starting in version 9.5.0.66, this method also supports loading the JWK (JSON
	// Web Key) format.
	// 
	bool LoadFromFile(const wchar_t *path);

	// Loads a public key from any string format, such as PEM, XML, or encoded binary
	// DER (such as base64 encoded binary DER). The format of the keyString is
	// auto-recognized.
	// 
	// Starting in version 9.5.0.66, this method also supports loading the JWK (JSON
	// Web Key) format.
	// 
	bool LoadFromString(const wchar_t *keyString);

	// This method is deprecated. Applications should call LoadFromBinary instead.
	// 
	// Loads a public key from in-memory PKCS8 DER formatted byte data.
	// 
	bool LoadOpenSslDer(CkByteData &data);

	// This method is deprecated. Applications should call LoadFromFile instead.
	// 
	// Loads a public key from an PKCS8 DER format file.
	// 
	bool LoadOpenSslDerFile(const wchar_t *path);

	// This method is deprecated. Applications should call LoadFromString instead.
	// 
	// Loads a public key from an PKCS8 PEM string.
	// 
	bool LoadOpenSslPem(const wchar_t *str);

	// This method is deprecated. Applications should call LoadFromFile instead.
	// 
	// Loads a public key from an PKCS8 PEM file.
	// 
	bool LoadOpenSslPemFile(const wchar_t *path);

	// This method is deprecated. Applications should call LoadFromString instead.
	// 
	// Loads an RSA public key from PKCS#1 PEM format.
	// 
	bool LoadPkcs1Pem(const wchar_t *str);

	// This method is deprecated. Applications should call LoadFromBinary instead.
	// 
	// Loads a public key from in-memory PKCS1 DER formatted byte data.
	// 
	bool LoadRsaDer(CkByteData &data);

	// This method is deprecated. Applications should call LoadFromFile instead.
	// 
	// Loads a public key from an PKCS1 DER formatted file.
	// 
	bool LoadRsaDerFile(const wchar_t *path);

	// This method is deprecated. Applications should call LoadFromString instead.
	// 
	// Loads a public key from an XML string.
	// 
	bool LoadXml(const wchar_t *xml);

	// This method is deprecated. Applications should call LoadFromFile instead.
	// 
	// Loads a public key from an XML file.
	// 
	bool LoadXmlFile(const wchar_t *path);

	// Saves the public key to a file in binary DER format. If the key type (such as
	// RSA) supports both PKCS1 and PKCS8 formats, then preferPkcs1 determine which format is
	// returned.
	bool SaveDerFile(bool preferPkcs1, const wchar_t *path);

	// This method is deprecated. Applications should call SaveDerFile with preference
	// for PKCS8 instead.
	// 
	// Saves the public key to an PKCS8 DER format file.
	// 
	bool SaveOpenSslDerFile(const wchar_t *path);

	// This method is deprecated. Applications should call SavePemFile with preference
	// for PKCS8 instead.
	// 
	// Saves the public key to an PKCS8 PEM format file.
	// 
	bool SaveOpenSslPemFile(const wchar_t *path);

	// Saves the public key to a file in PEM format. If the key type (such as RSA)
	// supports both PKCS1 and PKCS8 formats, then preferPkcs1 determine which format is
	// returned.
	bool SavePemFile(bool preferPkcs1, const wchar_t *path);

	// This method is deprecated. Applications should call SaveDerFile with preference
	// for PKCS1 instead.
	// 
	// Saves the public key to an PKCS1 DER format file.
	// 
	bool SaveRsaDerFile(const wchar_t *path);

	// Saves the public key to an XML file.
	bool SaveXmlFile(const wchar_t *path);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
