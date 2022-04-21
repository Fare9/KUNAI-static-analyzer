// CkPrivateKeyW.h: interface for the CkPrivateKeyW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkPrivateKeyW_H
#define _CkPrivateKeyW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkWideCharBase.h"

class CkByteData;
class CkPublicKeyW;
class CkStringBuilderW;
class CkBinDataW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkPrivateKeyW
class CK_VISIBLE_PUBLIC CkPrivateKeyW  : public CkWideCharBase
{
	

	private:
	
	// Don't allow assignment or copying these objects.
	CkPrivateKeyW(const CkPrivateKeyW &);
	CkPrivateKeyW &operator=(const CkPrivateKeyW &);

    public:
	CkPrivateKeyW(void);
	virtual ~CkPrivateKeyW(void);

	

	static CkPrivateKeyW *createNew(void);
	

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The bit length (strength) of the private key.
	int get_BitLength(void);

	// The type of private key. Can be "empty", "rsa", "dsa", "ecc" (i.e. ECDSA), or
	// "ed25519".
	void get_KeyType(CkString &str);
	// The type of private key. Can be "empty", "rsa", "dsa", "ecc" (i.e. ECDSA), or
	// "ed25519".
	const wchar_t *keyType(void);

	// The encryption algorithm to be used when exporting the key to encrypted PKCS8.
	// The default value is "3des". Possible choices also include "aes128", "aes192",
	// and "aes256". All of the encryption algorithm choices use CBC mode.
	void get_Pkcs8EncryptAlg(CkString &str);
	// The encryption algorithm to be used when exporting the key to encrypted PKCS8.
	// The default value is "3des". Possible choices also include "aes128", "aes192",
	// and "aes256". All of the encryption algorithm choices use CBC mode.
	const wchar_t *pkcs8EncryptAlg(void);
	// The encryption algorithm to be used when exporting the key to encrypted PKCS8.
	// The default value is "3des". Possible choices also include "aes128", "aes192",
	// and "aes256". All of the encryption algorithm choices use CBC mode.
	void put_Pkcs8EncryptAlg(const wchar_t *newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Gets the private key in JWK (JSON Web Key) format.
	// 
	// RSA keys have this JWK format:
	//          {"kty":"RSA",
	//           "n":"0vx7agoebGcQ ... JzKnqDKgw",
	//           "e":"AQAB",
	//           "d":"X4cTteJY_gn4F ... 4jfcKoAC8Q",
	//           "p":"83i-7IvMGXoMX ... vn7O0nVbfs",
	//           "q":"3dfOR9cuYq-0S ... 4vIcb6yelxk",
	//           "dp":"G4sPXkc6Ya9 ... 8YeiKkTiBj0",
	//           "dq":"s9lAH9fggBso ... w494Q_cgk",
	//           "qi":"GyM_p6JrXySi ... zTKhAVRU"}
	// 
	// ECC keys have this JWK format.
	//          {"kty":"EC",
	//           "crv":"P-256",
	//           "x":"MKBCTNIcKUSDii11ySs3526iDZ8AiTo7Tu6KPAqv7D4",
	//           "y":"4Etl6SRW2YiLUrN5vfvVHuhp7x8PxltmWWlbbM4IFyM",
	//           "d":"870MB6gfuTJ4HtUnUvYMyJpr5eUZNP4Bk43bVdj3eAE"}
	// 
	// Ed25519 keys (added in v9.5.0.83) have this JWK format.
	//          {"kty": "OKP",
	//          "crv": "Ed25519",
	//          "x": "SE2Kne5xt51z1eciMH2T2ftDQp96Gl6FhY6zSQujiP0",
	//          "d": "O-eRXewadF0sNyB0U9omcnt8Qg2ZmeK3WSXPYgqe570",
	//          "use": "sig"}
	// 
	bool GetJwk(CkString &outStr);
	// Gets the private key in JWK (JSON Web Key) format.
	// 
	// RSA keys have this JWK format:
	//          {"kty":"RSA",
	//           "n":"0vx7agoebGcQ ... JzKnqDKgw",
	//           "e":"AQAB",
	//           "d":"X4cTteJY_gn4F ... 4jfcKoAC8Q",
	//           "p":"83i-7IvMGXoMX ... vn7O0nVbfs",
	//           "q":"3dfOR9cuYq-0S ... 4vIcb6yelxk",
	//           "dp":"G4sPXkc6Ya9 ... 8YeiKkTiBj0",
	//           "dq":"s9lAH9fggBso ... w494Q_cgk",
	//           "qi":"GyM_p6JrXySi ... zTKhAVRU"}
	// 
	// ECC keys have this JWK format.
	//          {"kty":"EC",
	//           "crv":"P-256",
	//           "x":"MKBCTNIcKUSDii11ySs3526iDZ8AiTo7Tu6KPAqv7D4",
	//           "y":"4Etl6SRW2YiLUrN5vfvVHuhp7x8PxltmWWlbbM4IFyM",
	//           "d":"870MB6gfuTJ4HtUnUvYMyJpr5eUZNP4Bk43bVdj3eAE"}
	// 
	// Ed25519 keys (added in v9.5.0.83) have this JWK format.
	//          {"kty": "OKP",
	//          "crv": "Ed25519",
	//          "x": "SE2Kne5xt51z1eciMH2T2ftDQp96Gl6FhY6zSQujiP0",
	//          "d": "O-eRXewadF0sNyB0U9omcnt8Qg2ZmeK3WSXPYgqe570",
	//          "use": "sig"}
	// 
	const wchar_t *getJwk(void);
	// Gets the private key in JWK (JSON Web Key) format.
	// 
	// RSA keys have this JWK format:
	//          {"kty":"RSA",
	//           "n":"0vx7agoebGcQ ... JzKnqDKgw",
	//           "e":"AQAB",
	//           "d":"X4cTteJY_gn4F ... 4jfcKoAC8Q",
	//           "p":"83i-7IvMGXoMX ... vn7O0nVbfs",
	//           "q":"3dfOR9cuYq-0S ... 4vIcb6yelxk",
	//           "dp":"G4sPXkc6Ya9 ... 8YeiKkTiBj0",
	//           "dq":"s9lAH9fggBso ... w494Q_cgk",
	//           "qi":"GyM_p6JrXySi ... zTKhAVRU"}
	// 
	// ECC keys have this JWK format.
	//          {"kty":"EC",
	//           "crv":"P-256",
	//           "x":"MKBCTNIcKUSDii11ySs3526iDZ8AiTo7Tu6KPAqv7D4",
	//           "y":"4Etl6SRW2YiLUrN5vfvVHuhp7x8PxltmWWlbbM4IFyM",
	//           "d":"870MB6gfuTJ4HtUnUvYMyJpr5eUZNP4Bk43bVdj3eAE"}
	// 
	// Ed25519 keys (added in v9.5.0.83) have this JWK format.
	//          {"kty": "OKP",
	//          "crv": "Ed25519",
	//          "x": "SE2Kne5xt51z1eciMH2T2ftDQp96Gl6FhY6zSQujiP0",
	//          "d": "O-eRXewadF0sNyB0U9omcnt8Qg2ZmeK3WSXPYgqe570",
	//          "use": "sig"}
	// 
	const wchar_t *jwk(void);

	// Returns the JWK thumbprint for the private key. This is the thumbprint of the
	// JSON Web Key (JWK) as per RFC 7638.
	bool GetJwkThumbprint(const wchar_t *hashAlg, CkString &outStr);
	// Returns the JWK thumbprint for the private key. This is the thumbprint of the
	// JSON Web Key (JWK) as per RFC 7638.
	const wchar_t *getJwkThumbprint(const wchar_t *hashAlg);
	// Returns the JWK thumbprint for the private key. This is the thumbprint of the
	// JSON Web Key (JWK) as per RFC 7638.
	const wchar_t *jwkThumbprint(const wchar_t *hashAlg);

	// Gets the private key in unencrypted binary DER format, preferring PKCS1 if
	// possible.
	// 
	// RSA keys are returned in PKCS1 ASN.1 DER format:
	// RSAPrivateKey ::= SEQUENCE {
	//     version           Version,
	//     modulus           INTEGER,  -- n
	//     publicExponent    INTEGER,  -- e
	//     privateExponent   INTEGER,  -- d
	//     prime1            INTEGER,  -- p
	//     prime2            INTEGER,  -- q
	//     exponent1         INTEGER,  -- d mod (p-1)
	//     exponent2         INTEGER,  -- d mod (q-1)
	//     coefficient       INTEGER,  -- (inverse of q) mod p
	//     otherPrimeInfos   OtherPrimeInfos OPTIONAL
	// }
	// 
	// DSA keys are returned in this ASN.1 DER format:
	// SEQUENCE(6 elem)
	//     INTEGER 0
	//     INTEGER(2048 bit) (p) 
	//     INTEGER(160 bit) (q) 
	//     INTEGER(2044 bit) (g) 
	//     INTEGER(2040 bit) (y - public key) 
	//     INTEGER(156 bit) (x - private key) 
	// 
	// ECC keys are returned in this ASN.1 DER format:
	// (from RFC 5915)
	// ECPrivateKey ::= SEQUENCE {
	//     version        INTEGER { ecPrivkeyVer1(1) } (ecPrivkeyVer1),
	//     privateKey     OCTET STRING,
	//     parameters [0] ECParameters {{ NamedCurve }} OPTIONAL,
	//     publicKey  [1] BIT STRING OPTIONAL (This is the ANSI X9.63 public key format.)
	// 
	bool GetPkcs1(CkByteData &outBytes);

	// Gets the private key in unencrypted binary DER format, preferring PKCS1 if
	// possible, and returns in an encoded string, as specified by the encoding argument.
	// 
	// RSA keys are returned in PKCS1 ASN.1 DER format:
	// RSAPrivateKey ::= SEQUENCE {
	//     version           Version,
	//     modulus           INTEGER,  -- n
	//     publicExponent    INTEGER,  -- e
	//     privateExponent   INTEGER,  -- d
	//     prime1            INTEGER,  -- p
	//     prime2            INTEGER,  -- q
	//     exponent1         INTEGER,  -- d mod (p-1)
	//     exponent2         INTEGER,  -- d mod (q-1)
	//     coefficient       INTEGER,  -- (inverse of q) mod p
	//     otherPrimeInfos   OtherPrimeInfos OPTIONAL
	// }
	// 
	// DSA keys are returned in this ASN.1 DER format:
	// SEQUENCE(6 elem)
	//     INTEGER 0
	//     INTEGER(2048 bit) (p) 
	//     INTEGER(160 bit) (q) 
	//     INTEGER(2044 bit) (g) 
	//     INTEGER(2040 bit) (y - public key) 
	//     INTEGER(156 bit) (x - private key) 
	// 
	// ECC keys are returned in this ASN.1 DER format:
	// (from RFC 5915)
	// ECPrivateKey ::= SEQUENCE {
	//     version        INTEGER { ecPrivkeyVer1(1) } (ecPrivkeyVer1),
	//     privateKey     OCTET STRING,
	//     parameters [0] ECParameters {{ NamedCurve }} OPTIONAL,
	//     publicKey  [1] BIT STRING OPTIONAL (This is the ANSI X9.63 public key format.)
	// 
	bool GetPkcs1ENC(const wchar_t *encoding, CkString &outStr);
	// Gets the private key in unencrypted binary DER format, preferring PKCS1 if
	// possible, and returns in an encoded string, as specified by the encoding argument.
	// 
	// RSA keys are returned in PKCS1 ASN.1 DER format:
	// RSAPrivateKey ::= SEQUENCE {
	//     version           Version,
	//     modulus           INTEGER,  -- n
	//     publicExponent    INTEGER,  -- e
	//     privateExponent   INTEGER,  -- d
	//     prime1            INTEGER,  -- p
	//     prime2            INTEGER,  -- q
	//     exponent1         INTEGER,  -- d mod (p-1)
	//     exponent2         INTEGER,  -- d mod (q-1)
	//     coefficient       INTEGER,  -- (inverse of q) mod p
	//     otherPrimeInfos   OtherPrimeInfos OPTIONAL
	// }
	// 
	// DSA keys are returned in this ASN.1 DER format:
	// SEQUENCE(6 elem)
	//     INTEGER 0
	//     INTEGER(2048 bit) (p) 
	//     INTEGER(160 bit) (q) 
	//     INTEGER(2044 bit) (g) 
	//     INTEGER(2040 bit) (y - public key) 
	//     INTEGER(156 bit) (x - private key) 
	// 
	// ECC keys are returned in this ASN.1 DER format:
	// (from RFC 5915)
	// ECPrivateKey ::= SEQUENCE {
	//     version        INTEGER { ecPrivkeyVer1(1) } (ecPrivkeyVer1),
	//     privateKey     OCTET STRING,
	//     parameters [0] ECParameters {{ NamedCurve }} OPTIONAL,
	//     publicKey  [1] BIT STRING OPTIONAL (This is the ANSI X9.63 public key format.)
	// 
	const wchar_t *getPkcs1ENC(const wchar_t *encoding);
	// Gets the private key in unencrypted binary DER format, preferring PKCS1 if
	// possible, and returns in an encoded string, as specified by the encoding argument.
	// 
	// RSA keys are returned in PKCS1 ASN.1 DER format:
	// RSAPrivateKey ::= SEQUENCE {
	//     version           Version,
	//     modulus           INTEGER,  -- n
	//     publicExponent    INTEGER,  -- e
	//     privateExponent   INTEGER,  -- d
	//     prime1            INTEGER,  -- p
	//     prime2            INTEGER,  -- q
	//     exponent1         INTEGER,  -- d mod (p-1)
	//     exponent2         INTEGER,  -- d mod (q-1)
	//     coefficient       INTEGER,  -- (inverse of q) mod p
	//     otherPrimeInfos   OtherPrimeInfos OPTIONAL
	// }
	// 
	// DSA keys are returned in this ASN.1 DER format:
	// SEQUENCE(6 elem)
	//     INTEGER 0
	//     INTEGER(2048 bit) (p) 
	//     INTEGER(160 bit) (q) 
	//     INTEGER(2044 bit) (g) 
	//     INTEGER(2040 bit) (y - public key) 
	//     INTEGER(156 bit) (x - private key) 
	// 
	// ECC keys are returned in this ASN.1 DER format:
	// (from RFC 5915)
	// ECPrivateKey ::= SEQUENCE {
	//     version        INTEGER { ecPrivkeyVer1(1) } (ecPrivkeyVer1),
	//     privateKey     OCTET STRING,
	//     parameters [0] ECParameters {{ NamedCurve }} OPTIONAL,
	//     publicKey  [1] BIT STRING OPTIONAL (This is the ANSI X9.63 public key format.)
	// 
	const wchar_t *pkcs1ENC(const wchar_t *encoding);

	// Gets the private key in non-encrypted PEM format, preferring PKCS1 over PKCS8 if
	// possible for the key type.
	bool GetPkcs1Pem(CkString &outStr);
	// Gets the private key in non-encrypted PEM format, preferring PKCS1 over PKCS8 if
	// possible for the key type.
	const wchar_t *getPkcs1Pem(void);
	// Gets the private key in non-encrypted PEM format, preferring PKCS1 over PKCS8 if
	// possible for the key type.
	const wchar_t *pkcs1Pem(void);

	// Gets the private key in unencrypted PKCS8 format.
	// 
	// RSA keys are returned in PKCS8 ASN.1 DER format:
	// SEQUENCE                  // PrivateKeyInfo
	// +- INTEGER                // Version - 0 (v1998)
	// +- SEQUENCE               // AlgorithmIdentifier
	//    +- OID                 // 1.2.840.113549.1.1.1
	//    +- NULL                // Optional Parameters
	// +- OCTETSTRING            // PrivateKey
	//    +- SEQUENCE            // RSAPrivateKey
	//       +- INTEGER(0)       // Version - v1998(0)
	//       +- INTEGER(N)       // N
	//       +- INTEGER(E)       // E
	//       +- INTEGER(D)       // D
	//       +- INTEGER(P)       // P
	//       +- INTEGER(Q)       // Q
	//       +- INTEGER(DP)      // d mod p-1
	//       +- INTEGER(DQ)      // d mod q-1
	//       +- INTEGER(Inv Q)   // INV(q) mod p
	// 
	// DSA keys are returned in this ASN.1 DER format:
	// SEQUENCE                 // PrivateKeyInfo
	// +- INTEGER                 // Version
	// +- SEQUENCE              // AlgorithmIdentifier
	//     +- OID                       // 1.2.840.10040.4.1
	//     +- SEQUENCE           // DSS-Params (Optional Parameters)
	// 	+- INTEGER(P)      // P
	// 	+- INTEGER(Q)      // Q
	// 	+- INTEGER(G)      // G
	//     +- OCTETSTRING             // PrivateKey
	// 	+- INTEGER(X)      // DSAPrivateKey X
	// 
	// ECC keys are returned in this ASN.1 DER format:
	// (from RFC 5915)
	// ECPrivateKey ::= SEQUENCE {
	//     version        INTEGER { ecPrivkeyVer1(1) } (ecPrivkeyVer1),
	//     privateKey     OCTET STRING,
	//     parameters [0] ECParameters {{ NamedCurve }} OPTIONAL,
	//     publicKey  [1] BIT STRING OPTIONAL (This is the ANSI X9.63 public key format.)
	// 
	bool GetPkcs8(CkByteData &outData);

	// Gets the private key in unencrypted PKCS8 format and returned in an encoded
	// string, as specified by the encoding argument.
	bool GetPkcs8ENC(const wchar_t *encoding, CkString &outStr);
	// Gets the private key in unencrypted PKCS8 format and returned in an encoded
	// string, as specified by the encoding argument.
	const wchar_t *getPkcs8ENC(const wchar_t *encoding);
	// Gets the private key in unencrypted PKCS8 format and returned in an encoded
	// string, as specified by the encoding argument.
	const wchar_t *pkcs8ENC(const wchar_t *encoding);

	// Writes the private key to password-protected PKCS8 format. The Pkcs8EncryptAlg
	// property controls the encryption algorithm used to encrypt.
	bool GetPkcs8Encrypted(const wchar_t *password, CkByteData &outBytes);

	// Writes the private key to password-protected PKCS8 format and returns as an
	// encoded string as specified by the encoding argument. The Pkcs8EncryptAlg property
	// controls the encryption algorithm used to encrypt.
	bool GetPkcs8EncryptedENC(const wchar_t *encoding, const wchar_t *password, CkString &outStr);
	// Writes the private key to password-protected PKCS8 format and returns as an
	// encoded string as specified by the encoding argument. The Pkcs8EncryptAlg property
	// controls the encryption algorithm used to encrypt.
	const wchar_t *getPkcs8EncryptedENC(const wchar_t *encoding, const wchar_t *password);
	// Writes the private key to password-protected PKCS8 format and returns as an
	// encoded string as specified by the encoding argument. The Pkcs8EncryptAlg property
	// controls the encryption algorithm used to encrypt.
	const wchar_t *pkcs8EncryptedENC(const wchar_t *encoding, const wchar_t *password);

	// Writes the private key to password-protected PKCS8 PEM format. The
	// Pkcs8EncryptAlg property controls the encryption algorithm used to encrypt.
	bool GetPkcs8EncryptedPem(const wchar_t *password, CkString &outStr);
	// Writes the private key to password-protected PKCS8 PEM format. The
	// Pkcs8EncryptAlg property controls the encryption algorithm used to encrypt.
	const wchar_t *getPkcs8EncryptedPem(const wchar_t *password);
	// Writes the private key to password-protected PKCS8 PEM format. The
	// Pkcs8EncryptAlg property controls the encryption algorithm used to encrypt.
	const wchar_t *pkcs8EncryptedPem(const wchar_t *password);

	// Gets the private key in PKCS8 PEM format.
	bool GetPkcs8Pem(CkString &outStr);
	// Gets the private key in PKCS8 PEM format.
	const wchar_t *getPkcs8Pem(void);
	// Gets the private key in PKCS8 PEM format.
	const wchar_t *pkcs8Pem(void);

	// Returns the public key portion of the private key as a public key object.
	// The caller is responsible for deleting the object returned by this method.
	CkPublicKeyW *GetPublicKey(void);

	// Returns the private key in raw hex format (lowercase). The public key is written
	// to pubKey.
	// 
	// Ed25519 private and public keys are 32-byte each (64 chars in hex format).
	// 
	// The length of an EC key depends on the curve. The private key is a single hex
	// string. The public key is a hex string composed of the "x" and "y" parts of the
	// public key like this:
	//     04||HEX(x)||HEX(y)
	// 
	// Note: This method is only applicable to Ed25519 and ECDSA keys. An RSA key
	// cannot be returned in such as simple raw format because it is composed of
	// multiple parts (modulus, exponent, and more).
	// 
	bool GetRawHex(CkStringBuilderW &pubKey, CkString &outStr);
	// Returns the private key in raw hex format (lowercase). The public key is written
	// to pubKey.
	// 
	// Ed25519 private and public keys are 32-byte each (64 chars in hex format).
	// 
	// The length of an EC key depends on the curve. The private key is a single hex
	// string. The public key is a hex string composed of the "x" and "y" parts of the
	// public key like this:
	//     04||HEX(x)||HEX(y)
	// 
	// Note: This method is only applicable to Ed25519 and ECDSA keys. An RSA key
	// cannot be returned in such as simple raw format because it is composed of
	// multiple parts (modulus, exponent, and more).
	// 
	const wchar_t *getRawHex(CkStringBuilderW &pubKey);
	// Returns the private key in raw hex format (lowercase). The public key is written
	// to pubKey.
	// 
	// Ed25519 private and public keys are 32-byte each (64 chars in hex format).
	// 
	// The length of an EC key depends on the curve. The private key is a single hex
	// string. The public key is a hex string composed of the "x" and "y" parts of the
	// public key like this:
	//     04||HEX(x)||HEX(y)
	// 
	// Note: This method is only applicable to Ed25519 and ECDSA keys. An RSA key
	// cannot be returned in such as simple raw format because it is composed of
	// multiple parts (modulus, exponent, and more).
	// 
	const wchar_t *rawHex(CkStringBuilderW &pubKey);

	// Gets the private key in PKCS1 DER format. This method is deprecated and is
	// replaced by the GetPkcs1Der method (given that this object may contain a non-RSA
	// key).
	bool GetRsaDer(CkByteData &outData);

	// Gets the private key in PKCS1 PEM format. This method is deprecated and is
	// replaced by the GetPkcs1Pem and GetPkcs8Pem methods (given that this object may
	// contain a non-RSA key).
	bool GetRsaPem(CkString &outStr);
	// Gets the private key in PKCS1 PEM format. This method is deprecated and is
	// replaced by the GetPkcs1Pem and GetPkcs8Pem methods (given that this object may
	// contain a non-RSA key).
	const wchar_t *getRsaPem(void);
	// Gets the private key in PKCS1 PEM format. This method is deprecated and is
	// replaced by the GetPkcs1Pem and GetPkcs8Pem methods (given that this object may
	// contain a non-RSA key).
	const wchar_t *rsaPem(void);

	// Returns the private key in XML format. The private key is returned unencrypted
	// and the parts are base64 encoded.
	// 
	// RSA keys have this XML format:
	//   ...  ...  
	// 
	// ...
	// 
	//   ...  ...  ...  ...  ...
	// 
	// DSA keys have this XML format:
	// 
	// ...
	// 
	// ............
	// 
	// ECC keys have this XML format. The CURVE_NAME could be one of secp256r1,
	// secp384r1, secp521r1, secp256k1 (or others as new curves are supported.)
	// ...
	// 
	bool GetXml(CkString &outStr);
	// Returns the private key in XML format. The private key is returned unencrypted
	// and the parts are base64 encoded.
	// 
	// RSA keys have this XML format:
	//   ...  ...  
	// 
	// ...
	// 
	//   ...  ...  ...  ...  ...
	// 
	// DSA keys have this XML format:
	// 
	// ...
	// 
	// ............
	// 
	// ECC keys have this XML format. The CURVE_NAME could be one of secp256r1,
	// secp384r1, secp521r1, secp256k1 (or others as new curves are supported.)
	// ...
	// 
	const wchar_t *getXml(void);
	// Returns the private key in XML format. The private key is returned unencrypted
	// and the parts are base64 encoded.
	// 
	// RSA keys have this XML format:
	//   ...  ...  
	// 
	// ...
	// 
	//   ...  ...  ...  ...  ...
	// 
	// DSA keys have this XML format:
	// 
	// ...
	// 
	// ............
	// 
	// ECC keys have this XML format. The CURVE_NAME could be one of secp256r1,
	// secp384r1, secp521r1, secp256k1 (or others as new curves are supported.)
	// ...
	// 
	const wchar_t *xml(void);

	// Loads a private key from any format (PKCS1, PKCS8, PEM, JWK, PVK, etc.). The
	// contents of the key (binary or text) is passed in privKeyData. The password is optional and
	// should be specified if needed.
	bool LoadAnyFormat(CkBinDataW &privKeyData, const wchar_t *password);

	// Loads a private key from a file in any format (PKCS1, PKCS8, PEM, JWK, PVK,
	// etc.). The password is optional and should be specified if needed.
	bool LoadAnyFormatFile(const wchar_t *path, const wchar_t *password);

	// Loads the private key object with an ed25519 key pair. The privKey is the 32-byte
	// private key as a hex string. The pubKey is the 32-byte public key as a hex string.
	// pubKey may be an empty string, in which case the public key is automatically
	// computed from the private key.
	bool LoadEd25519(const wchar_t *privKey, const wchar_t *pubKey);

	// Loads the private key from an in-memory encrypted PEM string. An encrypted PEM
	// contains the private key in encrypted PKCS#8 format, where the data begins and
	// ends with the following tags:
	// -----BEGIN ENCRYPTED PRIVATE KEY-----
	// BASE64 ENCODED DATA
	// -----END ENCRYPTED PRIVATE KEY-----
	// 
	// For those requiring a deeper understanding: The base64 data contains ASN.1 DER
	// with the following structure:
	// EncryptedPrivateKeyInfo ::= SEQUENCE {
	//   encryptionAlgorithm  EncryptionAlgorithmIdentifier,
	//   encryptedData        EncryptedData
	// }
	// 
	// EncryptionAlgorithmIdentifier ::= AlgorithmIdentifier
	// 
	// EncryptedData ::= OCTET STRING
	// 
	// Note: Each of the private key Load* methods willl auto-recognize the content and
	// will parse appropriately. The private key should be successfully loaded even
	// when the wrong format data is passed to the wrong method.
	// 
	bool LoadEncryptedPem(const wchar_t *pemStr, const wchar_t *password);

	// Loads a private key from an encrypted PEM file.
	// 
	// Note: Each of the private key Load* methods willl auto-recognize the content and
	// will parse appropriately. The private key should be successfully loaded even
	// when the wrong format data is passed to the wrong method.
	// 
	bool LoadEncryptedPemFile(const wchar_t *path, const wchar_t *password);

	// Loads a private key from an JWK (JSON Web Key) string.
	// 
	// RSA keys have this JWK format:
	//          {"kty":"RSA",
	//           "n":"0vx7agoebGcQ ... JzKnqDKgw",
	//           "e":"AQAB",
	//           "d":"X4cTteJY_gn4F ... 4jfcKoAC8Q",
	//           "p":"83i-7IvMGXoMX ... vn7O0nVbfs",
	//           "q":"3dfOR9cuYq-0S ... 4vIcb6yelxk",
	//           "dp":"G4sPXkc6Ya9 ... 8YeiKkTiBj0",
	//           "dq":"s9lAH9fggBso ... w494Q_cgk",
	//           "qi":"GyM_p6JrXySi ... zTKhAVRU"}
	// 
	// ECC keys have this JWK format.
	//          {"kty":"EC",
	//           "crv":"P-256",
	//           "x":"MKBCTNIcKUSDii11ySs3526iDZ8AiTo7Tu6KPAqv7D4",
	//           "y":"4Etl6SRW2YiLUrN5vfvVHuhp7x8PxltmWWlbbM4IFyM",
	//           "d":"870MB6gfuTJ4HtUnUvYMyJpr5eUZNP4Bk43bVdj3eAE"}
	// 
	// Ed25519 keys (added in v9.5.0.83) have this JWK format.
	//          {"kty": "OKP",
	//          "crv": "Ed25519",
	//          "x": "SE2Kne5xt51z1eciMH2T2ftDQp96Gl6FhY6zSQujiP0",
	//          "d": "O-eRXewadF0sNyB0U9omcnt8Qg2ZmeK3WSXPYgqe570",
	//          "use": "sig"}
	// 
	// Note: Each of the private key Load* methods willl auto-recognize the content and
	// will parse appropriately. The private key should be successfully loaded even
	// when the wrong format data is passed to the wrong method.
	// 
	bool LoadJwk(const wchar_t *jsonStr);

	// Loads the private key from an in-memory PEM string. If the PEM contains an
	// encrypted private key, then the LoadEncryptedPem method should instead be
	// called. This method is for loading an unencrypted private key stored in PEM
	// using PKCS#1 or PKCS#8.
	// 
	// A private key stored in PKCS#1 format begins and ends with the tags:
	// -----BEGIN RSA PRIVATE KEY-----
	// BASE64 ENCODED DATA
	// -----END RSA PRIVATE KEY-----
	// 
	// For those requiring a deeper understanding, the PKCS1 base64 contains ASN.1 in
	// DER encoding with the following structure:
	// RSAPrivateKey ::= SEQUENCE {
	//   version           Version,
	//   modulus           INTEGER,  -- n
	//   publicExponent    INTEGER,  -- e
	//   privateExponent   INTEGER,  -- d
	//   prime1            INTEGER,  -- p
	//   prime2            INTEGER,  -- q
	//   exponent1         INTEGER,  -- d mod (p-1)
	//   exponent2         INTEGER,  -- d mod (q-1)
	//   coefficient       INTEGER,  -- (inverse of q) mod p
	//   otherPrimeInfos   OtherPrimeInfos OPTIONAL
	// }
	// 
	// A private key stored in PKCS#8 format begins and ends with the tags:
	// -----BEGIN PRIVATE KEY-----
	// BASE64 ENCODED DATA
	// -----END PRIVATE KEY-----
	// 
	// For those requiring a deeper understanding, the PKCS8 base64 contains ASN.1 in
	// DER encoding with the following structure:
	// PrivateKeyInfo ::= SEQUENCE {
	//   version         Version,
	//   algorithm       AlgorithmIdentifier,
	//   PrivateKey      BIT STRING
	// }
	// 
	// AlgorithmIdentifier ::= SEQUENCE {
	//   algorithm       OBJECT IDENTIFIER,
	//   parameters      ANY DEFINED BY algorithm OPTIONAL
	// }
	// 
	// Note: Each of the private key Load* methods willl auto-recognize the content and
	// will parse appropriately. The private key should be successfully loaded even
	// when the wrong format data is passed to the wrong method.
	// 
	bool LoadPem(const wchar_t *str);

	// Loads a private key from a PEM file.
	bool LoadPemFile(const wchar_t *path);

	// Loads an RSA, ECC, or DSA private key from binary DER.
	// 
	// Note: Each of the private key Load* methods willl auto-recognize the content and
	// will parse appropriately. The private key should be successfully loaded even
	// when the wrong format data is passed to the wrong method.
	// 
	bool LoadPkcs1(CkByteData &data);

	// Loads a private key from a PKCS1 file.
	// 
	// Note: Each of the private key Load* methods willl auto-recognize the content and
	// will parse appropriately. The private key should be successfully loaded even
	// when the wrong format data is passed to the wrong method.
	// 
	bool LoadPkcs1File(const wchar_t *path);

	// Loads a private key from in-memory PKCS8 byte data.
	// 
	// For those requiring a deeper understanding, the PKCS8 contains ASN.1 in DER
	// encoding with the following structure:
	// PrivateKeyInfo ::= SEQUENCE {
	//   version         Version,
	//   algorithm       AlgorithmIdentifier,
	//   PrivateKey      BIT STRING
	// }
	// 
	// AlgorithmIdentifier ::= SEQUENCE {
	//   algorithm       OBJECT IDENTIFIER,
	//   parameters      ANY DEFINED BY algorithm OPTIONAL
	// }
	// 
	// Note: Each of the private key Load* methods willl auto-recognize the content and
	// will parse appropriately. The private key should be successfully loaded even
	// when the wrong format data is passed to the wrong method.
	// 
	bool LoadPkcs8(CkByteData &data);

	// Loads a private key from in-memory password-protected PKCS8 byte data.
	// 
	// For those requiring a deeper understanding, the encrypted PKCS8 contains ASN.1
	// in DER encoding with the following structure:
	// EncryptedPrivateKeyInfo ::= SEQUENCE {
	//   encryptionAlgorithm  EncryptionAlgorithmIdentifier,
	//   encryptedData        EncryptedData
	// }
	// 
	// EncryptionAlgorithmIdentifier ::= AlgorithmIdentifier
	// 
	// EncryptedData ::= OCTET STRING
	// 
	// Note: Each of the private key Load* methods willl auto-recognize the content and
	// will parse appropriately. The private key should be successfully loaded even
	// when the wrong format data is passed to the wrong method.
	// 
	bool LoadPkcs8Encrypted(CkByteData &data, const wchar_t *password);

	// Loads a private key from an encrypted PKCS8 file.
	// 
	// Note: Each of the private key Load* methods willl auto-recognize the content and
	// will parse appropriately. The private key should be successfully loaded even
	// when the wrong format data is passed to the wrong method.
	// 
	bool LoadPkcs8EncryptedFile(const wchar_t *path, const wchar_t *password);

	// Loads a private key from a PKCS8 file.
	// 
	// Note: Each of the private key Load* methods willl auto-recognize the content and
	// will parse appropriately. The private key should be successfully loaded even
	// when the wrong format data is passed to the wrong method.
	// 
	bool LoadPkcs8File(const wchar_t *path);

#if defined(CK_CRYPTOAPI_INCLUDED)
	// Loads a private key from in-memory PVK byte data.
	// 
	// Note: Each of the private key Load* methods willl auto-recognize the content and
	// will parse appropriately. The private key should be successfully loaded even
	// when the wrong format data is passed to the wrong method.
	// 
	bool LoadPvk(CkByteData &data, const wchar_t *password);
#endif

#if defined(CK_CRYPTOAPI_INCLUDED)
	// Loads a private key from a PVK format file.
	// 
	// Note: Each of the private key Load* methods willl auto-recognize the content and
	// will parse appropriately. The private key should be successfully loaded even
	// when the wrong format data is passed to the wrong method.
	// 
	bool LoadPvkFile(const wchar_t *path, const wchar_t *password);
#endif

	// This method is deprecated. Deprecated methods will be removed at some point in
	// the future. Applications should instead call LoadPkcs1.
	// 
	// Loads a private key from in-memory RSA PKCS#1 DER byte data.
	// 
	// For those requiring a deeper understanding, the PKCS1 contains ASN.1 in DER
	// encoding with the following structure:
	// RSAPrivateKey ::= SEQUENCE {
	//   version           Version,
	//   modulus           INTEGER,  -- n
	//   publicExponent    INTEGER,  -- e
	//   privateExponent   INTEGER,  -- d
	//   prime1            INTEGER,  -- p
	//   prime2            INTEGER,  -- q
	//   exponent1         INTEGER,  -- d mod (p-1)
	//   exponent2         INTEGER,  -- d mod (q-1)
	//   coefficient       INTEGER,  -- (inverse of q) mod p
	//   otherPrimeInfos   OtherPrimeInfos OPTIONAL
	// }
	// 
	// Note: Each of the private key Load* methods willl auto-recognize the content and
	// will parse appropriately. The private key should be successfully loaded even
	// when the wrong format data is passed to the wrong method.
	// 
	bool LoadRsaDer(CkByteData &data);

	// This method is deprecated. Deprecated methods will be removed at some point in
	// the future. Applications should instead call LoadPkcs1File.
	// 
	// Loads a private key from an RSA DER format file.
	// 
	// Note: Each of the private key Load* methods willl auto-recognize the content and
	// will parse appropriately. The private key should be successfully loaded even
	// when the wrong format data is passed to the wrong method.
	// 
	bool LoadRsaDerFile(const wchar_t *path);

	// Loads a private key from an XML string.
	// 
	// RSA keys have this XML format:
	//   ...  ...  
	// 
	// ...
	// 
	//   ...  ...  ...  ...  ...
	// 
	// DSA keys have this XML format:
	// 
	// ...
	// 
	// ............
	// 
	// ECC keys have this XML format. The CURVE_NAME could be one of secp256r1,
	// secp384r1, secp521r1, secp256k1 (or others as new curves are supported.)
	// ...
	// 
	// Note: Each of the private key Load* methods willl auto-recognize the content and
	// will parse appropriately. The private key should be successfully loaded even
	// when the wrong format data is passed to the wrong method.
	// 
	bool LoadXml(const wchar_t *xml);

	// Loads a private key from an XML file.
	// 
	// Note: Each of the private key Load* methods willl auto-recognize the content and
	// will parse appropriately. The private key should be successfully loaded even
	// when the wrong format data is passed to the wrong method.
	// 
	bool LoadXmlFile(const wchar_t *path);

	// Saves the private key to an unencrypted PKCS1 PEM format file.
	bool SavePemFile(const wchar_t *path);

	// Saves the private key to an unencrypted binary PKCS1 format file.
	bool SavePkcs1File(const wchar_t *path);

	// Saves the private key to a password-protected PKCS8 format file. The
	// Pkcs8EncryptAlg property controls the encryption algorithm used to encrypt.
	bool SavePkcs8EncryptedFile(const wchar_t *password, const wchar_t *path);

	// Saves the private key to a password-protected PKCS8 PEM format file. The
	// Pkcs8EncryptAlg property controls the encryption algorithm used to encrypt.
	bool SavePkcs8EncryptedPemFile(const wchar_t *password, const wchar_t *path);

	// Saves the private key to an unencrypted binary PKCS8 format file.
	bool SavePkcs8File(const wchar_t *path);

	// Saves the private key to a PKCS8 PEM format file.
	bool SavePkcs8PemFile(const wchar_t *path);

	// This method is deprecated and applications should instead call SavePkcs1File.
	// 
	// Saves the private key to a binary PKCS1 DER format file.
	// 
	bool SaveRsaDerFile(const wchar_t *path);

	// This method is deprecated. Applications should instead call SavePemFile.
	// 
	// Saves the private key to a PKCS1 PEM format file.
	// 
	bool SaveRsaPemFile(const wchar_t *path);

	// Saves the private key to an XML file.
	bool SaveXmlFile(const wchar_t *path);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
