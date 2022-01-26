// CkJavaKeyStore.h: interface for the CkJavaKeyStore class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkJavaKeyStore_H
#define _CkJavaKeyStore_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkPfx;
class CkCert;
class CkCertChain;
class CkPrivateKey;
class CkBinData;
class CkByteData;
class CkJsonObject;
class CkStringBuilder;
class CkPem;
class CkXmlCertVault;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkJavaKeyStore
class CK_VISIBLE_PUBLIC CkJavaKeyStore  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkJavaKeyStore(const CkJavaKeyStore &);
	CkJavaKeyStore &operator=(const CkJavaKeyStore &);

    public:
	CkJavaKeyStore(void);
	virtual ~CkJavaKeyStore(void);

	static CkJavaKeyStore *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The number of private keys contained within the keystore. Each private key has
	// an alias and certificate chain associated with it.
	int get_NumPrivateKeys(void);

	// The number of secret keys (such as AES keys) contained within the keystore. Each
	// secret key can have an alias associated with it.
	int get_NumSecretKeys(void);

	// The number of trusted certificates contained within the keystore. Each
	// certificate has an alias (identifying string) associated with it.
	int get_NumTrustedCerts(void);

	// If true, then adding a private key to the JKS only succeeds if the certificate
	// chain can be completed to the root certificate. A root certificate is either a
	// trusted CA root or a self-signed certificate. If false, then incomplete
	// certificate chains are allowed. The default value is true.
	bool get_RequireCompleteChain(void);
	// If true, then adding a private key to the JKS only succeeds if the certificate
	// chain can be completed to the root certificate. A root certificate is either a
	// trusted CA root or a self-signed certificate. If false, then incomplete
	// certificate chains are allowed. The default value is true.
	void put_RequireCompleteChain(bool newVal);

	// If true, then the keystore's keyed digest is required to pass validation
	// (password required) for any of the load methods (such as LoadFile, LoadBinary,
	// or LoadEncoded). If false, then a keystore may be loaded into memory without
	// password validation (if a null or empty string is passed to the load method).
	// The default value of this property is true.
	bool get_VerifyKeyedDigest(void);
	// If true, then the keystore's keyed digest is required to pass validation
	// (password required) for any of the load methods (such as LoadFile, LoadBinary,
	// or LoadEncoded). If false, then a keystore may be loaded into memory without
	// password validation (if a null or empty string is passed to the load method).
	// The default value of this property is true.
	void put_VerifyKeyedDigest(bool newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Adds the contents of a PFX or PKCS #12 (.p12) to the Java keystore object. One
	// JKS entry per private key found in the PKCS12 is added. The certs found within
	// the PCKS12 are used to build the certificate chains for each private key. (A
	// typical PFX file contains a single private key along with its associated
	// certificate, and the certificates in the chain of authentication to the root CA
	// cert.)
	// 
	// This method does not add trusted certificate entries to the JKS.
	// 
	// The specified alias is applied to the 1st private key found. If the alias is
	// empty, then the alias is obtained from the cert/PFX in the following order of
	// preference:
	//     Certificate's subject common name
	//     Certificate's subject email address
	//     Certificate's friendly name found in the PKCS9 attributes of the PKCS12
	//     Certificate's serial number
	// 
	// If multiple private keys are found in the PKCS12, then all but the first will
	// automaticallly be assigned aliases using the preference just described.
	// 
	// The UseCertVault method may be called to provide additional certificates for the
	// automatic construction of the certificate chains. If the RequireCompleteChain
	// property is set to true, then this method will fail if any certificate chain
	// is not completed to the root. The TrustedRoots class may be used to provide a
	// source for obtaining trusted CA roots if these are not already present within
	// the PKCS12.
	// 
	bool AddPfx(CkPfx &pfx, const char *alias, const char *password);


	// Adds a private key entry to the JKS. Both the private key and certificate chain
	// are obtained from the certificate object that is passed in the 1st argument.
	// 
	// If the alias is empty, then the alias is automatically chosen based on the
	// certificate's information, in the following order of preference:
	//     Certificate's subject common name
	//     Certificate's subject email address
	//     Certificate's serial number
	// 
	// The UseCertVault method may be called to provide additional certificates for the
	// automatic construction of the certificate chains. If the RequireCompleteChain
	// property is set to true, then this method will fail if the certificate chain
	// is not completed to the root. The TrustedRoots class may be used to provide a
	// source for obtaining trusted CA roots.
	// 
	bool AddPrivateKey(CkCert &cert, const char *alias, const char *password);


	// Adds a secret (symmetric) key entry to the JKS. This adds a symmetric key, which
	// is simply a number of binary bytes (such as 16 bytes for a 128-bit AES key). The
	// encodedKeyBytes provides the actual bytes of the symmetric key, in an encoded string form.
	// The encoding indicates the encoding of encodedKeyBytes (such as "base64", "hex", "base64url",
	// etc.) The algorithm describes the symmetric algorithm, such as "AES". The alias is the
	// password used to seal (encrypt) the key bytes.
	// 
	// Note: The algorithm describes the usage of the encodedKeyBytes. For example, if encodedKeyBytes contains
	// the 16 bytes of a 128-bit AES key, then algorithm should be set to "AES". The actual
	// encryption algorithm used to seal the key within the JCEKS is
	// PBEWithMD5AndTripleDES, which is part of the JCEKS specification.
	// 
	bool AddSecretKey(const char *encodedKeyBytes, const char *encoding, const char *algorithm, const char *alias, const char *password);


	// Adds a trusted certificate to the Java keystore object.
	bool AddTrustedCert(CkCert &cert, const char *alias);


	// Changes the password for a private key.
	bool ChangePassword(int index, const char *oldPassword, const char *newPassword);


	// Finds and returns the certificate chain for the private key with the specified
	// alias.
	// The caller is responsible for deleting the object returned by this method.
	CkCertChain *FindCertChain(const char *alias, bool caseSensitive);


	// Finds and returns the private key with the specified alias.
	// The caller is responsible for deleting the object returned by this method.
	CkPrivateKey *FindPrivateKey(const char *password, const char *alias, bool caseSensitive);


	// Finds and returns the trusted certificate with the specified alias.
	// The caller is responsible for deleting the object returned by this method.
	CkCert *FindTrustedCert(const char *alias, bool caseSensitive);


	// Returns the certificate chain associated with the Nth private key contained
	// within the keystore. The 1st private key is at index 0.
	// The caller is responsible for deleting the object returned by this method.
	CkCertChain *GetCertChain(int index);


	// Returns the Nth private key contained within the keystore. The 1st private key
	// is at index 0.
	// The caller is responsible for deleting the object returned by this method.
	CkPrivateKey *GetPrivateKey(const char *password, int index);


	// Returns the Nth private key alias contained within the keystore. The 1st private
	// key is at index 0.
	bool GetPrivateKeyAlias(int index, CkString &outStr);

	// Returns the Nth private key alias contained within the keystore. The 1st private
	// key is at index 0.
	const char *getPrivateKeyAlias(int index);
	// Returns the Nth private key alias contained within the keystore. The 1st private
	// key is at index 0.
	const char *privateKeyAlias(int index);


	// Returns the Nth secret key contained within the keystore. The 1st secret key is
	// at index 0. The bytes of the secret key are returned in the specified encoding.
	// (such as hex, base64, base64url, etc.)
	bool GetSecretKey(const char *password, int index, const char *encoding, CkString &outStr);

	// Returns the Nth secret key contained within the keystore. The 1st secret key is
	// at index 0. The bytes of the secret key are returned in the specified encoding.
	// (such as hex, base64, base64url, etc.)
	const char *getSecretKey(const char *password, int index, const char *encoding);
	// Returns the Nth secret key contained within the keystore. The 1st secret key is
	// at index 0. The bytes of the secret key are returned in the specified encoding.
	// (such as hex, base64, base64url, etc.)
	const char *secretKey(const char *password, int index, const char *encoding);


	// Returns the Nth secret key alias contained within the keystore. The 1st secret
	// key is at index 0.
	bool GetSecretKeyAlias(int index, CkString &outStr);

	// Returns the Nth secret key alias contained within the keystore. The 1st secret
	// key is at index 0.
	const char *getSecretKeyAlias(int index);
	// Returns the Nth secret key alias contained within the keystore. The 1st secret
	// key is at index 0.
	const char *secretKeyAlias(int index);


	// Returns the Nth trusted certificate contained within the keystore. The 1st
	// certificate is at index 0.
	// The caller is responsible for deleting the object returned by this method.
	CkCert *GetTrustedCert(int index);


	// Returns the Nth trusted certificate alias contained within the keystore. The 1st
	// certificate is at index 0.
	bool GetTrustedCertAlias(int index, CkString &outStr);

	// Returns the Nth trusted certificate alias contained within the keystore. The 1st
	// certificate is at index 0.
	const char *getTrustedCertAlias(int index);
	// Returns the Nth trusted certificate alias contained within the keystore. The 1st
	// certificate is at index 0.
	const char *trustedCertAlias(int index);


	// Loads a Java keystore from the contents of bd.
	bool LoadBd(const char *password, CkBinData &bd);


	// Loads a Java keystore from in-memory byte data.
	bool LoadBinary(const char *password, CkByteData &jksData);


	// Loads a Java keystore from an encoded string (such as base64, hex, etc.)
	bool LoadEncoded(const char *password, const char *jksEncData, const char *encoding);


	// Loads a Java keystore from a file.
	bool LoadFile(const char *password, const char *path);


	// Loads the Java KeyStore from a JSON Web Key (JWK) Set.
	bool LoadJwkSet(const char *password, CkJsonObject &jwkSet);


	// Removes the Nth trusted certificate or private key entry from the keystore. The
	// entryType indicates whether it is a trusted root or private key entry (1 = trusted
	// certificate entry, 2 = private key entry). The 1st entry is at index 0.
	bool RemoveEntry(int entryType, int index);


	// Sets the alias name for a trusted certificate or private key entry. The entryType
	// indicates whether it is a trusted root or private key entry (1 = trusted
	// certificate entry, 2 = private key entry). The 1st entry is at index 0.
	bool SetAlias(int entryType, int index, const char *alias);


	// Writes the key store to in-memory bytes. The password is used for the keyed hash of
	// the entire JKS file. (Each private key within the file may use different
	// passwords, and these are provided when the private key is added via the
	// AddPrivateKey method.)
	bool ToBinary(const char *password, CkByteData &outBytes);


	// Writes the key store to an encoded string. The encoding can be any encoding such as
	// "base64" or "hex". The password is used for the keyed hash of the entire JKS file.
	// (Each private key within the file may use different passwords, and these are
	// provided when the private key is added via the AddPrivateKey method.)
	bool ToEncodedString(const char *password, const char *encoding, CkString &outStr);

	// Writes the key store to an encoded string. The encoding can be any encoding such as
	// "base64" or "hex". The password is used for the keyed hash of the entire JKS file.
	// (Each private key within the file may use different passwords, and these are
	// provided when the private key is added via the AddPrivateKey method.)
	const char *toEncodedString(const char *password, const char *encoding);

	// Writes the key store to a file. The password is used for the keyed hash of the
	// entire JKS file. (Each private key within the file may use different passwords,
	// and these are provided when the private key is added via the AddPrivateKey
	// method.)
	bool ToFile(const char *password, const char *path);


	// Returns the private keys in JSON JWK Set format. The JWK identifier (kid) will
	// be set from the key alias in the store.
	bool ToJwkSet(const char *password, CkStringBuilder &sbJwkSet);


	// Returns the Java KeyStore as a Pem object.
	// The caller is responsible for deleting the object returned by this method.
	CkPem *ToPem(const char *password);


	// Returns the Java KeyStore as a Pfx object.
	// The caller is responsible for deleting the object returned by this method.
	CkPfx *ToPfx(const char *password);


	// Unlocks the component allowing for the full functionality to be used. If a
	// purchased unlock code is passed, there is no expiration. Any other string
	// automatically begins a fully-functional 30-day trial the first time
	// UnlockComponent is called.
	bool UnlockComponent(const char *unlockCode);


	// Adds an XML certificate vault to the object's internal list of sources to be
	// searched for certificates for help in building certificate chains to a root
	// certificate.
	bool UseCertVault(CkXmlCertVault &vault);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
