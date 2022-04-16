// CkPfx.h: interface for the CkPfx class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkPfx_H
#define _CkPfx_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkCert;
class CkPrivateKey;
class CkCertChain;
class CkJsonObject;
class CkByteData;
class CkJavaKeyStore;
class CkXmlCertVault;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkPfx
class CK_VISIBLE_PUBLIC CkPfx  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkPfx(const CkPfx &);
	CkPfx &operator=(const CkPfx &);

    public:
	CkPfx(void);
	virtual ~CkPfx(void);

	static CkPfx *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The encryption algorithm to be used when writing a PFX. After loading a PFX,
	// this property is set to the encryption algorithm used by the loaded PFX. (This
	// is the algorithm used for the "shrouded key bag", which is internal to the PFX.)
	// 
	// The default value (for backward compatibility) is
	// "pbeWithSHAAnd3_KeyTripleDES_CBC". Can be set to "pbes2", in which case the
	// Pbes2CryptAlg and Pbes2HmacAlg properies will be set to the algorithms to be
	// used when writing, or the algorithms used by the loaded PFX.
	// 
	void get_AlgorithmId(CkString &str);
	// The encryption algorithm to be used when writing a PFX. After loading a PFX,
	// this property is set to the encryption algorithm used by the loaded PFX. (This
	// is the algorithm used for the "shrouded key bag", which is internal to the PFX.)
	// 
	// The default value (for backward compatibility) is
	// "pbeWithSHAAnd3_KeyTripleDES_CBC". Can be set to "pbes2", in which case the
	// Pbes2CryptAlg and Pbes2HmacAlg properies will be set to the algorithms to be
	// used when writing, or the algorithms used by the loaded PFX.
	// 
	const char *algorithmId(void);
	// The encryption algorithm to be used when writing a PFX. After loading a PFX,
	// this property is set to the encryption algorithm used by the loaded PFX. (This
	// is the algorithm used for the "shrouded key bag", which is internal to the PFX.)
	// 
	// The default value (for backward compatibility) is
	// "pbeWithSHAAnd3_KeyTripleDES_CBC". Can be set to "pbes2", in which case the
	// Pbes2CryptAlg and Pbes2HmacAlg properies will be set to the algorithms to be
	// used when writing, or the algorithms used by the loaded PFX.
	// 
	void put_AlgorithmId(const char *newVal);

	// The number of certificates contained in the PFX.
	int get_NumCerts(void);

	// The number of private keys contained in the PFX.
	int get_NumPrivateKeys(void);

	// If the AlgorithmId property equals "pbes2", then this is the encryption
	// algorithm to be used when writing the PFX, or used by the PFX that was loaded.
	// If the AlgorithmId is not equal to "pbes2", then the value of this property is
	// meaningless.
	// 
	// Possible values are:
	//     aes256-cbc
	//     aes192-cbc
	//     aes128-cbc
	//     3des-cbc
	// 
	// The default value (for writing) is "aes256-cbc". Note: The algorithm specified
	// by this property is only used when the Algorithmid = "pbes2".
	// 
	void get_Pbes2CryptAlg(CkString &str);
	// If the AlgorithmId property equals "pbes2", then this is the encryption
	// algorithm to be used when writing the PFX, or used by the PFX that was loaded.
	// If the AlgorithmId is not equal to "pbes2", then the value of this property is
	// meaningless.
	// 
	// Possible values are:
	//     aes256-cbc
	//     aes192-cbc
	//     aes128-cbc
	//     3des-cbc
	// 
	// The default value (for writing) is "aes256-cbc". Note: The algorithm specified
	// by this property is only used when the Algorithmid = "pbes2".
	// 
	const char *pbes2CryptAlg(void);
	// If the AlgorithmId property equals "pbes2", then this is the encryption
	// algorithm to be used when writing the PFX, or used by the PFX that was loaded.
	// If the AlgorithmId is not equal to "pbes2", then the value of this property is
	// meaningless.
	// 
	// Possible values are:
	//     aes256-cbc
	//     aes192-cbc
	//     aes128-cbc
	//     3des-cbc
	// 
	// The default value (for writing) is "aes256-cbc". Note: The algorithm specified
	// by this property is only used when the Algorithmid = "pbes2".
	// 
	void put_Pbes2CryptAlg(const char *newVal);

	// If the AlgorithmId property equals "pbes2", then this is the HMAC hash algorithm
	// to be used when writing the PFX, or used by the PFX that was loaded. If the
	// AlgorithmId is not equal to "pbes2", then the value of this property is
	// meaningless.
	// 
	// Possible values are:
	//     hmacWithSha256
	//     hmacWithSha384
	//     hmacWithSha512
	//     hmacWithSha1
	// 
	// The default value (for writing) is "hmacWithSha256". Note: The algorithm
	// specified by this property is only used when the Algorithmid = "pbes2".
	// 
	void get_Pbes2HmacAlg(CkString &str);
	// If the AlgorithmId property equals "pbes2", then this is the HMAC hash algorithm
	// to be used when writing the PFX, or used by the PFX that was loaded. If the
	// AlgorithmId is not equal to "pbes2", then the value of this property is
	// meaningless.
	// 
	// Possible values are:
	//     hmacWithSha256
	//     hmacWithSha384
	//     hmacWithSha512
	//     hmacWithSha1
	// 
	// The default value (for writing) is "hmacWithSha256". Note: The algorithm
	// specified by this property is only used when the Algorithmid = "pbes2".
	// 
	const char *pbes2HmacAlg(void);
	// If the AlgorithmId property equals "pbes2", then this is the HMAC hash algorithm
	// to be used when writing the PFX, or used by the PFX that was loaded. If the
	// AlgorithmId is not equal to "pbes2", then the value of this property is
	// meaningless.
	// 
	// Possible values are:
	//     hmacWithSha256
	//     hmacWithSha384
	//     hmacWithSha512
	//     hmacWithSha1
	// 
	// The default value (for writing) is "hmacWithSha256". Note: The algorithm
	// specified by this property is only used when the Algorithmid = "pbes2".
	// 
	void put_Pbes2HmacAlg(const char *newVal);

	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "LegacyOrder" - Introduced in v9.5.0.83. Write the internal ContentInfos in
	//     the order Chilkat traditionally used in previous versions.
	//     "NoTruncatePfxPassword64" - Introduced in v9.5.0.87. Microsoft systems/tools
	//     such as certmgr.msc would typically truncate extremely long passwords to 64
	//     bytes/chars. Other systems did not. Chilkat will by default truncate passwords
	//     to 64 chars. Add this keyword to prevent truncation.
	// 
	void get_UncommonOptions(CkString &str);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "LegacyOrder" - Introduced in v9.5.0.83. Write the internal ContentInfos in
	//     the order Chilkat traditionally used in previous versions.
	//     "NoTruncatePfxPassword64" - Introduced in v9.5.0.87. Microsoft systems/tools
	//     such as certmgr.msc would typically truncate extremely long passwords to 64
	//     bytes/chars. Other systems did not. Chilkat will by default truncate passwords
	//     to 64 chars. Add this keyword to prevent truncation.
	// 
	const char *uncommonOptions(void);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "LegacyOrder" - Introduced in v9.5.0.83. Write the internal ContentInfos in
	//     the order Chilkat traditionally used in previous versions.
	//     "NoTruncatePfxPassword64" - Introduced in v9.5.0.87. Microsoft systems/tools
	//     such as certmgr.msc would typically truncate extremely long passwords to 64
	//     bytes/chars. Other systems did not. Chilkat will by default truncate passwords
	//     to 64 chars. Add this keyword to prevent truncation.
	// 
	void put_UncommonOptions(const char *newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Adds a certificate, its private key (if it exists), and potentially its
	// certificate chain to the PFX. If includeChain is true, then the certificate must have
	// a private key. The certificate's private key is automatically obtained
	// (internally) via the cert's ExportPrivateKey method. If the certificate's chain
	// of authentication is to be added, it is automatically constructed and added
	// using whatever resources are at hand (such as certs provided via the
	// UseCertVault method, the trusted roots from Chilkat's TrustedRoots class, etc.
	// If a certificate chain is to be added, which is the typical case, then the chain
	// must be completed to the root to succeed.
	bool AddCert(CkCert &cert, bool includeChain);


	// Adds a private key and certificate chain to the PFX. The private key should be
	// such that it is associated with the 1st certificate in the chain. In other
	// words, the 1st certificate in the chain has a public key (embedded within the
	// X.509 structure of the cert itself) that is the counterpart to the private key.
	bool AddPrivateKey(CkPrivateKey &privKey, CkCertChain &certChain);


	// Finds and returns the certificate (in the PFX) that has a cert bag "localKeyId"
	// attribute with the specified value. The localKeyId is specifid using the encoding
	// (such as "decimal", "base64", "hex") specified by encoding.
	// The caller is responsible for deleting the object returned by this method.
	CkCert *FindCertByLocalKeyId(const char *localKeyId, const char *encoding);


	// Returns the Nth certificate in the PFX. (The 1st certificate is at index 0.)
	// The caller is responsible for deleting the object returned by this method.
	CkCert *GetCert(int index);


	// Returns the Nth private key in the PFX. (The 1st private key is at index 0.)
	// The caller is responsible for deleting the object returned by this method.
	CkPrivateKey *GetPrivateKey(int index);


	// Can be called to get one of the following safebag attributes for the Nth private
	// key or certificate in the PFX. forPrivateKey should be true for a private key, and
	// false for a certificate. The index is the index of the certificate or key in
	// the PFX. The attrName can be one of the following:
	// 
	//     "localKeyId" : Returns the decimal representation of the local key ID. The
	//     local key ID is used to associate the certificate contained in the PFX with this
	//     private key. (The certificate will include a "localKeyId" attribute in its cert
	//     bag of attributes within the PFX.)
	//     "keyContainerName" : Returns the key container name (or key name) of the
	//     private key. For more information about the directories where the Windows OS
	//     stores private keys,
	//     seehttps://docs.microsoft.com/en-us/windows/win32/seccng/key-storage-and-retrieva
	//     l
	//     <https://docs.microsoft.com/en-us/windows/win32/seccng/key-storage-and-retrieval>
	//     "storageProvider" : Returns the name of the Cryptographic Storage Provider
	//     to be used for the key.
	// 
	// Note: It is not required that any of the above attributes are present in the
	// PFX.
	// 
	bool GetSafeBagAttr(bool forPrivateKey, int index, const char *attrName, CkString &outStr);

	// Can be called to get one of the following safebag attributes for the Nth private
	// key or certificate in the PFX. forPrivateKey should be true for a private key, and
	// false for a certificate. The index is the index of the certificate or key in
	// the PFX. The attrName can be one of the following:
	// 
	//     "localKeyId" : Returns the decimal representation of the local key ID. The
	//     local key ID is used to associate the certificate contained in the PFX with this
	//     private key. (The certificate will include a "localKeyId" attribute in its cert
	//     bag of attributes within the PFX.)
	//     "keyContainerName" : Returns the key container name (or key name) of the
	//     private key. For more information about the directories where the Windows OS
	//     stores private keys,
	//     seehttps://docs.microsoft.com/en-us/windows/win32/seccng/key-storage-and-retrieva
	//     l
	//     <https://docs.microsoft.com/en-us/windows/win32/seccng/key-storage-and-retrieval>
	//     "storageProvider" : Returns the name of the Cryptographic Storage Provider
	//     to be used for the key.
	// 
	// Note: It is not required that any of the above attributes are present in the
	// PFX.
	// 
	const char *getSafeBagAttr(bool forPrivateKey, int index, const char *attrName);
	// Can be called to get one of the following safebag attributes for the Nth private
	// key or certificate in the PFX. forPrivateKey should be true for a private key, and
	// false for a certificate. The index is the index of the certificate or key in
	// the PFX. The attrName can be one of the following:
	// 
	//     "localKeyId" : Returns the decimal representation of the local key ID. The
	//     local key ID is used to associate the certificate contained in the PFX with this
	//     private key. (The certificate will include a "localKeyId" attribute in its cert
	//     bag of attributes within the PFX.)
	//     "keyContainerName" : Returns the key container name (or key name) of the
	//     private key. For more information about the directories where the Windows OS
	//     stores private keys,
	//     seehttps://docs.microsoft.com/en-us/windows/win32/seccng/key-storage-and-retrieva
	//     l
	//     <https://docs.microsoft.com/en-us/windows/win32/seccng/key-storage-and-retrieval>
	//     "storageProvider" : Returns the name of the Cryptographic Storage Provider
	//     to be used for the key.
	// 
	// Note: It is not required that any of the above attributes are present in the
	// PFX.
	// 
	const char *safeBagAttr(bool forPrivateKey, int index, const char *attrName);


#if defined(CK_WINCERTSTORE_INCLUDED)
	// Imports the certificates and private keys contained in the PFX to Windows
	// certificate store(s).
	// 
	// If exportable is true, imported keys are marked as exportable.
	// 
	// If userProtected is true, the user is to be notified through a dialog box or other
	// method when certain attempts to use this key are made. The precise behavior is
	// specified by the cryptographic service provider (CSP) being used.
	// 
	// If machineKeyset is true, the private keys are stored under the local computer and not
	// under the current user.
	// 
	// If allowOverwriteKey is true, allow overwrite of the existing key. Specify this flag when
	// you encounter a scenario in which you must import a PFX file that contains a key
	// name that already exists. For example, when you import a PFX file, it is
	// possible that a container of the same name is already present because there is
	// no unique namespace for key containers. If you have created a "TestKey" on your
	// computer, and then you import a PFX file that also has "TestKey" as the key
	// container, this flag allows the key to be overwritten.
	// 
	// if allowExport is true, then the key is marked as exportable, which allows for it to
	// be re-exported to a PFX.
	// 
	// The leafStore, intermediateStore, and rootStore are the Windows certificate store names indicating
	// where to import certificates of each given type. A leafStore is a certificate that is
	// not the issuer of any other certificate in the PFX. An intermediateStore is any certificate
	// that is not a root (or self-signed) but is also the issuer of some other
	// certificate in the PFX. A rootStore is a self-signed or root certificate.
	// 
	// The possible store names for leafStore, intermediateStore, and rootStore are as follows:
	//     "None": Do not import certificates of the given type into any Windows
	//     certicate store.
	//     "AddressBook": Certificate store for other users.
	//     "AuthRoot": Certificate store for third-party certification authorities
	//     (CAs).
	//     "CertificationAuthority": Certificate store for intermediate certification
	//     authorities (CAs).
	//     "My": Certificate store for personal certificates. (Leaf certificates are
	//     typically imported into this store.)
	//     "Root": Certificate store for trusted root certification authorities (CAs).
	//     "TrustedPeople": Certificate store for directly trusted people and
	//     resources.
	//     "TrustedPublisher": Certificate store for directly trusted publishers.
	// 
	// The rootStore is reserved for any future options that may be needed. At this time,
	// pass an empty string.
	// 
	bool ImportToWindows(bool exportable, bool userProtected, bool machineKeyset, bool allowOverwriteKey, bool allowExport, const char *leafStore, const char *intermediateStore, const char *rootStore, const char *extraOptions);

#endif

	// Provides information about what transpired in the last method called.
	// The caller is responsible for deleting the object returned by this method.
	CkJsonObject *LastJsonData(void);


	// Loads a PFX from a PEM formatted string. The PEM can contain the private key,
	// the certificate, and certificates in the chain of authentication up to the CA
	// root. For example:
	//  -----BEGIN RSA PRIVATE KEY-----
	// ...
	// ... the private key associated with the main certificate.
	// ...
	// -----END RSA PRIVATE KEY-----
	// -----BEGIN CERTIFICATE-----
	// ...
	// ... the main certificate
	// ...
	// -----END CERTIFICATE-----
	// -----BEGIN CERTIFICATE-----
	// ...
	// ... an intermediate CA certificate (if present)
	// ...
	// -----END CERTIFICATE-----
	// -----BEGIN CERTIFICATE-----
	// ...
	// ... the root CA certificate
	// ...
	// -----END CERTIFICATE----- 
	bool LoadPem(const char *pemStr, const char *password);


	// Loads a PFX from in-memory bytes.
	// 
	// If the .pfx/.p12 uses different passwords for integrity and private keys, then
	// the password argument may contain JSON to specify the passwords. See the LoadPfxFile
	// method (below) for details.
	// 
	bool LoadPfxBytes(CkByteData &pfxData, const char *password);


	// Loads a PFX from encoded byte data. The encoding can by any encoding, such as
	// "Base64", "modBase64", "Base32", "UU", "QP" (for quoted-printable), "URL" (for
	// url-encoding), "Hex", "Q", "B", "url_oath", "url_rfc1738", "url_rfc2396", and
	// "url_rfc3986".
	// 
	// If the .pfx/.p12 uses different passwords for integrity and private keys, then
	// the encoding argument may contain JSON to specify the passwords. See the LoadPfxFile
	// method (below) for details.
	// 
	bool LoadPfxEncoded(const char *encodedData, const char *encoding, const char *password);


	// Loads a PFX from a file.
	// 
	// Starting in v9.5.0.75, a .pfx/.p12 file with different passwords for integrity
	// and private keys can be loaded by passing the following JSON for the password.
	//     {
	//       "integrity": "password1",
	//       "privKeys": "password2",
	//      }
	// If it is desired to open the .pfx/.p12 without access to the private keys, then
	// add "skipPrivateKeys" like this:
	//     {
	//       "integrity": "password1",
	//       "privKeys": "not used",
	//        "skipPrivateKeys": true
	//      }
	// 
	bool LoadPfxFile(const char *path, const char *password);


	// Sets a safe bag attribute for the Nth private key or certificate in the PFX.
	// Safe bag attributes can be added by calling this method once for each attribute
	// to be added to each certificate or key. forPrivateKey should be true for a private key,
	// and false for a certificate. The index is the index of the certificate or key
	// in the PFX. (The 1st item is at index 0.) See the example below for more
	// information. The encoding indicates a binary encoding such as "base64", "hex",
	// "decimal", "fingerprint", etc if the value contains binary (non-text) data.
	// 
	// A safe bag attribute can be removed by passing an empty string for the value.
	// 
	bool SetSafeBagAttr(bool forPrivateKey, int index, const char *name, const char *value, const char *encoding);


	// Write the PFX to in-memory bytes.
	bool ToBinary(const char *password, CkByteData &outBytes);


	// Write the PFX to an encoded string. The encoding can be any encoding such as
	// "base64" or "hex".
	bool ToEncodedString(const char *password, const char *encoding, CkString &outStr);

	// Write the PFX to an encoded string. The encoding can be any encoding such as
	// "base64" or "hex".
	const char *toEncodedString(const char *password, const char *encoding);

	// Write the PFX to a file. PFX and PKCS12 are essentially the same. Standard
	// filename extensions are ".pfx" or ".p12".
	bool ToFile(const char *password, const char *path);


	// Converts the PFX (PKCS12) to a JavaKeyStore object. One JKS entry per private
	// key found in the PKCS12 is added. The certs found within the PCKS12 are used to
	// build the certificate chains for each private key. (A typical PFX file contains
	// a single private key along with its associated certificate, and the certificates
	// in the chain of authentication to the root CA cert.)
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
	// The caller is responsible for deleting the object returned by this method.
	CkJavaKeyStore *ToJavaKeyStore(const char *alias, const char *password);


	// Write the PFX to a PEM formatted string. The resultant PEM will contain the
	// private key, as well as the certs in the chain of authentication (or whatever
	// certs are available in the PFX). For example:
	//  -----BEGIN RSA PRIVATE KEY-----
	// ...
	// ... the private key associated with the main certificate.
	// ...
	// -----END RSA PRIVATE KEY-----
	// -----BEGIN CERTIFICATE-----
	// ...
	// ... the main certificate
	// ...
	// -----END CERTIFICATE-----
	// -----BEGIN CERTIFICATE-----
	// ...
	// ... an intermediate CA certificate (if present)
	// ...
	// -----END CERTIFICATE-----
	// -----BEGIN CERTIFICATE-----
	// ...
	// ... the root CA certificate
	// ...
	// -----END CERTIFICATE----- 
	bool ToPem(CkString &outStr);

	// Write the PFX to a PEM formatted string. The resultant PEM will contain the
	// private key, as well as the certs in the chain of authentication (or whatever
	// certs are available in the PFX). For example:
	//  -----BEGIN RSA PRIVATE KEY-----
	// ...
	// ... the private key associated with the main certificate.
	// ...
	// -----END RSA PRIVATE KEY-----
	// -----BEGIN CERTIFICATE-----
	// ...
	// ... the main certificate
	// ...
	// -----END CERTIFICATE-----
	// -----BEGIN CERTIFICATE-----
	// ...
	// ... an intermediate CA certificate (if present)
	// ...
	// -----END CERTIFICATE-----
	// -----BEGIN CERTIFICATE-----
	// ...
	// ... the root CA certificate
	// ...
	// -----END CERTIFICATE----- 
	const char *toPem(void);

	// Write the PFX to a PEM formatted string. If extendedAttrs is true, then extended
	// properties (Bag Attributes and Key Attributes) are output. If noKeys is true,
	// then no private keys are output. If noCerts is true, then no certificates are
	// output. If noCaCerts is true, then no CA certs or intermediate CA certs are output.
	// If encryptAlg is not empty, it indicates the encryption algorithm to be used for
	// encrypting the private keys (otherwise the private keys are output unencrypted).
	// The possible choices for the encryptAlg are "des3", "aes128", "aes192", and "aes256".
	// (All encryption algorithm choices use CBC mode.) If the private keys are to be
	// encrypted, then password is the password to be used. Otherwise, password may be left
	// empty. For example:
	// Bag Attributes
	//     Microsoft Local Key set: localKeyID: 01 00 00 00 
	//     friendlyName: le-2b09a3d2-9037-4a05-95cc-4d44518e8607
	//     Microsoft CSP Name: Microsoft RSA SChannel Cryptographic Provider
	// Key Attributes
	//     X509v3 Key Usage: 10 
	//  -----BEGIN RSA PRIVATE KEY-----
	// ...
	// ... the private key associated with the main certificate.
	// ...
	// -----END RSA PRIVATE KEY-----
	// Bag Attributes
	//     localKeyID: 01 00 00 00 
	//     1.3.6.1.4.1.311.17.3.92: 00 08 00 00 
	//     1.3.6.1.4.1.311.17.3.20: C2 53 54 F3 ...
	//     1.3.6.1.4.1.311.17.3.71: 49 00 43 00 ...
	//     1.3.6.1.4.1.311.17.3.75: 31 00 42 00 ...
	// subject=/OU=Domain Control Validated/OU=PositiveSSL/CN=something.com
	// issuer=/C=GB/ST=Greater Manchester/L=Salford/O=COMODO CA Limited/CN=COMODO RSA Domain Validation Secure Server CA
	// -----BEGIN CERTIFICATE-----
	// ...
	// ... the main certificate
	// ...
	// -----END CERTIFICATE-----
	// ...
	// -----BEGIN CERTIFICATE-----
	// ...
	// ... an intermediate CA certificate (if present)
	// ...
	// -----END CERTIFICATE-----
	// ...
	// -----BEGIN CERTIFICATE-----
	// ...
	// ... the root CA certificate
	// ...
	// -----END CERTIFICATE----- 
	bool ToPemEx(bool extendedAttrs, bool noKeys, bool noCerts, bool noCaCerts, const char *encryptAlg, const char *password, CkString &outStr);

	// Write the PFX to a PEM formatted string. If extendedAttrs is true, then extended
	// properties (Bag Attributes and Key Attributes) are output. If noKeys is true,
	// then no private keys are output. If noCerts is true, then no certificates are
	// output. If noCaCerts is true, then no CA certs or intermediate CA certs are output.
	// If encryptAlg is not empty, it indicates the encryption algorithm to be used for
	// encrypting the private keys (otherwise the private keys are output unencrypted).
	// The possible choices for the encryptAlg are "des3", "aes128", "aes192", and "aes256".
	// (All encryption algorithm choices use CBC mode.) If the private keys are to be
	// encrypted, then password is the password to be used. Otherwise, password may be left
	// empty. For example:
	// Bag Attributes
	//     Microsoft Local Key set: localKeyID: 01 00 00 00 
	//     friendlyName: le-2b09a3d2-9037-4a05-95cc-4d44518e8607
	//     Microsoft CSP Name: Microsoft RSA SChannel Cryptographic Provider
	// Key Attributes
	//     X509v3 Key Usage: 10 
	//  -----BEGIN RSA PRIVATE KEY-----
	// ...
	// ... the private key associated with the main certificate.
	// ...
	// -----END RSA PRIVATE KEY-----
	// Bag Attributes
	//     localKeyID: 01 00 00 00 
	//     1.3.6.1.4.1.311.17.3.92: 00 08 00 00 
	//     1.3.6.1.4.1.311.17.3.20: C2 53 54 F3 ...
	//     1.3.6.1.4.1.311.17.3.71: 49 00 43 00 ...
	//     1.3.6.1.4.1.311.17.3.75: 31 00 42 00 ...
	// subject=/OU=Domain Control Validated/OU=PositiveSSL/CN=something.com
	// issuer=/C=GB/ST=Greater Manchester/L=Salford/O=COMODO CA Limited/CN=COMODO RSA Domain Validation Secure Server CA
	// -----BEGIN CERTIFICATE-----
	// ...
	// ... the main certificate
	// ...
	// -----END CERTIFICATE-----
	// ...
	// -----BEGIN CERTIFICATE-----
	// ...
	// ... an intermediate CA certificate (if present)
	// ...
	// -----END CERTIFICATE-----
	// ...
	// -----BEGIN CERTIFICATE-----
	// ...
	// ... the root CA certificate
	// ...
	// -----END CERTIFICATE----- 
	const char *toPemEx(bool extendedAttrs, bool noKeys, bool noCerts, bool noCaCerts, const char *encryptAlg, const char *password);

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
