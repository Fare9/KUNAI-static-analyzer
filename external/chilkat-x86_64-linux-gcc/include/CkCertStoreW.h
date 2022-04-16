// CkCertStoreW.h: interface for the CkCertStoreW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkCertStoreW_H
#define _CkCertStoreW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkWideCharBase.h"

class CkCertW;
class CkByteData;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkCertStoreW
class CK_VISIBLE_PUBLIC CkCertStoreW  : public CkWideCharBase
{
	

	private:
	
	// Don't allow assignment or copying these objects.
	CkCertStoreW(const CkCertStoreW &);
	CkCertStoreW &operator=(const CkCertStoreW &);

    public:
	CkCertStoreW(void);
	virtual ~CkCertStoreW(void);

	

	static CkCertStoreW *createNew(void);
	

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// Applies only when running on a Microsoft Windows operating system. If true,
	// then any method that returns a certificate will not try to also access the
	// associated private key, assuming one exists. This is useful if the certificate
	// was installed with high-security such that a private key access would trigger
	// the Windows OS to display a security warning dialog. The default value of this
	// property is false.
	bool get_AvoidWindowsPkAccess(void);
	// Applies only when running on a Microsoft Windows operating system. If true,
	// then any method that returns a certificate will not try to also access the
	// associated private key, assuming one exists. This is useful if the certificate
	// was installed with high-security such that a private key access would trigger
	// the Windows OS to display a security warning dialog. The default value of this
	// property is false.
	void put_AvoidWindowsPkAccess(bool newVal);

	// The number of certificates held in the certificate store.
	int get_NumCertificates(void);

#if defined(CK_WINCERTSTORE_INCLUDED)
	// (This property only available on Microsoft Windows operating systems.)
	// The number of certificates that can be used for sending secure email within this
	// store.
	int get_NumEmailCerts(void);
#endif

	// Can be set to the PIN value for a certificate / private key stored on a smart
	// card.
	void get_SmartCardPin(CkString &str);
	// Can be set to the PIN value for a certificate / private key stored on a smart
	// card.
	const wchar_t *smartCardPin(void);
	// Can be set to the PIN value for a certificate / private key stored on a smart
	// card.
	void put_SmartCardPin(const wchar_t *newVal);

	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string, and should typically remain empty.
	void get_UncommonOptions(CkString &str);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string, and should typically remain empty.
	const wchar_t *uncommonOptions(void);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string, and should typically remain empty.
	void put_UncommonOptions(const wchar_t *newVal);



	// ----------------------
	// Methods
	// ----------------------
#if defined(CK_WINCERTSTORE_INCLUDED)
	// (This method only available on Microsoft Windows operating systems.)
	// Adds a certificate to the store. If the certificate is already in the store, it
	// is updated with the new information.
	bool AddCertificate(CkCertW &cert);
#endif

#if defined(CK_WINCERTSTORE_INCLUDED)
	// (This method only available on Microsoft Windows operating systems.)
	// Creates a new file-based certificate store. Certificates may be saved to this
	// store by calling AddCertificate.
	bool CreateFileStore(const wchar_t *filename);
#endif

#if defined(CK_WINCERTSTORE_INCLUDED)
	// (This method only available on Microsoft Windows operating systems.)
	// Creates an in-memory certificate store. Certificates may be added by calling
	// AddCertificate.
	bool CreateMemoryStore(void);
#endif

#if defined(CK_WINCERTSTORE_INCLUDED)
	// (This method only available on Microsoft Windows operating systems.)
	// Creates a registry-based certificate store. regRoot must be "CurrentUser" or
	// "LocalMachine". regPath is a registry path such as
	// "Software/MyApplication/Certificates".
	bool CreateRegistryStore(const wchar_t *regRoot, const wchar_t *regPath);
#endif

	// Finds a certificate by it's key container name.
	// The caller is responsible for deleting the object returned by this method.
	CkCertW *FindCertByKeyContainer(const wchar_t *name);

	// Locates and returns a certificate by its RFC 822 name.
	// 
	// If multiple certificates match, then non-expired certificates will take
	// precedence over expired certificates. In other words, Chilkat will aways return
	// the non-expired certificate over the expired certificate.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkCertW *FindCertByRfc822Name(const wchar_t *name);

	// Finds and returns the certificate that has the matching serial number.
	// 
	// Returns _NULL_ on failure.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkCertW *FindCertBySerial(const wchar_t *str);

	// Finds a certificate by it's SHA-1 thumbprint. The thumbprint is a hexidecimal
	// string.
	// 
	// Returns _NULL_ on failure.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkCertW *FindCertBySha1Thumbprint(const wchar_t *str);

	// Finds a certificate where one of the Subject properties (SubjectCN, SubjectE,
	// SubjectO, SubjectOU, SubjectL, SubjectST, SubjectC) matches exactly (but case
	// insensitive) with the passed string. A match in SubjectCN will be tried first,
	// followed by SubjectE, and SubjectO. After that, the first match found in
	// SubjectOU, SubjectL, SubjectST, or SubjectC, but in no guaranteed order, is
	// returned. All matches are case insensitive.
	// 
	// Returns _NULL_ on failure.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkCertW *FindCertBySubject(const wchar_t *str);

	// Finds a certificate where the SubjectCN property (common name) matches exactly
	// (but case insensitive) with the passed string
	// 
	// Returns _NULL_ on failure.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkCertW *FindCertBySubjectCN(const wchar_t *str);

	// Finds a certificate where the SubjectE property (email address) matches exactly
	// (but case insensitive) with the passed string. This function differs from
	// FindCertForEmail in that the certificate does not need to match the
	// ForSecureEmail property.
	// 
	// Returns _NULL_ on failure.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkCertW *FindCertBySubjectE(const wchar_t *str);

	// Finds a certificate where the SubjectO property (organization) matches exactly
	// (but case insensitive) with the passed string.
	// 
	// Returns _NULL_ on failure.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkCertW *FindCertBySubjectO(const wchar_t *str);

	// (This method only available on Microsoft Windows operating systems.)
	// Finds a certificate that can be used to send secure email to the passed email
	// address. A certificate matches only if the ForSecureEmail property is TRUE, and
	// the email address matches exactly (but case insensitive) with the SubjectE
	// property. Returns NULL if no matches are found.
	// 
	// Returns _NULL_ on failure.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkCertW *FindCertForEmail(const wchar_t *emailAddress);

	// Returns the Nth certificate in the store. The first certificate is at index 0.
	// 
	// Returns _NULL_ on failure.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkCertW *GetCertificate(int index);

#if defined(CK_WINCERTSTORE_INCLUDED)
	// (This method only available on Microsoft Windows operating systems.)
	// Returns the Nth email certificate in the store. The first certificate is at
	// index 0. Use the NumEmailCertificates property to get the number of email
	// certificates.
	// 
	// Returns _NULL_ on failure.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkCertW *GetEmailCert(int index);
#endif

	// Loads the certificates contained within a PEM formatted file.
	bool LoadPemFile(const wchar_t *pemPath);

	// Loads the certificates contained within an in-memory PEM formatted string.
	bool LoadPemStr(const wchar_t *pemString);

	// Loads a PFX from an in-memory image of a PFX file. Once loaded, the certificates
	// within the PFX may be searched via the Find* methods. It is also possible to
	// iterate from 0 to NumCertficates-1, calling GetCertificate for each index, to
	// retrieve each certificate within the PFX.
	bool LoadPfxData(CkByteData &pfxData, const wchar_t *password);

#if !defined(CHILKAT_MONO)
	// Loads a PFX from an in-memory image of a PFX file. Once loaded, the certificates
	// within the PFX may be searched via the Find* methods. It is also possible to
	// iterate from 0 to NumCertficates-1, calling GetCertificate for each index, to
	// retrieve each certificate within the PFX.
	bool LoadPfxData2(const void *pByteData, unsigned long szByteData, const wchar_t *password);
#endif

	// Loads a PFX file. Once loaded, the certificates within the PFX may be searched
	// via the Find* methods. It is also possible to iterate from 0 to
	// NumCertficates-1, calling GetCertificate for each index, to retrieve each
	// certificate within the PFX.
	// 
	// Note: This method does not import certificates into the Windows certificate
	// stores. The purpose of this method is to load a .pfx/.p12 into this object so
	// that other API methods can be called to explore or search the contents of the
	// PFX. The Chilkat Pfx class also provides similar functionality.
	// 
	bool LoadPfxFile(const wchar_t *pfxFilename, const wchar_t *password);

#if defined(CK_WINCERTSTORE_INCLUDED)
	// (This method is only available on Microsoft Windows operating systems.)
	// Opens the registry-based Current-User\Personal certificate store. Set readOnly =
	// true if only fetching certificates and not updating the certificate store
	// (i.e. certificates will not be added or removed). Setting readOnly = true causes
	// the certificate store to be opened read-only, and will prevent "permission
	// denied" errors caused by the need for read-write permission.
	// 
	// Once loaded, the certificates within the store may be searched via the Find*
	// methods. An application may also iterate from 0 to NumCertficates-1 and call
	// GetCertificate to access each certificate by index.
	// 
	bool OpenCurrentUserStore(bool readOnly);
#endif

#if defined(CK_WINCERTSTORE_INCLUDED)
	// (This method only available on Microsoft Windows operating systems.)
	// Opens a file-based certificate store.
	// 
	// Once loaded, the certificates within the store may be searched via the Find*
	// methods. An application may also iterate from 0 to NumCertficates-1 and call
	// GetCertificate to access each certificate by index.
	// 
	bool OpenFileStore(const wchar_t *filename, bool readOnly);
#endif

#if defined(CK_WINCERTSTORE_INCLUDED)
	// (This method is only available on Microsoft Windows operating systems.)
	// Opens the registry-based Local-Computer\Personal certificate store. Set readOnly =
	// true if only fetching certificates and not updating the certificate store
	// (i.e. certificates will not be added or removed). Setting readOnly = true causes
	// the certificate store to be opened read-only, and will prevent "permission
	// denied" errors caused by the need for read-write permission.
	// 
	// Once loaded, the certificates within the store may be searched via the Find*
	// methods. An application may also iterate from 0 to NumCertficates-1 and call
	// GetCertificate to access each certificate by index.
	// 
	bool OpenLocalSystemStore(bool readOnly);
#endif

#if defined(CK_WINCERTSTORE_INCLUDED)
	// (This method only available on Microsoft Windows operating systems.)
	// Opens an arbitrary registry-based certificate store. regRoot must be "CurrentUser"
	// or "LocalMachine". regPath is a registry path such as
	// "Software/MyApplication/Certificates".
	// 
	// Setting readOnly = true causes the certificate store to be opened read-only, and
	// will prevent "permission denied" errors caused by the need for read-write
	// permission.
	// 
	// Once loaded, the certificates within the store may be searched via the Find*
	// methods. An application may also iterate from 0 to NumCertficates-1 and call
	// GetCertificate to access each certificate by index.
	// 
	bool OpenRegistryStore(const wchar_t *regRoot, const wchar_t *regPath, bool readOnly);
#endif

#if defined(CK_WINCERTSTORE_INCLUDED)
	// Opens the certificate store on the smartcard currently in the reader or USB
	// token.
	// 
	// The csp can be set to the name of the CSP (Cryptographic Service Provider) that
	// should be used. If csp is an empty string, then the 1st CSP found matching one
	// of the following names will be used:
	// 
	//     Microsoft Smart Card Key Storage Provider
	//     Microsoft Base Smart Card Crypto Provider
	//     Bit4id Universal Middleware Provider
	//     YubiHSM Key Storage Provider (starting in v9.5.0.83)
	//     eToken Base Cryptographic Provider
	//     FTSafe ePass1000 RSA Cryptographic Service Provider
	//     SecureStoreCSP
	//     EnterSafe ePass2003 CSP v2.0
	//     Gemalto Classic Card CSP
	//     PROXKey CSP India V1.0
	//     PROXKey CSP India V2.0
	//     TRUST KEY CSP V1.0
	//     Watchdata Brazil CSP V1.0
	//     Luna Cryptographic Services for Microsoft Windows
	//     Luna SChannel Cryptographic Services for Microsoft Windows
	//     Safenet RSA Full Cryptographic Provider
	//     nCipher Enhanced Cryptographic Provider
	//     SafeSign Standard Cryptographic Service Provider
	//     SafeSign Standard RSA and AES Cryptographic Service Provider
	//     MySmartLogon NFC CSP
	//     NFC Connector Enterprise
	//     ActivClient Cryptographic Service Provider
	//     EnterSafe ePass2003 CSP v1.0
	//     Oberthur Card Systems Cryptographic Provider
	//     Athena ASECard Crypto CSP"
	// 
	// (This method is only available on Microsoft Windows operating systems.)
	// 
	bool OpenSmartcard(const wchar_t *csp);
#endif

#if defined(CK_WINCERTSTORE_INCLUDED)
	// (This method only available on Microsoft Windows operating systems.)
	// Opens a Microsoft Windows certificate store. storeLocation must be "CurrentUser" or
	// "LocalMachine". storeName is the name of the certificate store to open. It may be any
	// of the following:
	//     AddressBook: Certificate store for other users.
	//     AuthRoot: Certificate store for third-party certification authorities (CAs).
	//     CertificationAuthority: Certificate store for intermediate certification
	//     authorities (CAs).
	//     Disallowed: Certificate store for revoked certificates.
	//     My: Certificate store for personal certificates.
	//     Root: Certificate store for trusted root certification authorities (CAs).
	//     TrustedPeople: Certificate store for directly trusted people and resources.
	//     TrustedPublisher: Certificate store for directly trusted publishers.
	// 
	// Setting readOnly = true causes the certificate store to be opened read-only, and
	// will prevent "permission denied" errors caused by the need for read-write
	// permission.
	// 
	// Once loaded, the certificates within the store may be searched via the Find*
	// methods. An application may also iterate from 0 to NumCertficates-1 and call
	// GetCertificate to access each certificate by index.
	// 
	bool OpenWindowsStore(const wchar_t *storeLocation, const wchar_t *storeName, bool readOnly);
#endif

#if defined(CK_WINCERTSTORE_INCLUDED)
	// (This method only available on Microsoft Windows operating systems.)
	// Removes the passed certificate from the store. The certificate object passed as
	// the argument can no longer be used once removed.
	bool RemoveCertificate(CkCertW &cert);
#endif





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
