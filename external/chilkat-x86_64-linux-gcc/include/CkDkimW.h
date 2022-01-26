// CkDkimW.h: interface for the CkDkimW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkDkimW_H
#define _CkDkimW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacksW.h"

class CkByteData;
class CkBinDataW;
class CkTaskW;
class CkPrivateKeyW;
class CkBaseProgressW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkDkimW
class CK_VISIBLE_PUBLIC CkDkimW  : public CkClassWithCallbacksW
{
	private:
	bool m_cbOwned;

	private:
	
	// Don't allow assignment or copying these objects.
	CkDkimW(const CkDkimW &);
	CkDkimW &operator=(const CkDkimW &);

    public:
	CkDkimW(void);
	virtual ~CkDkimW(void);

	

	static CkDkimW *createNew(void);
	

	CkDkimW(bool bCallbackOwned);
	static CkDkimW *createNew(bool bCallbackOwned);

	
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
	// When set to true, causes the currently running method to abort. Methods that
	// always finish quickly (i.e.have no length file operations or network
	// communications) are not affected. If no method is running, then this property is
	// automatically reset to false when the next method is called. When the abort
	// occurs, this property is reset to false. Both synchronous and asynchronous
	// method calls can be aborted. (A synchronous method call could be aborted by
	// setting this property from a separate thread.)
	bool get_AbortCurrent(void);
	// When set to true, causes the currently running method to abort. Methods that
	// always finish quickly (i.e.have no length file operations or network
	// communications) are not affected. If no method is running, then this property is
	// automatically reset to false when the next method is called. When the abort
	// occurs, this property is reset to false. Both synchronous and asynchronous
	// method calls can be aborted. (A synchronous method call could be aborted by
	// setting this property from a separate thread.)
	void put_AbortCurrent(bool newVal);

	// The signing algorithm to be used in creating the DKIM-Signature. Possible values
	// are "rsa-sha256" and "rsa-sha1". The default value is "rsa-sha256".
	void get_DkimAlg(CkString &str);
	// The signing algorithm to be used in creating the DKIM-Signature. Possible values
	// are "rsa-sha256" and "rsa-sha1". The default value is "rsa-sha256".
	const wchar_t *dkimAlg(void);
	// The signing algorithm to be used in creating the DKIM-Signature. Possible values
	// are "rsa-sha256" and "rsa-sha1". The default value is "rsa-sha256".
	void put_DkimAlg(const wchar_t *newVal);

	// Optional body length count to set a maximum number of body bytes to be hashed
	// when creating the DKIM-Signature field. The default value is 0, indicating that
	// the entire body should be hashed.
	int get_DkimBodyLengthCount(void);
	// Optional body length count to set a maximum number of body bytes to be hashed
	// when creating the DKIM-Signature field. The default value is 0, indicating that
	// the entire body should be hashed.
	void put_DkimBodyLengthCount(int newVal);

	// Canonicalization algorithm to be used for both header and body when creating the
	// DKIM-Signature. Possible values are "simple" and "relaxed". The default value is
	// "relaxed".
	void get_DkimCanon(CkString &str);
	// Canonicalization algorithm to be used for both header and body when creating the
	// DKIM-Signature. Possible values are "simple" and "relaxed". The default value is
	// "relaxed".
	const wchar_t *dkimCanon(void);
	// Canonicalization algorithm to be used for both header and body when creating the
	// DKIM-Signature. Possible values are "simple" and "relaxed". The default value is
	// "relaxed".
	void put_DkimCanon(const wchar_t *newVal);

	// The domain name of the signing domain when creating the DKIM-Signature.
	void get_DkimDomain(CkString &str);
	// The domain name of the signing domain when creating the DKIM-Signature.
	const wchar_t *dkimDomain(void);
	// The domain name of the signing domain when creating the DKIM-Signature.
	void put_DkimDomain(const wchar_t *newVal);

	// A colon-separated list of header field names that identify headers presented to
	// the signing algorithm when creating the DKIM-Signature. The default value is:
	// "mime-version:date:message-id:subject:from:to:content-type".
	void get_DkimHeaders(CkString &str);
	// A colon-separated list of header field names that identify headers presented to
	// the signing algorithm when creating the DKIM-Signature. The default value is:
	// "mime-version:date:message-id:subject:from:to:content-type".
	const wchar_t *dkimHeaders(void);
	// A colon-separated list of header field names that identify headers presented to
	// the signing algorithm when creating the DKIM-Signature. The default value is:
	// "mime-version:date:message-id:subject:from:to:content-type".
	void put_DkimHeaders(const wchar_t *newVal);

	// The selector to be used to form the DNS query for the public key. This property
	// applies to creating a DKIM-Signature. For example, if the selector is
	// "reykjavik" and the domain is "example-code.com", then the DNS query would be
	// for "reykjavik._domainkey.example-code.com".
	void get_DkimSelector(CkString &str);
	// The selector to be used to form the DNS query for the public key. This property
	// applies to creating a DKIM-Signature. For example, if the selector is
	// "reykjavik" and the domain is "example-code.com", then the DNS query would be
	// for "reykjavik._domainkey.example-code.com".
	const wchar_t *dkimSelector(void);
	// The selector to be used to form the DNS query for the public key. This property
	// applies to creating a DKIM-Signature. For example, if the selector is
	// "reykjavik" and the domain is "example-code.com", then the DNS query would be
	// for "reykjavik._domainkey.example-code.com".
	void put_DkimSelector(const wchar_t *newVal);

	// The signing algorithm to be used in creating the DomainKey-Signature. The only
	// possible value is "rsa-sha1". The default value is "rsa-sha1".
	void get_DomainKeyAlg(CkString &str);
	// The signing algorithm to be used in creating the DomainKey-Signature. The only
	// possible value is "rsa-sha1". The default value is "rsa-sha1".
	const wchar_t *domainKeyAlg(void);
	// The signing algorithm to be used in creating the DomainKey-Signature. The only
	// possible value is "rsa-sha1". The default value is "rsa-sha1".
	void put_DomainKeyAlg(const wchar_t *newVal);

	// Canonicalization algorithm to be used for when creating the DomainKey-Signature.
	// Possible values are "simple" and "nofws". The default value is "nofws".
	void get_DomainKeyCanon(CkString &str);
	// Canonicalization algorithm to be used for when creating the DomainKey-Signature.
	// Possible values are "simple" and "nofws". The default value is "nofws".
	const wchar_t *domainKeyCanon(void);
	// Canonicalization algorithm to be used for when creating the DomainKey-Signature.
	// Possible values are "simple" and "nofws". The default value is "nofws".
	void put_DomainKeyCanon(const wchar_t *newVal);

	// The domain name of the signing domain when creating the DomainKey-Signature.
	void get_DomainKeyDomain(CkString &str);
	// The domain name of the signing domain when creating the DomainKey-Signature.
	const wchar_t *domainKeyDomain(void);
	// The domain name of the signing domain when creating the DomainKey-Signature.
	void put_DomainKeyDomain(const wchar_t *newVal);

	// A colon-separated list of header field names that identify headers presented to
	// the signing algorithm when creating the DomainKey-Signature. The default value
	// is: "mime-version:date:message-id:subject:from:to:content-type".
	void get_DomainKeyHeaders(CkString &str);
	// A colon-separated list of header field names that identify headers presented to
	// the signing algorithm when creating the DomainKey-Signature. The default value
	// is: "mime-version:date:message-id:subject:from:to:content-type".
	const wchar_t *domainKeyHeaders(void);
	// A colon-separated list of header field names that identify headers presented to
	// the signing algorithm when creating the DomainKey-Signature. The default value
	// is: "mime-version:date:message-id:subject:from:to:content-type".
	void put_DomainKeyHeaders(const wchar_t *newVal);

	// The selector to be used to form the DNS query for the public key. This property
	// applies to creating a DomainKey-Signature. For example, if the selector is
	// "reykjavik" and the domain is "example-code.com", then the DNS query would be
	// for "reykjavik._domainkey.example-code.com".
	void get_DomainKeySelector(CkString &str);
	// The selector to be used to form the DNS query for the public key. This property
	// applies to creating a DomainKey-Signature. For example, if the selector is
	// "reykjavik" and the domain is "example-code.com", then the DNS query would be
	// for "reykjavik._domainkey.example-code.com".
	const wchar_t *domainKeySelector(void);
	// The selector to be used to form the DNS query for the public key. This property
	// applies to creating a DomainKey-Signature. For example, if the selector is
	// "reykjavik" and the domain is "example-code.com", then the DNS query would be
	// for "reykjavik._domainkey.example-code.com".
	void put_DomainKeySelector(const wchar_t *newVal);

	// The number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any method call prior to
	// completion. If HeartbeatMs is 0 (the default), no AbortCheck event callbacks
	// will fire.
	int get_HeartbeatMs(void);
	// The number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any method call prior to
	// completion. If HeartbeatMs is 0 (the default), no AbortCheck event callbacks
	// will fire.
	void put_HeartbeatMs(int newVal);

	// Contains JSON with information about the last DKIM or DomainKey signature
	// verified (or verification failure). The JSON will contain information like this:
	// {
	//   "domain": "amazonses.com",
	//   "selector": "7v7vs6w47njt4pimodk5mmttbegzsi6n",
	//   "publicKey": "MIGfMA0GCSq...z6uqeQIDAQAB",
	//   "canonicalization": "relaxed/simple",
	//   "algorithm": "rsa-sha256",
	//   "signedHeaders": "Subject:From:To:Date:Mime-Version:Content-Type:References:Message-Id:Feedback-ID",
	//   "verified": "yes"
	// }
	void get_VerifyInfo(CkString &str);
	// Contains JSON with information about the last DKIM or DomainKey signature
	// verified (or verification failure). The JSON will contain information like this:
	// {
	//   "domain": "amazonses.com",
	//   "selector": "7v7vs6w47njt4pimodk5mmttbegzsi6n",
	//   "publicKey": "MIGfMA0GCSq...z6uqeQIDAQAB",
	//   "canonicalization": "relaxed/simple",
	//   "algorithm": "rsa-sha256",
	//   "signedHeaders": "Subject:From:To:Date:Mime-Version:Content-Type:References:Message-Id:Feedback-ID",
	//   "verified": "yes"
	// }
	const wchar_t *verifyInfo(void);



	// ----------------------
	// Methods
	// ----------------------
	// Constructs and prepends a DKIM-Signature header to the MIME. Prior to calling
	// this method, your program must set both the DkimDomain and DkimSelector
	// properties, and it must load a private key by calling LoadDkimPk or
	// LoadDkimPkFile.
	// 
	// Note: This method is deprecated. Use the DkimSign method instead.
	// 
	bool AddDkimSignature(CkByteData &mimeIn, CkByteData &outBytes);

	// Constructs and prepends a DomainKey-Signature header to the MIME. Prior to
	// calling this method, your program must set both the DomainKeyDomain and
	// DomainKeySelector properties, and it must load a private key by calling
	// LoadDomainKeyPk or LoadDomainKeyPkFile.
	// 
	// Note: This method is deprecated. Use the DomainKeySign method instead.
	// 
	bool AddDomainKeySignature(CkByteData &mimeIn, CkByteData &outBytes);

	// Constructs and prepends a DKIM-Signature header to the MIME passed in mimeData.
	// Prior to calling this method, your program must set both the DkimDomain and
	// DkimSelector properties, and it must load a private key by calling
	// SetDkimPrivateKey.
	bool DkimSign(CkBinDataW &mimeData);

	// Verifies the Nth DKIM-Signature header in the mimeData. (In most cases, there is
	// only one signature.) The 1st signature is at sigIndex 0.
	// 
	// On return, the VerifyInfo property will contain additional information about the
	// DKIM-Signature that was verified (or not verified).
	// 
	bool DkimVerify(int sigIndex, CkBinDataW &mimeData);

	// Constructs and prepends a DomainKey-Signature header to the MIME passed in mimeData.
	// Prior to calling this method, your program must set both the DomainKeyDomain and
	// DomainKeySelector properties, and it must load a private key by calling
	// SetDomainKeyPrivateKey.
	bool DomainKeySign(CkBinDataW &mimeData);

	// Verifies the Nth DomainKey-Signature header in the mimeData. (In most cases, there
	// is only one signature.) The 1st signature is at sigIndex 0.
	// 
	// On return, the VerifyInfo property will contain additional information about the
	// DKIM-Signature that was verified (or not verified).
	// 
	bool DomainKeyVerify(int sigIndex, CkBinDataW &mimeData);

	// Loads an RSA private key to be used for creating a DKIM-Signature. Any valid RSA
	// private key format that is not binary, such as PEM or XML, may be passed to this
	// method. This method will automatically determine the format and parse it
	// correctly. A password is only required if key is encrypted, such as for
	// encrypted PEM.
	// 
	// Note: This method is deprecated. Use the SetDkimPrivateKey method instead.
	// 
	bool LoadDkimPk(const wchar_t *privateKey, const wchar_t *optionalPassword);

	// Loads an RSA private key to be used for creating a DKIM-Signature. Any valid RSA
	// private key binary format, such as DER, may be passed to this method. This
	// method will automatically determine the format and parse it correctly. A
	// password is only required if key is encrypted.
	// 
	// Note: This method is deprecated. Use the SetDkimPrivateKey method instead.
	// 
	bool LoadDkimPkBytes(CkByteData &privateKeyDer, const wchar_t *optionalPassword);

	// Loads an RSA private key file to be used for creating a DKIM-Signature. The file
	// name contains any valid RSA private key format, such as DER, PKCS8, PEM, XML,
	// etc. This method will automatically determine the format and parse it correctly.
	// A password is only required if key is encrypted, such as for encrypted PEM or
	// encrypted PKCS8.
	// 
	// Note: This method is deprecated. Use the SetDkimPrivateKey method instead.
	// 
	bool LoadDkimPkFile(const wchar_t *privateKeyFilePath, const wchar_t *optionalPassword);

	// Loads an RSA private key to be used for creating a DomainKey-Signature. Any
	// valid RSA private key format that is not binary, such as PEM or XML, may be
	// passed to this method. This method will automatically determine the format and
	// parse it correctly. A password is only required if key is encrypted, such as for
	// encrypted PEM.
	// 
	// Note: This method is deprecated. Use the SetDomainKeyPrivateKey method instead.
	// 
	bool LoadDomainKeyPk(const wchar_t *privateKey, const wchar_t *optionalPassword);

	// Loads an RSA private key to be used for creating a DomainKey-Signature. Any
	// valid RSA private key binary format, such as DER, may be passed to this method.
	// This method will automatically determine the format and parse it correctly. A
	// password is only required if key is encrypted.
	// 
	// Note: This method is deprecated. Use the SetDomainKeyPrivateKey method instead.
	// 
	bool LoadDomainKeyPkBytes(CkByteData &privateKeyDer, const wchar_t *optionalPassword);

	// Loads an RSA private key file to be used for creating a DomainKey-Signature. The
	// file name contains any valid RSA private key format, such as DER, PKCS8, PEM,
	// XML, etc. This method will automatically determine the format and parse it
	// correctly. A password is only required if key is encrypted, such as for
	// encrypted PEM or encrypted PKCS8.
	// 
	// Note: This method is deprecated. Use the SetDomainKeyPrivateKey method instead.
	// 
	bool LoadDomainKeyPkFile(const wchar_t *privateKeyFilePath, const wchar_t *optionalPassword);

	// Caches a public key to be used for verifying DKIM and DomainKey signatures for a
	// given selector and domain. The publicKey is a string containing an RSA public key in
	// any text format, such as XML, PEM, etc. This method will automatically detect
	// the format and load the public key correctly. This method is useful for testing
	// DKIM and DomainKey verification when your public key has not yet been installed
	// in DNS.
	bool LoadPublicKey(const wchar_t *selector, const wchar_t *domain, const wchar_t *publicKey);

	// Caches a public key to be used for verifying DKIM and DomainKey signatures for a
	// given selector and domain. The publicKeyFilepath is a filepath of an RSA public key in any
	// format. This method will automatically detect the format and load the public key
	// correctly. This method is useful for testing DKIM and DomainKey verification
	// when your public key has not yet been installed in DNS.
	bool LoadPublicKeyFile(const wchar_t *selector, const wchar_t *domain, const wchar_t *publicKeyFilepath);

	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTaskW &task);

	// Returns the number of DKIM-Signature header fields found in mimeData.
	int NumDkimSignatures(CkByteData &mimeData);

	// Returns the number of DKIM-Signature header fields found in mimeData.
	int NumDkimSigs(CkBinDataW &mimeData);

	// Returns the number of DomainKey-Signature header fields found in mimeData.
	int NumDomainKeySignatures(CkByteData &mimeData);

	// Returns the number of DomainKey-Signature header fields found in mimeData.
	int NumDomainKeySigs(CkBinDataW &mimeData);

	// Useful if your application is going to verify many emails from a single domain
	// (or a few domains). This eliminates the need to do a DNS lookup to fetch the
	// public key each time an email's DKIM or DomainKey signature is verified.
	// 
	// This method may be called multiple times -- once for each selector/domain to be
	// pre-fetched. The verify methods (VerifyDkimSignature and
	// VerifyDomainKeySignature) will use a pre-fetched key if the selector and domain
	// match.
	// 
	bool PrefetchPublicKey(const wchar_t *selector, const wchar_t *domain);

	// Creates an asynchronous task to call the PrefetchPublicKey method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *PrefetchPublicKeyAsync(const wchar_t *selector, const wchar_t *domain);

	// Sets the private key file to be used for creating a DKIM-Signature.
	bool SetDkimPrivateKey(CkPrivateKeyW &privateKey);

	// Sets the private key file to be used for creating a DomainKey-Signature.
	bool SetDomainKeyPrivateKey(CkPrivateKeyW &privateKey);

	// Unlocks the component allowing for the full functionality to be used. If this
	// method unexpectedly returns false, examine the contents of the LastErrorText
	// property to determine the reason for failure.
	bool UnlockComponent(const wchar_t *unlockCode);

	// Verifies the Nth DKIM-Signature header in the mimeData. (In most cases, there is
	// only one signature.) The 1st signature is at sigIndex 0.
	// 
	// Important: Many anti-virus programs, such as AVG, will modify the MIME of an
	// email as it is received. This will cause DKIM signature verification to fail
	// because the body of the MIME is modified.
	// 
	// Note: This method is deprecated. Use the DkimVerify method instead.
	// 
	bool VerifyDkimSignature(int sigIndex, CkByteData &mimeData);

	// Creates an asynchronous task to call the VerifyDkimSignature method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *VerifyDkimSignatureAsync(int sigIndex, CkByteData &mimeData);

	// Verifies the Nth DomainKey-Signature header in the mimeData. (In most cases, there
	// is only one signature.) The 1st signature is at sigIndex 0.
	// 
	// Important: Many anti-virus programs, such as AVG, will modify the MIME of an
	// email as it is received. This will cause DomainKey signature verification to
	// fail because the body of the MIME is modified.
	// 
	// Note: This method is deprecated. Use the DomainKeyVerify method instead.
	// 
	bool VerifyDomainKeySignature(int sigIndex, CkByteData &mimeData);

	// Creates an asynchronous task to call the VerifyDomainKeySignature method with
	// the arguments provided. (Async methods are available starting in Chilkat
	// v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *VerifyDomainKeySignatureAsync(int sigIndex, CkByteData &mimeData);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
