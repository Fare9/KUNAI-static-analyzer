// CkXmlCertVault.h: interface for the CkXmlCertVault class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkXmlCertVault_H
#define _CkXmlCertVault_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkCert;
class CkByteData;
class CkCertChain;
class CkPfx;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkXmlCertVault
class CK_VISIBLE_PUBLIC CkXmlCertVault  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkXmlCertVault(const CkXmlCertVault &);
	CkXmlCertVault &operator=(const CkXmlCertVault &);

    public:
	CkXmlCertVault(void);
	virtual ~CkXmlCertVault(void);

	static CkXmlCertVault *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The master password for the vault. Certificates are stored unencrypted, but
	// private keys are stored 256-bit AES encrypted using the individual PFX
	// passwords. The PFX passwords are stored 256-bit AES encrypted using the master
	// password. The master password is required to acces any of the private keys
	// stored within the XML vault.
	void get_MasterPassword(CkString &str);
	// The master password for the vault. Certificates are stored unencrypted, but
	// private keys are stored 256-bit AES encrypted using the individual PFX
	// passwords. The PFX passwords are stored 256-bit AES encrypted using the master
	// password. The master password is required to acces any of the private keys
	// stored within the XML vault.
	const char *masterPassword(void);
	// The master password for the vault. Certificates are stored unencrypted, but
	// private keys are stored 256-bit AES encrypted using the individual PFX
	// passwords. The PFX passwords are stored 256-bit AES encrypted using the master
	// password. The master password is required to acces any of the private keys
	// stored within the XML vault.
	void put_MasterPassword(const char *newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Adds a certificate to the XML vault.
	bool AddCert(CkCert &cert);


	// Adds a certificate to the XML vault from any binary format, such as DER.
	bool AddCertBinary(CkByteData &certBytes);


	// Adds a chain of certificates to the XML vault.
	bool AddCertChain(CkCertChain &certChain);


	// Adds a certificate to the XML vault where certificate is passed directly from
	// encoded bytes (such as Base64, Hex, etc.). The encoding is indicated by encoding.
	bool AddCertEncoded(const char *encodedBytes, const char *encoding);


	// Adds a certificate to the XML vault.
	bool AddCertFile(const char *path);


	// Adds a certificate from any string representation format such as PEM.
	bool AddCertString(const char *certData);


	// Adds the contents of a PEM file to the XML vault. The PEM file may be encrypted,
	// and it may contain one or more certificates and/or private keys. The password is
	// optional and only required if the PEM file contains encrypted content that
	// requires a password.
	bool AddPemFile(const char *path, const char *password);


	// Adds a PFX (PKCS12) to the XML vault.
	bool AddPfx(CkPfx &pfx);


	// Adds a PFX to the XML vault where PFX is passed directly from in-memory binary
	// bytes.
	bool AddPfxBinary(CkByteData &pfxBytes, const char *password);


	// Adds a PFX to the XML vault where PFX is passed directly from encoded bytes
	// (such as Base64, Hex, etc.). The encoding is indicated by encoding.
	bool AddPfxEncoded(const char *encodedBytes, const char *encoding, const char *password);


	// Adds a PFX file to the XML vault.
	bool AddPfxFile(const char *path, const char *password);


	// Returns the contents of the cert vault as an XML string.
	bool GetXml(CkString &outStr);

	// Returns the contents of the cert vault as an XML string.
	const char *getXml(void);
	// Returns the contents of the cert vault as an XML string.
	const char *xml(void);


	// Loads from an XML string.
	bool LoadXml(const char *xml);


	// Loads from an XML file.
	bool LoadXmlFile(const char *path);


	// Saves the contents to an XML file.
	bool SaveXml(const char *path);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
