// CkCert.h: interface for the CkCert class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkCert_H
#define _CkCert_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkByteData;
class CkBinData;
class CkPrivateKey;
class CkPublicKey;
class CkCertChain;
class CkDateTime;
class CkTask;
class CkXmlCertVault;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkCert
class CK_VISIBLE_PUBLIC CkCert  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkCert(const CkCert &);
	CkCert &operator=(const CkCert &);

    public:
	CkCert(void);
	virtual ~CkCert(void);

	static CkCert *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The authority key identifier of the certificate in base64 string format. This is
	// only present if the certificate contains the extension OID 2.5.29.35.
	void get_AuthorityKeyId(CkString &str);
	// The authority key identifier of the certificate in base64 string format. This is
	// only present if the certificate contains the extension OID 2.5.29.35.
	const char *authorityKeyId(void);

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

	// The version of the certificate (1, 2, or 3). A value of 0 indicates an error --
	// the most likely cause being that the certificate object is empty (i.e. was never
	// loaded with a certificate). Note: This is not the version of the software, it is
	// the version of the X.509 certificate object. The version of the Chilkat
	// certificate software is indicated by the Version property.
	int get_CertVersion(void);

	// (Relevant only when running on a Microsoft Windows operating system.) If the
	// HasKeyContainer property is true, then the certificate is linked to a key
	// container and this property contains the name of the associated CSP
	// (cryptographic service provider). When a certificate is linked to a key
	// container , the following properties will provide information about the key
	// container and private key: CspName, KeyContainerName, MachineKeyset, and Silent.
	void get_CspName(CkString &str);
	// (Relevant only when running on a Microsoft Windows operating system.) If the
	// HasKeyContainer property is true, then the certificate is linked to a key
	// container and this property contains the name of the associated CSP
	// (cryptographic service provider). When a certificate is linked to a key
	// container , the following properties will provide information about the key
	// container and private key: CspName, KeyContainerName, MachineKeyset, and Silent.
	const char *cspName(void);

	// Has a value of true if the certificate or any certificate in the chain of
	// authority has expired. (This information is not available when running on
	// Windows 95/98 computers.)
	bool get_Expired(void);

	// Returns a string containing a comma separated list of keywords with the extended
	// key usages of the certificate. The list of possible extended key usages are:
	//     serverAuth - TLS WWW server authentication
	//     clientAuth - TLS WWW client authentication
	//     codeSigning - Signing of downloadable executable code
	//     emailProtection - Email protection
	//     timeStamping - Binding the hash of an object to a time
	//     OCSPSigning - Signing OCSP responses
	void get_ExtendedKeyUsage(CkString &str);
	// Returns a string containing a comma separated list of keywords with the extended
	// key usages of the certificate. The list of possible extended key usages are:
	//     serverAuth - TLS WWW server authentication
	//     clientAuth - TLS WWW client authentication
	//     codeSigning - Signing of downloadable executable code
	//     emailProtection - Email protection
	//     timeStamping - Binding the hash of an object to a time
	//     OCSPSigning - Signing OCSP responses
	const char *extendedKeyUsage(void);

	// true if this certificate can be used for client authentication, otherwise
	// false.
	bool get_ForClientAuthentication(void);

	// true if this certificate can be used for code signing, otherwise false.
	bool get_ForCodeSigning(void);

	// true if this certificate can be used for sending secure email, otherwise
	// false.
	bool get_ForSecureEmail(void);

	// true if this certificate can be used for server authentication, otherwise
	// false.
	bool get_ForServerAuthentication(void);

	// true if this certificate can be used for time stamping, otherwise false.
	bool get_ForTimeStamping(void);

	// (Relevant only when running on a Microsoft Windows operating system.) Indicates
	// whether this certificate is linked to a key container. If true then the
	// certificate is linked to a key container (usually containing a private key). If
	// false, then it is not.
	// 
	// When a certificate is linked to a key container , the following properties will
	// provide information about the key container and private key: CspName,
	// KeyContainerName, MachineKeyset, and Silent.
	bool get_HasKeyContainer(void);

	// Bitflags indicating the intended usages of the certificate. The flags are:
	// Digital Signature: 0x80
	// Non-Repudiation: 0x40
	// Key Encipherment: 0x20
	// Data Encipherment: 0x10
	// Key Agreement: 0x08
	// Certificate Signing: 0x04
	// CRL Signing: 0x02
	// Encipher-Only: 0x01
	unsigned long get_IntendedKeyUsage(void);

	// true if this is the root certificate, otherwise false.
	bool get_IsRoot(void);

	// The certificate issuer's country.
	void get_IssuerC(CkString &str);
	// The certificate issuer's country.
	const char *issuerC(void);

	// The certificate issuer's common name.
	void get_IssuerCN(CkString &str);
	// The certificate issuer's common name.
	const char *issuerCN(void);

	// The issuer's full distinguished name.
	void get_IssuerDN(CkString &str);
	// The issuer's full distinguished name.
	const char *issuerDN(void);

	// The certificate issuer's email address.
	void get_IssuerE(CkString &str);
	// The certificate issuer's email address.
	const char *issuerE(void);

	// The certificate issuer's locality, which could be a city, count, township, or
	// other geographic region.
	void get_IssuerL(CkString &str);
	// The certificate issuer's locality, which could be a city, count, township, or
	// other geographic region.
	const char *issuerL(void);

	// The certificate issuer's organization, which is typically the company name.
	void get_IssuerO(CkString &str);
	// The certificate issuer's organization, which is typically the company name.
	const char *issuerO(void);

	// The certificate issuer's organizational unit, which is the unit within the
	// organization.
	void get_IssuerOU(CkString &str);
	// The certificate issuer's organizational unit, which is the unit within the
	// organization.
	const char *issuerOU(void);

	// The certificate issuer's state or province.
	void get_IssuerS(CkString &str);
	// The certificate issuer's state or province.
	const char *issuerS(void);

	// (Relevant only when running on a Microsoft Windows operating system.) If the
	// HasKeyContainer property is true, then the certificate is linked to a key
	// container and this property contains the name of the key container.
	// 
	// When a certificate is linked to a key container , the following properties will
	// provide information about the key container and private key: CspName,
	// KeyContainerName, MachineKeyset, and Silent.
	void get_KeyContainerName(CkString &str);
	// (Relevant only when running on a Microsoft Windows operating system.) If the
	// HasKeyContainer property is true, then the certificate is linked to a key
	// container and this property contains the name of the key container.
	// 
	// When a certificate is linked to a key container , the following properties will
	// provide information about the key container and private key: CspName,
	// KeyContainerName, MachineKeyset, and Silent.
	const char *keyContainerName(void);

	// (Relevant only when running on a Microsoft Windows operating system.) If the
	// HasKeyContainer property is true, then the certificate is linked to a key
	// container and this property indicates whether the key container is in the
	// machine's keyset or in the keyset specific to the logged on user's account. If
	// true, the key container is within the machine keyset. If false, it's in the
	// user's keyset.
	// 
	// When a certificate is linked to a key container , the following properties will
	// provide information about the key container and private key: CspName,
	// KeyContainerName, MachineKeyset, and Silent.
	bool get_MachineKeyset(void);

	// If present in the certificate's extensions, returns the OCSP URL of the
	// certificate. (The Online Certificate Status Protocol (OCSP) is an Internet
	// protocol used for obtaining the revocation status of an X.509 digital
	// certificate.)
	void get_OcspUrl(CkString &str);
	// If present in the certificate's extensions, returns the OCSP URL of the
	// certificate. (The Online Certificate Status Protocol (OCSP) is an Internet
	// protocol used for obtaining the revocation status of an X.509 digital
	// certificate.)
	const char *ocspUrl(void);

	// (Relevant only when running on a Microsoft Windows operating system.) Indicates
	// whether the private key was installed with security settings that allow it to be
	// re-exported.
	bool get_PrivateKeyExportable(void);

	// true if the certificate or any certificate in the chain of authority has been
	// revoked. This information is not available when running on Windows 95/98
	// computers. Note: If this property is false, it could mean that it was not able
	// to check the revocation status. Because of this uncertainty, a CheckRevoked
	// method has been added. It returns an integer indicating one of three possible
	// states: 1 (revoked) , 0 (not revoked), -1 (unable to check revocation status).
	bool get_Revoked(void);

	// The RFC822 name of the certificate. (The RFC822 name is one part of the Subject
	// Alternative Name extension of a certificate, if it exists. It is often the only
	// part of the SAN.)
	// 
	// If the certificate contains a list of RFC822 names then this property will
	// return the comma separated list of names.
	// 
	// Starting in Chilkat v9.5.0.85, to get the complete Subject Alternative Name
	// extension as XML, use the SubjectAlternativeName property.
	// 
	void get_Rfc822Name(CkString &str);
	// The RFC822 name of the certificate. (The RFC822 name is one part of the Subject
	// Alternative Name extension of a certificate, if it exists. It is often the only
	// part of the SAN.)
	// 
	// If the certificate contains a list of RFC822 names then this property will
	// return the comma separated list of names.
	// 
	// Starting in Chilkat v9.5.0.85, to get the complete Subject Alternative Name
	// extension as XML, use the SubjectAlternativeName property.
	// 
	const char *rfc822Name(void);

	// true if this is a self-signed certificate, otherwise false.
	bool get_SelfSigned(void);

	// The certificate's serial number as a decimal string.
	void get_SerialDecimal(CkString &str);
	// The certificate's serial number as a decimal string.
	const char *serialDecimal(void);

	// The certificate's serial number as a hexidecimal string.
	void get_SerialNumber(CkString &str);
	// The certificate's serial number as a hexidecimal string.
	const char *serialNumber(void);

	// Hexidecimal string of the SHA-1 thumbprint for the certificate. (This is the
	// SHA1 hash of the binary DER representation of the entire X.509 certificate.)
	void get_Sha1Thumbprint(CkString &str);
	// Hexidecimal string of the SHA-1 thumbprint for the certificate. (This is the
	// SHA1 hash of the binary DER representation of the entire X.509 certificate.)
	const char *sha1Thumbprint(void);

	// Returns true if the certificate and all certificates in the chain of authority
	// have valid signatures, otherwise returns false.
	bool get_SignatureVerified(void);

	// (Relevant only when running on a Microsoft Windows operating system.)
	// 
	// If the HasKeyContainer property is true, then the certificate is linked to a
	// key container and this property indicates that the key container will attempt to
	// open any keys silently without any user interface prompts.
	// 
	// When a certificate is linked to a key container , the following properties will
	// provide information about the key container and private key: CspName,
	// KeyContainerName, MachineKeyset, and Silent.
	// 
	bool get_Silent(void);

	// If set to true, then no dialog will automatically popup if the SmartCardPin is
	// incorrect. Instead, the method requiring the private key on the smart card will
	// fail. The default value of this property is false, which means that if the
	// SmartCardPin property is incorrect, a dialog with prompt will be displayed.
	bool get_SmartCardNoDialog(void);
	// If set to true, then no dialog will automatically popup if the SmartCardPin is
	// incorrect. Instead, the method requiring the private key on the smart card will
	// fail. The default value of this property is false, which means that if the
	// SmartCardPin property is incorrect, a dialog with prompt will be displayed.
	void put_SmartCardNoDialog(bool newVal);

	// Can be set to the PIN value for a certificate / private key stored on a smart
	// card.
	void get_SmartCardPin(CkString &str);
	// Can be set to the PIN value for a certificate / private key stored on a smart
	// card.
	const char *smartCardPin(void);
	// Can be set to the PIN value for a certificate / private key stored on a smart
	// card.
	void put_SmartCardPin(const char *newVal);

	// The subject alternative name (SAN) name of the certificate returned as XML. See
	// the examples linked below.
	void get_SubjectAlternativeName(CkString &str);
	// The subject alternative name (SAN) name of the certificate returned as XML. See
	// the examples linked below.
	const char *subjectAlternativeName(void);

	// The certificate subject's country.
	void get_SubjectC(CkString &str);
	// The certificate subject's country.
	const char *subjectC(void);

	// The certificate subject's common name.
	void get_SubjectCN(CkString &str);
	// The certificate subject's common name.
	const char *subjectCN(void);

	// The certificate subject's full distinguished name.
	void get_SubjectDN(CkString &str);
	// The certificate subject's full distinguished name.
	const char *subjectDN(void);

	// The certificate subject's email address.
	void get_SubjectE(CkString &str);
	// The certificate subject's email address.
	const char *subjectE(void);

	// The subject key identifier of the certificate in base64 string format. This is
	// only present if the certificate contains the extension OID 2.5.29.14.
	void get_SubjectKeyId(CkString &str);
	// The subject key identifier of the certificate in base64 string format. This is
	// only present if the certificate contains the extension OID 2.5.29.14.
	const char *subjectKeyId(void);

	// The certificate subject's locality, which could be a city, count, township, or
	// other geographic region.
	void get_SubjectL(CkString &str);
	// The certificate subject's locality, which could be a city, count, township, or
	// other geographic region.
	const char *subjectL(void);

	// The certificate subject's organization, which is typically the company name.
	void get_SubjectO(CkString &str);
	// The certificate subject's organization, which is typically the company name.
	const char *subjectO(void);

	// The certificate subject's organizational unit, which is the unit within the
	// organization.
	void get_SubjectOU(CkString &str);
	// The certificate subject's organizational unit, which is the unit within the
	// organization.
	const char *subjectOU(void);

	// The certificate subject's state or province.
	void get_SubjectS(CkString &str);
	// The certificate subject's state or province.
	const char *subjectS(void);

	// Returns true if the certificate has a trusted root authority, otherwise
	// returns false.
	// 
	// Note: As of version 9.5.0.41, the notion of what your application deems as
	// trusted becomes more specific. The TrustedRoots class/object was added in
	// v9.5.0.0. Prior to this, a certificate was considered to be anchored by a
	// trusted root if the certificate chain could be established to a root
	// (self-signed) certificate, AND if the root certificate was located somewhere in
	// the Windows registry-based certificate stores. There are two problems with this:
	// (1) it's a Windows-only solution. This property would always return false on
	// non-Windows systems, and (2) it might be considered not a strong enough set of
	// conditions for trusting a root certificate.
	// 
	// As of version 9.5.0.41, this property pays attention to the new TrustedRoots
	// class/object, which allows for an application to specificallly indicate which
	// root certificates are to be trusted. Certificates may be added to the
	// TrustedRoots object via the LoadCaCertsPem or AddCert methods, and then
	// activated by calling the TrustedRoots.Activate method. The activated trusted
	// roots are deemed to be trusted in any Chilkat API method/property that needs to
	// make this determination. In addition, the TrustedRoots object has a property
	// named TrustSystemCaRoots, which defaults to true, which allows for backward
	// compatibility. It will trust CA certificates stored in the Windows
	// registry-based certificate stores, or if on Linux, will trust certificates found
	// in /etc/ssl/certs/ca-certificates.crt.
	// 
	bool get_TrustedRoot(void);

	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string, and should typically remain empty.
	void get_UncommonOptions(CkString &str);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string, and should typically remain empty.
	const char *uncommonOptions(void);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string, and should typically remain empty.
	void put_UncommonOptions(const char *newVal);

	// The date this certificate becomes (or became) valid. It is a GMT/UTC date that
	// is returned.
	void get_ValidFrom(SYSTEMTIME &outSysTime);

	// The date (in RFC822 string format) that this certificate becomes (or became)
	// valid. It is a GMT/UTC date that is returned.
	void get_ValidFromStr(CkString &str);
	// The date (in RFC822 string format) that this certificate becomes (or became)
	// valid. It is a GMT/UTC date that is returned.
	const char *validFromStr(void);

	// The date this certificate becomes (or became) invalid. It is a GMT/UTC date that
	// is returned.
	void get_ValidTo(SYSTEMTIME &outSysTime);

	// The date (in RFC822 string format) that this certificate becomes (or became)
	// invalid. It is a GMT/UTC date that is returned.
	void get_ValidToStr(CkString &str);
	// The date (in RFC822 string format) that this certificate becomes (or became)
	// invalid. It is a GMT/UTC date that is returned.
	const char *validToStr(void);



	// ----------------------
	// Methods
	// ----------------------
	// Returns 1 if the certificate has been revoked, 0 if not revoked, and -1 if
	// unable to check the revocation status.
	// 
	// Note: This method is only implemented on Windows systems. It uses the underlying
	// Microsoft CertVerifyRevocation Platform SDK function to check the revocation
	// status of a certificate. (Search "CertVerifyRevocation" to get information about
	// it.)
	// 
	// Non-Windows (and Windows) applications can send an OCSP request as shown in the
	// example below.
	// 
	int CheckRevoked(void);


	// Verifies that the SmartCardPin property setting is correct. Returns 1 if
	// correct, 0 if incorrect, and -1 if unable to check because the underlying CSP
	// does not support the functionality.
	int CheckSmartCardPin(void);


	// Exports the digital certificate to ASN.1 DER format.
	bool ExportCertDer(CkByteData &outData);


	// Exports the digital certificate in ASN.1 DER format to a BinData object.
	bool ExportCertDerBd(CkBinData &cerData);


	// Exports the digital certificate to ASN.1 DER format binary file.
	bool ExportCertDerFile(const char *path);


	// Exports the digital certificate to an unencrypted PEM formatted string.
	bool ExportCertPem(CkString &outStr);

	// Exports the digital certificate to an unencrypted PEM formatted string.
	const char *exportCertPem(void);

	// Exports the digital certificate to an unencrypted PEM formatted file.
	bool ExportCertPemFile(const char *path);


	// Exports a certificate to an XML format where the XML tags are the names of the
	// ASN.1 objects that compose the X.509 certificate. Binary data is either hex or
	// base64 encoded. (The binary data for a "bits" ASN.1 tag is hex encoded, whereas
	// for all other ASN.1 tags, such as "octets", it is base64.)
	bool ExportCertXml(CkString &outStr);

	// Exports a certificate to an XML format where the XML tags are the names of the
	// ASN.1 objects that compose the X.509 certificate. Binary data is either hex or
	// base64 encoded. (The binary data for a "bits" ASN.1 tag is hex encoded, whereas
	// for all other ASN.1 tags, such as "octets", it is base64.)
	const char *exportCertXml(void);

	// Exports the certificate's private key.
	// The caller is responsible for deleting the object returned by this method.
	CkPrivateKey *ExportPrivateKey(void);


	// Exports the certificate's public key.
	// The caller is responsible for deleting the object returned by this method.
	CkPublicKey *ExportPublicKey(void);


	// Exports the certificate and private key (if available) to pfxData. The password is what
	// will be required to access the PFX contents at a later time. If includeCertChain is true,
	// then the certificates in the chain of authority are also included in the PFX.
	bool ExportToPfxBd(const char *password, bool includeCertChain, CkBinData &pfxData);


	// Exports the certificate and private key (if available) to an in-memory PFX
	// image. The password is what will be required to access the PFX contents at a later
	// time. If includeCertChain is true, then the certificates in the chain of authority are
	// also included in the PFX.
	bool ExportToPfxData(const char *password, bool includeCertChain, CkByteData &outBytes);


	// Exports the certificate and private key (if available) to a PFX (.pfx or .p12)
	// file. The output PFX is secured using the pfxPassword. If bIncludeCertChain is true, then the
	// certificates in the chain of authority are also included in the PFX output file.
	bool ExportToPfxFile(const char *pfxFilename, const char *pfxPassword, bool bIncludeCertChain);


	// Finds and returns the issuer certificate. If the certificate is a root or
	// self-issued, then the certificate returned is a copy of the caller certificate.
	// (The IsRoot property can be check to see if the certificate is a root (or
	// self-issued) certificate.)
	// The caller is responsible for deleting the object returned by this method.
	CkCert *FindIssuer(void);


	// Returns a certficate chain object containing all the certificates (including
	// this one), in the chain of authentication to the trusted root (if possible). If
	// this certificate object was loaded from a PFX, then the certiicates contained in
	// the PFX are automatically available for building the certificate chain. The
	// UseCertVault method can be called to provide additional certificates that might
	// be required to build the cert chain. Finally, the TrustedRoots object can be
	// used to provide a way of making trusted root certificates available.
	// 
	// Note: Prior to v9.5.0.50, this method would fail if the certificate chain could
	// not be completed to the root. Starting in v9.5.0.50, the incomplete certificate
	// chain will be returned. The certificate chain's ReachesRoot property can be
	// examined to see if the chain was completed to the root.
	// 
	// On Windows systems, the registry-based certificate stores are automatically
	// consulted if needed to locate intermediate or root certificates in the chain.
	// Chilkat searches certificate stores in the following order. SeeSystem Store
	// Locations
	// <https://docs.microsoft.com/en-us/windows/desktop/seccrypto/system-store-location
	// s> for more information.
	//     Current-User "CA" Certificate Store
	//     Local-Machine "CA" Certificate Store
	//     Current-User "Root" Certificate Store
	//     Local-Machine "Root" Certificate Store
	//     Current-User "MY" Certificate Store
	//     Local-Machine "MY" Certificate Store
	//     Current-User "ADDRESSBOOK" Certificate Store (if it exists)
	//     Local-Machine "ADDRESSBOOK" Certificate Store (if it exists)
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkCertChain *GetCertChain(void);


	// Returns a base64 encoded string representation of the certificate's binary DER
	// format, which can be passed to SetFromEncoded to recreate the certificate
	// object.
	bool GetEncoded(CkString &outStr);

	// Returns a base64 encoded string representation of the certificate's binary DER
	// format, which can be passed to SetFromEncoded to recreate the certificate
	// object.
	const char *getEncoded(void);
	// Returns a base64 encoded string representation of the certificate's binary DER
	// format, which can be passed to SetFromEncoded to recreate the certificate
	// object.
	const char *encoded(void);


	// Returns the certificate extension data as a string. This method should only be
	// called for those extensions with text values NOT stored as binary ASN.1. In most
	// cases, applications should call GetExtensionAsXml because most extensions
	// contain ASN.1 values that need to be decoded..
	bool GetExtensionAsText(const char *oid, CkString &outStr);

	// Returns the certificate extension data as a string. This method should only be
	// called for those extensions with text values NOT stored as binary ASN.1. In most
	// cases, applications should call GetExtensionAsXml because most extensions
	// contain ASN.1 values that need to be decoded..
	const char *getExtensionAsText(const char *oid);
	// Returns the certificate extension data as a string. This method should only be
	// called for those extensions with text values NOT stored as binary ASN.1. In most
	// cases, applications should call GetExtensionAsXml because most extensions
	// contain ASN.1 values that need to be decoded..
	const char *extensionAsText(const char *oid);


	// Returns the certificate extension data in XML format (converted from ASN.1). The
	// oid is an OID, such as the ones listed here:
	// http://www.alvestrand.no/objectid/2.5.29.html
	// 
	// Note: In many cases, the data within the XML is returned base64 encoded. An
	// application may need to take one further step to base64 decode the information
	// contained within the XML.
	// 
	bool GetExtensionAsXml(const char *oid, CkString &outStr);

	// Returns the certificate extension data in XML format (converted from ASN.1). The
	// oid is an OID, such as the ones listed here:
	// http://www.alvestrand.no/objectid/2.5.29.html
	// 
	// Note: In many cases, the data within the XML is returned base64 encoded. An
	// application may need to take one further step to base64 decode the information
	// contained within the XML.
	// 
	const char *getExtensionAsXml(const char *oid);
	// Returns the certificate extension data in XML format (converted from ASN.1). The
	// oid is an OID, such as the ones listed here:
	// http://www.alvestrand.no/objectid/2.5.29.html
	// 
	// Note: In many cases, the data within the XML is returned base64 encoded. An
	// application may need to take one further step to base64 decode the information
	// contained within the XML.
	// 
	const char *extensionAsXml(const char *oid);


	// Exports the certificate's private key to a PEM string (if the private key is
	// available).
	bool GetPrivateKeyPem(CkString &outStr);

	// Exports the certificate's private key to a PEM string (if the private key is
	// available).
	const char *getPrivateKeyPem(void);
	// Exports the certificate's private key to a PEM string (if the private key is
	// available).
	const char *privateKeyPem(void);


	// Returns the SPKI Fingerprint suitable for use in pinning. (See RFC 7469.) An
	// SPKI Fingerprint is defined as the output of a known cryptographic hash
	// algorithm whose input is the DER-encoded ASN.1 representation of the Subject
	// Public Key Info (SPKI) of an X.509 certificate. The hashAlg specifies the hash
	// algorithm and may be "sha256", "sha384", "sha512", "sha1", "md2", "md5",
	// "haval", "ripemd128", "ripemd160","ripemd256", or "ripemd320". The encoding
	// specifies the encoding, and may be "base64", "hex", or any of the encoding modes
	// specified in the article at the link below.
	bool GetSpkiFingerprint(const char *hashAlg, const char *encoding, CkString &outStr);

	// Returns the SPKI Fingerprint suitable for use in pinning. (See RFC 7469.) An
	// SPKI Fingerprint is defined as the output of a known cryptographic hash
	// algorithm whose input is the DER-encoded ASN.1 representation of the Subject
	// Public Key Info (SPKI) of an X.509 certificate. The hashAlg specifies the hash
	// algorithm and may be "sha256", "sha384", "sha512", "sha1", "md2", "md5",
	// "haval", "ripemd128", "ripemd160","ripemd256", or "ripemd320". The encoding
	// specifies the encoding, and may be "base64", "hex", or any of the encoding modes
	// specified in the article at the link below.
	const char *getSpkiFingerprint(const char *hashAlg, const char *encoding);
	// Returns the SPKI Fingerprint suitable for use in pinning. (See RFC 7469.) An
	// SPKI Fingerprint is defined as the output of a known cryptographic hash
	// algorithm whose input is the DER-encoded ASN.1 representation of the Subject
	// Public Key Info (SPKI) of an X.509 certificate. The hashAlg specifies the hash
	// algorithm and may be "sha256", "sha384", "sha512", "sha1", "md2", "md5",
	// "haval", "ripemd128", "ripemd160","ripemd256", or "ripemd320". The encoding
	// specifies the encoding, and may be "base64", "hex", or any of the encoding modes
	// specified in the article at the link below.
	const char *spkiFingerprint(const char *hashAlg, const char *encoding);


	// Returns a part of the certificate's subject by name or OID. The partNameOrOid can be a
	// part name, such as "CN", "O", "OU", "E", "S", "L", "C", or "SERIALNUMBER", or it
	// can be an OID such as "2.5.4.3".
	bool GetSubjectPart(const char *partNameOrOid, CkString &outStr);

	// Returns a part of the certificate's subject by name or OID. The partNameOrOid can be a
	// part name, such as "CN", "O", "OU", "E", "S", "L", "C", or "SERIALNUMBER", or it
	// can be an OID such as "2.5.4.3".
	const char *getSubjectPart(const char *partNameOrOid);
	// Returns a part of the certificate's subject by name or OID. The partNameOrOid can be a
	// part name, such as "CN", "O", "OU", "E", "S", "L", "C", or "SERIALNUMBER", or it
	// can be an OID such as "2.5.4.3".
	const char *subjectPart(const char *partNameOrOid);


	// Returns the date/time this certificate becomes (or became) valid.
	// The caller is responsible for deleting the object returned by this method.
	CkDateTime *GetValidFromDt(void);


	// Returns the date/time this certificate becomes (or became) invalid.
	// The caller is responsible for deleting the object returned by this method.
	CkDateTime *GetValidToDt(void);


	// Returns an encoded hash of a particular part of the certificate. The part may be
	// one of the following:
	//     IssuerDN
	//     IssuerPublicKey
	//     SubjectDN
	//     SubjectPublicKey
	// 
	// The hashAlg is the name of the hash algorithm, such as "sha1", "sha256", "sha384",
	// "sha512", "md5", etc. The encoding is the format to return, such as "hex", "base64",
	// etc.
	// 
	bool HashOf(const char *part, const char *hashAlg, const char *encoding, CkString &outStr);

	// Returns an encoded hash of a particular part of the certificate. The part may be
	// one of the following:
	//     IssuerDN
	//     IssuerPublicKey
	//     SubjectDN
	//     SubjectPublicKey
	// 
	// The hashAlg is the name of the hash algorithm, such as "sha1", "sha256", "sha384",
	// "sha512", "md5", etc. The encoding is the format to return, such as "hex", "base64",
	// etc.
	// 
	const char *hashOf(const char *part, const char *hashAlg, const char *encoding);

	// Returns true if a private key associated with the certificate is available.
	bool HasPrivateKey(void);


	// (Relevant only when running on a Microsoft Windows operating system.) Searches
	// the Windows Local Machine and Current User registry-based certificate stores for
	// a certificate having the common name specified. If found, the certificate is
	// loaded and ready for use.
	bool LoadByCommonName(const char *cn);


	// (Relevant only when running on a Microsoft Windows operating system.) Searches
	// the Windows Local Machine and Current User registry-based certificate stores for
	// a certificate containing the email address specified. If found, the certificate
	// is loaded and ready for use.
	bool LoadByEmailAddress(const char *emailAddress);


	// (Relevant only when running on a Microsoft Windows operating system.) Searches
	// the Windows Local Machine and Current User registry-based certificate stores for
	// a certificate matching the issuerCN and having an issuer matching the serialNumber. If
	// found, the certificate is loaded and ready for use.
	// 
	// Note: The hex serial number should be uppercase. Starting in Chilkat v9.5.0.88,
	// the hex serial number is case-insensitive.
	// 
	bool LoadByIssuerAndSerialNumber(const char *issuerCN, const char *serialNumber);


	// (Relevant only when running on a Microsoft Windows operating system.) Searches
	// the Windows Local Machine and Current User registry-based certificate stores for
	// a certificate containing a subject part matching the oid and value.
	bool LoadBySubjectOid(const char *oid, const char *value);


	// (Relevant only when running on a Microsoft Windows operating system.) Searches
	// the Windows Local Machine and Current User registry-based certificate stores for
	// a certificate having an MD5 or SHA1 thumbprint equal to the thumbprint. The hash (i.e.
	// thumbprint) is passed as a string using the encoding specified by encoding (such as
	// "base64", "hex", etc.).
	bool LoadByThumbprint(const char *thumbprint, const char *encoding);


	// Loads an ASN.1 or DER encoded certificate represented in a Base64 string.
	bool LoadFromBase64(const char *encodedCert);


	// Loads an X.509 certificate from data contained in certBytes.
	// 
	// Note: The certBytes may contain the certificate in any format. It can be binary DER
	// (ASN.1), PEM, Base64, etc. Chilkat will automatically detect the format.
	// 
	bool LoadFromBd(CkBinData &certBytes);


	// Loads an X.509 certificate from ASN.1 DER encoded bytes.
	// 
	// Note: The data may contain the certificate in any format. It can be binary DER
	// (ASN.1), PEM, Base64, etc. Chilkat will automatically detect the format.
	// 
	bool LoadFromBinary(CkByteData &data);


#if !defined(CHILKAT_MONO)
	// The same as LoadFromBinary, but instead of using a CkByteData object, the
	// pointer to the byte data and length (in number of bytes) are specified directly
	// in the method arguments.
	bool LoadFromBinary2(const void *pByteData, unsigned long szByteData);

#endif

	// Loads a certificate from a .cer, .crt, .p7b, or .pem file. This method accepts
	// certificates from files in any of the following formats:
	// 1. DER encoded binary X.509 (.CER)
	// 2. Base-64 encoded X.509 (.CER)
	// 3. Cryptographic Message Syntax Standard - PKCS #7 Certificates (.P7B)
	// 4. PEM format
	// This method decodes the certificate based on the contents if finds within the
	// file, and not based on the file extension. If your certificate is in a file
	// having a different extension, try loading it using this method before assuming
	// it won't work. This method does not load .p12 or .pfx (PKCS #12) files.
	bool LoadFromFile(const char *path);


	// Starting in Chilkat v9.5.0.87, the csp can be a string that specifies the
	// certificate to be loaded by either Subject Common Name (CN) or hex serial
	// number. For example, instead of passing a CSP name, your application would pass
	// a string such as "CN=The cert subject common name" or "serial=01020304". See the
	// linked examples below. If a certificate is specified by CN or Serial, then each
	// connected smartcard and USB token is searched for the matching certificate. If
	// the certificate is found, it is loaded and this method returns true.
	// 
	// Otherwise, this method loads the X.509 certificate from the smartcard currently
	// in the reader, or from a USB token.
	// 
	// If the smartcard contains multiple certificates, this method arbitrarily picks
	// one.
	// 
	// If the csp does not begin with "CN=" or "serial=", then the csp can be set to
	// the name of the CSP (Cryptographic Service Provider) that should be used. If
	// csp is an empty string, then the 1st CSP found matching one of the following
	// names will be used:
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
	bool LoadFromSmartcard(const char *csp);


	// Loads the certificate from a PEM string.
	bool LoadPem(const char *strPem);


	// Loads the certificate from the PFX contained in pfxData. Note: If the PFX contains
	// multiple certificates, the 1st certificate in the PFX is loaded.
	bool LoadPfxBd(CkBinData &pfxData, const char *password);


	// Loads a PFX from an in-memory image of a PFX file. Note: If the PFX contains
	// multiple certificates, the 1st certificate in the PFX is loaded.
	bool LoadPfxData(CkByteData &pfxData, const char *password);


#if !defined(CHILKAT_MONO)
	// Loads a PFX from an in-memory image of a PFX file. Note: If the PFX contains
	// multiple certificates, the 1st certificate in the PFX is loaded.
	bool LoadPfxData2(const void *pByteData, unsigned long szByteData, const char *password);

#endif

	// Loads a PFX file. Note: If the PFX contains multiple certificates, the 1st
	// certificate in the PFX is loaded.
	bool LoadPfxFile(const char *pfxPath, const char *password);


	// Loads the certificate from a completed asynchronous task.
	bool LoadTaskResult(CkTask &task);


	// Converts a PEM file to a DER file.
	bool PemFileToDerFile(const char *fromPath, const char *toPath);


	// Saves a certificate object to a .cer file.
	bool SaveToFile(const char *path);


	// Initializes the certificate object from a base64 encoded string representation
	// of the certificate's binary DER format.
	bool SetFromEncoded(const char *encodedCert);


	// Used to associate a private key with the certificate for subsequent (PKCS7)
	// signature creation or decryption.
	bool SetPrivateKey(CkPrivateKey &privKey);


	// Same as SetPrivateKey, but the key is provided in unencrypted PEM format. (Note:
	// The privKeyPem is not a file path, it is the actual PEM text.)
	bool SetPrivateKeyPem(const char *privKeyPem);


	// Adds an XML certificate vault to the object's internal list of sources to be
	// searched for certificates for help in building certificate chains and verifying
	// the certificate signature to the trusted root.
	bool UseCertVault(CkXmlCertVault &vault);


	// Verifies the certificate signature, as well as the signatures of all
	// certificates in the chain of authentication to the trusted root. Returns true
	// if all signatures are verified to the trusted root. Otherwise returns false.
	bool VerifySignature(void);


	// Returns the base64 representation of an X509PKIPathv1 containing just the
	// calling certificate. This is typically used in an X.509 Binary Security Token.
	// It is a PKIPath that contains an ordered list of X.509 public certificates
	// packaged in a PKIPath. The X509PKIPathv1 token type may be used to represent a
	// certificate path. (This is sometimes used in XAdES signatures.)
	bool X509PKIPathv1(CkString &outStr);

	// Returns the base64 representation of an X509PKIPathv1 containing just the
	// calling certificate. This is typically used in an X.509 Binary Security Token.
	// It is a PKIPath that contains an ordered list of X.509 public certificates
	// packaged in a PKIPath. The X509PKIPathv1 token type may be used to represent a
	// certificate path. (This is sometimes used in XAdES signatures.)
	const char *x509PKIPathv1(void);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
