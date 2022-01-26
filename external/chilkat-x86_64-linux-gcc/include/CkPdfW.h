// CkPdfW.h: interface for the CkPdfW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkPdfW_H
#define _CkPdfW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacksW.h"

class CkCertW;
class CkJsonObjectW;
class CkBinDataW;
class CkHttpW;
class CkPrivateKeyW;
class CkTaskW;
class CkBaseProgressW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkPdfW
class CK_VISIBLE_PUBLIC CkPdfW  : public CkClassWithCallbacksW
{
	private:
	bool m_cbOwned;

	private:
	
	// Don't allow assignment or copying these objects.
	CkPdfW(const CkPdfW &);
	CkPdfW &operator=(const CkPdfW &);

    public:
	CkPdfW(void);
	virtual ~CkPdfW(void);

	

	static CkPdfW *createNew(void);
	

	CkPdfW(bool bCallbackOwned);
	static CkPdfW *createNew(bool bCallbackOwned);

	
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
	// The number of pages in the currently open PDF.
	int get_NumPages(void);

	// The number of digital signatures present in the currently open PDF.
	int get_NumSignatures(void);

	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string. It can be set to a list of one or more of the
	// following comma separated keywords:
	//     "WriteStandardXref" - When writing the PDF, write the cross reference
	//     section in standard format if possible. (The "standard format" is the older
	//     non-compressed format.)
	//     "NO_VERIFY_CERT_SIGNATURES" - When countersigning a PDF (i.e. adding a new
	//     signature to a PDF that already contains one or more signatures), Chilkat will
	//     automatically validate the existing signatures and their certificates. (The
	//     signing certificates are typically embedded within a signature.) If any of these
	//     validations fail, the new signature is not added. Sometimes, an existing
	//     signature is validated, but the certs in the chain of authentication, such as
	//     issuer certs or root CA certs, are not included and not available to check. In
	//     this case, you can add the "NO_VERIFY_CERT_SIGNATURES" to skip the existing
	//     signature certificate validations.
	void get_UncommonOptions(CkString &str);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string. It can be set to a list of one or more of the
	// following comma separated keywords:
	//     "WriteStandardXref" - When writing the PDF, write the cross reference
	//     section in standard format if possible. (The "standard format" is the older
	//     non-compressed format.)
	//     "NO_VERIFY_CERT_SIGNATURES" - When countersigning a PDF (i.e. adding a new
	//     signature to a PDF that already contains one or more signatures), Chilkat will
	//     automatically validate the existing signatures and their certificates. (The
	//     signing certificates are typically embedded within a signature.) If any of these
	//     validations fail, the new signature is not added. Sometimes, an existing
	//     signature is validated, but the certs in the chain of authentication, such as
	//     issuer certs or root CA certs, are not included and not available to check. In
	//     this case, you can add the "NO_VERIFY_CERT_SIGNATURES" to skip the existing
	//     signature certificate validations.
	const wchar_t *uncommonOptions(void);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string. It can be set to a list of one or more of the
	// following comma separated keywords:
	//     "WriteStandardXref" - When writing the PDF, write the cross reference
	//     section in standard format if possible. (The "standard format" is the older
	//     non-compressed format.)
	//     "NO_VERIFY_CERT_SIGNATURES" - When countersigning a PDF (i.e. adding a new
	//     signature to a PDF that already contains one or more signatures), Chilkat will
	//     automatically validate the existing signatures and their certificates. (The
	//     signing certificates are typically embedded within a signature.) If any of these
	//     validations fail, the new signature is not added. Sometimes, an existing
	//     signature is validated, but the certs in the chain of authentication, such as
	//     issuer certs or root CA certs, are not included and not available to check. In
	//     this case, you can add the "NO_VERIFY_CERT_SIGNATURES" to skip the existing
	//     signature certificate validations.
	void put_UncommonOptions(const wchar_t *newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Adds a certificate to be used for PDF signing. To sign with more than one
	// certificate, call AddSigningCert once per certificate.
	// 
	// Note: This method is used to provide the ability to sign once with multiple
	// certificates. This is different than signing with one certificate, and then
	// signing again with a different certificate.
	// 
	bool AddSigningCert(CkCertW &cert);

	// Gets the contents of the PDF's Document Security Store (/DSS) if it exists.
	// Returns the information in JSON format (in json). If there is no /DSS then an
	// empty JSON document "{}" is returned in json.
	bool GetDss(CkJsonObjectW &json);

	// This method can be used to get the signer certificate after calling
	// VerifySignature. The index should be the same value as the index passed to
	// VerifySignature. If successful, and if the signer certificate is fully available
	// within the signature, the cert is loaded with the signer certificate.
	bool GetSignerCert(int index, CkCertW &cert);

	// Provides information about what transpired in the last method called. For many
	// methods, there is no information. For some methods, details about what
	// transpired can be obtained via LastJsonData. For example, after calling a method
	// to verify a signature, the LastJsonData will return JSON with details about the
	// algorithms used for signature verification.
	// The caller is responsible for deleting the object returned by this method.
	CkJsonObjectW *LastJsonData(void);

	// Loads the PDF file contained in pdfData.
	bool LoadBd(CkBinDataW &pdfData);

	// Load a PDF file into this object in memory.
	bool LoadFile(const wchar_t *filePath);

	// Sets the HTTP object to be used to communicate with the timestamp authority
	// (TSA) server for cases where long term validation (LTV) of signatures is
	// desired. The http is used to send the requests, and it allows for connection
	// related settings and timeouts to be set. For example, if HTTP or SOCKS proxies
	// are required, these features can be specified on the http.
	// 
	// The http is also used to send OCSP requests to store OCSP responses in the PDF's
	// document security store (DSS).
	// 
	void SetHttpObj(CkHttpW &http);

	// Provides an optional JPG image to be included in the signature appearance. The
	// JPG data is passed in jpgData.
	bool SetSignatureJpeg(CkBinDataW &jpgData);

	// Specifies a certificate to be used when signing the PDF. Signing requires both a
	// certificate and private key. In this case, the private key is implicitly
	// specified if the certificate originated from a PFX that contains the
	// corresponding private key, or if on a Windows-based computer where the
	// certificate and corresponding private key are pre-installed.
	bool SetSigningCert(CkCertW &cert);

	// Specifies a digital certificate and private key to be used for signing the PDF.
	bool SetSigningCert2(CkCertW &cert, CkPrivateKeyW &privateKey);

	// Signs the open PDF and if successful writes the signed PDF to the ARG3. The jsonOptions
	// contains information and instructions about the signature. See the examples
	// below for more detailed information about the JSON options listed here.
	// 
	// Summary of PDF Signing Options
	//     appearance.fillUnsignedSignatureField - Can be set to true) to tell Chilkat
	//     to find the 1st existing unsigned signature field and use its location and size.
	//     Chilkat will automatically scale the visual appearance (text + graphics) to fit
	//     the pre-existing signature field. When fillUnsignedSignatureField is specified,
	//     it is not necessary to set appearance.x, appearance.y, appearance.fontScale,
	//     etc.
	//     appearance.fontScale - The font scale (in pts) to be used, such as "10.0".
	//     appearance.height - Optional to specify the exact height of the visible
	//     signature rectangle in points, where 72 points equals 1 inch. If the
	//     appearance.height is set, then appearance.width should be set to "auto" (or left
	//     unset). Chilkat will compute the font scale to achieve the desired rectangle
	//     height, and the resulting width will depend on the text.
	//     appearance.image - Indicates an image will be included in the signature. Set
	//     to the keyword "custom-jpg" to use an image set by calling the SetSignatureJpeg
	//     method. Otherwise can be set to one of the following keywords to indicate a
	//     built-in SVG graphic. (These are graphics embedded within the Chilkat library
	//     itself.)
	//         green-check-grey-circle
	//         green-check-green-circle
	//         application-approved
	//         application-rejected
	//         document-accepted
	//         approved
	//         blue-check-mark
	//         green-check-mark
	//         green-check-grey-circle
	//         red-x-red-circle
	//         rejected
	//         result-failure
	//         result-pass
	//         signature
	//         document-check
	//         document-x
	//         red-x-grey-circle
	//     appearance.imageOpacity - Sets the image opacity. Can be an integer from 1
	//     to 100.
	//     appearance.imagePlacement - Sets the image placment within the signature
	//     rectangle. Can be "left", "right", or "center". Images placed in the center
	//     typically have opacity 50% or less because the text is displayed over the image
	//     (i.e. it is a background image). Images placed left or right are not background
	//     images. The signature rectangle is divided into two sub-rectangles, one for the
	//     image, and one for the text.
	//     appearance.margin_x - If the appearance.x is set to "left", then this can
	//     optionally be use to specify the position from the left edge. The default margin
	//     is 10.0 (10 points, where 72 points equals 1 inch).
	//     appearance.margin_y - If the appearance.y is set to "top", then this can
	//     optionally be use to specify the position from the top. The default margin for y
	//     is 20.0 (20 points, where 72 points equals 1 inch).
	//     appearance.text[i] - The text that should appear in the signature box. Each
	//     line is specified in a JSON array item, where "i" is an integer starting with
	//     "0" as the 1st line. The text can contain the following keywords which are
	//     replaced with actual values:
	//         cert_country - The signing certificate's subject country (C).
	//         cert_cn - The signing certificate's subject common name (CN).
	//         cert_dn - The signing certificate's DN (distinguished name).
	//         cert_email - The signing certificate's subject email address (E).
	//         cert_issuer_cn - The signing certificate's issuer's common name (CN).
	//         cert_locality - The signing certificate's subject locality (L).
	//         cert_organization - The signing certificate's subject organization (O).
	//         cert_org_id - The signing certificate's organization ID.
	//         cert_ou - The signing certificate's subject organizational unit (OU).
	//         cert_san_rfc822name - The signing certificate's RFC822 subject
	//         alternative name.
	//         cert_serial_dec - The signing certificate's serial number in decimal
	//         format.
	//         cert_serial_hex - The signing certificate's serial number in hex format.
	//         cert_state - The signing certificate's subject state (S).
	//         cert_thumbprint - The signing certificate's thumbprint (The SHA1 hash of
	//         the binary DER representation in hex format).
	//         current_datetime - Current local date/time in the format such as "Sep 11
	//         2020 16:30:54".
	//         current_dt - Current local date/time in PDF date/time string format,
	//         such as YYYY.MM.DD hh:mm:ss -05'00'.
	//         current_rfc822_dt_gmt - Current GMT date/time in RFC822 format, such as
	//         "Mon, 22 Nov 2021 15:58:41 GMT".
	//         current_rfc822_dt_local - Current local date/time in RFC822 format, such
	//         as "Mon, 22 Nov 2021 09:56:16 -0600".
	//         current_timestamp_gmt - Current GMT date/time in timestamp format, such
	//         as "1990-12-31T23:59:60Z".
	//         current_timestamp_local - Current local date/time in timestamp format,
	//         such as "2019-10-12T03:20:50.52-04:00".
	//     appearance.width - Optional to specify the exact width of the visible
	//     signature rectangle in points, where 72 points equals 1 inch. If the
	//     appearance.width is set, then appearance.height should be set to "auto" (or left
	//     unset). Chilkat will compute the font scale to achieve the desired rectangle
	//     width, and the resulting height will depend on the text and number of lines of
	//     text.
	//     appearance.x - The horizontal position on the page of the left edge of the
	//     signature box. Can be a keyword such as "left", "right", or "middle" to specify
	//     a typical placement with default margins. Otherwise can be a floating point
	//     number where 0.0 is the leftmost coordinate of a page, and 612.0 is the right
	//     most. Can also be the keyword "after" to place the signature just to the right
	//     of the rightmost signature on the page.
	//     appearance.y - The vertical position on the page of the top edge of the
	//     signature box. Can be a keyword such as "top", "middle", or "bottom" to specify
	//     a typical placement with default margins. Otherwise can be a floating point
	//     number where 0.0 is the bottom coordinate of a page, and 792.0 is the top. Can
	//     also be the keyword "below" to place the signature just under the bottommost
	//     signature on the page.
	//     contactInfo - Optional to provide free-form text with contact information
	//     about the signer.(
	//     docMDP.add - Set this boolean value to true to include the Document MDP
	//     Permissions with a certifying signature. When a certifying signature is desired,
	//     both "lockAfterSigning" and "docMDP.add" should be specified. The default Doc
	//     MDP permission level is 2.
	//     docMDP.accessPermissions - Include this if the docMDP.add is specified, and
	//     a permission level different from the default value of 2 is desired. Possible
	//     values are:
	//         1: No changes to the document are permitted and any changes invalidate
	//         the signature.
	//         2: Permitted changes include filling in forms, instantiating page
	//         templates and signing.
	//         3: Same as 2, but also allow annotation creation, deletion, and
	//         modification.
	//     embedCertChain - Boolean to control whether the certificate chain is
	//     included in the signature. The default is to include the certificates in the
	//     chain of authentication. Set this to false to only include the signing
	//     certificate (this is not common).
	//     hashAlgorithm - If the signing certificate is RSA-based, and the signing
	//     scheme (padding scheme) is RSA-PSS, then this specifies the PSS hash algorithm.
	//     Can be "sha1", "sha256", "sha384", or "sha512". ("sha256" is what should be
	//     commonly chosen.)
	//     includeRootCert - Boolean to control whether the root CA certificate is
	//     included in the certificate chain (assuming the certificate chain is included).
	//     The default is to include the root CA certificate. Set this to false to exclude
	//     the root CA certificate from being included (this is not common).
	//     invisibleSignature - Set this boolean to true to create an invisible
	//     signature with no appearance.
	//     legalAttestation - Set to provide a free-form text legal attestation.
	//     location - Optional to provide free-form text with a description of the
	//     geographic location where the PDF was signed.(
	//     lockAfterSigning - Set this boolean to true to certify and lock a PDF as
	//     opposed to adding an approval signature (the default) which allows for
	//     additional countersignatures.
	//     ltvOcsp - Set this boolean to true to create an LTV-enabled signature.
	//     page - The page number where the signature will be placed. Page 1 is the 1st
	//     page.
	//     reason - Optional to provide text indicating the reason for the signature.(
	//     signatureAlgorithm - If the signing certificate is RSA-based, then chooses
	//     the RSA padding scheme. Possible values are "pkcs" for PKCS-v1_5 or "pss" for
	//     RSASSA-PSS.
	//     signingCertificateV2 - Set to "1" to include the "SigningCertificateV2"
	//     authenticated attribute. This is desired in most cases.
	//     signingTime - Set to "1" to include the "SigningTime" authenticated
	//     attribute. This is desired in most cases. Note: This is not the same as
	//     time-stamping. This is a fundamental authenticated attribute that should be
	//     included with or without the addition of time-stamping.
	//     sigTextLabel - Set to provide free-form text for the signatures annotation
	//     text label.
	//     timestampToken.enabled - Set to true to tell Chilkat to request a timestamp
	//     from a TSA server and include the timestamp token in the signature's
	//     authentication attributes
	//     timestampToken.tsaUrl - The timestamp server's URL.
	//     timestampToken.tsaUsername, timestampToken.tsaPassword - If the timestamp
	//     server requires a login and password.
	//     timestampToken.requestTsaCert - Set to true to ask the timestamp server to
	//     include its certificate in the timestamp token.
	// 
	bool SignPdf(CkJsonObjectW &jsonOptions, const wchar_t *outFilePath);

	// Creates an asynchronous task to call the SignPdf method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SignPdfAsync(CkJsonObjectW &jsonOptions, const wchar_t *outFilePath);

	// Verifies the Nth signature contained in the PDF, where the 1st signature is
	// indicated by an index of 0. Returns true if the signature valid, otherwise
	// returns false. The sigInfo is an output argument and is populated with
	// information about the validated or unvalidated signature.
	bool VerifySignature(int index, CkJsonObjectW &sigInfo);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
