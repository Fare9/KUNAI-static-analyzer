// CkXmlDSigW.h: interface for the CkXmlDSigW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkXmlDSigW_H
#define _CkXmlDSigW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkWideCharBase.h"

class CkStringArrayW;
class CkXmlW;
class CkPublicKeyW;
class CkBinDataW;
class CkStringBuilderW;
class CkXmlCertVaultW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkXmlDSigW
class CK_VISIBLE_PUBLIC CkXmlDSigW  : public CkWideCharBase
{
	

	private:
	
	// Don't allow assignment or copying these objects.
	CkXmlDSigW(const CkXmlDSigW &);
	CkXmlDSigW &operator=(const CkXmlDSigW &);

    public:
	CkXmlDSigW(void);
	virtual ~CkXmlDSigW(void);

	

	static CkXmlDSigW *createNew(void);
	

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// May contain a set of directory paths specifying where referenced external files
	// are located. Directory paths should be separated using a semicolon character.
	// The default value of this property is the empty string which means no
	// directories are automatically searched.
	// 
	// This property can be used if the external file referenced in the XML signature
	// has the same filename as the file in the local filesystem.
	// 
	void get_ExternalRefDirs(CkString &str);
	// May contain a set of directory paths specifying where referenced external files
	// are located. Directory paths should be separated using a semicolon character.
	// The default value of this property is the empty string which means no
	// directories are automatically searched.
	// 
	// This property can be used if the external file referenced in the XML signature
	// has the same filename as the file in the local filesystem.
	// 
	const wchar_t *externalRefDirs(void);
	// May contain a set of directory paths specifying where referenced external files
	// are located. Directory paths should be separated using a semicolon character.
	// The default value of this property is the empty string which means no
	// directories are automatically searched.
	// 
	// This property can be used if the external file referenced in the XML signature
	// has the same filename as the file in the local filesystem.
	// 
	void put_ExternalRefDirs(const wchar_t *newVal);

	// If true, then ignore failures caused by external references not being
	// available. This allows for the XML signature to be at least partially validated
	// if the external referenced files are not available. The default value of this
	// property is false.
	bool get_IgnoreExternalRefs(void);
	// If true, then ignore failures caused by external references not being
	// available. This allows for the XML signature to be at least partially validated
	// if the external referenced files are not available. The default value of this
	// property is false.
	void put_IgnoreExternalRefs(bool newVal);

	// The number of data objects referenced in the XML digital signature. A data
	// object may be self-contained within the loaded XML signature, or it may be an
	// external URI reference. An application can check each reference to see if it's
	// external by calling IsReferenceExternal, and can get each reference URI by
	// calling ReferenceUri.
	int get_NumReferences(void);

	// The number of digital signatures found within the loaded XML. Each digital
	// signature is composed of XML having the following structure:
	//   _LT_Signature ID?_GT_ 
	//      _LT_SignedInfo_GT_
	//        _LT_CanonicalizationMethod/_GT_
	//        _LT_SignatureMethod/_GT_
	//        (_LT_Reference URI? _GT_
	//          (_LT_Transforms_GT_)?
	//          _LT_DigestMethod_GT_
	//          _LT_DigestValue_GT_
	//        _LT_/Reference_GT_)+
	//      _LT_/SignedInfo_GT_
	//      _LT_SignatureValue_GT_ 
	//     (_LT_KeyInfo_GT_)?
	//     (_LT_Object ID?_GT_)*
	//  _LT_/Signature_GT_
	// Note: The "Signature" and other XML tags may be namespace prefixed.
	// 
	// The Selector property is used to select which XML signature is in effect when
	// validating or calling other methods or properties.
	// 
	int get_NumSignatures(void);

	// Indicates the failure reason for the last call to VerifyReferenceDigest.
	// Possible values are:
	//     0: No failure, the reference digest was valid.
	//     1: The computed digest differs from the digest stored in the XML.
	//     2: An external file is referenced, but it is unavailable for computing the
	//     digest.
	//     3: The index argument passed to VerifyReferenceDigest was out of range.
	//     4: Unable to find the Signature.
	//     5: A transformation specified some sort of XML canonicalization that is not
	//     supported.
	//     99: Unknown. (Should never get this value.)
	int get_RefFailReason(void);

	// If the loaded XML contains multiple signatures, this property can be set to
	// specify which signature is in effect when calling other methods or properties.
	// In most cases, the loaded XML contains a single signature and this property will
	// remain at the default value of 0.
	int get_Selector(void);
	// If the loaded XML contains multiple signatures, this property can be set to
	// specify which signature is in effect when calling other methods or properties.
	// In most cases, the loaded XML contains a single signature and this property will
	// remain at the default value of 0.
	void put_Selector(int newVal);

	// Note: This property is not actually used because the "with/without comments"
	// behavior is passed as an argument to the CanonicalizeXml and
	// CanonicalizeFragment methods.
	// 
	// Determines whether XML is canonicalized with or without comments. The default
	// value is false. (Set to true to canonicalize with XML comments.)
	// 
	bool get_WithComments(void);
	// Note: This property is not actually used because the "with/without comments"
	// behavior is passed as an argument to the CanonicalizeXml and
	// CanonicalizeFragment methods.
	// 
	// Determines whether XML is canonicalized with or without comments. The default
	// value is false. (Set to true to canonicalize with XML comments.)
	// 
	void put_WithComments(bool newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Applies XML canonicalization to a fragment of the passed-in XML, and returns the
	// canonicalized XML string. The fragmentId identifies the XML element where output
	// begins. It is the XML element having an "id" attribute equal to fragmentId. The version
	// may be one of the following:
	//     "C14N" -- for Inclusive Canonical XML
	//     "EXCL_C14N" -- for Exclusive Canonical XML
	// 
	// The prefixList only applies when the version is set to "EXCL_C14N". The prefixList may be an
	// empty string, or a SPACE separated list of namespace prefixes. It is the
	// InclusiveNamespaces PrefixList, which is the list of namespaces that are
	// propagated from ancestor elements for canonicalization purposes.
	// 
	// If withComments is true, then XML comments are included in the output. If withComments is
	// false, then XML comments are excluded from the output.
	// 
	bool CanonicalizeFragment(const wchar_t *xml, const wchar_t *fragmentId, const wchar_t *version, const wchar_t *prefixList, bool withComments, CkString &outStr);
	// Applies XML canonicalization to a fragment of the passed-in XML, and returns the
	// canonicalized XML string. The fragmentId identifies the XML element where output
	// begins. It is the XML element having an "id" attribute equal to fragmentId. The version
	// may be one of the following:
	//     "C14N" -- for Inclusive Canonical XML
	//     "EXCL_C14N" -- for Exclusive Canonical XML
	// 
	// The prefixList only applies when the version is set to "EXCL_C14N". The prefixList may be an
	// empty string, or a SPACE separated list of namespace prefixes. It is the
	// InclusiveNamespaces PrefixList, which is the list of namespaces that are
	// propagated from ancestor elements for canonicalization purposes.
	// 
	// If withComments is true, then XML comments are included in the output. If withComments is
	// false, then XML comments are excluded from the output.
	// 
	const wchar_t *canonicalizeFragment(const wchar_t *xml, const wchar_t *fragmentId, const wchar_t *version, const wchar_t *prefixList, bool withComments);

	// Applies XML canonicalization to the passed-in XML, and returns the canonicalized
	// XML string. The version may be one of the following:
	//     "C14N" -- for Inclusive Canonical XML
	//     "EXCL_C14N" -- for Exclusive Canonical XML
	// 
	// If withComments is true, then XML comments are included in the output. If withComments is
	// false, then XML comments are excluded from the output.
	// 
	bool CanonicalizeXml(const wchar_t *xml, const wchar_t *version, bool withComments, CkString &outStr);
	// Applies XML canonicalization to the passed-in XML, and returns the canonicalized
	// XML string. The version may be one of the following:
	//     "C14N" -- for Inclusive Canonical XML
	//     "EXCL_C14N" -- for Exclusive Canonical XML
	// 
	// If withComments is true, then XML comments are included in the output. If withComments is
	// false, then XML comments are excluded from the output.
	// 
	const wchar_t *canonicalizeXml(const wchar_t *xml, const wchar_t *version, bool withComments);

	// Returns the certificates found in the signature indicated by the Selector
	// property. The base64 representation of each certificate is returned.
	bool GetCerts(CkStringArrayW &sa);

	// Returns the KeyInfo XML for the signature indicated by the Selector property.
	// Returns _NULL_ if no KeyInfo exists.
	// The caller is responsible for deleting the object returned by this method.
	CkXmlW *GetKeyInfo(void);

	// Returns the public key from the KeyInfo XML for the signature indicated by the
	// Selector property. Returns _NULL_ if no KeyInfo exists, or if no public key is
	// actually contained in the KeyInfo.
	// The caller is responsible for deleting the object returned by this method.
	CkPublicKeyW *GetPublicKey(void);

	// Returns true if the reference at index is external. Each external reference
	// would first need to be provided by the application prior to validating the
	// signature.
	bool IsReferenceExternal(int index);

	// Loads an XML document containing 1 or more XML digital signatures. An
	// application would first load XML containing digital signature(s), then validate.
	// After loading, it is also possible to use various methods and properties to get
	// information about the signature, such as the key info, references, etc. If
	// external data is referenced by the signature, it may be necessary to provide the
	// referenced data prior to validating.
	// 
	// Note: When loading an XML document, the Selector property is automatically reset
	// to 0, and the NumSignatures property is set to the number of XML digital
	// signatures found.
	// 
	bool LoadSignature(const wchar_t *xmlSig);

	// Loads an XML document containing one or more XML digital signatures from the
	// contents of a BinData object. An application would first load the XML, then
	// validate. After loading, it is also possible to use various methods and
	// properties to get information about the signature, such as the key info,
	// references, etc. If external data is referenced by the signature, it may be
	// necessary to provide the referenced data prior to validating.
	// 
	// Note: When loading an XML document, the Selector property is automatically reset
	// to 0, and the NumSignatures property is set to the number of XML digital
	// signatures found.
	// 
	bool LoadSignatureBd(CkBinDataW &binData);

	// Loads an XML document containing one or more XML digital signatures from the
	// contents of a StringBuilder object. An application would first load the XML,
	// then validate. After loading, it is also possible to use various methods and
	// properties to get information about the signature, such as the key info,
	// references, etc. If external data is referenced by the signature, it may be
	// necessary to provide the referenced data prior to validating.
	// 
	// Note: When loading an XML document, the Selector property is automatically reset
	// to 0, and the NumSignatures property is set to the number of XML digital
	// signatures found.
	// 
	bool LoadSignatureSb(CkStringBuilderW &sbXmlSig);

	// Returns the URI of the Nth reference specified by index. (The 1st reference is at
	// index 0.) URI's beginning with a pound sign ('#') are internal references.
	bool ReferenceUri(int index, CkString &outStr);
	// Returns the URI of the Nth reference specified by index. (The 1st reference is at
	// index 0.) URI's beginning with a pound sign ('#') are internal references.
	const wchar_t *referenceUri(int index);

	// Sets the HMAC key to be used if the Signature used an HMAC signing algorithm.
	// The encoding specifies the encoding of key, and can be "hex", "base64", "ascii", or
	// any of the binary encodings supported by Chilkat in the link below.
	bool SetHmacKey(const wchar_t *key, const wchar_t *encoding);

	// Sets the public key to be used for verifying the signature indicated by the
	// Selector property. A public key only needs to be explicitly provided by the
	// application for those XML signatures where the public key is not already
	// available within the KeyInfo of the Signature. In some cases, the KeyInfo within
	// the Signature contains information about what public key was used for signing.
	// The application can use this information to retrieve the matching public key and
	// provide it via this method.
	bool SetPublicKey(CkPublicKeyW &pubKey);

	// Provides the binary data for the external reference at index.
	bool SetRefDataBd(int index, CkBinDataW &binData);

	// Provides the data for the external reference at index. When validating the
	// signature, the data contained in path will be streamed to compute the digest for
	// validation.
	bool SetRefDataFile(int index, const wchar_t *path);

	// Provides the text data for the external reference at index. The charset specifies
	// the byte representation of the text, such as "utf-8", "utf-16", "windows-1252",
	// etc. (If in doubt, try utf-8 first.)
	bool SetRefDataSb(int index, CkStringBuilderW &sb, const wchar_t *charset);

	// Adds an XML certificate vault to the object's internal list of sources to be
	// searched for certificates having public keys when verifying an XML signature. A
	// single XML certificate vault may be used. If UseCertVault is called multiple
	// times, only the last certificate vault will be used, as each call to
	// UseCertVault will replace the certificate vault provided in previous calls.
	bool UseCertVault(CkXmlCertVaultW &certVault);

	// This method allows for an application to verify the digest for each reference
	// separately. This can be helpful if the full XMLDSIG validation fails, then one
	// can test each referenced data's digest to see which, if any, fail to match.
	bool VerifyReferenceDigest(int index);

	// Verifies the signature indicated by the Selector property. If verifyReferenceDigests is true,
	// then the digest of each Reference within the signature's SignedInfo is also
	// validated.
	bool VerifySignature(bool verifyReferenceDigests);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
