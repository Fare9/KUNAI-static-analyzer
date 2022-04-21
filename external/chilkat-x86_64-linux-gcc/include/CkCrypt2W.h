// CkCrypt2W.h: interface for the CkCrypt2W class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkCrypt2W_H
#define _CkCrypt2W_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacksW.h"

class CkByteData;
class CkCertW;
class CkTaskW;
class CkBinDataW;
class CkStringBuilderW;
class CkSecureStringW;
class CkStreamW;
class CkCertChainW;
class CkJsonObjectW;
class CkPrivateKeyW;
class CkHttpW;
class CkXmlCertVaultW;
class CkBaseProgressW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkCrypt2W
class CK_VISIBLE_PUBLIC CkCrypt2W  : public CkClassWithCallbacksW
{
	private:
	bool m_cbOwned;

	private:
	
	// Don't allow assignment or copying these objects.
	CkCrypt2W(const CkCrypt2W &);
	CkCrypt2W &operator=(const CkCrypt2W &);

    public:
	CkCrypt2W(void);
	virtual ~CkCrypt2W(void);

	

	static CkCrypt2W *createNew(void);
	

	CkCrypt2W(bool bCallbackOwned);
	static CkCrypt2W *createNew(bool bCallbackOwned);

	
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

	// The BCrypt work factor to be used for the BCryptHash and BCryptVerify. This is
	// the log2 of the number of rounds of hashing to apply. For example, if the work
	// (cost) factor is 12, then 2^12 rounds of hashing are applied. The purpose of
	// this cost factor is to make the BCrypt computation expensive enought to prevent
	// brute-force attacks. (Any complaints about BCrypt "not being fast enough" will
	// be ignored.)
	// 
	// This property must have a value ranging from 4 to 31 inclusive.
	// 
	// The default value is 10.
	// 
	int get_BCryptWorkFactor(void);
	// The BCrypt work factor to be used for the BCryptHash and BCryptVerify. This is
	// the log2 of the number of rounds of hashing to apply. For example, if the work
	// (cost) factor is 12, then 2^12 rounds of hashing are applied. The purpose of
	// this cost factor is to make the BCrypt computation expensive enought to prevent
	// brute-force attacks. (Any complaints about BCrypt "not being fast enough" will
	// be ignored.)
	// 
	// This property must have a value ranging from 4 to 31 inclusive.
	// 
	// The default value is 10.
	// 
	void put_BCryptWorkFactor(int newVal);

	// The block-size (in bytes) of the selected encryption algorithm. For example, if
	// the CryptAlgorithm property is set to "aes", the BlockSize property is
	// automatically set to 16. The block-size for the ARC4 streaming encryption
	// algorithm is 1.
	int get_BlockSize(void);

	// Applies to all methods that create PKCS7 signatures. To create a CAdES-BES
	// signature, set this property equal to true. The default value of this property
	// is false.
	bool get_CadesEnabled(void);
	// Applies to all methods that create PKCS7 signatures. To create a CAdES-BES
	// signature, set this property equal to true. The default value of this property
	// is false.
	void put_CadesEnabled(bool newVal);

	// This is the base64 hash of the policy document located at the CadesSigPolicyUri.
	// You can use either the SHA256 or SHA1 hash. You may use this online tool to
	// compute the base64 hash:Compute Base64 Hash for CaDES Signature Policy URL
	// <http://tools.chilkat.io/hashFileAtUrl.cshtml>
	// 
	// Note: This property applies to all methods that create PKCS7 signatures. To
	// create a CAdES-EPES signature, set the CadesEnabled property = true, and also
	// provide values for each of the following properties: CadesSigPolicyHash,
	// CadesSigPolicyId, and CadesSigPolicyUri. For example (in pseudo-code):
	// cryptObj.CadesSigPolicyId = "2.16.76.1.7.1.1.1"
	// cryptObj.CadesSigPolicyUri = "http://politicas.icpbrasil.gov.br/PA_AD_RB.der"
	// cryptObj.CadesSigPolicyHash = "rySugyKaMhiMR8Y/o5yuU2A2bF0="
	// Note: Do NOT use the values above. They are only provided as an example to show
	// valid values. For example, the Policy ID is an OID. The Policy URI is a
	// typically a URL to a DER encoded policy file, and the Policy Hash is a base64
	// encoded hash.
	// 
	void get_CadesSigPolicyHash(CkString &str);
	// This is the base64 hash of the policy document located at the CadesSigPolicyUri.
	// You can use either the SHA256 or SHA1 hash. You may use this online tool to
	// compute the base64 hash:Compute Base64 Hash for CaDES Signature Policy URL
	// <http://tools.chilkat.io/hashFileAtUrl.cshtml>
	// 
	// Note: This property applies to all methods that create PKCS7 signatures. To
	// create a CAdES-EPES signature, set the CadesEnabled property = true, and also
	// provide values for each of the following properties: CadesSigPolicyHash,
	// CadesSigPolicyId, and CadesSigPolicyUri. For example (in pseudo-code):
	// cryptObj.CadesSigPolicyId = "2.16.76.1.7.1.1.1"
	// cryptObj.CadesSigPolicyUri = "http://politicas.icpbrasil.gov.br/PA_AD_RB.der"
	// cryptObj.CadesSigPolicyHash = "rySugyKaMhiMR8Y/o5yuU2A2bF0="
	// Note: Do NOT use the values above. They are only provided as an example to show
	// valid values. For example, the Policy ID is an OID. The Policy URI is a
	// typically a URL to a DER encoded policy file, and the Policy Hash is a base64
	// encoded hash.
	// 
	const wchar_t *cadesSigPolicyHash(void);
	// This is the base64 hash of the policy document located at the CadesSigPolicyUri.
	// You can use either the SHA256 or SHA1 hash. You may use this online tool to
	// compute the base64 hash:Compute Base64 Hash for CaDES Signature Policy URL
	// <http://tools.chilkat.io/hashFileAtUrl.cshtml>
	// 
	// Note: This property applies to all methods that create PKCS7 signatures. To
	// create a CAdES-EPES signature, set the CadesEnabled property = true, and also
	// provide values for each of the following properties: CadesSigPolicyHash,
	// CadesSigPolicyId, and CadesSigPolicyUri. For example (in pseudo-code):
	// cryptObj.CadesSigPolicyId = "2.16.76.1.7.1.1.1"
	// cryptObj.CadesSigPolicyUri = "http://politicas.icpbrasil.gov.br/PA_AD_RB.der"
	// cryptObj.CadesSigPolicyHash = "rySugyKaMhiMR8Y/o5yuU2A2bF0="
	// Note: Do NOT use the values above. They are only provided as an example to show
	// valid values. For example, the Policy ID is an OID. The Policy URI is a
	// typically a URL to a DER encoded policy file, and the Policy Hash is a base64
	// encoded hash.
	// 
	void put_CadesSigPolicyHash(const wchar_t *newVal);

	// See the description for the CadesSigPolicyHash property above.
	void get_CadesSigPolicyId(CkString &str);
	// See the description for the CadesSigPolicyHash property above.
	const wchar_t *cadesSigPolicyId(void);
	// See the description for the CadesSigPolicyHash property above.
	void put_CadesSigPolicyId(const wchar_t *newVal);

	// See the description for the CadesSigPolicyHash property above.
	void get_CadesSigPolicyUri(CkString &str);
	// See the description for the CadesSigPolicyHash property above.
	const wchar_t *cadesSigPolicyUri(void);
	// See the description for the CadesSigPolicyHash property above.
	void put_CadesSigPolicyUri(const wchar_t *newVal);

	// Controls the character encoding of the text encrypted, signed, hashed or
	// compressed. This property is relevant wherever strings are used as inputs or
	// outputs.
	// 
	// This property defaults to the ANSI charset of the computer. For example, the
	// default ANSI code page on Windows computers in the USA and Western Europe would
	// be "windows-1252".
	// 
	// When working with strings, it is important to know the exact bytes that are
	// being encrypted/hashed/signed/compressed. This is critical when interoperating
	// with other systems. If your application is sending an encrypted string to
	// another system that will decrypt it, you will need to know the encoding of the
	// string that is expected on the receiving end (after decryption). If you pass
	// Unicode data (2 byte per character) to the encryptor, subsequent decryption will
	// reproduce the original Unicode. However, it may be that your program works with
	// Unicode strings, but the recipient of the encrypted data works with iso-8859-1
	// strings. In such a case, setting the Charset property to "iso-8859-1" causes the
	// character data to be automatically converted to the Charset before being
	// encrypted (or compressed, or hashed, or signed). The set of valid charsets is
	// listed below:
	// hex
	// base64
	//     * "hex" and "base64" are special values that allow for binary (non-text) encoded data to be passed to any method where the input data is a string.
	//        Rather than converting to an actual charset (such as utf-8, iso-8859-1), the binary data is decoded, and the decoded bytes are passed
	//         to the underlying encryptor, hashing, signing, etc.
	// ANSI
	// us-ascii
	// unicode
	// unicodefffe
	// iso-8859-1
	// iso-8859-2
	// iso-8859-3
	// iso-8859-4
	// iso-8859-5
	// iso-8859-6
	// iso-8859-7
	// iso-8859-8
	// iso-8859-9
	// iso-8859-13
	// iso-8859-15
	// windows-874
	// windows-1250
	// windows-1251
	// windows-1252
	// windows-1253
	// windows-1254
	// windows-1255
	// windows-1256
	// windows-1257
	// windows-1258
	// utf-7
	// utf-8
	// utf-32
	// utf-32be
	// shift_jis
	// gb2312
	// ks_c_5601-1987
	// big5
	// iso-2022-jp
	// iso-2022-kr
	// euc-jp
	// euc-kr
	// macintosh
	// x-mac-japanese
	// x-mac-chinesetrad
	// x-mac-korean
	// x-mac-arabic
	// x-mac-hebrew
	// x-mac-greek
	// x-mac-cyrillic
	// x-mac-chinesesimp
	// x-mac-romanian
	// x-mac-ukrainian
	// x-mac-thai
	// x-mac-ce
	// x-mac-icelandic
	// x-mac-turkish
	// x-mac-croatian
	// asmo-708
	// dos-720
	// dos-862
	// ibm037
	// ibm437
	// ibm500
	// ibm737
	// ibm775
	// ibm850
	// ibm852
	// ibm855
	// ibm857
	// ibm00858
	// ibm860
	// ibm861
	// ibm863
	// ibm864
	// ibm865
	// cp866
	// ibm869
	// ibm870
	// cp875
	// koi8-r
	// koi8-u
	// 
	void get_Charset(CkString &str);
	// Controls the character encoding of the text encrypted, signed, hashed or
	// compressed. This property is relevant wherever strings are used as inputs or
	// outputs.
	// 
	// This property defaults to the ANSI charset of the computer. For example, the
	// default ANSI code page on Windows computers in the USA and Western Europe would
	// be "windows-1252".
	// 
	// When working with strings, it is important to know the exact bytes that are
	// being encrypted/hashed/signed/compressed. This is critical when interoperating
	// with other systems. If your application is sending an encrypted string to
	// another system that will decrypt it, you will need to know the encoding of the
	// string that is expected on the receiving end (after decryption). If you pass
	// Unicode data (2 byte per character) to the encryptor, subsequent decryption will
	// reproduce the original Unicode. However, it may be that your program works with
	// Unicode strings, but the recipient of the encrypted data works with iso-8859-1
	// strings. In such a case, setting the Charset property to "iso-8859-1" causes the
	// character data to be automatically converted to the Charset before being
	// encrypted (or compressed, or hashed, or signed). The set of valid charsets is
	// listed below:
	// hex
	// base64
	//     * "hex" and "base64" are special values that allow for binary (non-text) encoded data to be passed to any method where the input data is a string.
	//        Rather than converting to an actual charset (such as utf-8, iso-8859-1), the binary data is decoded, and the decoded bytes are passed
	//         to the underlying encryptor, hashing, signing, etc.
	// ANSI
	// us-ascii
	// unicode
	// unicodefffe
	// iso-8859-1
	// iso-8859-2
	// iso-8859-3
	// iso-8859-4
	// iso-8859-5
	// iso-8859-6
	// iso-8859-7
	// iso-8859-8
	// iso-8859-9
	// iso-8859-13
	// iso-8859-15
	// windows-874
	// windows-1250
	// windows-1251
	// windows-1252
	// windows-1253
	// windows-1254
	// windows-1255
	// windows-1256
	// windows-1257
	// windows-1258
	// utf-7
	// utf-8
	// utf-32
	// utf-32be
	// shift_jis
	// gb2312
	// ks_c_5601-1987
	// big5
	// iso-2022-jp
	// iso-2022-kr
	// euc-jp
	// euc-kr
	// macintosh
	// x-mac-japanese
	// x-mac-chinesetrad
	// x-mac-korean
	// x-mac-arabic
	// x-mac-hebrew
	// x-mac-greek
	// x-mac-cyrillic
	// x-mac-chinesesimp
	// x-mac-romanian
	// x-mac-ukrainian
	// x-mac-thai
	// x-mac-ce
	// x-mac-icelandic
	// x-mac-turkish
	// x-mac-croatian
	// asmo-708
	// dos-720
	// dos-862
	// ibm037
	// ibm437
	// ibm500
	// ibm737
	// ibm775
	// ibm850
	// ibm852
	// ibm855
	// ibm857
	// ibm00858
	// ibm860
	// ibm861
	// ibm863
	// ibm864
	// ibm865
	// cp866
	// ibm869
	// ibm870
	// cp875
	// koi8-r
	// koi8-u
	// 
	const wchar_t *charset(void);
	// Controls the character encoding of the text encrypted, signed, hashed or
	// compressed. This property is relevant wherever strings are used as inputs or
	// outputs.
	// 
	// This property defaults to the ANSI charset of the computer. For example, the
	// default ANSI code page on Windows computers in the USA and Western Europe would
	// be "windows-1252".
	// 
	// When working with strings, it is important to know the exact bytes that are
	// being encrypted/hashed/signed/compressed. This is critical when interoperating
	// with other systems. If your application is sending an encrypted string to
	// another system that will decrypt it, you will need to know the encoding of the
	// string that is expected on the receiving end (after decryption). If you pass
	// Unicode data (2 byte per character) to the encryptor, subsequent decryption will
	// reproduce the original Unicode. However, it may be that your program works with
	// Unicode strings, but the recipient of the encrypted data works with iso-8859-1
	// strings. In such a case, setting the Charset property to "iso-8859-1" causes the
	// character data to be automatically converted to the Charset before being
	// encrypted (or compressed, or hashed, or signed). The set of valid charsets is
	// listed below:
	// hex
	// base64
	//     * "hex" and "base64" are special values that allow for binary (non-text) encoded data to be passed to any method where the input data is a string.
	//        Rather than converting to an actual charset (such as utf-8, iso-8859-1), the binary data is decoded, and the decoded bytes are passed
	//         to the underlying encryptor, hashing, signing, etc.
	// ANSI
	// us-ascii
	// unicode
	// unicodefffe
	// iso-8859-1
	// iso-8859-2
	// iso-8859-3
	// iso-8859-4
	// iso-8859-5
	// iso-8859-6
	// iso-8859-7
	// iso-8859-8
	// iso-8859-9
	// iso-8859-13
	// iso-8859-15
	// windows-874
	// windows-1250
	// windows-1251
	// windows-1252
	// windows-1253
	// windows-1254
	// windows-1255
	// windows-1256
	// windows-1257
	// windows-1258
	// utf-7
	// utf-8
	// utf-32
	// utf-32be
	// shift_jis
	// gb2312
	// ks_c_5601-1987
	// big5
	// iso-2022-jp
	// iso-2022-kr
	// euc-jp
	// euc-kr
	// macintosh
	// x-mac-japanese
	// x-mac-chinesetrad
	// x-mac-korean
	// x-mac-arabic
	// x-mac-hebrew
	// x-mac-greek
	// x-mac-cyrillic
	// x-mac-chinesesimp
	// x-mac-romanian
	// x-mac-ukrainian
	// x-mac-thai
	// x-mac-ce
	// x-mac-icelandic
	// x-mac-turkish
	// x-mac-croatian
	// asmo-708
	// dos-720
	// dos-862
	// ibm037
	// ibm437
	// ibm500
	// ibm737
	// ibm775
	// ibm850
	// ibm852
	// ibm855
	// ibm857
	// ibm00858
	// ibm860
	// ibm861
	// ibm863
	// ibm864
	// ibm865
	// cp866
	// ibm869
	// ibm870
	// cp875
	// koi8-r
	// koi8-u
	// 
	void put_Charset(const wchar_t *newVal);

	// Controls the cipher mode for block encryption algorithms (AES, Blowfish,TwoFish,
	// DES, 3DES, RC2). Possible values are "CBC" (the default) , "ECB", "CTR", "OFB",
	// "GCM", and "CFB". These acronyms have the following meanings:
	// 
	//     CBC: Cipher Block Chaining,
	//     ECB: Electronic CookBook
	//     CTR: Counter Mode
	//     CFB: Cipher Feedback
	//     OFB: Output Feedback
	//     GCM: Galois/Counter Mode
	// 
	// (see http://en.wikipedia.org/wiki/Block_cipher_modes_of_operation )
	// 
	// Note: Prior to Chilkat v9.5.0.55, the CFB mode is only implemented for AES,
	// Blowfish, and DES/3DES, and the CTR mode is only implemented for AES.
	// 
	// Starting in v9.5.0.55 CFB and OFB modes are useable with all encryption
	// algorithms, and GCM (Galois/Counter Mode) is available with any cipher having a
	// 16-byte block size, such as AES and Twofish. CFB, OFB, CTR, and GCM modes
	// convert block ciphers into stream ciphers. In these modes of operation, the
	// PaddingScheme property is unused because no padding occurs.
	// 
	void get_CipherMode(CkString &str);
	// Controls the cipher mode for block encryption algorithms (AES, Blowfish,TwoFish,
	// DES, 3DES, RC2). Possible values are "CBC" (the default) , "ECB", "CTR", "OFB",
	// "GCM", and "CFB". These acronyms have the following meanings:
	// 
	//     CBC: Cipher Block Chaining,
	//     ECB: Electronic CookBook
	//     CTR: Counter Mode
	//     CFB: Cipher Feedback
	//     OFB: Output Feedback
	//     GCM: Galois/Counter Mode
	// 
	// (see http://en.wikipedia.org/wiki/Block_cipher_modes_of_operation )
	// 
	// Note: Prior to Chilkat v9.5.0.55, the CFB mode is only implemented for AES,
	// Blowfish, and DES/3DES, and the CTR mode is only implemented for AES.
	// 
	// Starting in v9.5.0.55 CFB and OFB modes are useable with all encryption
	// algorithms, and GCM (Galois/Counter Mode) is available with any cipher having a
	// 16-byte block size, such as AES and Twofish. CFB, OFB, CTR, and GCM modes
	// convert block ciphers into stream ciphers. In these modes of operation, the
	// PaddingScheme property is unused because no padding occurs.
	// 
	const wchar_t *cipherMode(void);
	// Controls the cipher mode for block encryption algorithms (AES, Blowfish,TwoFish,
	// DES, 3DES, RC2). Possible values are "CBC" (the default) , "ECB", "CTR", "OFB",
	// "GCM", and "CFB". These acronyms have the following meanings:
	// 
	//     CBC: Cipher Block Chaining,
	//     ECB: Electronic CookBook
	//     CTR: Counter Mode
	//     CFB: Cipher Feedback
	//     OFB: Output Feedback
	//     GCM: Galois/Counter Mode
	// 
	// (see http://en.wikipedia.org/wiki/Block_cipher_modes_of_operation )
	// 
	// Note: Prior to Chilkat v9.5.0.55, the CFB mode is only implemented for AES,
	// Blowfish, and DES/3DES, and the CTR mode is only implemented for AES.
	// 
	// Starting in v9.5.0.55 CFB and OFB modes are useable with all encryption
	// algorithms, and GCM (Galois/Counter Mode) is available with any cipher having a
	// 16-byte block size, such as AES and Twofish. CFB, OFB, CTR, and GCM modes
	// convert block ciphers into stream ciphers. In these modes of operation, the
	// PaddingScheme property is unused because no padding occurs.
	// 
	void put_CipherMode(const wchar_t *newVal);

	// A JSON string for controlling extra CMS (PKCS7) signature and validation
	// options.
	void get_CmsOptions(CkString &str);
	// A JSON string for controlling extra CMS (PKCS7) signature and validation
	// options.
	const wchar_t *cmsOptions(void);
	// A JSON string for controlling extra CMS (PKCS7) signature and validation
	// options.
	void put_CmsOptions(const wchar_t *newVal);

	// This property is deprecated. The only possible value is "BZIP2". The compression
	// functionality in Crypt2 is legacy and existed long before the general
	// compression functionality that is currently offered in Chilkat.Compression. The
	// Chilkat.Compression API should be used instead.
	void get_CompressionAlgorithm(CkString &str);
	// This property is deprecated. The only possible value is "BZIP2". The compression
	// functionality in Crypt2 is legacy and existed long before the general
	// compression functionality that is currently offered in Chilkat.Compression. The
	// Chilkat.Compression API should be used instead.
	const wchar_t *compressionAlgorithm(void);
	// This property is deprecated. The only possible value is "BZIP2". The compression
	// functionality in Crypt2 is legacy and existed long before the general
	// compression functionality that is currently offered in Chilkat.Compression. The
	// Chilkat.Compression API should be used instead.
	void put_CompressionAlgorithm(const wchar_t *newVal);

	// Selects the encryption algorithm for encrypting and decrypting. Possible values
	// are: "chacha20", "pki", "aes", "blowfish2", "des", "3des", "rc2", "arc4",
	// "twofish", "pbes1" and "pbes2". The "pki" encryption algorithm isn't a specific
	// algorithm, but instead tells the component to encrypt/decrypt using public-key
	// encryption with digital certificates. The other choices are symmetric encryption
	// algorithms that do not involve digital certificates and public/private keys.
	// 
	// The default value is "aes"
	// 
	// The original Chilkat implementation of Blowfish (in 2004) has a 4321
	// byte-swapping issue (the results are 4321 byte-swapped). The newer
	// implementation (in 2006 and named "blowfish2") does not byte swap. This should
	// be used for compatibility with other Blowfish software. If an application needs
	// to decrypt something encrypted with the old 4321 byte-swapped blowfish, set the
	// property to "blowfish_old".
	// 
	// Password-based encryption (PBE) is selected by setting this property to "pbes1"
	// or "pbes2". Password-based encryption is defined in the PKCS5 Password-Based
	// Cryptography Standard at https://tools.ietf.org/html/rfc2898. If PBE is used,
	// the underlying encryption algorithm is specified by the PbesAlgorithm property.
	// The underlying encryption (PbesAlgorithm) for PBES1 is limited to 56-bit DES or
	// 64-bit RC2.
	// 
	// Note:The chacha20 algorithm is introduced in Chilkat v9.5.0.55.
	// 
	void get_CryptAlgorithm(CkString &str);
	// Selects the encryption algorithm for encrypting and decrypting. Possible values
	// are: "chacha20", "pki", "aes", "blowfish2", "des", "3des", "rc2", "arc4",
	// "twofish", "pbes1" and "pbes2". The "pki" encryption algorithm isn't a specific
	// algorithm, but instead tells the component to encrypt/decrypt using public-key
	// encryption with digital certificates. The other choices are symmetric encryption
	// algorithms that do not involve digital certificates and public/private keys.
	// 
	// The default value is "aes"
	// 
	// The original Chilkat implementation of Blowfish (in 2004) has a 4321
	// byte-swapping issue (the results are 4321 byte-swapped). The newer
	// implementation (in 2006 and named "blowfish2") does not byte swap. This should
	// be used for compatibility with other Blowfish software. If an application needs
	// to decrypt something encrypted with the old 4321 byte-swapped blowfish, set the
	// property to "blowfish_old".
	// 
	// Password-based encryption (PBE) is selected by setting this property to "pbes1"
	// or "pbes2". Password-based encryption is defined in the PKCS5 Password-Based
	// Cryptography Standard at https://tools.ietf.org/html/rfc2898. If PBE is used,
	// the underlying encryption algorithm is specified by the PbesAlgorithm property.
	// The underlying encryption (PbesAlgorithm) for PBES1 is limited to 56-bit DES or
	// 64-bit RC2.
	// 
	// Note:The chacha20 algorithm is introduced in Chilkat v9.5.0.55.
	// 
	const wchar_t *cryptAlgorithm(void);
	// Selects the encryption algorithm for encrypting and decrypting. Possible values
	// are: "chacha20", "pki", "aes", "blowfish2", "des", "3des", "rc2", "arc4",
	// "twofish", "pbes1" and "pbes2". The "pki" encryption algorithm isn't a specific
	// algorithm, but instead tells the component to encrypt/decrypt using public-key
	// encryption with digital certificates. The other choices are symmetric encryption
	// algorithms that do not involve digital certificates and public/private keys.
	// 
	// The default value is "aes"
	// 
	// The original Chilkat implementation of Blowfish (in 2004) has a 4321
	// byte-swapping issue (the results are 4321 byte-swapped). The newer
	// implementation (in 2006 and named "blowfish2") does not byte swap. This should
	// be used for compatibility with other Blowfish software. If an application needs
	// to decrypt something encrypted with the old 4321 byte-swapped blowfish, set the
	// property to "blowfish_old".
	// 
	// Password-based encryption (PBE) is selected by setting this property to "pbes1"
	// or "pbes2". Password-based encryption is defined in the PKCS5 Password-Based
	// Cryptography Standard at https://tools.ietf.org/html/rfc2898. If PBE is used,
	// the underlying encryption algorithm is specified by the PbesAlgorithm property.
	// The underlying encryption (PbesAlgorithm) for PBES1 is limited to 56-bit DES or
	// 64-bit RC2.
	// 
	// Note:The chacha20 algorithm is introduced in Chilkat v9.5.0.55.
	// 
	void put_CryptAlgorithm(const wchar_t *newVal);

	// Controls the encoding of binary data to a printable string for many methods. The
	// valid modes are "Base64", "modBase64", "base64url", "Base32", "Base58", "UU",
	// "QP" (for quoted-printable), "URL" (for url-encoding), "Hex", "Q", "B",
	// "url_oauth", "url_rfc1738", "url_rfc2396", "url_rfc3986", "fingerprint", or
	// "decimal".
	// 
	// The default value is "base64"
	// 
	// The "fingerprint" and"decimal" encodings are introduced in Chilkat v9.5.0.55.
	// 
	// The "fingerprint" encoding is a lowercase hex encoding where each hex digit is
	// separated by a colon character. For example:
	// 6a:de:e0:af:56:f8:0c:04:11:5b:ef:4d:49:ad:09:23
	// 
	// The "decimal" encoding is for converting large decimal integers to/from a
	// big-endian binary representation. For example, the decimal string
	// "72623859790382856" converts to the bytes 0x01 0x02 0x03 0x04 0x05 0x06 0x07
	// 0x08.
	// 
	void get_EncodingMode(CkString &str);
	// Controls the encoding of binary data to a printable string for many methods. The
	// valid modes are "Base64", "modBase64", "base64url", "Base32", "Base58", "UU",
	// "QP" (for quoted-printable), "URL" (for url-encoding), "Hex", "Q", "B",
	// "url_oauth", "url_rfc1738", "url_rfc2396", "url_rfc3986", "fingerprint", or
	// "decimal".
	// 
	// The default value is "base64"
	// 
	// The "fingerprint" and"decimal" encodings are introduced in Chilkat v9.5.0.55.
	// 
	// The "fingerprint" encoding is a lowercase hex encoding where each hex digit is
	// separated by a colon character. For example:
	// 6a:de:e0:af:56:f8:0c:04:11:5b:ef:4d:49:ad:09:23
	// 
	// The "decimal" encoding is for converting large decimal integers to/from a
	// big-endian binary representation. For example, the decimal string
	// "72623859790382856" converts to the bytes 0x01 0x02 0x03 0x04 0x05 0x06 0x07
	// 0x08.
	// 
	const wchar_t *encodingMode(void);
	// Controls the encoding of binary data to a printable string for many methods. The
	// valid modes are "Base64", "modBase64", "base64url", "Base32", "Base58", "UU",
	// "QP" (for quoted-printable), "URL" (for url-encoding), "Hex", "Q", "B",
	// "url_oauth", "url_rfc1738", "url_rfc2396", "url_rfc3986", "fingerprint", or
	// "decimal".
	// 
	// The default value is "base64"
	// 
	// The "fingerprint" and"decimal" encodings are introduced in Chilkat v9.5.0.55.
	// 
	// The "fingerprint" encoding is a lowercase hex encoding where each hex digit is
	// separated by a colon character. For example:
	// 6a:de:e0:af:56:f8:0c:04:11:5b:ef:4d:49:ad:09:23
	// 
	// The "decimal" encoding is for converting large decimal integers to/from a
	// big-endian binary representation. For example, the decimal string
	// "72623859790382856" converts to the bytes 0x01 0x02 0x03 0x04 0x05 0x06 0x07
	// 0x08.
	// 
	void put_EncodingMode(const wchar_t *newVal);

	// Chilkat Crypt2 provides the ability to feed the encryption/decryption methods
	// with chunks of data. This allows a large amount of data, or a data stream, to be
	// fed piecemeal for encrypting or decrypting. It applies to all symmetric
	// algorithms currently supported (AES, Blowfish, Twofish, 3DES, RC2, DES, ARC4),
	// and all algorithms supported in the future.
	// 
	// The default value for both FirstChunk and LastChunk is true. This means when
	// an Encrypt* or Decrypt* method is called, it is both the first and last chunk
	// (i.e. it's the entire amount of data to be encrypted or decrypted).
	// 
	// If you wish to feed the data piecemeal, do this:
	// 
	//     Set FirstChunk = true, LastChunk = false for the first chunk of data.
	//     For all "middle" chunks (i.e. all chunks except for the final chunk) set
	//     FirstChunk = false and LastChunk = false.
	//     For the final chunk, set FirstChunk = false and LastChunk = true
	// 
	// There is no need to worry about feeding data according to the block size of the
	// encryption algorithm. For example, AES has a block size of 16 bytes. Data may be
	// fed in chunks of any size. The Chilkat Crypt2 component will buffer the data.
	// When the final chunk is passed, the output is padded to the algorithm's block
	// size according to the PaddingScheme.
	// 
	bool get_FirstChunk(void);
	// Chilkat Crypt2 provides the ability to feed the encryption/decryption methods
	// with chunks of data. This allows a large amount of data, or a data stream, to be
	// fed piecemeal for encrypting or decrypting. It applies to all symmetric
	// algorithms currently supported (AES, Blowfish, Twofish, 3DES, RC2, DES, ARC4),
	// and all algorithms supported in the future.
	// 
	// The default value for both FirstChunk and LastChunk is true. This means when
	// an Encrypt* or Decrypt* method is called, it is both the first and last chunk
	// (i.e. it's the entire amount of data to be encrypted or decrypted).
	// 
	// If you wish to feed the data piecemeal, do this:
	// 
	//     Set FirstChunk = true, LastChunk = false for the first chunk of data.
	//     For all "middle" chunks (i.e. all chunks except for the final chunk) set
	//     FirstChunk = false and LastChunk = false.
	//     For the final chunk, set FirstChunk = false and LastChunk = true
	// 
	// There is no need to worry about feeding data according to the block size of the
	// encryption algorithm. For example, AES has a block size of 16 bytes. Data may be
	// fed in chunks of any size. The Chilkat Crypt2 component will buffer the data.
	// When the final chunk is passed, the output is padded to the algorithm's block
	// size according to the PaddingScheme.
	// 
	void put_FirstChunk(bool newVal);

	// Selects the hash algorithm used by methods that create hashes. The valid choices
	// are "sha1", "sha256", "sha384", "sha512", "sha3-224", "sha3-256", "sha3-384",
	// "sha3-512", "md2", "md5", "haval", "ripemd128", "ripemd160","ripemd256", or
	// "ripemd320".
	// 
	// Note: SHA-2 designates a set of cryptographic hash functions that includes
	// SHA-256, SHA-384, and SHA-512. Chilkat by definition supports "SHA-2" because it
	// supports these algorithms.
	// 
	// The default value is "sha1".
	// 
	// Note: The HAVAL hash algorithm is affected by two other properties: HavalRounds
	// and KeyLength.
	// 
	//     The HavalRounds may have values of 3, 4, or 5.
	//     The KeyLength may have values of 128, 160, 192, 224, or 256.
	// 
	// Note: The "sha3-224", "sha3-256", "sha3-384", "sha3-512" algorithms are added in
	// Chilkat v9.5.0.83.
	// 
	void get_HashAlgorithm(CkString &str);
	// Selects the hash algorithm used by methods that create hashes. The valid choices
	// are "sha1", "sha256", "sha384", "sha512", "sha3-224", "sha3-256", "sha3-384",
	// "sha3-512", "md2", "md5", "haval", "ripemd128", "ripemd160","ripemd256", or
	// "ripemd320".
	// 
	// Note: SHA-2 designates a set of cryptographic hash functions that includes
	// SHA-256, SHA-384, and SHA-512. Chilkat by definition supports "SHA-2" because it
	// supports these algorithms.
	// 
	// The default value is "sha1".
	// 
	// Note: The HAVAL hash algorithm is affected by two other properties: HavalRounds
	// and KeyLength.
	// 
	//     The HavalRounds may have values of 3, 4, or 5.
	//     The KeyLength may have values of 128, 160, 192, 224, or 256.
	// 
	// Note: The "sha3-224", "sha3-256", "sha3-384", "sha3-512" algorithms are added in
	// Chilkat v9.5.0.83.
	// 
	const wchar_t *hashAlgorithm(void);
	// Selects the hash algorithm used by methods that create hashes. The valid choices
	// are "sha1", "sha256", "sha384", "sha512", "sha3-224", "sha3-256", "sha3-384",
	// "sha3-512", "md2", "md5", "haval", "ripemd128", "ripemd160","ripemd256", or
	// "ripemd320".
	// 
	// Note: SHA-2 designates a set of cryptographic hash functions that includes
	// SHA-256, SHA-384, and SHA-512. Chilkat by definition supports "SHA-2" because it
	// supports these algorithms.
	// 
	// The default value is "sha1".
	// 
	// Note: The HAVAL hash algorithm is affected by two other properties: HavalRounds
	// and KeyLength.
	// 
	//     The HavalRounds may have values of 3, 4, or 5.
	//     The KeyLength may have values of 128, 160, 192, 224, or 256.
	// 
	// Note: The "sha3-224", "sha3-256", "sha3-384", "sha3-512" algorithms are added in
	// Chilkat v9.5.0.83.
	// 
	void put_HashAlgorithm(const wchar_t *newVal);

	// Applies to the HAVAL hash algorithm only and must be set to the integer value 3,
	// 4, or 5. The default value is 3.
	int get_HavalRounds(void);
	// Applies to the HAVAL hash algorithm only and must be set to the integer value 3,
	// 4, or 5. The default value is 3.
	void put_HavalRounds(int newVal);

	// The number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort some methods call prior to
	// completion. If HeartbeatMs is 0 (the default), no AbortCheck event callbacks
	// will fire.
	// 
	// The methods with event callbacks are: CkDecryptFile, CkEncryptFile, HashFile,
	// and HashFileENC.
	// 
	int get_HeartbeatMs(void);
	// The number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort some methods call prior to
	// completion. If HeartbeatMs is 0 (the default), no AbortCheck event callbacks
	// will fire.
	// 
	// The methods with event callbacks are: CkDecryptFile, CkEncryptFile, HashFile,
	// and HashFileENC.
	// 
	void put_HeartbeatMs(int newVal);

	// Only applies when creating digital signatures. If true (the default), then
	// additional certificates (if any) in the chain of authentication are included in
	// the PKCS7 digital signature.
	bool get_IncludeCertChain(void);
	// Only applies when creating digital signatures. If true (the default), then
	// additional certificates (if any) in the chain of authentication are included in
	// the PKCS7 digital signature.
	void put_IncludeCertChain(bool newVal);

	// The initial counter for the ChaCha20 encryption algorithm. The default value is
	// 0.
	int get_InitialCount(void);
	// The initial counter for the ChaCha20 encryption algorithm. The default value is
	// 0.
	void put_InitialCount(int newVal);

	// Iteration count to be used with password-based encryption (PBE). Password-based
	// encryption is defined in the PKCS5 Password-Based Cryptography Standard at
	// http://www.rsa.com/rsalabs/node.asp?id=2127
	// 
	// The purpose of the iteration count is to increase the computation required to
	// encrypt and decrypt. A larger iteration count makes cracking via exhaustive
	// search more difficult. The default value is 1024.
	// 
	int get_IterationCount(void);
	// Iteration count to be used with password-based encryption (PBE). Password-based
	// encryption is defined in the PKCS5 Password-Based Cryptography Standard at
	// http://www.rsa.com/rsalabs/node.asp?id=2127
	// 
	// The purpose of the iteration count is to increase the computation required to
	// encrypt and decrypt. A larger iteration count makes cracking via exhaustive
	// search more difficult. The default value is 1024.
	// 
	void put_IterationCount(int newVal);

	// The initialization vector to be used with symmetric encryption algorithms (AES,
	// Blowfish, Twofish, etc.). If left unset, no initialization vector is used.
	void get_IV(CkByteData &outBytes);
	// The initialization vector to be used with symmetric encryption algorithms (AES,
	// Blowfish, Twofish, etc.). If left unset, no initialization vector is used.
	void put_IV(const CkByteData &inBytes);

	// The key length in bits for symmetric encryption algorithms. The default value is
	// 256.
	int get_KeyLength(void);
	// The key length in bits for symmetric encryption algorithms. The default value is
	// 256.
	void put_KeyLength(int newVal);

	// (See the description for the FirstChunk property.)
	bool get_LastChunk(void);
	// (See the description for the FirstChunk property.)
	void put_LastChunk(bool newVal);

	// Selects the MAC algorithm to be used for any of the Mac methods, such as
	// MacStringENC, MacBytes, etc. The default value is "hmac". Possible values are
	// "hmac" and "poly1305".
	void get_MacAlgorithm(CkString &str);
	// Selects the MAC algorithm to be used for any of the Mac methods, such as
	// MacStringENC, MacBytes, etc. The default value is "hmac". Possible values are
	// "hmac" and "poly1305".
	const wchar_t *macAlgorithm(void);
	// Selects the MAC algorithm to be used for any of the Mac methods, such as
	// MacStringENC, MacBytes, etc. The default value is "hmac". Possible values are
	// "hmac" and "poly1305".
	void put_MacAlgorithm(const wchar_t *newVal);

	// This property is set when a digital signature is verified. It contains the
	// number of signer certificates. Each signing certificate can be retrieved by
	// calling the GetSignerCert method, passing an index from 0 to NumSignerCerts-1.
	int get_NumSignerCerts(void);

	// Selects the hash algorithm for use within OAEP padding when encrypting using
	// "pki" with RSAES-OAEP. The valid choices are "sha1", "sha256", "sha384",
	// "sha512",
	// 
	// The default value is "sha256"
	// 
	void get_OaepHash(CkString &str);
	// Selects the hash algorithm for use within OAEP padding when encrypting using
	// "pki" with RSAES-OAEP. The valid choices are "sha1", "sha256", "sha384",
	// "sha512",
	// 
	// The default value is "sha256"
	// 
	const wchar_t *oaepHash(void);
	// Selects the hash algorithm for use within OAEP padding when encrypting using
	// "pki" with RSAES-OAEP. The valid choices are "sha1", "sha256", "sha384",
	// "sha512",
	// 
	// The default value is "sha256"
	// 
	void put_OaepHash(const wchar_t *newVal);

	// Selects the MGF hash algorithm for use within OAEP padding when encrypting using
	// "pki" with RSAES-OAEP. The valid choices are "sha1", "sha256", "sha384",
	// "sha512", The default is "sha1".
	void get_OaepMgfHash(CkString &str);
	// Selects the MGF hash algorithm for use within OAEP padding when encrypting using
	// "pki" with RSAES-OAEP. The valid choices are "sha1", "sha256", "sha384",
	// "sha512", The default is "sha1".
	const wchar_t *oaepMgfHash(void);
	// Selects the MGF hash algorithm for use within OAEP padding when encrypting using
	// "pki" with RSAES-OAEP. The valid choices are "sha1", "sha256", "sha384",
	// "sha512", The default is "sha1".
	void put_OaepMgfHash(const wchar_t *newVal);

	// Selects the RSA encryption scheme when encrypting using "pki" (with a
	// certificate and private key). The default value is false, which selects
	// RSAES_PKCS1-V1_5. If set to true, then RSAES_OAEP is used.
	bool get_OaepPadding(void);
	// Selects the RSA encryption scheme when encrypting using "pki" (with a
	// certificate and private key). The default value is false, which selects
	// RSAES_PKCS1-V1_5. If set to true, then RSAES_OAEP is used.
	void put_OaepPadding(bool newVal);

	// The padding scheme used by block encryption algorithms such as AES (Rijndael),
	// Blowfish, Twofish, RC2, DES, 3DES, etc. Block encryption algorithms pad
	// encrypted data to a multiple of algorithm's block size. The default value of
	// this property is 0.
	// 
	// Possible values are:
	// 
	// 0 = RFC 1423 padding scheme: Each padding byte is set to the number of padding
	// bytes. If the data is already a multiple of algorithm's block size bytes, an
	// extra block is appended each having a value equal to the block size. (for
	// example, if the algorithm's block size is 16, then 16 bytes having the value
	// 0x10 are added.). (This is also known as PKCS5 padding: PKCS #5 padding string
	// consists of a sequence of bytes, each of which is equal to the total number of
	// padding bytes added. )
	// 
	// 1 = FIPS81 (Federal Information Processing Standards 81) where the last byte
	// contains the number of padding bytes, including itself, and the other padding
	// bytes are set to random values.
	// 
	// 2 = Each padding byte is set to a random value. The decryptor must know how many
	// bytes are in the original unencrypted data.
	// 
	// 3 = Pad with NULLs. (If already a multiple of the algorithm's block size, no
	// padding is added).
	// 
	// 4 = Pad with SPACE chars(0x20). (If already a multiple of algorithm's block
	// size, no padding is added).
	// 
	int get_PaddingScheme(void);
	// The padding scheme used by block encryption algorithms such as AES (Rijndael),
	// Blowfish, Twofish, RC2, DES, 3DES, etc. Block encryption algorithms pad
	// encrypted data to a multiple of algorithm's block size. The default value of
	// this property is 0.
	// 
	// Possible values are:
	// 
	// 0 = RFC 1423 padding scheme: Each padding byte is set to the number of padding
	// bytes. If the data is already a multiple of algorithm's block size bytes, an
	// extra block is appended each having a value equal to the block size. (for
	// example, if the algorithm's block size is 16, then 16 bytes having the value
	// 0x10 are added.). (This is also known as PKCS5 padding: PKCS #5 padding string
	// consists of a sequence of bytes, each of which is equal to the total number of
	// padding bytes added. )
	// 
	// 1 = FIPS81 (Federal Information Processing Standards 81) where the last byte
	// contains the number of padding bytes, including itself, and the other padding
	// bytes are set to random values.
	// 
	// 2 = Each padding byte is set to a random value. The decryptor must know how many
	// bytes are in the original unencrypted data.
	// 
	// 3 = Pad with NULLs. (If already a multiple of the algorithm's block size, no
	// padding is added).
	// 
	// 4 = Pad with SPACE chars(0x20). (If already a multiple of algorithm's block
	// size, no padding is added).
	// 
	void put_PaddingScheme(int newVal);

	// If the CryptAlgorithm property is set to "pbes1" or "pbes2", this property
	// specifies the underlying encryption algorithm to be used with password-based
	// encryption (PBE). Password-based encryption is defined in the PKCS5
	// Password-Based Cryptography Standard at
	// http://www.rsa.com/rsalabs/node.asp?id=2127
	void get_PbesAlgorithm(CkString &str);
	// If the CryptAlgorithm property is set to "pbes1" or "pbes2", this property
	// specifies the underlying encryption algorithm to be used with password-based
	// encryption (PBE). Password-based encryption is defined in the PKCS5
	// Password-Based Cryptography Standard at
	// http://www.rsa.com/rsalabs/node.asp?id=2127
	const wchar_t *pbesAlgorithm(void);
	// If the CryptAlgorithm property is set to "pbes1" or "pbes2", this property
	// specifies the underlying encryption algorithm to be used with password-based
	// encryption (PBE). Password-based encryption is defined in the PKCS5
	// Password-Based Cryptography Standard at
	// http://www.rsa.com/rsalabs/node.asp?id=2127
	void put_PbesAlgorithm(const wchar_t *newVal);

	// The password to be used with password-based encryption (PBE). Password-based
	// encryption is defined in the PKCS5 Password-Based Cryptography Standard at
	// http://www.rsa.com/rsalabs/node.asp?id=2127
	void get_PbesPassword(CkString &str);
	// The password to be used with password-based encryption (PBE). Password-based
	// encryption is defined in the PKCS5 Password-Based Cryptography Standard at
	// http://www.rsa.com/rsalabs/node.asp?id=2127
	const wchar_t *pbesPassword(void);
	// The password to be used with password-based encryption (PBE). Password-based
	// encryption is defined in the PKCS5 Password-Based Cryptography Standard at
	// http://www.rsa.com/rsalabs/node.asp?id=2127
	void put_PbesPassword(const wchar_t *newVal);

	// When the CryptAlgorithm property is "PKI" to select PKCS7 public-key encryption,
	// this selects the underlying symmetric encryption algorithm. Possible values are:
	// "aes", "des", "3des", and "rc2". The default value is "aes".
	void get_Pkcs7CryptAlg(CkString &str);
	// When the CryptAlgorithm property is "PKI" to select PKCS7 public-key encryption,
	// this selects the underlying symmetric encryption algorithm. Possible values are:
	// "aes", "des", "3des", and "rc2". The default value is "aes".
	const wchar_t *pkcs7CryptAlg(void);
	// When the CryptAlgorithm property is "PKI" to select PKCS7 public-key encryption,
	// this selects the underlying symmetric encryption algorithm. Possible values are:
	// "aes", "des", "3des", and "rc2". The default value is "aes".
	void put_Pkcs7CryptAlg(const wchar_t *newVal);

	// The effective key length (in bits) for the RC2 encryption algorithm. When RC2 is
	// used, both the KeyLength and Rc2EffectiveKeyLength properties should be set. For
	// RC2, both should be between 8 and 1024 (inclusive).
	// 
	// The default value is 128
	// 
	int get_Rc2EffectiveKeyLength(void);
	// The effective key length (in bits) for the RC2 encryption algorithm. When RC2 is
	// used, both the KeyLength and Rc2EffectiveKeyLength properties should be set. For
	// RC2, both should be between 8 and 1024 (inclusive).
	// 
	// The default value is 128
	// 
	void put_Rc2EffectiveKeyLength(int newVal);

	// The salt to be used with password-based encryption (PBE). Password-based
	// encryption is defined in the PKCS5 Password-Based Cryptography Standard at
	// http://www.rsa.com/rsalabs/node.asp?id=2127
	// 
	// To clarify: This property is used in encryption when the CryptAlgorithm is set
	// to "pbes1" or "pbes2". Also note that it is not used by the Pbkdf1 or Pbkdf2
	// methods, as the salt is passed in an argument to those methods.
	// 
	void get_Salt(CkByteData &outBytes);
	// The salt to be used with password-based encryption (PBE). Password-based
	// encryption is defined in the PKCS5 Password-Based Cryptography Standard at
	// http://www.rsa.com/rsalabs/node.asp?id=2127
	// 
	// To clarify: This property is used in encryption when the CryptAlgorithm is set
	// to "pbes1" or "pbes2". Also note that it is not used by the Pbkdf1 or Pbkdf2
	// methods, as the salt is passed in an argument to those methods.
	// 
	void put_Salt(const CkByteData &inBytes);

	// The binary secret key used for symmetric encryption (Aes, Blowfish, Twofish,
	// ChaCha20, ARC4, 3DES, RC2, etc.). The secret key must be identical for
	// decryption to succeed. The length in bytes of the SecretKey must equal the
	// KeyLength/8.
	void get_SecretKey(CkByteData &outBytes);
	// The binary secret key used for symmetric encryption (Aes, Blowfish, Twofish,
	// ChaCha20, ARC4, 3DES, RC2, etc.). The secret key must be identical for
	// decryption to succeed. The length in bytes of the SecretKey must equal the
	// KeyLength/8.
	void put_SecretKey(const CkByteData &inBytes);

	// This property selects the signature algorithm for the OpaqueSign*, Sign*, and
	// CreateDetachedSignature, CreateP7M, and CreateP7S methods. The default value is
	// "PKCS1-v1_5". This can be set to "RSASSA-PSS" (or simply "pss") to use the
	// RSASSA-PSS signature scheme.
	// 
	// Note: This property only applies when the private key is an RSA private key. It
	// does not apply for ECC or DSA private keys.
	// 
	void get_SigningAlg(CkString &str);
	// This property selects the signature algorithm for the OpaqueSign*, Sign*, and
	// CreateDetachedSignature, CreateP7M, and CreateP7S methods. The default value is
	// "PKCS1-v1_5". This can be set to "RSASSA-PSS" (or simply "pss") to use the
	// RSASSA-PSS signature scheme.
	// 
	// Note: This property only applies when the private key is an RSA private key. It
	// does not apply for ECC or DSA private keys.
	// 
	const wchar_t *signingAlg(void);
	// This property selects the signature algorithm for the OpaqueSign*, Sign*, and
	// CreateDetachedSignature, CreateP7M, and CreateP7S methods. The default value is
	// "PKCS1-v1_5". This can be set to "RSASSA-PSS" (or simply "pss") to use the
	// RSASSA-PSS signature scheme.
	// 
	// Note: This property only applies when the private key is an RSA private key. It
	// does not apply for ECC or DSA private keys.
	// 
	void put_SigningAlg(const wchar_t *newVal);

	// Contains JSON to specify the authenticated (signed) attributes or
	// unauthenticated (unsigned) attributes that are to be included in CMS signatures.
	// The default value is:
	// {
	//     "contentType": 1,
	//     "signingTime": 1,
	//     "messageDigest": 1
	// }
	// 
	// Other possible values that can be added are:
	//     signingCertificateV2
	//     signingCertificate
	//     sMIMECapabilities
	//     microsoftRecipientInfo
	//     encrypKeyPref
	//     cmsAlgorithmProtection
	// Contact Chilkat (support@chilkatsoft.com) about other signed/unsigned attributes
	// that may be needed for CAdES signatures.
	// 
	void get_SigningAttributes(CkString &str);
	// Contains JSON to specify the authenticated (signed) attributes or
	// unauthenticated (unsigned) attributes that are to be included in CMS signatures.
	// The default value is:
	// {
	//     "contentType": 1,
	//     "signingTime": 1,
	//     "messageDigest": 1
	// }
	// 
	// Other possible values that can be added are:
	//     signingCertificateV2
	//     signingCertificate
	//     sMIMECapabilities
	//     microsoftRecipientInfo
	//     encrypKeyPref
	//     cmsAlgorithmProtection
	// Contact Chilkat (support@chilkatsoft.com) about other signed/unsigned attributes
	// that may be needed for CAdES signatures.
	// 
	const wchar_t *signingAttributes(void);
	// Contains JSON to specify the authenticated (signed) attributes or
	// unauthenticated (unsigned) attributes that are to be included in CMS signatures.
	// The default value is:
	// {
	//     "contentType": 1,
	//     "signingTime": 1,
	//     "messageDigest": 1
	// }
	// 
	// Other possible values that can be added are:
	//     signingCertificateV2
	//     signingCertificate
	//     sMIMECapabilities
	//     microsoftRecipientInfo
	//     encrypKeyPref
	//     cmsAlgorithmProtection
	// Contact Chilkat (support@chilkatsoft.com) about other signed/unsigned attributes
	// that may be needed for CAdES signatures.
	// 
	void put_SigningAttributes(const wchar_t *newVal);

	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "UseConstructedOctets" - Introduced in v9.5.0.83. When creating opaque CMS
	//     signatures (signatures that embed the data being signed), will use the
	//     "constructed octets" form of the ASN.1 that holds the data. This is to satify
	//     some validators that are brittle/fragile/picky and require a particular format,
	//     such as for the ICP-Brazil Digital Signature Standard.
	// 
	void get_UncommonOptions(CkString &str);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "UseConstructedOctets" - Introduced in v9.5.0.83. When creating opaque CMS
	//     signatures (signatures that embed the data being signed), will use the
	//     "constructed octets" form of the ASN.1 that holds the data. This is to satify
	//     some validators that are brittle/fragile/picky and require a particular format,
	//     such as for the ICP-Brazil Digital Signature Standard.
	// 
	const wchar_t *uncommonOptions(void);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "UseConstructedOctets" - Introduced in v9.5.0.83. When creating opaque CMS
	//     signatures (signatures that embed the data being signed), will use the
	//     "constructed octets" form of the ASN.1 that holds the data. This is to satify
	//     some validators that are brittle/fragile/picky and require a particular format,
	//     such as for the ICP-Brazil Digital Signature Standard.
	// 
	void put_UncommonOptions(const wchar_t *newVal);

	// When UU encoding, this is the filename to be embedded in UU encoded output. The
	// default is "file.dat". When UU decoding, this is the filename found in the UU
	// encoded input.
	void get_UuFilename(CkString &str);
	// When UU encoding, this is the filename to be embedded in UU encoded output. The
	// default is "file.dat". When UU decoding, this is the filename found in the UU
	// encoded input.
	const wchar_t *uuFilename(void);
	// When UU encoding, this is the filename to be embedded in UU encoded output. The
	// default is "file.dat". When UU decoding, this is the filename found in the UU
	// encoded input.
	void put_UuFilename(const wchar_t *newVal);

	// When UU encoding, this is the file permissions mode to be embedded in UU encoded
	// output. The default is "644". When UU decoding, this property is set to the mode
	// found in the UU encoded input.
	void get_UuMode(CkString &str);
	// When UU encoding, this is the file permissions mode to be embedded in UU encoded
	// output. The default is "644". When UU decoding, this property is set to the mode
	// found in the UU encoded input.
	const wchar_t *uuMode(void);
	// When UU encoding, this is the file permissions mode to be embedded in UU encoded
	// output. The default is "644". When UU decoding, this property is set to the mode
	// found in the UU encoded input.
	void put_UuMode(const wchar_t *newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Adds a certificate to be used for public-key encryption. (To use public-key
	// encryption with digital certificates, set the CryptAlgorithm property = "pki".)
	// To encrypt with more than one certificate , call AddEncryptCert once per
	// certificate.
	void AddEncryptCert(CkCertW &cert);

	// Adds a PFX to the object's internal list of sources to be searched for
	// certificates and private keys when decrypting. Multiple PFX sources can be added
	// by calling this method once for each. (On the Windows operating system, the
	// registry-based certificate stores are also automatically searched, so it is
	// commonly not required to explicitly add PFX sources.)
	// 
	// The pfxBytes contains the bytes of a PFX file (also known as PKCS12 or .p12).
	// 
	bool AddPfxSourceData(CkByteData &pfxBytes, const wchar_t *pfxPassword);

	// Adds a PFX file to the object's internal list of sources to be searched for
	// certificates and private keys when decrypting. Multiple PFX files can be added
	// by calling this method once for each. (On the Windows operating system, the
	// registry-based certificate stores are also automatically searched, so it is
	// commonly not required to explicitly add PFX sources.)
	// 
	// The pfxFilePath contains the bytes of a PFX file (also known as PKCS12 or .p12).
	// 
	bool AddPfxSourceFile(const wchar_t *pfxFilePath, const wchar_t *pfxPassword);

	// Adds a certificate to be used for signing. To sign with more than one
	// certificate, call AddSigningCert once per certificate.
	// 
	// Note: This method was added in v9.5.0.83. The SetSigningCert and SetSigningCert2
	// methods are used to set the signing certificate for signatures with one signer.
	// 
	bool AddSigningCert(CkCertW &cert);

	// Implements the AES Key Wrap Algorithm (RFC 3394) for unwrapping. The kek is the
	// Key Encryption Key (the AES key used to unwrap the wrappedKeyData). The arguments and
	// return value are binary encoded strings using the encoding specified by encoding
	// (which can be "base64", "hex", "base64url", etc.) The full list of supported
	// encodings is available at the link below.
	// 
	// The kek should be an AES key of 16 bytes, 24 bytes, or 32 bytes (i.e. 128-bits,
	// 192- bits, or 256-bits). For example, if passed as a hex string, then the kek
	// should be 32 chars in length, 48 chars, or 64 chars (because each byte is
	// represented as 2 chars in hex).
	// 
	// The wrappedKeyData contains the data to be unwrapped. The result, if decoded, is 8 bytes
	// less than the wrapped key data. For example, if a 256-bit AES key (32 bytes) is
	// wrapped, the size of the wrapped key data is 40 bytes. Unwrapping restores it to
	// the original 32 bytes.
	// 
	bool AesKeyUnwrap(const wchar_t *kek, const wchar_t *wrappedKeyData, const wchar_t *encoding, CkString &outStr);
	// Implements the AES Key Wrap Algorithm (RFC 3394) for unwrapping. The kek is the
	// Key Encryption Key (the AES key used to unwrap the wrappedKeyData). The arguments and
	// return value are binary encoded strings using the encoding specified by encoding
	// (which can be "base64", "hex", "base64url", etc.) The full list of supported
	// encodings is available at the link below.
	// 
	// The kek should be an AES key of 16 bytes, 24 bytes, or 32 bytes (i.e. 128-bits,
	// 192- bits, or 256-bits). For example, if passed as a hex string, then the kek
	// should be 32 chars in length, 48 chars, or 64 chars (because each byte is
	// represented as 2 chars in hex).
	// 
	// The wrappedKeyData contains the data to be unwrapped. The result, if decoded, is 8 bytes
	// less than the wrapped key data. For example, if a 256-bit AES key (32 bytes) is
	// wrapped, the size of the wrapped key data is 40 bytes. Unwrapping restores it to
	// the original 32 bytes.
	// 
	const wchar_t *aesKeyUnwrap(const wchar_t *kek, const wchar_t *wrappedKeyData, const wchar_t *encoding);

	// Implements the AES Key Wrap Algorithm (RFC 3394). The kek is the Key Encryption
	// Key (the AES key used to encrypt the keyData). The arguments and return value are
	// binary encoded strings using the encoding specified by encoding (which can be
	// "base64", "hex", "base64url", etc.) The full list of supported encodings is
	// available at the link below.
	// 
	// The kek should be an AES key of 16 bytes, 24 bytes, or 32 bytes (i.e. 128-bits,
	// 192- bits, or 256-bits). For example, if passed as a hex string, then the kek
	// should be 32 chars in length, 48 chars, or 64 chars (because each byte is
	// represented as 2 chars in hex).
	// 
	// The keyData contains the data to be key wrapped. It must be a multiple of 64-bits
	// in length. In other words, if the keyData is decoded to binary, it should be a
	// number of bytes that is a multiple of 8.
	// 
	// The return string, if decoded to binary bytes, is equal to the size of the key
	// data + 8 additional bytes.
	// 
	bool AesKeyWrap(const wchar_t *kek, const wchar_t *keyData, const wchar_t *encoding, CkString &outStr);
	// Implements the AES Key Wrap Algorithm (RFC 3394). The kek is the Key Encryption
	// Key (the AES key used to encrypt the keyData). The arguments and return value are
	// binary encoded strings using the encoding specified by encoding (which can be
	// "base64", "hex", "base64url", etc.) The full list of supported encodings is
	// available at the link below.
	// 
	// The kek should be an AES key of 16 bytes, 24 bytes, or 32 bytes (i.e. 128-bits,
	// 192- bits, or 256-bits). For example, if passed as a hex string, then the kek
	// should be 32 chars in length, 48 chars, or 64 chars (because each byte is
	// represented as 2 chars in hex).
	// 
	// The keyData contains the data to be key wrapped. It must be a multiple of 64-bits
	// in length. In other words, if the keyData is decoded to binary, it should be a
	// number of bytes that is a multiple of 8.
	// 
	// The return string, if decoded to binary bytes, is equal to the size of the key
	// data + 8 additional bytes.
	// 
	const wchar_t *aesKeyWrap(const wchar_t *kek, const wchar_t *keyData, const wchar_t *encoding);

	// Computes and returns a bcrypt hash of the password. The number of rounds of hashing
	// is determined by the BCryptWorkFactor property.
	// 
	// Starting in v9.5.0.76, if the password is prefixed with "$2b$" then the output will
	// use the $2b version of bcrypt. For example, to create a "$2b$" bcrypt has for
	// the password "secret", pass in the string "$2b$secret" for password.
	// 
	bool BCryptHash(const wchar_t *password, CkString &outStr);
	// Computes and returns a bcrypt hash of the password. The number of rounds of hashing
	// is determined by the BCryptWorkFactor property.
	// 
	// Starting in v9.5.0.76, if the password is prefixed with "$2b$" then the output will
	// use the $2b version of bcrypt. For example, to create a "$2b$" bcrypt has for
	// the password "secret", pass in the string "$2b$secret" for password.
	// 
	const wchar_t *bCryptHash(const wchar_t *password);

	// Verifies the password against a previously computed BCrypt hash. Returns true if
	// the password matches the bcryptHash. Returns false if the password does not match.
	bool BCryptVerify(const wchar_t *password, const wchar_t *bcryptHash);

	// Utility method to convert bytes to a string -- interpreting the bytes according
	// to the charset specified.
	bool BytesToString(CkByteData &inData, const wchar_t *charset, CkString &outStr);
	// Utility method to convert bytes to a string -- interpreting the bytes according
	// to the charset specified.
	const wchar_t *bytesToString(CkByteData &inData, const wchar_t *charset);

	// Convenience method for byte swapping between little-endian byte ordering and
	// big-endian byte ordering.
	bool ByteSwap4321(CkByteData &data, CkByteData &outBytes);

	// File-to-file decryption. There is no limit to the size of the file that can be
	// decrypted because the component will operate in streaming mode internally.
	bool CkDecryptFile(const wchar_t *srcFile, const wchar_t *destFile);

	// Creates an asynchronous task to call the CkDecryptFile method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CkDecryptFileAsync(const wchar_t *srcFile, const wchar_t *destFile);

	// File-to-file encryption. There is no limit to the size of the file that can be
	// encrypted because the component will operate in streaming mode internally.
	bool CkEncryptFile(const wchar_t *srcFile, const wchar_t *destFile);

	// Creates an asynchronous task to call the CkEncryptFile method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CkEncryptFileAsync(const wchar_t *srcFile, const wchar_t *destFile);

	// Clears the internal list of digital certificates to be used for public-key
	// encryption.
	void ClearEncryptCerts(void);

	// Clears the set of certificates to be used in signing.
	void ClearSigningCerts(void);

	// Bzip2 compresses a byte array and returns the compressed bytes.
	// 
	// This is a legacy method that should not be used in new development. It will not
	// be marked as deprecated or removed from future APIs because existing
	// applications may have data already compressed using this method.
	// 
	// The output of this method includes an 8-byte header composed of a 4-byte magic
	// number (0xB394A7E1) and the 4-byte length of the uncompressed data.
	// 
	bool CompressBytes(CkByteData &data, CkByteData &outData);

	// Same as CompressBytes, except an encoded string is returned. The output encoding
	// is specified by the EncodingMode property.
	bool CompressBytesENC(CkByteData &data, CkString &outStr);
	// Same as CompressBytes, except an encoded string is returned. The output encoding
	// is specified by the EncodingMode property.
	const wchar_t *compressBytesENC(CkByteData &data);

	// Compresses a string and returns the compressed bytes. Prior to compressing, the
	// string is converted to a byte representation such as utf-8, utf-16, etc. as
	// determined by the Charset property. Otherwise, this method is the same as the
	// CompressBytes method.
	bool CompressString(const wchar_t *str, CkByteData &outData);

	// Compresses a string and returns the encoded compressed bytes. Prior to
	// compressing, the string is converted to a byte representation such as utf-8,
	// utf-16, etc. as determined by the Charset property. The output encoding is
	// specified by the EncodingMode property. Otherwise, this method is the same as
	// the CompressBytes method.
	bool CompressStringENC(const wchar_t *str, CkString &outStr);
	// Compresses a string and returns the encoded compressed bytes. Prior to
	// compressing, the string is converted to a byte representation such as utf-8,
	// utf-16, etc. as determined by the Charset property. The output encoding is
	// specified by the EncodingMode property. Otherwise, this method is the same as
	// the CompressBytes method.
	const wchar_t *compressStringENC(const wchar_t *str);

	// Co-sign's an existing CMS signature. bdIn contains the existing CMS signature.
	// If successful, cert is the output co-signed CMS signature.
	bool CoSign(CkBinDataW &bdIn, CkCertW &cert, CkBinDataW &bdOut);

	// Calculates a CRC for in-memory byte data. To compute the CRC used in the Zip
	// file format, pass "CRC-32" (or CRC32, case insensitive) for the crcAlg. (The crcAlg
	// argument provides the flexibility to add additional CRC algorithms on an
	// as-needed basis in the future.)
	// 
	// Starting in v9.5.0.88, crc8 can be computed by passing "CRC8" in crcAlg.
	// 
	unsigned long CrcBytes(const wchar_t *crcAlg, CkByteData &byteData);

	// Calculates a CRC for the contents of a file. To compute the CRC used in the Zip
	// file format, pass "CRC-32" for the crcAlg. (The crcAlg argument provides the
	// flexibility to add additional CRC algorithms on an as-needed basis in the
	// future.) A value of 0 is returned if the file is unable to be read. Given that
	// there is a 1 in 4 billion chance of having an actual CRC of 0, an application
	// might choose to react to a 0 return value by testing to see if the file can be
	// opened and read.
	// 
	// Starting in v9.5.0.88, crc8 can be computed by passing "CRC8" in crcAlg.
	// 
	unsigned long CrcFile(const wchar_t *crcAlg, const wchar_t *path);

	// Creates an asynchronous task to call the CrcFile method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CrcFileAsync(const wchar_t *crcAlg, const wchar_t *path);

	// Digitally signs a file and writes the digital signature to a separate output
	// file (a PKCS#7 signature file). The input file (inFilePath) is unmodified. A
	// certificate for signing must be specified by calling SetSigningCert or
	// SetSigningCert2 prior to calling this method.
	// 
	// This method is equivalent to CreateP7S. The CreateP7S method was added to
	// clarify the format of the signature file that is created.
	// 
	bool CreateDetachedSignature(const wchar_t *inFilePath, const wchar_t *sigFilePath);

	// Digitally signs a file and creates a .p7m (PKCS #7 Message) file that contains
	// both the signature and original file content. The input file (inFilename) is
	// unmodified. A certificate for signing must be specified by calling
	// SetSigningCert or SetSigningCert2 prior to calling this method.
	// 
	// To sign with a particular hash algorithm, set the HashAlgorithm property. Valid
	// hash algorithms for signing are "sha256", "sha1", "sha384", "sha512", "md5", and
	// "md2".
	// 
	// Note: The CreateP7M method creates an opaque signature. To do the same thing
	// entirely in memory, your application would call any of the OpaqueSign* methods,
	// such as OpaqueSignBd, OpaqueSignString, OpaqueSignStringENC, etc.
	// 
	bool CreateP7M(const wchar_t *inFilename, const wchar_t *p7mPath);

	// Creates an asynchronous task to call the CreateP7M method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CreateP7MAsync(const wchar_t *inFilename, const wchar_t *p7mPath);

	// Digitally signs a file and creates a .p7s (PKCS #7 Signature) signature file.
	// The input file (inFilename) is unmodified. The output file (p7sPath) contains only the
	// signature and not the original data. A certificate for signing must be specified
	// by calling SetSigningCert or SetSigningCert2 prior to calling this method.
	// 
	// To sign with a particular hash algorithm, set the HashAlgorithm property. Valid
	// hash algorithms for signing are "sha256", "sha1", "sha384", "sha512", "md5", and
	// "md2".
	// 
	// Note: The CreateP7S method creates a detached signature. To do the same thing
	// entirely in memory, your application would call any of the Sign* methods, such
	// as SignBdENC, SignString, SignStringENC, SignSbENC, etc.
	// 
	bool CreateP7S(const wchar_t *inFilename, const wchar_t *p7sPath);

	// Creates an asynchronous task to call the CreateP7S method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CreateP7SAsync(const wchar_t *inFilename, const wchar_t *p7sPath);

	// Decode binary data from an encoded string. The encoding can be set to any of the
	// following strings: "base64", "hex", "quoted-printable", "url", "base32", "Q",
	// "B", "url_rc1738", "url_rfc2396", "url_rfc3986", "url_oauth", "uu", "modBase64",
	// or "html" (for HTML entity encoding).
	bool Decode(const wchar_t *str, const wchar_t *encoding, CkByteData &outData);

	// Decodes from an encoding back to the original string. The encoding can be set to any
	// of the following strings: "base64", "hex", "quoted-printable", "url", "base32",
	// "Q", "B", "url_rc1738", "url_rfc2396", "url_rfc3986", "url_oauth", "uu",
	// "modBase64", or "html" (for HTML entity encoding).
	bool DecodeString(const wchar_t *inStr, const wchar_t *charset, const wchar_t *encoding, CkString &outStr);
	// Decodes from an encoding back to the original string. The encoding can be set to any
	// of the following strings: "base64", "hex", "quoted-printable", "url", "base32",
	// "Q", "B", "url_rc1738", "url_rfc2396", "url_rfc3986", "url_oauth", "uu",
	// "modBase64", or "html" (for HTML entity encoding).
	const wchar_t *decodeString(const wchar_t *inStr, const wchar_t *charset, const wchar_t *encoding);

	// In-place decrypts the contents of bd. The minimal set of properties that
	// should be set before decrypting are: CryptAlgorithm, SecretKey. Other properties
	// that control encryption are: CipherMode, PaddingScheme, KeyLength, IV.
	bool DecryptBd(CkBinDataW &bd);

	// Decrypts a byte array and returns the unencrypted byte array. The property
	// settings used when encrypting the data must match the settings when decrypting.
	// Specifically, the CryptAlgorithm, CipherMode, PaddingScheme, KeyLength, IV, and
	// SecretKey properties must match.
	bool DecryptBytes(CkByteData &data, CkByteData &outData);

	// The same as DecryptBytes except the binary data is passed via a pointer and
	// length.
	bool DecryptBytes2(const void *pByteData, unsigned long szByteData, CkByteData &outBytes);

	// Decrypts string-encoded encrypted data and returns the unencrypted byte array.
	// Data encrypted with EncryptBytesENC can be decrypted with this method. The
	// property settings used when encrypting the data must match the settings when
	// decrypting. Specifically, the EncodingMode, CryptAlgorithm, CipherMode,
	// PaddingScheme, KeyLength, IV, and SecretKey properties must match.
	bool DecryptBytesENC(const wchar_t *str, CkByteData &outData);

	// Encrypted data is passed to this method as an encoded string (base64, hex,
	// etc.). This method first decodes the input data according to the EncodingMode
	// property setting. It then decrypts and re-encodes using the EncodingMode
	// setting, and returns the decrypted data in encoded string form.
	bool DecryptEncoded(const wchar_t *encodedEncryptedData, CkString &outStr);
	// Encrypted data is passed to this method as an encoded string (base64, hex,
	// etc.). This method first decodes the input data according to the EncodingMode
	// property setting. It then decrypts and re-encodes using the EncodingMode
	// setting, and returns the decrypted data in encoded string form.
	const wchar_t *decryptEncoded(const wchar_t *encodedEncryptedData);

	// Decrypts the contents of bdIn to sbOut. The decrypted string is appended to sbOut.
	// The minimal set of properties that should be set before ecrypting are:
	// CryptAlgorithm, SecretKey. Other properties that control encryption are:
	// CipherMode, PaddingScheme, KeyLength, IV.
	bool DecryptSb(CkBinDataW &bdIn, CkStringBuilderW &sbOut);

	// Identical to DecryptStringENC, except the decrypts the cipherText and appends the
	// decrypted string to the secureStr.
	bool DecryptSecureENC(const wchar_t *cipherText, CkSecureStringW &secureStr);

	// Decrypts a stream. Internally, the strm's source is read, decrypted, and the
	// decrypted data written to the strm's sink. It does this in streaming fashion.
	// Extremely large or even infinite streams can be decrypted with stable ungrowing
	// memory usage.
	bool DecryptStream(CkStreamW &strm);

	// Creates an asynchronous task to call the DecryptStream method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *DecryptStreamAsync(CkStreamW &strm);

	// The reverse of EncryptString.
	// 
	// Decrypts encrypted byte data and returns the original string. The property
	// settings used when encrypting the string must match the settings when
	// decrypting. Specifically, the Charset, CryptAlgorithm, CipherMode,
	// PaddingScheme, KeyLength, IV, and SecretKey properties must match.
	// 
	bool DecryptString(CkByteData &data, CkString &outStr);
	// The reverse of EncryptString.
	// 
	// Decrypts encrypted byte data and returns the original string. The property
	// settings used when encrypting the string must match the settings when
	// decrypting. Specifically, the Charset, CryptAlgorithm, CipherMode,
	// PaddingScheme, KeyLength, IV, and SecretKey properties must match.
	// 
	const wchar_t *decryptString(CkByteData &data);

	// The reverse of EncryptStringENC.
	// 
	// Decrypts string-encoded encrypted data and returns the original string. The
	// property settings used when encrypting the string must match the settings when
	// decrypting. Specifically, the Charset, EncodingMode, CryptAlgorithm, CipherMode,
	// PaddingScheme, KeyLength, IV, and SecretKey properties must match.
	// 
	bool DecryptStringENC(const wchar_t *str, CkString &outStr);
	// The reverse of EncryptStringENC.
	// 
	// Decrypts string-encoded encrypted data and returns the original string. The
	// property settings used when encrypting the string must match the settings when
	// decrypting. Specifically, the Charset, EncodingMode, CryptAlgorithm, CipherMode,
	// PaddingScheme, KeyLength, IV, and SecretKey properties must match.
	// 
	const wchar_t *decryptStringENC(const wchar_t *str);

	// Encode binary data to base64, hex, quoted-printable, or URL-encoding. The encoding
	// can be set to any of the following strings: "base64", "hex", "quoted-printable"
	// (or "qp"), "url", "base32", "Q", "B", "url_rc1738", "url_rfc2396",
	// "url_rfc3986", "url_oauth", "uu", "modBase64", or "html" (for HTML entity
	// encoding).
	bool Encode(CkByteData &byteData, const wchar_t *encoding, CkString &outStr);
	// Encode binary data to base64, hex, quoted-printable, or URL-encoding. The encoding
	// can be set to any of the following strings: "base64", "hex", "quoted-printable"
	// (or "qp"), "url", "base32", "Q", "B", "url_rc1738", "url_rfc2396",
	// "url_rfc3986", "url_oauth", "uu", "modBase64", or "html" (for HTML entity
	// encoding).
	const wchar_t *encode(CkByteData &byteData, const wchar_t *encoding);

	// Encode binary data to base64, hex, quoted-printable, or URL-encoding. The encoding
	// can be set to any of the following strings: "base64", "hex", "quoted-printable",
	// "url", "base32", "Q", "B", "url_rc1738", "url_rfc2396", "url_rfc3986",
	// "url_oauth", "uu", "modBase64", or "html" (for HTML entity encoding).
	// 
	// The pByteData points to the bytes to be encoded. The szByteData specifies the number of
	// bytes to encode.
	// 
	bool EncodeBytes(const void *pByteData, unsigned long szByteData, const wchar_t *encoding, CkString &outStr);
	// Encode binary data to base64, hex, quoted-printable, or URL-encoding. The encoding
	// can be set to any of the following strings: "base64", "hex", "quoted-printable",
	// "url", "base32", "Q", "B", "url_rc1738", "url_rfc2396", "url_rfc3986",
	// "url_oauth", "uu", "modBase64", or "html" (for HTML entity encoding).
	// 
	// The pByteData points to the bytes to be encoded. The szByteData specifies the number of
	// bytes to encode.
	// 
	const wchar_t *encodeBytes(const void *pByteData, unsigned long szByteData, const wchar_t *encoding);

	// Encodes an integer to N bytes and returns in the specified encoding. If littleEndian is
	// true, then little endian byte ordering is used. Otherwise big-endian byte
	// order is used.
	bool EncodeInt(int value, int numBytes, bool littleEndian, const wchar_t *encoding, CkString &outStr);
	// Encodes an integer to N bytes and returns in the specified encoding. If littleEndian is
	// true, then little endian byte ordering is used. Otherwise big-endian byte
	// order is used.
	const wchar_t *encodeInt(int value, int numBytes, bool littleEndian, const wchar_t *encoding);

	// Encodes a string. The toEncodingName can be set to any of the following strings: "base64",
	// "hex", "quoted-printable", "url", "base32", "Q", "B", "url_rc1738",
	// "url_rfc2396", "url_rfc3986", "url_oauth", "uu", "modBase64", or "html" (for
	// HTML entity encoding). The charsetName is important, and usually you'll want to specify
	// "ansi". For example, if the string "ABC" is to be encoded to "hex" using ANSI,
	// the result will be "414243". However, if "unicode" is used, the result is
	// "410042004300".
	bool EncodeString(const wchar_t *strToEncode, const wchar_t *charsetName, const wchar_t *toEncodingName, CkString &outStr);
	// Encodes a string. The toEncodingName can be set to any of the following strings: "base64",
	// "hex", "quoted-printable", "url", "base32", "Q", "B", "url_rc1738",
	// "url_rfc2396", "url_rfc3986", "url_oauth", "uu", "modBase64", or "html" (for
	// HTML entity encoding). The charsetName is important, and usually you'll want to specify
	// "ansi". For example, if the string "ABC" is to be encoded to "hex" using ANSI,
	// the result will be "414243". However, if "unicode" is used, the result is
	// "410042004300".
	const wchar_t *encodeString(const wchar_t *strToEncode, const wchar_t *charsetName, const wchar_t *toEncodingName);

	// In-place encrypts the contents of bd. The minimal set of properties that
	// should be set before encrypting are: CryptAlgorithm, SecretKey. Other properties
	// that control encryption are: CipherMode, PaddingScheme, KeyLength, IV. When
	// decrypting, all property settings must match otherwise the result is garbled
	// data.
	bool EncryptBd(CkBinDataW &bd);

	// Encrypts a byte array. The minimal set of properties that should be set before
	// encrypting are: CryptAlgorithm, SecretKey. Other properties that control
	// encryption are: CipherMode, PaddingScheme, KeyLength, IV. When decrypting, all
	// property settings must match otherwise garbled data is returned.
	bool EncryptBytes(CkByteData &data, CkByteData &outData);

	// The same as EncryptBytes except the binary data is passed via a pointer and
	// length.
	bool EncryptBytes2(const void *pByteData, unsigned long szByteData, CkByteData &outBytes);

	// Encrypts a byte array and returns the encrypted data as an encoded (printable)
	// string. The minimal set of properties that should be set before encrypting are:
	// CryptAlgorithm, SecretKey, EncodingMode. Other properties that control
	// encryption are: CipherMode, PaddingScheme, KeyLength, IV. When decrypting, all
	// property settings must match otherwise garbled data is returned. The encoding of
	// the string that is returned is controlled by the EncodingMode property, which
	// can be set to "Base64", "QP", or "Hex".
	bool EncryptBytesENC(CkByteData &data, CkString &outStr);
	// Encrypts a byte array and returns the encrypted data as an encoded (printable)
	// string. The minimal set of properties that should be set before encrypting are:
	// CryptAlgorithm, SecretKey, EncodingMode. Other properties that control
	// encryption are: CipherMode, PaddingScheme, KeyLength, IV. When decrypting, all
	// property settings must match otherwise garbled data is returned. The encoding of
	// the string that is returned is controlled by the EncodingMode property, which
	// can be set to "Base64", "QP", or "Hex".
	const wchar_t *encryptBytesENC(CkByteData &data);

	// The input string is first decoded according to the encoding algorithm specified
	// by the EncodingMode property (such as base64, hex, etc.) It is then encrypted
	// according to the encryption algorithm specified by CryptAlgorithm. The resulting
	// encrypted data is encoded (using EncodingMode) and returned.
	bool EncryptEncoded(const wchar_t *str, CkString &outStr);
	// The input string is first decoded according to the encoding algorithm specified
	// by the EncodingMode property (such as base64, hex, etc.) It is then encrypted
	// according to the encryption algorithm specified by CryptAlgorithm. The resulting
	// encrypted data is encoded (using EncodingMode) and returned.
	const wchar_t *encryptEncoded(const wchar_t *str);

	// Encrypts the contents of sbIn to bdOut. The minimal set of properties that should
	// be set before ecrypting are: CryptAlgorithm, SecretKey. Other properties that
	// control encryption are: CipherMode, PaddingScheme, KeyLength, IV.
	bool EncryptSb(CkStringBuilderW &sbIn, CkBinDataW &bdOut);

	// Identical to EncryptStringENC, except the clear-text contained within the secureStr
	// is encrypted and returned.
	bool EncryptSecureENC(CkSecureStringW &secureStr, CkString &outStr);
	// Identical to EncryptStringENC, except the clear-text contained within the secureStr
	// is encrypted and returned.
	const wchar_t *encryptSecureENC(CkSecureStringW &secureStr);

	// Encrypts a stream. Internally, the strm's source is read, encrypted, and the
	// encrypted data written to the strm's sink. It does this in streaming fashion.
	// Extremely large or even infinite streams can be encrypted with stable ungrowing
	// memory usage.
	bool EncryptStream(CkStreamW &strm);

	// Creates an asynchronous task to call the EncryptStream method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *EncryptStreamAsync(CkStreamW &strm);

	// Encrypts a string and returns the encrypted data as a byte array. The minimal
	// set of properties that should be set before encrypting are: CryptAlgorithm,
	// SecretKey, Charset. Other properties that control encryption are: CipherMode,
	// PaddingScheme, KeyLength, IV. When decrypting, all property settings must match
	// otherwise garbled data is returned. The Charset property controls the exact
	// bytes that get encrypted. Languages such as VB.NET, C#, and Visual Basic work
	// with Unicode strings, thus the input string is Unicode. If Unicode is to be
	// encrypted (i.e. 2 bytes per character) then set the Charset property to
	// "Unicode". To implicitly convert the string to another charset before the
	// encryption is applied, set the Charset property to something else, such as
	// "iso-8859-1", "Shift_JIS", "big5", "windows-1252", etc. The complete list of
	// possible charsets is listed here:
	// us-ascii
	// unicode
	// unicodefffe
	// iso-8859-1
	// iso-8859-2
	// iso-8859-3
	// iso-8859-4
	// iso-8859-5
	// iso-8859-6
	// iso-8859-7
	// iso-8859-8
	// iso-8859-9
	// iso-8859-13
	// iso-8859-15
	// windows-874
	// windows-1250
	// windows-1251
	// windows-1252
	// windows-1253
	// windows-1254
	// windows-1255
	// windows-1256
	// windows-1257
	// windows-1258
	// utf-7
	// utf-8
	// utf-32
	// utf-32be
	// shift_jis
	// gb2312
	// ks_c_5601-1987
	// big5
	// iso-2022-jp
	// iso-2022-kr
	// euc-jp
	// euc-kr
	// macintosh
	// x-mac-japanese
	// x-mac-chinesetrad
	// x-mac-korean
	// x-mac-arabic
	// x-mac-hebrew
	// x-mac-greek
	// x-mac-cyrillic
	// x-mac-chinesesimp
	// x-mac-romanian
	// x-mac-ukrainian
	// x-mac-thai
	// x-mac-ce
	// x-mac-icelandic
	// x-mac-turkish
	// x-mac-croatian
	// asmo-708
	// dos-720
	// dos-862
	// ibm037
	// ibm437
	// ibm500
	// ibm737
	// ibm775
	// ibm850
	// ibm852
	// ibm855
	// ibm857
	// ibm00858
	// ibm860
	// ibm861
	// ibm863
	// ibm864
	// ibm865
	// cp866
	// ibm869
	// ibm870
	// cp875
	// koi8-r
	// koi8-u
	bool EncryptString(const wchar_t *str, CkByteData &outData);

	// Encrypts a string and returns the encrypted data as an encoded (printable)
	// string. The minimal set of properties that should be set before encrypting are:
	// CryptAlgorithm, SecretKey, Charset, and EncodingMode. Other properties that
	// control encryption are: CipherMode, PaddingScheme, KeyLength, IV. When
	// decrypting (with DecryptStringENC), all property settings must match otherwise
	// garbled data is returned. The Charset property controls the exact bytes that get
	// encrypted. Languages such as VB.NET, C#, and Visual Basic work with Unicode
	// strings, thus the input string is Unicode. If Unicode is to be encrypted (i.e. 2
	// bytes per character) then set the Charset property to "Unicode". To implicitly
	// convert the string to another charset before the encryption is applied, set the
	// Charset property to something else, such as "iso-8859-1", "Shift_JIS", "big5",
	// "windows-1252", etc. (Refer to EncryptString for the complete list of charsets.)
	// 
	// The EncodingMode property controls the encoding of the string that is returned.
	// It can be set to "Base64", "QP", or "Hex".
	// 
	bool EncryptStringENC(const wchar_t *str, CkString &outStr);
	// Encrypts a string and returns the encrypted data as an encoded (printable)
	// string. The minimal set of properties that should be set before encrypting are:
	// CryptAlgorithm, SecretKey, Charset, and EncodingMode. Other properties that
	// control encryption are: CipherMode, PaddingScheme, KeyLength, IV. When
	// decrypting (with DecryptStringENC), all property settings must match otherwise
	// garbled data is returned. The Charset property controls the exact bytes that get
	// encrypted. Languages such as VB.NET, C#, and Visual Basic work with Unicode
	// strings, thus the input string is Unicode. If Unicode is to be encrypted (i.e. 2
	// bytes per character) then set the Charset property to "Unicode". To implicitly
	// convert the string to another charset before the encryption is applied, set the
	// Charset property to something else, such as "iso-8859-1", "Shift_JIS", "big5",
	// "windows-1252", etc. (Refer to EncryptString for the complete list of charsets.)
	// 
	// The EncodingMode property controls the encoding of the string that is returned.
	// It can be set to "Base64", "QP", or "Hex".
	// 
	const wchar_t *encryptStringENC(const wchar_t *str);

	// Important: In the v9.5.0.49 release, a bug involving this method was introduced:
	// The encoding is ignored and instead the encoding used is the current value of the
	// EncodingMode property. The workaround is to make sure the EncodingMode property
	// is set to the value of the desired output encoding. This problem will be fixed
	// in v9.5.0.50.
	// 
	// Identical to the GenerateSecretKey method, except it returns the binary secret
	// key as a string encoded according to encoding, which may be "base64", "hex", "url",
	// etc. Please see the documentation for GenerateSecretKey for more information.
	// 
	bool GenEncodedSecretKey(const wchar_t *password, const wchar_t *encoding, CkString &outStr);
	// Important: In the v9.5.0.49 release, a bug involving this method was introduced:
	// The encoding is ignored and instead the encoding used is the current value of the
	// EncodingMode property. The workaround is to make sure the EncodingMode property
	// is set to the value of the desired output encoding. This problem will be fixed
	// in v9.5.0.50.
	// 
	// Identical to the GenerateSecretKey method, except it returns the binary secret
	// key as a string encoded according to encoding, which may be "base64", "hex", "url",
	// etc. Please see the documentation for GenerateSecretKey for more information.
	// 
	const wchar_t *genEncodedSecretKey(const wchar_t *password, const wchar_t *encoding);

	// Hashes a string to a byte array that has the same number of bits as the current
	// value of the KeyLength property. For example, if KeyLength is equal to 128 bits,
	// then a 16-byte array is returned. This can be used to set the SecretKey
	// property. In order to decrypt, the SecretKey must match exactly. To use
	// "password-based" encryption, the password is passed to this method to generate a
	// binary secret key that can then be assigned to the SecretKey property.
	// 
	// IMPORTANT: If you are trying to decrypt something encrypted by another party
	// such that the other party provided you with the secret key, DO NOT use this
	// method. This method is for transforming an arbitrary-length password into a
	// binary secret key of the proper length. Please see this Chilkat blog
	// post:Getting Started with AES Decryption
	// <http://www.cknotes.com/?p=290>
	// 
	bool GenerateSecretKey(const wchar_t *password, CkByteData &outData);

	// Generates a random UUID string having standard UUID format, such as
	// "de305d54-75b4-431b-adb2-eb6b9e546014".
	// 
	// Note: This generates a "version 4 UUID" using random byte values. See RFC 4122.
	// 
	bool GenerateUuid(CkString &outStr);
	// Generates a random UUID string having standard UUID format, such as
	// "de305d54-75b4-431b-adb2-eb6b9e546014".
	// 
	// Note: This generates a "version 4 UUID" using random byte values. See RFC 4122.
	// 
	const wchar_t *generateUuid(void);

	// Generates numBytes random bytes and returns them as an encoded string. The encoding,
	// such as base64, hex, etc. is controlled by the EncodingMode property.
	bool GenRandomBytesENC(int numBytes, CkString &outStr);
	// Generates numBytes random bytes and returns them as an encoded string. The encoding,
	// such as base64, hex, etc. is controlled by the EncodingMode property.
	const wchar_t *genRandomBytesENC(int numBytes);

	// Returns the last certificate used for public-key decryption.
	// The caller is responsible for deleting the object returned by this method.
	CkCertW *GetDecryptCert(void);

	// Returns the authenticated additional data as an encoded string. The encoding
	// argument can be set to any of the following strings: "base64", "hex",
	// "quoted-printable", or "url".
	// 
	// The Aad is used when the CipherMode is "gcm" (Galois/Counter Mode), which is a
	// mode valid for symmetric ciphers that have a block size of 16 bytes, such as AES
	// or Twofish.
	// 
	bool GetEncodedAad(const wchar_t *encoding, CkString &outStr);
	// Returns the authenticated additional data as an encoded string. The encoding
	// argument can be set to any of the following strings: "base64", "hex",
	// "quoted-printable", or "url".
	// 
	// The Aad is used when the CipherMode is "gcm" (Galois/Counter Mode), which is a
	// mode valid for symmetric ciphers that have a block size of 16 bytes, such as AES
	// or Twofish.
	// 
	const wchar_t *getEncodedAad(const wchar_t *encoding);
	// Returns the authenticated additional data as an encoded string. The encoding
	// argument can be set to any of the following strings: "base64", "hex",
	// "quoted-printable", or "url".
	// 
	// The Aad is used when the CipherMode is "gcm" (Galois/Counter Mode), which is a
	// mode valid for symmetric ciphers that have a block size of 16 bytes, such as AES
	// or Twofish.
	// 
	const wchar_t *encodedAad(const wchar_t *encoding);

	// Returns the authentication tag as an encoded string. The encoding argument may be
	// set to any of the following strings: "base64", "hex", "quoted-printable", or
	// "url". The authentication tag is an output of authenticated encryption modes
	// such as GCM when encrypting. When GCM mode decrypting, the authenticate tag is
	// set by the application and is the expected result.
	// 
	// The authenticated tag plays a role when the CipherMode is "gcm" (Galois/Counter
	// Mode), which is a mode valid for symmetric block ciphers that have a block size
	// of 16 bytes, such as AES or Twofish.
	// 
	bool GetEncodedAuthTag(const wchar_t *encoding, CkString &outStr);
	// Returns the authentication tag as an encoded string. The encoding argument may be
	// set to any of the following strings: "base64", "hex", "quoted-printable", or
	// "url". The authentication tag is an output of authenticated encryption modes
	// such as GCM when encrypting. When GCM mode decrypting, the authenticate tag is
	// set by the application and is the expected result.
	// 
	// The authenticated tag plays a role when the CipherMode is "gcm" (Galois/Counter
	// Mode), which is a mode valid for symmetric block ciphers that have a block size
	// of 16 bytes, such as AES or Twofish.
	// 
	const wchar_t *getEncodedAuthTag(const wchar_t *encoding);
	// Returns the authentication tag as an encoded string. The encoding argument may be
	// set to any of the following strings: "base64", "hex", "quoted-printable", or
	// "url". The authentication tag is an output of authenticated encryption modes
	// such as GCM when encrypting. When GCM mode decrypting, the authenticate tag is
	// set by the application and is the expected result.
	// 
	// The authenticated tag plays a role when the CipherMode is "gcm" (Galois/Counter
	// Mode), which is a mode valid for symmetric block ciphers that have a block size
	// of 16 bytes, such as AES or Twofish.
	// 
	const wchar_t *encodedAuthTag(const wchar_t *encoding);

	// Returns the initialization vector as an encoded string. The encoding argument
	// can be set to any of the following strings: "base64", "hex", "quoted-printable",
	// or "url".
	bool GetEncodedIV(const wchar_t *encoding, CkString &outIV);
	// Returns the initialization vector as an encoded string. The encoding argument
	// can be set to any of the following strings: "base64", "hex", "quoted-printable",
	// or "url".
	const wchar_t *getEncodedIV(const wchar_t *encoding);
	// Returns the initialization vector as an encoded string. The encoding argument
	// can be set to any of the following strings: "base64", "hex", "quoted-printable",
	// or "url".
	const wchar_t *encodedIV(const wchar_t *encoding);

	// Returns the secret key as an encoded string. The encoding argument can be set to
	// any of the following strings: "base64", "hex", "quoted-printable", or "url".
	bool GetEncodedKey(const wchar_t *encoding, CkString &outKey);
	// Returns the secret key as an encoded string. The encoding argument can be set to
	// any of the following strings: "base64", "hex", "quoted-printable", or "url".
	const wchar_t *getEncodedKey(const wchar_t *encoding);
	// Returns the secret key as an encoded string. The encoding argument can be set to
	// any of the following strings: "base64", "hex", "quoted-printable", or "url".
	const wchar_t *encodedKey(const wchar_t *encoding);

	// Returns the password-based encryption (PBE) salt bytes as an encoded string. The
	// encoding argument can be set to any of the following strings: "base64", "hex",
	// "quoted-printable", or "url".
	bool GetEncodedSalt(const wchar_t *encoding, CkString &outStr);
	// Returns the password-based encryption (PBE) salt bytes as an encoded string. The
	// encoding argument can be set to any of the following strings: "base64", "hex",
	// "quoted-printable", or "url".
	const wchar_t *getEncodedSalt(const wchar_t *encoding);
	// Returns the password-based encryption (PBE) salt bytes as an encoded string. The
	// encoding argument can be set to any of the following strings: "base64", "hex",
	// "quoted-printable", or "url".
	const wchar_t *encodedSalt(const wchar_t *encoding);

	// Returns the last certificate used when verifying a signature. This method is
	// deprecated. Applications should instead call GetSignerCert with an index of 0.
	// The caller is responsible for deleting the object returned by this method.
	CkCertW *GetLastCert(void);

	// This method can be called after a digital signature is verified to retrieve the
	// signing time of the Nth certificate used for signing. The 1st
	// certificate/signing time is at index 0. The NumSignerCerts property contains the
	// total number of signing certificates. (Typically, a single certificate is used
	// in creating a digital signature.)
	// 
	// Note: An application should first check to see if a signing date/time is
	// available for the Nth certificate by calling the HasSignatureSigningTime method.
	// The indices for which there is no signing time available should be skipped.
	// 
	bool GetSignatureSigningTime(int index, SYSTEMTIME &outSysTime);

	// The same as GetSignatureSigningTime, except the date/time is returned in RFC822
	// string format.
	bool GetSignatureSigningTimeStr(int index, CkString &outStr);
	// The same as GetSignatureSigningTime, except the date/time is returned in RFC822
	// string format.
	const wchar_t *getSignatureSigningTimeStr(int index);
	// The same as GetSignatureSigningTime, except the date/time is returned in RFC822
	// string format.
	const wchar_t *signatureSigningTimeStr(int index);

	// Extracts the signed (authenticated) attributes for the Nth signer. In most
	// cases, a signature has only one signer, and the signerIndex should equal 0 to specify
	// the 1st (and only) signer.
	// 
	// The binary PKCS7 is passed in pkcs7Der. On success, the sbJson will contain the signed
	// attributes in JSON format.
	// 
	// Sample JSON output:
	// {
	//   "signedAttributes": [
	//     {
	//       "oid": "1.2.840.113549.1.9.3",
	//       "name": "Content Type"
	//     },
	//     {
	//       "oid": "1.2.840.113549.1.9.5",
	//       "name": "Signing Time"
	//     },
	//     {
	//       "oid": "1.2.840.113549.1.9.4",
	//       "name": "Message Digest"
	//     },
	//     {
	//       "oid": "1.2.840.113549.1.9.16.2.47",
	//       "name": "Signing Certificate V2"
	//     }
	//   ]
	// }
	// 
	bool GetSignedAttributes(int signerIndex, CkBinDataW &pkcs7Der, CkStringBuilderW &sbJson);

	// Gets the Nth certificate used for signing. This method can be called after
	// verifying a digital signature to get the signer certs. The 1st certificate is at
	// index 0. The NumSignerCerts property contains the total number of signing
	// certificates. (Typically, a single certificate is used in creating a digital
	// signature.)
	// The caller is responsible for deleting the object returned by this method.
	CkCertW *GetSignerCert(int index);

	// Returns the full certificate chain for the Nth certificate used to for signing.
	// Indexing begins at 0.
	// The caller is responsible for deleting the object returned by this method.
	CkCertChainW *GetSignerCertChain(int index);

	// Hashes the the bytes contained in bd and returns the hash as an encoded
	// string.
	// 
	// The hash algorithm is specified by the HashAlgorithm property, The encoding is
	// controlled by the EncodingMode property, which can be set to "base64", "hex",
	// "base64url", or any of the encodings listed at the link below.
	// 
	bool HashBdENC(CkBinDataW &bd, CkString &outStr);
	// Hashes the the bytes contained in bd and returns the hash as an encoded
	// string.
	// 
	// The hash algorithm is specified by the HashAlgorithm property, The encoding is
	// controlled by the EncodingMode property, which can be set to "base64", "hex",
	// "base64url", or any of the encodings listed at the link below.
	// 
	const wchar_t *hashBdENC(CkBinDataW &bd);

	// Begin hashing a byte stream. Call this method to hash the 1st chunk. Additional
	// chunks are hashed by calling HashMoreBytes 0 or more times followed by a final
	// call to HashFinal (or HashFinalENC) to retrieve the result. The hash algorithm
	// is selected by the HashAlgorithm property setting.
	bool HashBeginBytes(CkByteData &data);

	// The same as HashBeginBytes except the binary data is passed via a pointer and
	// length.
	bool HashBeginBytes2(const void *pByteData, unsigned long szByteData);

	// Begin hashing a text stream. Call this method to hash the 1st chunk. Additional
	// chunks are hashed by calling HashMoreString 0 or more times followed by a final
	// call to HashFinal (or HashFinalENC) to retrieve the result. The hash algorithm
	// is selected by the HashAlgorithm property setting.
	bool HashBeginString(const wchar_t *strData);

	// Hashes a byte array.
	// 
	// The hash algorithm is specified by the HashAlgorithm property, The encoding is
	// controlled by the EncodingMode property, which can be set to "base64", "hex",
	// "base64url", or any of the encodings listed at the link below.
	// 
	bool HashBytes(CkByteData &data, CkByteData &outData);

	// The same as HashBytes except the binary data is passed via a pointer and length.
	bool HashBytes2(const void *pByteData, unsigned long szByteData, CkByteData &outBytes);

	// Hashes a byte array and returns the hash as an encoded string.
	// 
	// The hash algorithm is specified by the HashAlgorithm property, The encoding is
	// controlled by the EncodingMode property, which can be set to "base64", "hex",
	// "base64url", or any of the encodings listed at the link below.
	// 
	bool HashBytesENC(CkByteData &data, CkString &outStr);
	// Hashes a byte array and returns the hash as an encoded string.
	// 
	// The hash algorithm is specified by the HashAlgorithm property, The encoding is
	// controlled by the EncodingMode property, which can be set to "base64", "hex",
	// "base64url", or any of the encodings listed at the link below.
	// 
	const wchar_t *hashBytesENC(CkByteData &data);

	// Hashes a file and returns the hash bytes.
	// 
	// The hash algorithm is specified by the HashAlgorithm property,
	// 
	// Any size file may be hashed because the file is hashed internally in streaming
	// mode (keeping memory usage low and constant).
	// 
	bool HashFile(const wchar_t *path, CkByteData &outBytes);

	// Creates an asynchronous task to call the HashFile method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *HashFileAsync(const wchar_t *path);

	// Hashes a file and returns the hash as an encoded string.
	// 
	// The hash algorithm is specified by the HashAlgorithm property, The encoding is
	// controlled by the EncodingMode property, which can be set to "base64", "hex",
	// "base64url", or any of the encodings listed at the link below.
	// 
	// Any size file is supported because the file is hashed internally in streaming
	// mode (keeping memory usage low and constant).
	// 
	bool HashFileENC(const wchar_t *path, CkString &outStr);
	// Hashes a file and returns the hash as an encoded string.
	// 
	// The hash algorithm is specified by the HashAlgorithm property, The encoding is
	// controlled by the EncodingMode property, which can be set to "base64", "hex",
	// "base64url", or any of the encodings listed at the link below.
	// 
	// Any size file is supported because the file is hashed internally in streaming
	// mode (keeping memory usage low and constant).
	// 
	const wchar_t *hashFileENC(const wchar_t *path);

	// Creates an asynchronous task to call the HashFileENC method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *HashFileENCAsync(const wchar_t *path);

	// Finalizes a multi-step hash computation and returns the hash bytes.
	bool HashFinal(CkByteData &outBytes);

	// Finalizes a multi-step hash computation and returns the hash bytes encoded
	// according to the EncodingMode property setting.
	bool HashFinalENC(CkString &outStr);
	// Finalizes a multi-step hash computation and returns the hash bytes encoded
	// according to the EncodingMode property setting.
	const wchar_t *hashFinalENC(void);

	// Adds more bytes to the hash currently under computation. (See HashBeginBytes)
	bool HashMoreBytes(CkByteData &data);

	// The same as HashMoreBytes except the binary data is passed via a pointer and
	// length.
	bool HashMoreBytes2(const void *pByteData, unsigned long szByteData);

	// Adds more text to the hash currently under computation. (See HashBeginString)
	bool HashMoreString(const wchar_t *strData);

	// Hashes a string and returns a binary hash. The hash algorithm is specified by
	// the HashAlgorithm property,
	// 
	// The Charset property controls the character encoding of the string that is
	// hashed. Languages such as VB.NET, C#, and Visual Basic work with Unicode
	// strings. If it is desired to hash Unicode directly (2 bytes/char) then set the
	// Charset property to "Unicode". To implicitly convert to another charset before
	// hashing, set the Charset property to the desired charset. For example, if
	// Charset is set to "iso-8859-1", the input string is first implicitly converted
	// to iso-8859-1 (1 byte per character) before hashing. The full list fo supported
	// charsets is listed in the EncryptString method description.
	// 
	// IMPORTANT: Hash algorithms hash bytes. Changing the bytes passed to a hash
	// algorithm changes the result. A character (i.e. a visible glyph) can have
	// different byte representations. The byte representation is defined by the
	// Charset. For example, 'A' in us-ascii is a single byte 0x41, whereas in utf-16
	// it is 2 bytes (0x41 0x00). The byte representation should be explicitly
	// specified, otherwise unexpected results may occur.
	// 
	bool HashString(const wchar_t *str, CkByteData &outData);

	// Hashes a string and returns the hash bytes as an encoded string.
	// 
	// The hash algorithm is specified by the HashAlgorithm property, The encoding is
	// controlled by the EncodingMode property, which can be set to "base64", "hex",
	// "base64url", or any of the encodings listed at the link below.
	// 
	// The Charset property controls the character encoding of the string that is
	// hashed. Languages such as VB.NET, C#, and Visual Basic work with Unicode
	// strings. If it is desired to hash Unicode directly (2 bytes/char) then set the
	// Charset property to "Unicode". To implicitly convert to another charset before
	// hashing, set the Charset property to the desired charset. For example, if
	// Charset is set to "iso-8859-1", the input string is first implicitly converted
	// to iso-8859-1 (1 byte per character) before hashing. The full list of supported
	// charsets is listed in the EncryptString method description.
	// 
	bool HashStringENC(const wchar_t *str, CkString &outStr);
	// Hashes a string and returns the hash bytes as an encoded string.
	// 
	// The hash algorithm is specified by the HashAlgorithm property, The encoding is
	// controlled by the EncodingMode property, which can be set to "base64", "hex",
	// "base64url", or any of the encodings listed at the link below.
	// 
	// The Charset property controls the character encoding of the string that is
	// hashed. Languages such as VB.NET, C#, and Visual Basic work with Unicode
	// strings. If it is desired to hash Unicode directly (2 bytes/char) then set the
	// Charset property to "Unicode". To implicitly convert to another charset before
	// hashing, set the Charset property to the desired charset. For example, if
	// Charset is set to "iso-8859-1", the input string is first implicitly converted
	// to iso-8859-1 (1 byte per character) before hashing. The full list of supported
	// charsets is listed in the EncryptString method description.
	// 
	const wchar_t *hashStringENC(const wchar_t *str);

	// This method can be called after a digital signature has been verified by one of
	// the Verify* methods. Returns true if a signing time for the Nth certificate is
	// available and can be retrieved by either the GetSignatureSigningTime or
	// GetSignatureSigningTimeStr methods.
	bool HasSignatureSigningTime(int index);

	// Computes a keyed-Hash Message Authentication Code (HMAC or KHMAC), which is a
	// type of message authentication code (MAC) calculated using a specific algorithm
	// involving a cryptographic hash function in combination with a secret key. As
	// with any MAC, it may be used to simultaneously verify both the data integrity
	// and the authenticity of a message. Any iterative cryptographic hash function,
	// such as MD5, SHA-1, SHA256, or any of the hash algorithms listed in the
	// HashAlgorithm property, may be used in the calculation of an HMAC; the resulting
	// MAC algorithm is termed HMAC-MD5, HMAC-SHA-1, etc. accordingly. The
	// cryptographic strength of the HMAC depends upon the cryptographic strength of
	// the underlying hash function, on the size and quality of the key and the size of
	// the hash output length in bits.
	// 
	// The secret key is set by calling one of the following methods prior to calling
	// this method: SetHmacKeyBytes, SetHmacKeyEncoded, or SetHmacKeyString.
	// 
	// The hash algorithm is specified by the HashAlgorithm property.
	// 
	// Note: If using Chilkat v9.5.0.55 or later, update your programs to use MacBytes
	// instead (with the MacAlgorithm property set to "hmac").
	// 
	bool HmacBytes(CkByteData &inBytes, CkByteData &outHmac);

	// Computes an HMAC using a secret key and hash algorithm. The result is encoded to
	// a string using the encoding (base64, hex, etc.) specified by the EncodingMode
	// property.
	// 
	// The secret key is set by calling one of the following methods prior to calling
	// this method: SetHmacKeyBytes, SetHmacKeyEncoded, or SetHmacKeyString.
	// 
	// The hash algorithm is specified by the HashAlgorithm property.
	// 
	// Note: If using Chilkat v9.5.0.55 or later, update your programs to use
	// MacBytesEnc instead (with the MacAlgorithm property set to "hmac").
	// 
	bool HmacBytesENC(CkByteData &inBytes, CkString &outEncodedHmac);
	// Computes an HMAC using a secret key and hash algorithm. The result is encoded to
	// a string using the encoding (base64, hex, etc.) specified by the EncodingMode
	// property.
	// 
	// The secret key is set by calling one of the following methods prior to calling
	// this method: SetHmacKeyBytes, SetHmacKeyEncoded, or SetHmacKeyString.
	// 
	// The hash algorithm is specified by the HashAlgorithm property.
	// 
	// Note: If using Chilkat v9.5.0.55 or later, update your programs to use
	// MacBytesEnc instead (with the MacAlgorithm property set to "hmac").
	// 
	const wchar_t *hmacBytesENC(CkByteData &inBytes);

	// Computes an HMAC using a secret key and hash algorithm.
	// 
	// The secret key is set by calling one of the following methods prior to calling
	// this method: SetHmacKeyBytes, SetHmacKeyEncoded, or SetHmacKeyString.
	// 
	// The hash algorithm is specified by the HashAlgorithm property.
	// 
	// Note: If using Chilkat v9.5.0.55 or later, update your programs to use MacString
	// instead (with the MacAlgorithm property set to "hmac").
	// 
	bool HmacString(const wchar_t *inText, CkByteData &outHmac);

	// Computes an HMAC using a secret key and hash algorithm. The result is encoded to
	// a string using the encoding (base64, hex, etc.) specified by the EncodingMode
	// property.
	// 
	// The secret key is set by calling one of the following methods prior to calling
	// this method: SetHmacKeyBytes, SetHmacKeyEncoded, or SetHmacKeyString.
	// 
	// The hash algorithm is specified by the HashAlgorithm property.
	// 
	// Note: If using Chilkat v9.5.0.55 or later, update your programs to use
	// MacStringENC instead (with the MacAlgorithm property set to "hmac").
	// 
	bool HmacStringENC(const wchar_t *inText, CkString &outEncodedHmac);
	// Computes an HMAC using a secret key and hash algorithm. The result is encoded to
	// a string using the encoding (base64, hex, etc.) specified by the EncodingMode
	// property.
	// 
	// The secret key is set by calling one of the following methods prior to calling
	// this method: SetHmacKeyBytes, SetHmacKeyEncoded, or SetHmacKeyString.
	// 
	// The hash algorithm is specified by the HashAlgorithm property.
	// 
	// Note: If using Chilkat v9.5.0.55 or later, update your programs to use
	// MacStringENC instead (with the MacAlgorithm property set to "hmac").
	// 
	const wchar_t *hmacStringENC(const wchar_t *inText);

	// Implements RFC 4226: HOTP: An HMAC-Based One-Time Password Algorithm. The
	// arguments to this method are:
	//     secret: The shared secret in an enocded representation such as base64, hex,
	//     ascii, etc.
	//     secretEnc: The encoding of the shared secret, such as "base64"
	//     counterHex: The 8-byte counter in hexidecimal format.
	//     numDigits: The number of decimal digits to return.
	//     truncOffset: Normally set this to -1 for dynamic truncation. Otherwise can be set
	//     in the range 0..15.
	//     hashAlg: Normally set to "sha1". Can be set to other hash algorithms such as
	//     "sha256", "sha512", etc.
	bool Hotp(const wchar_t *secret, const wchar_t *secretEnc, const wchar_t *counterHex, int numDigits, int truncOffset, const wchar_t *hashAlg, CkString &outStr);
	// Implements RFC 4226: HOTP: An HMAC-Based One-Time Password Algorithm. The
	// arguments to this method are:
	//     secret: The shared secret in an enocded representation such as base64, hex,
	//     ascii, etc.
	//     secretEnc: The encoding of the shared secret, such as "base64"
	//     counterHex: The 8-byte counter in hexidecimal format.
	//     numDigits: The number of decimal digits to return.
	//     truncOffset: Normally set this to -1 for dynamic truncation. Otherwise can be set
	//     in the range 0..15.
	//     hashAlg: Normally set to "sha1". Can be set to other hash algorithms such as
	//     "sha256", "sha512", etc.
	const wchar_t *hotp(const wchar_t *secret, const wchar_t *secretEnc, const wchar_t *counterHex, int numDigits, int truncOffset, const wchar_t *hashAlg);

	// Decompresses data that was compressed with CompressBytes.
	// 
	// This is a legacy method that should not be used in new development. It will not
	// be marked as deprecated or removed from future APIs because existing
	// applications may have data already compressed using CompressBytes.
	// 
	// This method expects the input to begin with an 8-byte header composed of a
	// 4-byte magic number (0xB394A7E1) and the 4-byte length of the uncompressed data.
	// 
	bool InflateBytes(CkByteData &data, CkByteData &outData);

	// The opposite of CompressBytesENC. The EncodingMode and CompressionAlgorithm
	// properties should match what was used when compressing.
	bool InflateBytesENC(const wchar_t *str, CkByteData &outData);

	// The opposite of CompressString. The Charset and CompressionAlgorithm properties
	// should match what was used when compressing.
	bool InflateString(CkByteData &data, CkString &outStr);
	// The opposite of CompressString. The Charset and CompressionAlgorithm properties
	// should match what was used when compressing.
	const wchar_t *inflateString(CkByteData &data);

	// The opposite of CompressStringENC. The Charset, EncodingMode, and
	// CompressionAlgorithm properties should match what was used when compressing.
	bool InflateStringENC(const wchar_t *str, CkString &outStr);
	// The opposite of CompressStringENC. The Charset, EncodingMode, and
	// CompressionAlgorithm properties should match what was used when compressing.
	const wchar_t *inflateStringENC(const wchar_t *str);

	// Returns true if the component is unlocked.
	bool IsUnlocked(void);

	// Provides information about what transpired in the last method called. For many
	// methods, there is no information. For some methods, details about what
	// transpired can be obtained via LastJsonData. For example, after calling a method
	// to verify a signature, the LastJsonData will return JSON with details about the
	// algorithms used for signature verification.
	// The caller is responsible for deleting the object returned by this method.
	CkJsonObjectW *LastJsonData(void);

	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTaskW &task);

	// Computes a Message Authentication Code on the bytes contained in bd, using the
	// MAC algorithm specified in the MacAlgorithm property. The result is encoded to a
	// string using the encoding (base64, hex, etc.) specified by the EncodingMode
	// property.
	bool MacBdENC(CkBinDataW &bd, CkString &outStr);
	// Computes a Message Authentication Code on the bytes contained in bd, using the
	// MAC algorithm specified in the MacAlgorithm property. The result is encoded to a
	// string using the encoding (base64, hex, etc.) specified by the EncodingMode
	// property.
	const wchar_t *macBdENC(CkBinDataW &bd);

	// Computes a Message Authentication Code using the MAC algorithm specified in the
	// MacAlgorithm property.
	bool MacBytes(CkByteData &inBytes, CkByteData &outBytes);

	// The same as MacBytes except the binary data is passed via a pointer and length.
	bool MacBytes2(const void *pByteData, unsigned long szByteData, CkByteData &outBytes);

	// Computes a Message Authentication Code using the MAC algorithm specified in the
	// MacAlgorithm property. The result is encoded to a string using the encoding
	// (base64, hex, etc.) specified by the EncodingMode property.
	bool MacBytesENC(CkByteData &inBytes, CkString &outStr);
	// Computes a Message Authentication Code using the MAC algorithm specified in the
	// MacAlgorithm property. The result is encoded to a string using the encoding
	// (base64, hex, etc.) specified by the EncodingMode property.
	const wchar_t *macBytesENC(CkByteData &inBytes);

	// Computes a Message Authentication Code using the MAC algorithm specified in the
	// MacAlgorithm property.
	bool MacString(const wchar_t *inText, CkByteData &outBytes);

	// Computes a Message Authentication Code using the MAC algorithm specified in the
	// MacAlgorithm property. The result is encoded to a string using the encoding
	// (base64, hex, etc.) specified by the EncodingMode property.
	bool MacStringENC(const wchar_t *inText, CkString &outStr);
	// Computes a Message Authentication Code using the MAC algorithm specified in the
	// MacAlgorithm property. The result is encoded to a string using the encoding
	// (base64, hex, etc.) specified by the EncodingMode property.
	const wchar_t *macStringENC(const wchar_t *inText);

	// Matches MySQL's AES_DECRYPT function. strEncryptedHex is a hex-encoded string of the AES
	// encrypted data. The return value is the original unencrypted string.
	bool MySqlAesDecrypt(const wchar_t *strEncryptedHex, const wchar_t *strPassword, CkString &outStr);
	// Matches MySQL's AES_DECRYPT function. strEncryptedHex is a hex-encoded string of the AES
	// encrypted data. The return value is the original unencrypted string.
	const wchar_t *mySqlAesDecrypt(const wchar_t *strEncryptedHex, const wchar_t *strPassword);

	// Matches MySQL's AES_ENCRYPT function. The return value is a hex-encoded string
	// of the encrypted data. The equivalent call in MySQL would look like this:
	// HEX(AES_ENCRYPT('The quick brown fox jumps over the lazy dog','password'))
	bool MySqlAesEncrypt(const wchar_t *strData, const wchar_t *strPassword, CkString &outStr);
	// Matches MySQL's AES_ENCRYPT function. The return value is a hex-encoded string
	// of the encrypted data. The equivalent call in MySQL would look like this:
	// HEX(AES_ENCRYPT('The quick brown fox jumps over the lazy dog','password'))
	const wchar_t *mySqlAesEncrypt(const wchar_t *strData, const wchar_t *strPassword);

	// In-place signs the contents of bd. The contents of bd is replaced with the
	// PKCS7/CMS format signature that embeds the data that was signed.
	bool OpaqueSignBd(CkBinDataW &bd);

	// Creates an asynchronous task to call the OpaqueSignBd method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *OpaqueSignBdAsync(CkBinDataW &bd);

	// Digitally signs a byte array and returns a PKCS7/CMS format signature. This is a
	// signature that contains both the original data as well as the signature. A
	// certificate must be set by calling SetSigningCert prior to calling this method.
	bool OpaqueSignBytes(CkByteData &data, CkByteData &outData);

	// Creates an asynchronous task to call the OpaqueSignBytes method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *OpaqueSignBytesAsync(CkByteData &data);

	// The same as OpaqueSignBytes except the binary data is passed via a pointer and
	// length.
	bool OpaqueSignBytes2(const void *pByteData, unsigned long szByteData, CkByteData &outBytes);

	// Digitally signs a byte array and returns a PKCS7/CMS format signature in encoded
	// string format (such as Base64 or hex). This is a signature that contains both
	// the original data as well as the signature. A certificate must be set by calling
	// SetSigningCert prior to calling this method. The EncodingMode property controls
	// the output encoding, which can be "Base64", "QP","Hex", etc. (See the
	// EncodingMode property.)
	bool OpaqueSignBytesENC(CkByteData &data, CkString &outStr);
	// Digitally signs a byte array and returns a PKCS7/CMS format signature in encoded
	// string format (such as Base64 or hex). This is a signature that contains both
	// the original data as well as the signature. A certificate must be set by calling
	// SetSigningCert prior to calling this method. The EncodingMode property controls
	// the output encoding, which can be "Base64", "QP","Hex", etc. (See the
	// EncodingMode property.)
	const wchar_t *opaqueSignBytesENC(CkByteData &data);

	// Creates an asynchronous task to call the OpaqueSignBytesENC method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *OpaqueSignBytesENCAsync(CkByteData &data);

	// Digitally signs a string and returns a PKCS7/CMS format signature. This is a
	// signature that contains both the original data as well as the signature. A
	// certificate must be set by calling SetSigningCert prior to calling this method.
	// The Charset property controls the character encoding of the string that is
	// signed. (Languages such as VB.NET, C#, and Visual Basic work with Unicode
	// strings.) To sign Unicode data (2 bytes per char), set the Charset property to
	// "Unicode". To implicitly convert the string to a mutlibyte charset such as
	// "iso-8859-1", "Shift_JIS", "utf-8", or something else, then set the Charset
	// property to the name of the charset before signing. The complete list of
	// charsets is listed in the EncryptString method description.
	bool OpaqueSignString(const wchar_t *str, CkByteData &outData);

	// Creates an asynchronous task to call the OpaqueSignString method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *OpaqueSignStringAsync(const wchar_t *str);

	// Digitally signs a string and returns a PKCS7/CMS format signature in encoded
	// string format (such as base64 or hex). This is a signature that contains both
	// the original data as well as the signature. A certificate must be set by calling
	// SetSigningCert prior to calling this method. The Charset property controls the
	// character encoding of the string that is signed. (Languages such as VB.NET, C#,
	// and Visual Basic work with Unicode strings.) To sign Unicode data (2 bytes per
	// char), set the Charset property to "Unicode". To implicitly convert the string
	// to a mutlibyte charset such as "iso-8859-1", "Shift_JIS", "utf-8", or something
	// else, then set the Charset property to the name of the charset before signing.
	// The complete list of charsets is listed in the EncryptString method description.
	// 
	// The EncodingMode property controls the output encoding, which can be "Base64",
	// "QP","Hex", etc. (See the EncodingMode property.)
	// 
	bool OpaqueSignStringENC(const wchar_t *str, CkString &outStr);
	// Digitally signs a string and returns a PKCS7/CMS format signature in encoded
	// string format (such as base64 or hex). This is a signature that contains both
	// the original data as well as the signature. A certificate must be set by calling
	// SetSigningCert prior to calling this method. The Charset property controls the
	// character encoding of the string that is signed. (Languages such as VB.NET, C#,
	// and Visual Basic work with Unicode strings.) To sign Unicode data (2 bytes per
	// char), set the Charset property to "Unicode". To implicitly convert the string
	// to a mutlibyte charset such as "iso-8859-1", "Shift_JIS", "utf-8", or something
	// else, then set the Charset property to the name of the charset before signing.
	// The complete list of charsets is listed in the EncryptString method description.
	// 
	// The EncodingMode property controls the output encoding, which can be "Base64",
	// "QP","Hex", etc. (See the EncodingMode property.)
	// 
	const wchar_t *opaqueSignStringENC(const wchar_t *str);

	// Creates an asynchronous task to call the OpaqueSignStringENC method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *OpaqueSignStringENCAsync(const wchar_t *str);

	// In-place verifies and unwraps the PKCS7/CMS contents of bd. If the signature
	// is verified, the contents of bd will be replaced with the original data, and
	// the method returns true. If the signature is not verified, then the contents
	// of bd remain unchanged and the method returns false.
	// 
	// Note: The signer certificates can be retrieved after any Verify* method call by
	// using the NumSignerCerts property and the GetSignerCert method.
	// 
	bool OpaqueVerifyBd(CkBinDataW &bd);

	// Verifies an opaque signature and returns the original data. If the signature
	// verification fails, the returned data will be 0 bytes in length.
	// 
	// Note: The signer certificates can be retrieved after any Verify* method call by
	// using the NumSignerCerts property and the GetSignerCert method.
	// 
	bool OpaqueVerifyBytes(CkByteData &p7s, CkByteData &outOriginal);

	// The same as OpaqueVerifyBytes except the binary data is passed via a pointer and
	// length.
	bool OpaqueVerifyBytes2(const void *pByteData, unsigned long szByteData, CkByteData &outBytes);

	// Verifies an opaque signature (encoded in string form) and returns the original
	// data. If the signature verification fails, the returned data will be 0 bytes in
	// length.
	// 
	// Note: The signer certificates can be retrieved after any Verify* method call by
	// using the NumSignerCerts property and the GetSignerCert method.
	// 
	bool OpaqueVerifyBytesENC(const wchar_t *p7s, CkByteData &outOriginal);

	// Verifies an opaque signature and returns the original string. If the signature
	// verification fails, the returned string will be 0 characters in length.
	// 
	// Note: The signer certificates can be retrieved after any Verify* method call by
	// using the NumSignerCerts property and the GetSignerCert method.
	// 
	bool OpaqueVerifyString(CkByteData &p7s, CkString &outOriginal);
	// Verifies an opaque signature and returns the original string. If the signature
	// verification fails, the returned string will be 0 characters in length.
	// 
	// Note: The signer certificates can be retrieved after any Verify* method call by
	// using the NumSignerCerts property and the GetSignerCert method.
	// 
	const wchar_t *opaqueVerifyString(CkByteData &p7s);

	// Verifies an opaque signature (encoded in string form) and returns the original
	// data string. If the signature verification fails, the returned string will be 0
	// characters in length.
	// 
	// Note: The signer certificates can be retrieved after any Verify* method call by
	// using the NumSignerCerts property and the GetSignerCert method.
	// 
	bool OpaqueVerifyStringENC(const wchar_t *p7s, CkString &outOriginal);
	// Verifies an opaque signature (encoded in string form) and returns the original
	// data string. If the signature verification fails, the returned string will be 0
	// characters in length.
	// 
	// Note: The signer certificates can be retrieved after any Verify* method call by
	// using the NumSignerCerts property and the GetSignerCert method.
	// 
	const wchar_t *opaqueVerifyStringENC(const wchar_t *p7s);

	// Implements the PBKDF1 algorithm (Password Based Key Derivation Function #1). The
	// password is converted to the character encoding represented by charset before being
	// passed (internally) to the key derivation function. The hashAlg may be "md5",
	// "sha1", "md2", etc. The salt should be random data at least 8 bytes (64 bits) in
	// length. (The GenRandomBytesENC method is good for generating a random salt
	// value.) The iterationCount should be no less than 1000. The length (in bits) of the
	// derived key output by this method is controlled by outputKeyBitLen. The encoding argument may
	// be "base64", "hex", etc. It controls the encoding of the output, and the
	// expected encoding of the salt. The derived key is returned.
	// 
	// Note: Starting in version 9.5.0.47, if the charset is set to one of the keywords
	// "hex" or "base64", then the password will be considered binary data that is hex
	// or base64 encoded. The bytes will be decoded and used directly as a binary
	// password.
	// 
	bool Pbkdf1(const wchar_t *password, const wchar_t *charset, const wchar_t *hashAlg, const wchar_t *salt, int iterationCount, int outputKeyBitLen, const wchar_t *encoding, CkString &outStr);
	// Implements the PBKDF1 algorithm (Password Based Key Derivation Function #1). The
	// password is converted to the character encoding represented by charset before being
	// passed (internally) to the key derivation function. The hashAlg may be "md5",
	// "sha1", "md2", etc. The salt should be random data at least 8 bytes (64 bits) in
	// length. (The GenRandomBytesENC method is good for generating a random salt
	// value.) The iterationCount should be no less than 1000. The length (in bits) of the
	// derived key output by this method is controlled by outputKeyBitLen. The encoding argument may
	// be "base64", "hex", etc. It controls the encoding of the output, and the
	// expected encoding of the salt. The derived key is returned.
	// 
	// Note: Starting in version 9.5.0.47, if the charset is set to one of the keywords
	// "hex" or "base64", then the password will be considered binary data that is hex
	// or base64 encoded. The bytes will be decoded and used directly as a binary
	// password.
	// 
	const wchar_t *pbkdf1(const wchar_t *password, const wchar_t *charset, const wchar_t *hashAlg, const wchar_t *salt, int iterationCount, int outputKeyBitLen, const wchar_t *encoding);

	// Implements the PBKDF2 algorithm (Password Based Key Derivation Function #2). The
	// password is converted to the character encoding represented by charset before being
	// passed (internally) to the key derivation function. The hashAlg may be "sha256",
	// "sha384", "sha512", "md5", "sha1", "md2", or any hash algorithm listed in the
	// HashAlgorithm property. The salt should be random data at least 8 bytes (64
	// bits) in length. (The GenRandomBytesENC method is good for generating a random
	// salt value.) The iterationCount should be no less than 1000. The length (in bits) of the
	// derived key output by this method is controlled by outputKeyBitLen. The encoding argument may
	// be "base64", "hex", etc. It controls the encoding of the output, and the
	// expected encoding of the salt. The derived key is returned.
	// 
	// Note: The PBKDF2 function (internally) utilizes a PRF that is a pseudorandom
	// function that is a keyed HMAC. The hash algorithm specified by hashAlg determines
	// this PRF. If hashAlg is "SHA256", then HMAC-SHA256 is used for the PRF. Likewise,
	// if the hash function is "SHA1", then HMAC-SHA1 is used. HMAC can be used with
	// any hash algorithm.
	// 
	// Note: Starting in version 9.5.0.47, if the charset is set to one of the keywords
	// "hex" or "base64", then the password will be considered binary data that is hex
	// or base64 encoded. The bytes will be decoded and used directly as a binary
	// password.
	// 
	bool Pbkdf2(const wchar_t *password, const wchar_t *charset, const wchar_t *hashAlg, const wchar_t *salt, int iterationCount, int outputKeyBitLen, const wchar_t *encoding, CkString &outStr);
	// Implements the PBKDF2 algorithm (Password Based Key Derivation Function #2). The
	// password is converted to the character encoding represented by charset before being
	// passed (internally) to the key derivation function. The hashAlg may be "sha256",
	// "sha384", "sha512", "md5", "sha1", "md2", or any hash algorithm listed in the
	// HashAlgorithm property. The salt should be random data at least 8 bytes (64
	// bits) in length. (The GenRandomBytesENC method is good for generating a random
	// salt value.) The iterationCount should be no less than 1000. The length (in bits) of the
	// derived key output by this method is controlled by outputKeyBitLen. The encoding argument may
	// be "base64", "hex", etc. It controls the encoding of the output, and the
	// expected encoding of the salt. The derived key is returned.
	// 
	// Note: The PBKDF2 function (internally) utilizes a PRF that is a pseudorandom
	// function that is a keyed HMAC. The hash algorithm specified by hashAlg determines
	// this PRF. If hashAlg is "SHA256", then HMAC-SHA256 is used for the PRF. Likewise,
	// if the hash function is "SHA1", then HMAC-SHA1 is used. HMAC can be used with
	// any hash algorithm.
	// 
	// Note: Starting in version 9.5.0.47, if the charset is set to one of the keywords
	// "hex" or "base64", then the password will be considered binary data that is hex
	// or base64 encoded. The bytes will be decoded and used directly as a binary
	// password.
	// 
	const wchar_t *pbkdf2(const wchar_t *password, const wchar_t *charset, const wchar_t *hashAlg, const wchar_t *salt, int iterationCount, int outputKeyBitLen, const wchar_t *encoding);

	// Extracts the signature digest contained within a PKCS7 signature. The PKCS7 is
	// passed in the encoding determined by the EncodingMode property (such as base64,
	// hex, etc.) an the signature digest is returned in the same encoding.
	bool Pkcs7ExtractDigest(int signerIndex, const wchar_t *pkcs7, CkString &outStr);
	// Extracts the signature digest contained within a PKCS7 signature. The PKCS7 is
	// passed in the encoding determined by the EncodingMode property (such as base64,
	// hex, etc.) an the signature digest is returned in the same encoding.
	const wchar_t *pkcs7ExtractDigest(int signerIndex, const wchar_t *pkcs7);

	// Sets the initialization vector to a random value.
	void RandomizeIV(void);

	// Sets the secret key to a random value.
	void RandomizeKey(void);

	// Convenience method to read an entire file and return as a byte array.
	bool ReadFile(const wchar_t *filename, CkByteData &outBytes);

	// Provides a means for converting from one encoding to another (such as base64 to
	// hex). This is helpful for programming environments where byte arrays are a real
	// pain-in-the-***. The fromEncoding and toEncoding may be (case-insensitive) "Base64",
	// "modBase64", "Base32", "Base58", "UU", "QP" (for quoted-printable), "URL" (for
	// url-encoding), "Hex", "Q", "B", "url_oauth", "url_rfc1738", "url_rfc2396", and
	// "url_rfc3986".
	bool ReEncode(const wchar_t *encodedData, const wchar_t *fromEncoding, const wchar_t *toEncoding, CkString &outStr);
	// Provides a means for converting from one encoding to another (such as base64 to
	// hex). This is helpful for programming environments where byte arrays are a real
	// pain-in-the-***. The fromEncoding and toEncoding may be (case-insensitive) "Base64",
	// "modBase64", "Base32", "Base58", "UU", "QP" (for quoted-printable), "URL" (for
	// url-encoding), "Hex", "Q", "B", "url_oauth", "url_rfc1738", "url_rfc2396", and
	// "url_rfc3986".
	const wchar_t *reEncode(const wchar_t *encodedData, const wchar_t *fromEncoding, const wchar_t *toEncoding);

	// Sets the digital certificate to be used for decryption when the CryptAlgorithm
	// property is set to "PKI". A private key is required for decryption. Because this
	// method only specifies the certificate, a prerequisite is that the certificate w/
	// private key must have been pre-installed on the computer. Private keys are
	// stored in the Windows Protected Store (either a user account specific store, or
	// the system-wide store). The Chilkat component will automatically locate and find
	// the certificate's corresponding private key from the protected store when
	// decrypting.
	bool SetDecryptCert(CkCertW &cert);

	// Sets the digital certificate to be used for decryption when the CryptAlgorithm
	// property is set to "PKI". The private key is supplied in the 2nd argument to
	// this method, so there is no requirement that the certificate be pre-installed on
	// a computer before decrypting (if this method is called).
	bool SetDecryptCert2(CkCertW &cert, CkPrivateKeyW &key);

	// Sets the authenticated additional data from an encoded string. The authenticated
	// additional data (AAD), if any, is used in authenticated encryption modes such as
	// GCM. The aadStr argument can be set to any of the following strings: "base64",
	// "hex", "quoted-printable", "ascii", or "url".
	// 
	// The Aad is used when the CipherMode is "gcm" (Galois/Counter Mode), which is a
	// mode valid for symmetric ciphers that have a block size of 16 bytes, such as AES
	// or Twofish.
	// 
	bool SetEncodedAad(const wchar_t *aadStr, const wchar_t *encoding);

	// Sets the expected authenticated tag from an encoded string. The authenticated
	// tag is used in authenticated encryption modes such as GCM. An application would
	// set the expected authenticated tag prior to decrypting. The authTagStr argument can be
	// set to any of the following strings: "base64", "hex", "quoted-printable",
	// "ascii", or "url".
	// 
	// The authenticated tag plays a role when the CipherMode is "gcm" (Galois/Counter
	// Mode), which is a mode valid for symmetric block ciphers that have a block size
	// of 16 bytes, such as AES or Twofish.
	// 
	bool SetEncodedAuthTag(const wchar_t *authTagStr, const wchar_t *encoding);

	// Sets the initialization vector from an encoded string. The encoding argument can
	// be set to any of the following strings: "base64", "hex", "quoted-printable",
	// "ascii", or "url".
	void SetEncodedIV(const wchar_t *ivStr, const wchar_t *encoding);

	// Sets the secret key from an encoded string. The encoding argument can be set to
	// any of the following strings: "base64", "hex", "quoted-printable", "ascii", or
	// "url".
	void SetEncodedKey(const wchar_t *keyStr, const wchar_t *encoding);

	// Sets the password-based encryption (PBE) salt bytes from an encoded string. The
	// encoding argument can be set to any of the following strings: "base64", "hex",
	// "quoted-printable", "ascii", or "url".
	void SetEncodedSalt(const wchar_t *saltStr, const wchar_t *encoding);

	// Tells the encryption library to use a specific digital certificate for
	// public-key encryption. To encrypt with multiple certificates, call
	// AddEncryptCert once for each certificate. (Calling this method is the equivalent
	// of calling ClearEncryptCerts followed by AddEncryptCert.)
	bool SetEncryptCert(CkCertW &cert);

	// Sets the HMAC key to be used for one of the HMAC methods.
	// 
	// Note: If using Chilkat v9.5.0.55 or later, update your programs to use
	// SetMacKeyBytes instead.
	// 
	void SetHmacKeyBytes(CkByteData &keyBytes);

	// Sets the secret key to be used for one of the HMAC methods. The encoding can be set
	// to any of the following strings: "base64", "hex", "quoted-printable", or "url".
	// 
	// Note: If using Chilkat v9.5.0.55 or later, update your programs to use
	// SetMacKeyEncoded instead.
	// 
	void SetHmacKeyEncoded(const wchar_t *key, const wchar_t *encoding);

	// Sets the MAC key to be used for one of the HMAC methods.
	// 
	// Note: If using Chilkat v9.5.0.55 or later, update your programs to use
	// SetMacKeyString instead.
	// 
	void SetHmacKeyString(const wchar_t *key);

	// Sets the initialization vector for a symmetric encryption algorithm (such as
	// AES, BlowFish, TwoFish, DES, etc.). IV's are used in CBC mode
	// (Cipher-Block-Chaining), but are not used in ECB mode (Electronic Cookbook). The
	// length of the IV should equal the block size of the algorithm. (It is not equal
	// to the key length). For AES and TwoFish, the block size (and thus IV size) is
	// always 16 bytes. For Blowfish it's 8 bytes. For DES and 3DES it's 8 bytes.
	void SetIV(const void *pByteData, unsigned long szByteData);

	// Sets the MAC key to be used for one of the Mac methods.
	bool SetMacKeyBytes(CkByteData &keyBytes);

	// Sets the MAC key to be used for one of the Mac methods. The encoding can be set to
	// any of the following strings: "base64", "hex", "quoted-printable", or "url".
	bool SetMacKeyEncoded(const wchar_t *key, const wchar_t *encoding);

	// Sets the MAC key to be used for one of the Mac methods.
	bool SetMacKeyString(const wchar_t *key);

	// Sets the value of the SecretKey property.
	void SetSecretKey(const void *pByteData, unsigned long szByteData);

	// Accepts a password string and (internally) generates a binary secret key of the
	// appropriate bit length and sets the SecretKey property. This method should only
	// be used if you are using Chilkat for both encryption and decryption because the
	// password-to-secret-key algorithm would need to be identical for the decryption
	// to match the encryption.
	// 
	// There is no minimum or maximum password length. The password string is
	// transformed to a binary secret key by computing the MD5 digest (of the utf-8
	// password) to obtain 16 bytes. If the KeyLength is greater than 16 bytes, then
	// the MD5 digest of the Base64 encoding of the utf-8 password is added. A max of
	// 32 bytes of key material is generated, and this is truncated to the actual
	// KeyLength required. The example below shows how to manually duplicate the
	// computation.
	// 
	void SetSecretKeyViaPassword(const wchar_t *password);

	// Specifies a certificate to be used when creating PKCS7 digital signatures.
	// Signing requires both a certificate and private key. In this case, the private
	// key is implicitly specified if the certificate originated from a PFX that
	// contains the corresponding private key, or if on a Windows-based computer where
	// the certificate and corresponding private key are pre-installed. (If a PFX file
	// is used, it is provided via the AddPfxSourceFile or AddPfxSourceData methods.)
	bool SetSigningCert(CkCertW &cert);

	// Specifies a digital certificate and private key to be used for creating PKCS7
	// digital signatures.
	bool SetSigningCert2(CkCertW &cert, CkPrivateKeyW &privateKey);

	// Sets the timestamp authority (TSA) options for cases where a CAdES-T signature
	// is to be created. The http is used to send the requests, and it allows for
	// connection related settings and timeouts to be set. For example, if HTTP or
	// SOCKS proxies are required, these features can be specified on the http.
	void SetTsaHttpObj(CkHttpW &http);

	// Sets the digital certificate to be used in verifying a signature.
	bool SetVerifyCert(CkCertW &cert);

	// Digitally signs the contents of dataToSign and returns the detached digital signature
	// in an encoded string (according to the EncodingMode property setting).
	bool SignBdENC(CkBinDataW &dataToSign, CkString &outStr);
	// Digitally signs the contents of dataToSign and returns the detached digital signature
	// in an encoded string (according to the EncodingMode property setting).
	const wchar_t *signBdENC(CkBinDataW &dataToSign);

	// Creates an asynchronous task to call the SignBdENC method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SignBdENCAsync(CkBinDataW &dataToSign);

	// Digitally signs a byte array and returns the detached digital signature. A
	// certificate must be set by calling SetSigningCert prior to calling this method.
	bool SignBytes(CkByteData &data, CkByteData &outData);

	// Creates an asynchronous task to call the SignBytes method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SignBytesAsync(CkByteData &data);

	// The same as SignBytes except the binary data is passed via a pointer and length.
	bool SignBytes2(const void *pByteData, unsigned long szByteData, CkByteData &outBytes);

	// Digitally signs a byte array and returns the detached digital signature encoded
	// as a printable string. A certificate must be set by calling SetSigningCert prior
	// to calling this method. The EncodingMode property controls the output encoding,
	// which can be "Base64", "QP", or "Hex".
	bool SignBytesENC(CkByteData &data, CkString &outStr);
	// Digitally signs a byte array and returns the detached digital signature encoded
	// as a printable string. A certificate must be set by calling SetSigningCert prior
	// to calling this method. The EncodingMode property controls the output encoding,
	// which can be "Base64", "QP", or "Hex".
	const wchar_t *signBytesENC(CkByteData &data);

	// Creates an asynchronous task to call the SignBytesENC method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SignBytesENCAsync(CkByteData &data);

	// Digitally signs a the contents of sb and returns the PKCS7 detached digital
	// signature as an encoded string according to the EncodingMode property setting.
	bool SignSbENC(CkStringBuilderW &sb, CkString &outStr);
	// Digitally signs a the contents of sb and returns the PKCS7 detached digital
	// signature as an encoded string according to the EncodingMode property setting.
	const wchar_t *signSbENC(CkStringBuilderW &sb);

	// Creates an asynchronous task to call the SignSbENC method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SignSbENCAsync(CkStringBuilderW &sb);

	// Digitally signs a string and returns the detached digital signature. A
	// certificate must be set by calling SetSigningCert prior to calling this method.
	// The Charset property controls the character encoding of the string that is
	// signed. (Languages such as VB.NET, C#, and Visual Basic work with Unicode
	// strings.) To sign Unicode data (2 bytes per char), set the Charset property to
	// "Unicode". To implicitly convert the string to a mutlibyte charset such as
	// "iso-8859-1", "Shift_JIS", "utf-8", or something else, then set the Charset
	// property to the name of the charset before signing. The complete list of
	// charsets is listed in the EncryptString method description.
	bool SignString(const wchar_t *str, CkByteData &outData);

	// Creates an asynchronous task to call the SignString method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SignStringAsync(const wchar_t *str);

	// Digitally signs a string and returns the PKCS7 detached digital signature as an
	// encoded string. A certificate must be set by calling SetSigningCert prior to
	// calling this method. The Charset property controls the character encoding of the
	// string that is signed. (Languages such as VB.NET, C#, and Visual Basic work with
	// Unicode strings.) To sign Unicode data (2 bytes per char), set the Charset
	// property to "Unicode". To implicitly convert the string to a mutlibyte charset
	// such as "iso-8859-1", "Shift_JIS", "utf-8", or something else, then set the
	// Charset property to the name of the charset before signing. The complete list of
	// charsets is listed in the EncryptString method description.
	// 
	// The encoding of the output string is controlled by the EncodingMode property,
	// which can be set to "Base64", "QP", or "Hex".
	// 
	bool SignStringENC(const wchar_t *str, CkString &outStr);
	// Digitally signs a string and returns the PKCS7 detached digital signature as an
	// encoded string. A certificate must be set by calling SetSigningCert prior to
	// calling this method. The Charset property controls the character encoding of the
	// string that is signed. (Languages such as VB.NET, C#, and Visual Basic work with
	// Unicode strings.) To sign Unicode data (2 bytes per char), set the Charset
	// property to "Unicode". To implicitly convert the string to a mutlibyte charset
	// such as "iso-8859-1", "Shift_JIS", "utf-8", or something else, then set the
	// Charset property to the name of the charset before signing. The complete list of
	// charsets is listed in the EncryptString method description.
	// 
	// The encoding of the output string is controlled by the EncodingMode property,
	// which can be set to "Base64", "QP", or "Hex".
	// 
	const wchar_t *signStringENC(const wchar_t *str);

	// Creates an asynchronous task to call the SignStringENC method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SignStringENCAsync(const wchar_t *str);

	// Convert a string to a byte array where the characters are encoded according to
	// the charset specified.
	bool StringToBytes(const wchar_t *inStr, const wchar_t *charset, CkByteData &outBytes);

	// Implements RFC 6238: TOTP: Time-Based One-Time Password Algorithm. The arguments
	// to this method are:
	//     secret: The shared secret in an enocded representation such as base64, hex,
	//     ascii, etc.
	//     secretEnc: The encoding of the shared secret, such as "base64"
	//     t0: The Unix time to start counting time steps. It is a number in decimal
	//     string form. A Unix time is the number of seconds elapsed since midnight UTC of
	//     January 1, 1970. "0" is a typical value used for this argument.
	//     tNow: The current Unix time in decimal string form. To use the current
	//     system date/time, pass an empty string for this argument.
	//     tStep: The time step in seconds. A typical value is 30. Note: Both client and
	//     server must pre-agree on the secret, the t0, and the tStep.
	//     numDigits: The number of decimal digits to return.
	//     truncOffset: Normally set this to -1 for dynamic truncation. Otherwise can be set
	//     in the range 0..15.
	//     hashAlg: Normally set to "sha1". Can be set to other hash algorithms such as
	//     "sha256", "sha512", etc.
	bool Totp(const wchar_t *secret, const wchar_t *secretEnc, const wchar_t *t0, const wchar_t *tNow, int tStep, int numDigits, int truncOffset, const wchar_t *hashAlg, CkString &outStr);
	// Implements RFC 6238: TOTP: Time-Based One-Time Password Algorithm. The arguments
	// to this method are:
	//     secret: The shared secret in an enocded representation such as base64, hex,
	//     ascii, etc.
	//     secretEnc: The encoding of the shared secret, such as "base64"
	//     t0: The Unix time to start counting time steps. It is a number in decimal
	//     string form. A Unix time is the number of seconds elapsed since midnight UTC of
	//     January 1, 1970. "0" is a typical value used for this argument.
	//     tNow: The current Unix time in decimal string form. To use the current
	//     system date/time, pass an empty string for this argument.
	//     tStep: The time step in seconds. A typical value is 30. Note: Both client and
	//     server must pre-agree on the secret, the t0, and the tStep.
	//     numDigits: The number of decimal digits to return.
	//     truncOffset: Normally set this to -1 for dynamic truncation. Otherwise can be set
	//     in the range 0..15.
	//     hashAlg: Normally set to "sha1". Can be set to other hash algorithms such as
	//     "sha256", "sha512", etc.
	const wchar_t *totp(const wchar_t *secret, const wchar_t *secretEnc, const wchar_t *t0, const wchar_t *tNow, int tStep, int numDigits, int truncOffset, const wchar_t *hashAlg);

	// Trim a string ending with a specific substring until the string no longer ends
	// with that substring.
	bool TrimEndingWith(const wchar_t *inStr, const wchar_t *ending, CkString &outStr);
	// Trim a string ending with a specific substring until the string no longer ends
	// with that substring.
	const wchar_t *trimEndingWith(const wchar_t *inStr, const wchar_t *ending);

	// Unlocks the component. This must be called once prior to calling any other
	// method.
	bool UnlockComponent(const wchar_t *unlockCode);

	// Adds an XML certificate vault to the object's internal list of sources to be
	// searched for certificates and private keys when encrypting/decrypting or
	// signing/verifying. Unlike the AddPfxSourceData and AddPfxSourceFile methods,
	// only a single XML certificate vault can be used. If UseCertVault is called
	// multiple times, only the last certificate vault will be used, as each call to
	// UseCertVault will replace the certificate vault provided in previous calls.
	bool UseCertVault(CkXmlCertVaultW &vault);

	// Verifies a digital signature against the original data contained in data.
	// Returns true if the signature is verified.
	// 
	// Note: The signer certificates can be retrieved after any Verify* method call by
	// using the NumSignerCerts property and the GetSignerCert method.
	// 
	bool VerifyBdENC(CkBinDataW &data, const wchar_t *encodedSig);

	// Verifies a byte array against a digital signature and returns true if the byte
	// array is unaltered.
	// 
	// Note: The signer certificates can be retrieved after any Verify* method call by
	// using the NumSignerCerts property and the GetSignerCert method.
	// 
	bool VerifyBytes(CkByteData &data, CkByteData &sig);

	// Verifies a byte array against a string-encoded digital signature and returns
	// true if the byte array is unaltered. This method can be used to verify a
	// signature produced by SignBytesENC. The EncodingMode property must be set prior
	// to calling to match the encoding of the digital signature string ("Base64",
	// "QP", or "Hex").
	// 
	// Note: The signer certificates can be retrieved after any Verify* method call by
	// using the NumSignerCerts property and the GetSignerCert method.
	// 
	bool VerifyBytesENC(CkByteData &data, const wchar_t *encodedSig);

	// Verifies a .p7s (PKCS #7 Signature) against the original file (or exact copy of
	// it). If the inFilename has not been modified, the return value is true, otherwise it
	// is false. This method is equivalent to VerifyP7S.
	// 
	// Note: The signer certificates can be retrieved after any Verify* method call by
	// using the NumSignerCerts property and the GetSignerCert method.
	// 
	bool VerifyDetachedSignature(const wchar_t *inFilename, const wchar_t *p7sFilename);

	// Verifies a .p7m file and extracts the original file from the .p7m. Returns
	// true if the signature is valid and the contents are unchanged. Otherwise
	// returns false.
	// 
	// Note: The signer certificates can be retrieved after any Verify* method call by
	// using the NumSignerCerts property and the GetSignerCert method.
	// 
	bool VerifyP7M(const wchar_t *p7mPath, const wchar_t *destPath);

	// Verifies a .p7s (PKCS #7 Signature) against the original file (or exact copy of
	// it). If the inFilename has not been modified, the return value is true, otherwise it
	// is false.
	// 
	// Note: The signer certificates can be retrieved after any Verify* method call by
	// using the NumSignerCerts property and the GetSignerCert method.
	// 
	bool VerifyP7S(const wchar_t *inFilename, const wchar_t *p7sFilename);

	// Verifies a digital signature against the original data contained in sb.
	// Returns true if the signature is verified.
	// 
	// Note: The signer certificates can be retrieved after any Verify* method call by
	// using the NumSignerCerts property and the GetSignerCert method.
	// 
	bool VerifySbENC(CkStringBuilderW &sb, const wchar_t *encodedSig);

	// Verifies a string against a binary digital signature and returns true if the
	// string is unaltered. This method can be used to verify a signature produced by
	// SignString. The Charset property must be set to the charset that was used when
	// creating the signature.
	// 
	// Note: The signer certificates can be retrieved after any Verify* method call by
	// using the NumSignerCerts property and the GetSignerCert method.
	// 
	bool VerifyString(const wchar_t *str, CkByteData &sig);

	// Verifies a string against a string-encoded digital signature and returns true if
	// the string is unaltered. This method can be used to verify a signature produced
	// by SignStringENC. The Charset and EncodingMode properties must be set to the
	// same values that were used when creating the signature.
	// 
	// Note: The signer certificates can be retrieved after any Verify* method call by
	// using the NumSignerCerts property and the GetSignerCert method.
	// 
	bool VerifyStringENC(const wchar_t *str, const wchar_t *encodedSig);

	// Convenience method to write an entire byte array to a file.
	bool WriteFile(const wchar_t *filename, CkByteData &fileData);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
