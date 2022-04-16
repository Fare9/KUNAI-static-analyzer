// CkScMinidriver.h: interface for the CkScMinidriver class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkScMinidriver_H
#define _CkScMinidriver_H
	
#include "chilkatDefs.h"
// This is a Windows-only class.  On non-Windows systems, the methods do nothing...
#include "CkString.h"
#include "CkMultiByteBase.h"

class CkCert;
class CkStringTable;
class CkJsonObject;
class CkPublicKey;
class CkPrivateKey;
class CkBinData;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkScMinidriver
class CK_VISIBLE_PUBLIC CkScMinidriver  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkScMinidriver(const CkScMinidriver &);
	CkScMinidriver &operator=(const CkScMinidriver &);

    public:
	CkScMinidriver(void);
	virtual ~CkScMinidriver(void);

	static CkScMinidriver *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The ATR of the card in the reader. This property is set by the AquireContext
	// method.
	void get_Atr(CkString &str);
	// The ATR of the card in the reader. This property is set by the AquireContext
	// method.
	const char *atr(void);

	// The name of the card in the reader. This property is set by the AquireContext
	// method.
	void get_CardName(CkString &str);
	// The name of the card in the reader. This property is set by the AquireContext
	// method.
	const char *cardName(void);

	// The maximum number of key containers available. The 1st key container is at
	// index 0. Each key container can potentially contain one signature key, and one
	// key exchange key.
	int get_MaxContainers(void);

	// If an RSA key is used for signing, this is the hash algorithm to used in
	// conjunction with the padding scheme. It can be "SHA1", "SHA256", "SHA384", or
	// "SHA512". The default is "SHA256".
	void get_RsaPaddingHash(CkString &str);
	// If an RSA key is used for signing, this is the hash algorithm to used in
	// conjunction with the padding scheme. It can be "SHA1", "SHA256", "SHA384", or
	// "SHA512". The default is "SHA256".
	const char *rsaPaddingHash(void);
	// If an RSA key is used for signing, this is the hash algorithm to used in
	// conjunction with the padding scheme. It can be "SHA1", "SHA256", "SHA384", or
	// "SHA512". The default is "SHA256".
	void put_RsaPaddingHash(const char *newVal);

	// If an RSA key is used for signing, this is the padding scheme to use. It can be
	// "PKCS" or "PSS". The default is "PSS".
	void get_RsaPaddingScheme(CkString &str);
	// If an RSA key is used for signing, this is the padding scheme to use. It can be
	// "PKCS" or "PSS". The default is "PSS".
	const char *rsaPaddingScheme(void);
	// If an RSA key is used for signing, this is the padding scheme to use. It can be
	// "PKCS" or "PSS". The default is "PSS".
	void put_RsaPaddingScheme(const char *newVal);

	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty.
	void get_UncommonOptions(CkString &str);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty.
	const char *uncommonOptions(void);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty.
	void put_UncommonOptions(const char *newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Initializes communication with the card inserted in the given reader. Reader
	// names can be discovered via the SCard.ListReaders or SCard.FindSmartcards
	// methods. If successful, the Atr and CardName properties will be set.
	bool AcquireContext(const char *readerName);


	// Deletes the file specified by dirName and fileName. dirName is the name of the directory
	// that contains the file, or the empty string for root.
	bool CardDeleteFile(const char *dirName, const char *fileName);


	// Deletes a certificate and optionally its associated private key from the smart
	// card. If delPrivKey is true, then the associated private key, if it exists, is also
	// deleted.
	bool DeleteCert(CkCert &cert, bool delPrivKey);


	// This function reverses the effect of AcquireContext and severs the communication
	// between the Base CSP/KSP and the card minidriver. The Atr and CardName
	// properties are cleared.
	bool DeleteContext(void);


	// Deletes the key container at the given containerIndex. This deletes both the "signature"
	// and "key exchange" keys that may be contained in the specified key container.
	bool DeleteKeyContainer(int containerIndex);


	// Get the list of files in the directory specified by dirName. Pass the empty string
	// for the root directory. The filenames are returned in st.
	bool EnumFiles(const char *dirName, CkStringTable &st);


	// Finds the certificate where the given certPart equals the partValue. Possible values for
	// certPart are: "subjectDN", "subjectDN_withTags", "subjectCN", "serial", or
	// "serial:issuerCN".
	// 
	// The cert is loaded with the certificate if successful.
	// 
	// Note: If successful, the cert will be linked internally with this ScMinidriver
	// session such that certificate can be used for signing on the smart card when
	// used in other Chilkat classes such as XmlDSigGen, Pdf, Crypt2, Mime, MailMan,
	// etc.
	// 
	bool FindCert(const char *certPart, const char *partValue, CkCert &cert);


	// Generates a key to be stored in either the "signature" or "key exchange"
	// location within a key container. Creates the key container if it does not
	// already exist. Otherwise replaces the key in the key container.
	// 
	// The keySpec can be "sig" or "kex" to specify either the "signature" or "key
	// exchange" location.
	// 
	// The keyType can be "ecc" or "rsa".
	// 
	// For RSA keys, the keySize is the size of the key in bits, such as 1024, 2048, 4096,
	// etc. (2048 is a typical value.) For ECC keys, the size can be 256, 384, or 521.
	// 
	// The pinId can be "user", or "3" through "7". (It is typically "user".)
	// 
	bool GenerateKey(int containerIndex, const char *keySpec, const char *keyType, int keySize, const char *pinId);


	// Gets all card properties and returns them in json. See the example below.
	bool GetCardProperties(CkJsonObject &json);


	// Get the certificate at the specified containerIndex and keySpec. The keySpec can be "sig" or
	// "kex" to specify either the "signature" or "key exchange" location within the
	// container. The containerIndex can be -1 to choose the first key container with a
	// certificate. The keySpec can also be "any" to choose either "sig" or "kex" based on
	// which is present, with preference given to "sig" if both are present.
	// 
	// The cert is loaded with the certificate if successful.
	// 
	// Note: If successful, the cert will be linked internally with this ScMinidriver
	// session such that certificate can be used for signing on the smart card when
	// used in other Chilkat classes such as XmlDSigGen, Pdf, Crypt2, Mime, MailMan,
	// etc.
	// 
	bool GetCert(int containerIndex, const char *keySpec, CkCert &cert);


	// Queries a key container to get the keys that are present. If the signature
	// public key is present, it is returned in sigKey. If the key exchange key is
	// present, it is returned in kexKey.
	bool GetContainerKeys(int containerIndex, CkPublicKey &sigKey, CkPublicKey &kexKey);


	// Returns the contents of the CSP container map file (cmapfile). The information
	// is returned in the json. This gives an overview of what key containers and
	// certificates exist in the smart card from a CSP's point of view. See the example
	// linked below.
	bool GetCspContainerMap(CkJsonObject &json);


	// Imports a certificate with its private key onto the smart card. The cert must
	// have an accessible private key, such as will be the case if the cert was loaded
	// from a .pfx/.p12, or if the cert was loaded from a Windows certificate store
	// where the private key exists (and can be exported from the Windows certificate
	// store).
	// 
	// The containerIndex is the container index. It can range from 0 to the MaxContainers-1.
	// 
	// The keySpec can be "sig" or "kex" to specify either the "signature" or "key
	// exchange" location within the container.
	// 
	// The pinId can be "user", or "3" through "7". (It is typically "user".)
	// 
	bool ImportCert(CkCert &cert, int containerIndex, const char *keySpec, const char *pinId);


	// Imports a key to be stored in either the "signature" or "key exchange" location
	// within a key container. Creates the key container if it does not already exist.
	// Otherwise replaces the specified key in the key container.
	// 
	// The keySpec can be "sig" or "kex" to specify either the "signature" or "key
	// exchange" location.
	// 
	// The privKey is the private key to import.
	// 
	// The ARG5 can be "user", or "3" through "7". (It is typically "user".)
	// 
	bool ImportKey(int containerIndex, const char *keySpec, CkPrivateKey &privKey, const char *pinId);


	// Lists the certs found on the smart card. The certPart indicates the information to
	// be returned from each certificate. Possible values are: "subjectDN",
	// "subjectDN_withTags", "subjectCN", "serial", or "serial:issuerCN". The
	// information is returned in st.
	bool ListCerts(const char *certPart, CkStringTable &st);


	// Performs regular PIN authentication. The pinId can be "user", "admin", or "3"
	// through "7". (It is typically "user".) The pin is the alphanumeric PIN.
	// 
	// Returns 0 for success. If not successful, the return value indicates the number
	// of attempts remaining before the PIN is locked. (The number of times an
	// incorrect PIN may be presented to the card before the PIN is blocked, and
	// requires the admin to unblock it.) If the PIN is already blocked, the return
	// value is -1. If the method fails for some other reason, such as if a context has
	// not yet been acquired, the return value is -2.
	// 
	int PinAuthenticate(const char *pinId, const char *pin);


	// The same as PinAutheneticate, but the PIN is passed as a hex string. For
	// example, to pass a PIN of 0x01, 0x02, 0x03, 0x04, pass "01020304".
	int PinAuthenticateHex(const char *pinId, const char *pin);


	// Changes a PIN. The pinId can be "user", "admin", or "3" through "7". (It is
	// typically "user".) The currentPin is the current alphanumeric PIN. The newPin is the new
	// PIN.
	// 
	// Returns 0 for success. If not successful, the return value indicates the number
	// of attempts remaining before the PIN is locked. (The number of times an
	// incorrect PIN may be presented to the card before the PIN is blocked, and
	// requires the admin to unblock it.) If the PIN is already blocked, the return
	// value is -1. If the method fails for some other reason, such as if a context has
	// not yet been acquired, the return value is -2.
	// 
	int PinChange(const char *pinId, const char *currentPin, const char *newPin);


	// Reverses a previous PIN authentication without resetting the card. The pinId can
	// be "user", "admin", or "3" through "7". (It is typically "user".)
	bool PinDeauthenticate(const char *pinId);


	// Reads the entire file specified by dirName and fileName into bd. dirName is the name of
	// the directory that contains the file, or the empty string for root.
	bool ReadFile(const char *dirName, const char *fileName, CkBinData &bd);


	// Signs the data passed in bdData. The hashDataAlg can be "sha1", "sha256", "sha384",
	// "sha512", or "none". If not equal to "none", then the hash of the data passed in
	// bdData is signed.
	// 
	// The containerIndex specifies the key container. By specifying the key container, you are
	// almost specifying the key. A key container can contain two keys: A signature
	// key, and a key-exchange key. The keySpec indicates which of these two keys to use.
	// keySpec should be set to "sig" or "kex".
	// 
	// Note: The type of signature created, such as RSA or ECC, is determined by the
	// type of key that exists in the key container (specified by containerIndex and keySpec). If it
	// is an RSA key, additional options can be specified via the RsaPaddingScheme and
	// RsaPaddingHash properties.
	// 
	// If successful, the signature is written to bdSignedData.
	// 
	bool SignData(int containerIndex, const char *keySpec, const char *hashDataAlg, CkBinData &bdData, CkBinData &bdSignedData);


	// Writes the entire file specified by dirName and fileName. dirName is the name of the
	// directory that contains the file, or the empty string for root. The entire
	// contents of bd are written to the file on the smart card.
	bool WriteFile(const char *dirName, const char *fileName, CkBinData &bd);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
