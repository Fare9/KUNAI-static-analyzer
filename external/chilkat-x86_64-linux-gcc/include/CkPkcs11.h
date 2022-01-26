// CkPkcs11.h: interface for the CkPkcs11 class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkPkcs11_H
#define _CkPkcs11_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkJsonObject;
class CkCert;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkPkcs11
class CK_VISIBLE_PUBLIC CkPkcs11  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkPkcs11(const CkPkcs11 &);
	CkPkcs11 &operator=(const CkPkcs11 &);

    public:
	CkPkcs11(void);
	virtual ~CkPkcs11(void);

	static CkPkcs11 *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The number of certificates contained on the smart card or USB token. This
	// property is set when FindAllCerts is called.
	int get_NumCerts(void);

	// On Windows systems, then should be set to the name of the DLL file (if the DLL
	// is located in C:\Windows\System32), or it can be the full path to the DLL.
	// 
	// On Linux, MacOSX, or other non-Windows systems, this can also be either the full
	// path to the .so or .dylib, or just the .so or .dylib filename. On these systems,
	// Chilkat calls thedlopen system function
	// <https://man7.org/linux/man-pages/man3/dlopen.3.html> to load the shared
	// library. If just the filename is passed in, the directories searched are those
	// indicated in the dlopen function description
	// athttps://man7.org/linux/man-pages/man3/dlopen.3.html
	// <https://man7.org/linux/man-pages/man3/dlopen.3.html>
	// 
	void get_SharedLibPath(CkString &str);
	// On Windows systems, then should be set to the name of the DLL file (if the DLL
	// is located in C:\Windows\System32), or it can be the full path to the DLL.
	// 
	// On Linux, MacOSX, or other non-Windows systems, this can also be either the full
	// path to the .so or .dylib, or just the .so or .dylib filename. On these systems,
	// Chilkat calls thedlopen system function
	// <https://man7.org/linux/man-pages/man3/dlopen.3.html> to load the shared
	// library. If just the filename is passed in, the directories searched are those
	// indicated in the dlopen function description
	// athttps://man7.org/linux/man-pages/man3/dlopen.3.html
	// <https://man7.org/linux/man-pages/man3/dlopen.3.html>
	// 
	const char *sharedLibPath(void);
	// On Windows systems, then should be set to the name of the DLL file (if the DLL
	// is located in C:\Windows\System32), or it can be the full path to the DLL.
	// 
	// On Linux, MacOSX, or other non-Windows systems, this can also be either the full
	// path to the .so or .dylib, or just the .so or .dylib filename. On these systems,
	// Chilkat calls thedlopen system function
	// <https://man7.org/linux/man-pages/man3/dlopen.3.html> to load the shared
	// library. If just the filename is passed in, the directories searched are those
	// indicated in the dlopen function description
	// athttps://man7.org/linux/man-pages/man3/dlopen.3.html
	// <https://man7.org/linux/man-pages/man3/dlopen.3.html>
	// 
	void put_SharedLibPath(const char *newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Closes the session on the token (i.e. smart card).
	// 
	// Note: Memory leaks can occur if the session is not properly closed.
	// 
	bool CloseSession(void);


	// Discovers the readers, smart cards, and USB tokens accessible via PKCS11 on the
	// computer (using the DLL/shared lib specified by SharedLibPath). The onlyTokensPresent
	// specifies if only slots (readers) with tokens (smart cards) present should be
	// returned. The information is written to the json. (For details, see the example
	// below.)
	bool Discover(bool onlyTokensPresent, CkJsonObject &json);


	// Finds all certificates contained on the smart card (or USB token). This sets the
	// NumCerts property. Each certificate can be obtained by calling GetCert(index)
	// where the 1st cert is at index 0.
	// 
	// Important: Private keys will not be seen unless the PKCS11 session is
	// authenticated. To authenticate, your application must call Login after calling
	// OpenSession.
	// 
	bool FindAllCerts(void);


	// Finds the certificate where the given certPart equals the partValue. Possible values for
	// certPart are: "privateKey", "subjectDN", "subjectDN_withTags", "subjectCN",
	// "serial", or "serial:issuerCN". If certPart equals "privateKey", then pass an empty
	// string in partValue. Specifying "privateKey" means to return the first certificate
	// having a private key.
	// 
	// The cert is loaded with the certificate if successful.
	// 
	// Important: Private keys will not be seen unless the PKCS11 session is
	// authenticated. To authenticate, your application must call Login after calling
	// OpenSession.
	// 
	// Note: If successful, the cert will be linked internally with this PKCS11 session
	// such that certificate can be used for signing on the smart card when used in
	// other Chilkat classes such as XmlDSigGen, Pdf, Crypt2, Mime, MailMan, etc.
	// 
	bool FindCert(const char *certPart, const char *partValue, CkCert &cert);


	// Loads cert with the Nth certificate indicated by index. The 1st certificate is at
	// index 0. The FindAllCerts method must be called beforehand to load the certs
	// from the smart card into this object. After calling FindAllCerts, the NumCerts
	// property is set and each certificate can be retrieved by calling GetCert.
	bool GetCert(int index, CkCert &cert);


	// Initializes the PKCS#11 library. Should be called after specifying the
	// SharedLibPath. The DLL (or .so/.dylib) is dynamically loaded and the PKCS#11 lib
	// is initialized.
	bool Initialize(void);


	// Initializes the normal user's PIN. This must be called from the security
	// officer's (SO) logged-in session.
	// 
	// Note: To unblock a smart card, login to the SO (Security Officer) session using
	// the PUK, and then call this with the new user PIN.
	// 
	bool InitPin(const char *pin);


	// Initializes a token. slotId is the slot ID of the token's slot.
	// 
	// If the token has not been initialized (i.e. new from the factory), then the pPin
	// parameter becomes the initial value of the SO (Security Officer) PIN. If the
	// token is being reinitialized, the pin parameter is checked against the existing
	// SO PIN to authorize the initialization operation.
	// 
	// When a token is initialized, all objects that can be destroyed are destroyed
	// (i.e., all except for “indestructible” objects such as keys built into the
	// token). Also, access by the normal user is disabled until the SO sets the normal
	// user’s PIN. Depending on the token, some “default” objects may be created, and
	// attributes of some objects may be set to default values.
	// 
	bool InitToken(int slotId, const char *pin, const char *tokenLabel);


	// Authenticates a session with a PIN. The userType can be one of the following integer
	// values:
	//     Security Officer (0)
	//     Normal User (1)
	//     Context Specific (2)
	// 
	// Except for special circumstances, you'll always login as the Normal User.
	// 
	bool Login(int userType, const char *pin);


	// Logs out from a token (smart card).
	bool Logout(void);


	// Opens a session on the token (i.e. smart card). The slotId is the ID of the slot
	// (not the index). Set slotId equal to -1 to choose the first available non-empty
	// slot. The readWrite indicates whether the session should be read-only or read-write.
	// 
	// The PKCS11 terminology is confusing:
	// 
	//     A "slot" corresponds to a connected smart card reader or USB hardware token,
	//     such as a Feitian ePass3003Auto token.
	//     A "token" corresponds to the smart card inserted into the reader. If we have
	//     a USB hardware token, such as the epass3003Auto (or many others), then
	//     technically there is always a "smart card" inserted, because the USB hardware
	//     token is effectively both the reader and smart card wrapped in one inseparable
	//     device.
	//     The PKCS11 DLL (or .so shared lib, or .dylib) is the vendor supplied PKCS11
	//     implementation (driver) that provides the low-level "C" PKCS11 functions (called
	//     by Chilkat internally).
	//     Generally, the number of slots will equal the number of connected smart
	//     cards or tokens belonging to the vendor of the DLL, or compatible with the DLL.
	//     In most cases you'll have your single reader with a single smart card inserted,
	//     and therefore only one slot exists.
	//     Some PKCS11 DLLs are provided by a 3rd party and support many smart cards.
	//     For example, A.E.T. Europe B.V.'s "SafeSign Identity Client Standard Version
	//     3.5" DLL is "aetpkss1.dll". It supports the following tokens:
	//         Defensiepas
	//         Defensiepas 2
	//         G&D Convego Join 4.01 40k/80k
	//         G&D SkySIM Hercules
	//         G&D SkySIM Scorpius
	//         G&D Sm@rtCafé Expert 3.2
	//         G&D Sm@rtCafé Expert 4.0
	//         G&D Sm@rtCafé Expert 5.0
	//         G&D Sm@rtCafé Expert 6.0
	//         G&D Sm@rtCafé Expert 7.0
	//         G&D Sm@rtCafé Expert 64
	//         Gemalto Desineo ICP D72 FXR1 Java
	//         Gemalto IDCore 30
	//         Gemalto MultiApp ID v2.1
	//         Gemalto Optelio D72 FR1
	//         Gemalto TOP DL v2
	//         Infineon Oracle JCOS Ed.1
	//         JCOP21 v2.3
	//         Morpho IDealCitiz v2.1
	//         Morpho JMV ProCL V3.0
	//         NXP J2A080 / J2A081 (JCOP 2.4.1 R3)
	//         NXP JD081 (JCOP 2.4.1 R3)
	//         NXP J3A080 (JCOP 2.4.1 R3)
	//         NXP JCOP 2.4.2 R3
	//         NXP JCOP 3 SecID P60
	//         Oberthur IDOne Cosmo v7.0
	//         RDW ABR kaart
	//         Rijkspas
	//         Rijkspas 2
	//         Sagem YpsID s2
	//         Sagem YpsID s3
	//         StarSign Crypto USB Token S
	//         Swissbit PS-100u SE
	//         UZI-pas
	//         UZI-pas 2
	bool OpenSession(int slotId, bool readWrite);


	// Modifies the PIN of the user that is currently logged in, or the Normal User PIN
	// if the session is not logged in.
	bool SetPin(const char *oldPin, const char *newPin);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
