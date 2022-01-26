// CkEdDSA.h: interface for the CkEdDSA class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkEdDSA_H
#define _CkEdDSA_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkPrng;
class CkPrivateKey;
class CkPublicKey;
class CkBinData;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkEdDSA
class CK_VISIBLE_PUBLIC CkEdDSA  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkEdDSA(const CkEdDSA &);
	CkEdDSA &operator=(const CkEdDSA &);

    public:
	CkEdDSA(void);
	virtual ~CkEdDSA(void);

	static CkEdDSA *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------


	// ----------------------
	// Methods
	// ----------------------
	// Generates an Ed25519 key. privKey is an output argument. The generated key is
	// created in privKey.
	bool GenEd25519Key(CkPrng &prng, CkPrivateKey &privKey);


	// Computes a shared secret given a private and public key. For example, Alice and
	// Bob can compute the identical shared secret by doing the following: Alice sends
	// Bob her public key, and Bob calls SharedSecretENC with his private key and
	// Alice's public key. Bob sends Alice his public key, and Alice calls
	// SharedSecretENC with her private key and Bob's public key. Both calls to
	// SharedSecretENC will produce the same result. The resulting bytes are returned
	// in encoded string form (hex, base64, etc) as specified by encoding.
	bool SharedSecretENC(CkPrivateKey &privkey, CkPublicKey &pubkey, const char *encoding, CkString &outStr);

	// Computes a shared secret given a private and public key. For example, Alice and
	// Bob can compute the identical shared secret by doing the following: Alice sends
	// Bob her public key, and Bob calls SharedSecretENC with his private key and
	// Alice's public key. Bob sends Alice his public key, and Alice calls
	// SharedSecretENC with her private key and Bob's public key. Both calls to
	// SharedSecretENC will produce the same result. The resulting bytes are returned
	// in encoded string form (hex, base64, etc) as specified by encoding.
	const char *sharedSecretENC(CkPrivateKey &privkey, CkPublicKey &pubkey, const char *encoding);

	// Signs the contents of bd and returns the signature according to encoding. The
	// encoding can be any encoding supported by Chilkat, such as "hex", "base64", etc.
	bool SignBdENC(CkBinData &bd, const char *encoding, CkPrivateKey &privkey, CkString &outStr);

	// Signs the contents of bd and returns the signature according to encoding. The
	// encoding can be any encoding supported by Chilkat, such as "hex", "base64", etc.
	const char *signBdENC(CkBinData &bd, const char *encoding, CkPrivateKey &privkey);

	// Verifies the signature against the contents of bd. The encodedSig is passed as an
	// encoded string (such as hex, base64, etc.) using the encoding specified by enocding.
	// The pubkey contains the Ed25519 public key used to verify.
	bool VerifyBdENC(CkBinData &bd, const char *encodedSig, const char *enocding, CkPublicKey &pubkey);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
