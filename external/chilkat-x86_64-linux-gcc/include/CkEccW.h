// CkEccW.h: interface for the CkEccW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkEccW_H
#define _CkEccW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkWideCharBase.h"

class CkPrivateKeyW;
class CkPrngW;
class CkPublicKeyW;
class CkBinDataW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkEccW
class CK_VISIBLE_PUBLIC CkEccW  : public CkWideCharBase
{
	

	private:
	
	// Don't allow assignment or copying these objects.
	CkEccW(const CkEccW &);
	CkEccW &operator=(const CkEccW &);

    public:
	CkEccW(void);
	virtual ~CkEccW(void);

	

	static CkEccW *createNew(void);
	

	
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
	// Generates an ECDSA private key. The curveName specifies the curve name which
	// determines the key size. The prng provides a source for generating the random
	// private key.
	// 
	// The following curve names are accepted:
	//     secp256r1 (also known as P-256 and prime256v1)
	//     secp384r1 (also known as P-384)
	//     secp521r1 (also known as P-521)
	//     secp256k1 (This is the curve used for Bitcoin)
	//     secp192r1
	//     secp224r1
	//     brainpoolP160r1
	//     brainpoolP192r1
	//     brainpoolP192r1
	//     brainpoolP224r1
	//     brainpoolP256r1
	//     brainpoolP320r1
	//     brainpoolP384r1
	//     brainpoolP512r1
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkPrivateKeyW *GenEccKey(const wchar_t *curveName, CkPrngW &prng);

	// Generates an ECDSA private key using a specified value for K. The curveName specifies
	// the curve name which determines the key size. The encodedK is the encoded value of
	// the private key. The encoding is the encoding used for encodedK, which can be "hex",
	// "base64", "decimal", etc.
	// 
	// Note: This method is typically used for testing -- such as when the same private
	// key is desired to produce results identical from run to run.
	// 
	// The following curve names are accepted:
	//     secp256r1 (also known as P-256 and prime256v1)
	//     secp384r1 (also known as P-384)
	//     secp521r1 (also known as P-521)
	//     secp256k1 (This is the curve used for Bitcoin)
	//     secp192r1
	//     secp224r1
	//     brainpoolP160r1
	//     brainpoolP192r1
	//     brainpoolP192r1
	//     brainpoolP224r1
	//     brainpoolP256r1
	//     brainpoolP320r1
	//     brainpoolP384r1
	//     brainpoolP512r1
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkPrivateKeyW *GenEccKey2(const wchar_t *curveName, const wchar_t *encodedK, const wchar_t *encoding);

	// Computes a shared secret given a private and public key. For example, Alice and
	// Bob can compute the identical shared secret by doing the following: Alice sends
	// Bob her public key, and Bob calls SharedSecretENC with his private key and
	// Alice's public key. Bob sends Alice his public key, and Alice calls
	// SharedSecretENC with her private key and Bob's public key. Both calls to
	// SharedSecretENC will produce the same result. The resulting bytes are returned
	// in encoded string form (hex, base64, etc) as specified by encoding.
	// 
	// Note: The private and public keys must both be keys on the same ECDSA curve.
	// 
	bool SharedSecretENC(CkPrivateKeyW &privKey, CkPublicKeyW &pubKey, const wchar_t *encoding, CkString &outStr);
	// Computes a shared secret given a private and public key. For example, Alice and
	// Bob can compute the identical shared secret by doing the following: Alice sends
	// Bob her public key, and Bob calls SharedSecretENC with his private key and
	// Alice's public key. Bob sends Alice his public key, and Alice calls
	// SharedSecretENC with her private key and Bob's public key. Both calls to
	// SharedSecretENC will produce the same result. The resulting bytes are returned
	// in encoded string form (hex, base64, etc) as specified by encoding.
	// 
	// Note: The private and public keys must both be keys on the same ECDSA curve.
	// 
	const wchar_t *sharedSecretENC(CkPrivateKeyW &privKey, CkPublicKeyW &pubKey, const wchar_t *encoding);

	// This method is the same as SignHashENC, except the actual data to be signed and
	// the name of the hash algorithm is passed in. The following hash algorithms are
	// supported: sha256, sha384, and sha512.
	bool SignBd(CkBinDataW &bdData, const wchar_t *hashAlg, const wchar_t *encoding, CkPrivateKeyW &privKey, CkPrngW &prng, CkString &outStr);
	// This method is the same as SignHashENC, except the actual data to be signed and
	// the name of the hash algorithm is passed in. The following hash algorithms are
	// supported: sha256, sha384, and sha512.
	const wchar_t *signBd(CkBinDataW &bdData, const wchar_t *hashAlg, const wchar_t *encoding, CkPrivateKeyW &privKey, CkPrngW &prng);

	// Computes an ECDSA signature on a hash. ECDSA signatures are computed and
	// verified on the hashes of data (such as SHA1, SHA256, etc.). The hash of the
	// data is passed in encodedHash. The encoding, such as "base64", "hex", etc. is passed in
	// encoding. The ECDSA private key is passed in the 3rd argument (privkey). Given that
	// creating an ECDSA signature involves the generation of random numbers, a PRNG is
	// passed in the 4th argument (prng). The signature is returned as an encoded
	// string using the encoding specified by the encoding argument.
	bool SignHashENC(const wchar_t *encodedHash, const wchar_t *encoding, CkPrivateKeyW &privkey, CkPrngW &prng, CkString &outStr);
	// Computes an ECDSA signature on a hash. ECDSA signatures are computed and
	// verified on the hashes of data (such as SHA1, SHA256, etc.). The hash of the
	// data is passed in encodedHash. The encoding, such as "base64", "hex", etc. is passed in
	// encoding. The ECDSA private key is passed in the 3rd argument (privkey). Given that
	// creating an ECDSA signature involves the generation of random numbers, a PRNG is
	// passed in the 4th argument (prng). The signature is returned as an encoded
	// string using the encoding specified by the encoding argument.
	const wchar_t *signHashENC(const wchar_t *encodedHash, const wchar_t *encoding, CkPrivateKeyW &privkey, CkPrngW &prng);

	// This method is the same as VerifyHashENC, except the actual data to be verified
	// and the name of the hash algorithm is passed in. The following hash algorithms
	// are supported: sha256, sha384, and sha512.
	int VerifyBd(CkBinDataW &bdData, const wchar_t *hashAlg, const wchar_t *encodedSig, const wchar_t *encoding, CkPublicKeyW &pubkey);

	// Verifies an ECDSA signature. ECDSA signatures are computed and verified on the
	// hashes of data (such as SHA1, SHA256, etc.). The hash of the data is passed in
	// encodedHash. The encoded signature is passed in encodedSig. The encoding of both the hash and
	// signature, such as "base64", "hex", etc. is passed in encoding. The ECDSA public key
	// is passed in the last argument (pubkey).
	// 
	// The method returns 1 for a valid signature, 0 for an invalid signature, and -1
	// for any other failure.
	// 
	int VerifyHashENC(const wchar_t *encodedHash, const wchar_t *encodedSig, const wchar_t *encoding, CkPublicKeyW &pubkey);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
