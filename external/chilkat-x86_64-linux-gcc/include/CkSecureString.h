// CkSecureString.h: interface for the CkSecureString class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkSecureString_H
#define _CkSecureString_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkStringBuilder;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkSecureString
class CK_VISIBLE_PUBLIC CkSecureString  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkSecureString(const CkSecureString &);
	CkSecureString &operator=(const CkSecureString &);

    public:
	CkSecureString(void);
	virtual ~CkSecureString(void);

	static CkSecureString *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// If set to the name of a hash algorithm, then a hash of the current string value
	// is maintained. This allows for the hash to be verified via the VerifyHash
	// method. Possible hash algorithm names are "sha1", "sha256", "sha384", "sha512",
	// "md5", "md2", "ripemd160", "ripemd128","ripemd256", and "ripemd320".
	void get_MaintainHash(CkString &str);
	// If set to the name of a hash algorithm, then a hash of the current string value
	// is maintained. This allows for the hash to be verified via the VerifyHash
	// method. Possible hash algorithm names are "sha1", "sha256", "sha384", "sha512",
	// "md5", "md2", "ripemd160", "ripemd128","ripemd256", and "ripemd320".
	const char *maintainHash(void);
	// If set to the name of a hash algorithm, then a hash of the current string value
	// is maintained. This allows for the hash to be verified via the VerifyHash
	// method. Possible hash algorithm names are "sha1", "sha256", "sha384", "sha512",
	// "md5", "md2", "ripemd160", "ripemd128","ripemd256", and "ripemd320".
	void put_MaintainHash(const char *newVal);

	// Can be set to true to make this secure string read-only (cannot be modified).
	bool get_ReadOnly(void);
	// Can be set to true to make this secure string read-only (cannot be modified).
	void put_ReadOnly(bool newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Returns the clear-text string value.
	bool Access(CkString &outStr);

	// Returns the clear-text string value.
	const char *access(void);

	// Appends a clear-text string to this secure string. The in-memory data will be
	// decrypted, the string will be appended, and then it will be re-encrypted. Can
	// return false if the string has been marked as read-only via the ReadOnly
	// property.
	bool Append(const char *str);


	// Appends a clear-text string contained in a StringBuilder to this secure string.
	// The in-memory data will be decrypted, the string will be appended, and then it
	// will be re-encrypted. Can return false if the string has been marked as
	// read-only via the ReadOnly property.
	bool AppendSb(CkStringBuilder &sb);


	// Appends the contents of a secure string to this secure string. The in-memory
	// data will be decrypted, the secure string will be appended, and then it will be
	// re-encrypted. Can return false if this string has been marked as read-only via
	// the ReadOnly property.
	bool AppendSecure(CkSecureString &secStr);


	// Returns the hash value for the current value of this secure string. The encoding
	// specifies the encoding to be used. It can be any of the binary encoding
	// algorithms, such as "base64", "hex", and many more listed atChilkat Binary
	// Encodings
	// <http://cknotes.com/chilkat-binary-encoding-list/>
	bool HashVal(const char *encoding, CkString &outStr);

	// Returns the hash value for the current value of this secure string. The encoding
	// specifies the encoding to be used. It can be any of the binary encoding
	// algorithms, such as "base64", "hex", and many more listed atChilkat Binary
	// Encodings
	// <http://cknotes.com/chilkat-binary-encoding-list/>
	const char *hashVal(const char *encoding);

	// Loads the contents of a file into this secure string. The current contents of
	// this object are replaced with the new text from the file.
	bool LoadFile(const char *path, const char *charset);


	// Returns true if the secStr equals the contents of this secure string.
	bool SecStrEquals(CkSecureString &secStr);


	// Verifies the hashVal against the hash value stored for the current value of this
	// secure string. The MaintainHash property must've previously been set for this
	// secure string to maintain an internal hash. The encoding specifies the encoding of
	// the hashVal. It can be any of the binary encoding algorithms, such as "base64",
	// "hex", and many more listed atChilkat Binary Encodings
	// <http://cknotes.com/chilkat-binary-encoding-list/>
	bool VerifyHash(const char *hashVal, const char *encoding);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
