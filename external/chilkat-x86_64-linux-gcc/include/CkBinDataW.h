// CkBinDataW.h: interface for the CkBinDataW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkBinDataW_H
#define _CkBinDataW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkWideCharBase.h"

class CkByteData;
class CkStringBuilderW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkBinDataW
class CK_VISIBLE_PUBLIC CkBinDataW  : public CkWideCharBase
{
	

	private:
	
	// Don't allow assignment or copying these objects.
	CkBinDataW(const CkBinDataW &);
	CkBinDataW &operator=(const CkBinDataW &);

    public:
	CkBinDataW(void);
	virtual ~CkBinDataW(void);

	const unsigned char *getBinaryDataPtr(void);

	static CkBinDataW *createNew(void);
	

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The number of bytes contained within the object.
	int get_NumBytes(void);



	// ----------------------
	// Methods
	// ----------------------
	// Appends the contents of another BinData to this object.
	bool AppendBd(CkBinDataW &binData);

	// Appends binary data to the current contents, if any.
	bool AppendBinary(CkByteData &data);

	// Appends binary data to the current contents, if any.
	bool AppendBinary2(const void *pByteData, unsigned long szByteData);

	// Appends the appropriate BOM (byte order mark), also known as a "preamble", for
	// the given charset. If the charset has no defined BOM, then nothing is appended. An
	// application would typically call this to append the utf-8, utf-16, or utf-32
	// BOM.
	bool AppendBom(const wchar_t *charset);

	// Appends a single byte. The byteValue should be a value from 0 to 255.
	bool AppendByte(int byteValue);

	// Appends encoded binary data to the current data. The encoding may be "Base64",
	// "modBase64", "base64Url", "Base32", "Base58", "QP" (for quoted-printable), "URL"
	// (for url-encoding), "Hex", or any of the encodings found atChilkat Binary
	// Encodings List
	// <http://cknotes.com/chilkat-binary-encoding-list/>.
	bool AppendEncoded(const wchar_t *encData, const wchar_t *encoding);

	// Decodes the contents of sb and appends the decoded bytes to this object. The
	// encoding may be "Base64", "modBase64", "base64Url", "Base32", "Base58", "QP" (for
	// quoted-printable), "URL" (for url-encoding), "Hex", or any of the encodings
	// found atChilkat Binary Encodings List
	// <http://cknotes.com/chilkat-binary-encoding-list/>.
	bool AppendEncodedSb(CkStringBuilderW &sb, const wchar_t *encoding);

	// Appends a 16-bit integer (2 bytes). If littleEndian is true, then the integer bytes
	// are appended in little-endian byte order, otherwise big-endian byte order is
	// used.
	bool AppendInt2(int value, bool littleEndian);

	// Appends a 32-bit integer (4 bytes). If littleEndian is true, then the integer bytes
	// are appended in little-endian byte order, otherwise big-endian byte order is
	// used.
	bool AppendInt4(int value, bool littleEndian);

	// Appends a string to this object, padded to the fieldLen with NULL or SPACE chars. If
	// padWithSpace is true, then SPACE chars are used and the string is not null-terminated.
	// If fieldLen is false, then null bytes are used. The charset controls the byte
	// representation to use, such as "utf-8".
	// 
	// Note: This call will always append a total number of bytes equal to fieldLen. If the
	// str is longer than fieldLen, the method returns false to indicate failure and
	// nothing is appended.
	// 
	bool AppendPadded(const wchar_t *str, const wchar_t *charset, bool padWithSpace, int fieldLen);

	// Appends the contents of a StringBuilder to this object.
	bool AppendSb(CkStringBuilderW &sb, const wchar_t *charset);

	// Appends a string to this object. (This does not append the BOM. If a BOM is
	// required, the AppendBom method can be called to append the appropriate BOM.)
	bool AppendString(const wchar_t *str, const wchar_t *charset);

	// Clears the contents.
	bool Clear(void);

	// Return true if the contents of this object equals the contents of binData.
	bool ContentsEqual(CkBinDataW &binData);

	// Return the index where the first occurrence of str is found. Return -1 if not
	// found. The startIdx indicates the byte index where the search begins. The charset
	// specifies the byte representation of str that is to be searched. For example,
	// it can be "utf-8", "windows-1252", "ansi", "utf-16", etc.
	int FindString(const wchar_t *str, int startIdx, const wchar_t *charset);

	// Retrieves the binary data contained within the object.
	bool GetBinary(CkByteData &outBytes);

	// Retrieves a chunk of the binary data contained within the object.
	bool GetBinaryChunk(int offset, int numBytes, CkByteData &outBytes);

	// Returns a pointer to the internal buffer. Be careful with this method because if
	// additional data is appended, the data within the object may be relocated and the
	// pointer may cease to be valid.
	const void *GetBytesPtr(void);

	// Retrieves the binary data as an encoded string. The encoding may be "Base64",
	// "modBase64", "base64Url", "Base32", "Base58", "QP" (for quoted-printable), "URL"
	// (for url-encoding), "Hex", or any of the encodings found atChilkat Binary
	// Encodings List
	// <http://cknotes.com/chilkat-binary-encoding-list/>.
	bool GetEncoded(const wchar_t *encoding, CkString &outStr);
	// Retrieves the binary data as an encoded string. The encoding may be "Base64",
	// "modBase64", "base64Url", "Base32", "Base58", "QP" (for quoted-printable), "URL"
	// (for url-encoding), "Hex", or any of the encodings found atChilkat Binary
	// Encodings List
	// <http://cknotes.com/chilkat-binary-encoding-list/>.
	const wchar_t *getEncoded(const wchar_t *encoding);
	// Retrieves the binary data as an encoded string. The encoding may be "Base64",
	// "modBase64", "base64Url", "Base32", "Base58", "QP" (for quoted-printable), "URL"
	// (for url-encoding), "Hex", or any of the encodings found atChilkat Binary
	// Encodings List
	// <http://cknotes.com/chilkat-binary-encoding-list/>.
	const wchar_t *encoded(const wchar_t *encoding);

	// Retrieves a chunk of the binary data and returns it in encoded form. The encoding
	// may be "Base64", "modBase64", "base64Url", "Base32", "Base58", "QP" (for
	// quoted-printable), "URL" (for url-encoding), "Hex", or any of the encodings
	// found atChilkat Binary Encodings List
	// <http://cknotes.com/chilkat-binary-encoding-list/>.
	bool GetEncodedChunk(int offset, int numBytes, const wchar_t *encoding, CkString &outStr);
	// Retrieves a chunk of the binary data and returns it in encoded form. The encoding
	// may be "Base64", "modBase64", "base64Url", "Base32", "Base58", "QP" (for
	// quoted-printable), "URL" (for url-encoding), "Hex", or any of the encodings
	// found atChilkat Binary Encodings List
	// <http://cknotes.com/chilkat-binary-encoding-list/>.
	const wchar_t *getEncodedChunk(int offset, int numBytes, const wchar_t *encoding);
	// Retrieves a chunk of the binary data and returns it in encoded form. The encoding
	// may be "Base64", "modBase64", "base64Url", "Base32", "Base58", "QP" (for
	// quoted-printable), "URL" (for url-encoding), "Hex", or any of the encodings
	// found atChilkat Binary Encodings List
	// <http://cknotes.com/chilkat-binary-encoding-list/>.
	const wchar_t *encodedChunk(int offset, int numBytes, const wchar_t *encoding);

	// Writes the encoded data to a StringBuilder. The encoding may be "Base64",
	// "modBase64", "base64Url", "Base32", "Base58", "QP" (for quoted-printable), "URL"
	// (for url-encoding), "Hex", or any of the encodings found atChilkat Binary
	// Encodings List
	// <http://cknotes.com/chilkat-binary-encoding-list/>.
	bool GetEncodedSb(const wchar_t *encoding, CkStringBuilderW &sb);

	// Returns the value of the 16-bit signed integer stored in big-endian or
	// little-endian byte ordering at the given index.
	int GetInt2(int index, bool littleEndian);

	// Returns the value of the 32-bit signed integer stored in big-endian or
	// little-endian byte ordering at the given index.
	int GetInt4(int index, bool littleEndian);

	// Interprets the bytes according to charset and returns the string. The charset can be
	// "utf-8", "utf-16", "ansi", "iso-8859-*", "windows-125*", or any of the supported
	// character encodings listed in the link below.
	bool GetString(const wchar_t *charset, CkString &outStr);
	// Interprets the bytes according to charset and returns the string. The charset can be
	// "utf-8", "utf-16", "ansi", "iso-8859-*", "windows-125*", or any of the supported
	// character encodings listed in the link below.
	const wchar_t *getString(const wchar_t *charset);
	// Interprets the bytes according to charset and returns the string. The charset can be
	// "utf-8", "utf-16", "ansi", "iso-8859-*", "windows-125*", or any of the supported
	// character encodings listed in the link below.
	const wchar_t *string(const wchar_t *charset);

	// Returns numBytes bytes starting at startIdx. The bytes are interpreted according to charset
	// (for example, "utf-8", "ansi", "utf-16", "windows-1252", etc.)
	bool GetTextChunk(int startIdx, int numBytes, const wchar_t *charset, CkString &outStr);
	// Returns numBytes bytes starting at startIdx. The bytes are interpreted according to charset
	// (for example, "utf-8", "ansi", "utf-16", "windows-1252", etc.)
	const wchar_t *getTextChunk(int startIdx, int numBytes, const wchar_t *charset);
	// Returns numBytes bytes starting at startIdx. The bytes are interpreted according to charset
	// (for example, "utf-8", "ansi", "utf-16", "windows-1252", etc.)
	const wchar_t *textChunk(int startIdx, int numBytes, const wchar_t *charset);

	// Returns the value of the 16-bit unsigned integer stored in big-endian or
	// little-endian byte ordering at the given index.
	unsigned long GetUInt2(int index, bool littleEndian);

	// Returns the value of the 32-bit unsigned integer stored in big-endian or
	// little-endian byte ordering at the given index.
	unsigned long GetUInt4(int index, bool littleEndian);

	// Loads binary data and replaces the current contents, if any.
	bool LoadBinary(CkByteData &data);

	// Loads binary data and replaces the current contents, if any.
	bool LoadBinary2(const void *pByteData, unsigned long szByteData);

	// Loads binary data from an encoded string, replacing the current contents, if
	// any. The encoding may be "Base64", "modBase64", "base64Url", "Base32", "Base58",
	// "QP" (for quoted-printable), "URL" (for url-encoding), "Hex", or any of the
	// encodings found atChilkat Binary Encodings List
	// <http://cknotes.com/chilkat-binary-encoding-list/>.
	bool LoadEncoded(const wchar_t *encData, const wchar_t *encoding);

	// Loads data from a file.
	bool LoadFile(const wchar_t *path);

	// Removes a chunk of bytes from the binary data.
	bool RemoveChunk(int offset, int numBytes);

	// Securely clears the contents by writing 0 bytes to the memory prior to
	// deallocating the internal memory.
	bool SecureClear(void);

	// Writes the contents to a file.
	bool WriteFile(const wchar_t *path);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
