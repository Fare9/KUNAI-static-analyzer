// CkAsnW.h: interface for the CkAsnW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkAsnW_H
#define _CkAsnW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkWideCharBase.h"

class CkByteData;
class CkBinDataW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkAsnW
class CK_VISIBLE_PUBLIC CkAsnW  : public CkWideCharBase
{
	

	private:
	
	// Don't allow assignment or copying these objects.
	CkAsnW(const CkAsnW &);
	CkAsnW &operator=(const CkAsnW &);

    public:
	CkAsnW(void);
	virtual ~CkAsnW(void);

	

	static CkAsnW *createNew(void);
	

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The ASN.1 item's boolean value if it is a boolean item.
	bool get_BoolValue(void);
	// The ASN.1 item's boolean value if it is a boolean item.
	void put_BoolValue(bool newVal);

	// true if this ASN.1 item is a constructed item. Sequence and Set items are
	// constructed and can contain sub-items. All other tags (boolean, integer, octets,
	// utf8String, etc.) are primitive (non-constructed).
	bool get_Constructed(void);

	// The ASN.1 item's content if it is an ASN.1 string type (such as Utf8String,
	// BmpString, PrintableString, VisibleString, T61String, IA5String, NumericString,
	// or UniversalString).
	void get_ContentStr(CkString &str);
	// The ASN.1 item's content if it is an ASN.1 string type (such as Utf8String,
	// BmpString, PrintableString, VisibleString, T61String, IA5String, NumericString,
	// or UniversalString).
	const wchar_t *contentStr(void);
	// The ASN.1 item's content if it is an ASN.1 string type (such as Utf8String,
	// BmpString, PrintableString, VisibleString, T61String, IA5String, NumericString,
	// or UniversalString).
	void put_ContentStr(const wchar_t *newVal);

	// The ASN.1 item's integer value if it is a small integer item.
	int get_IntValue(void);
	// The ASN.1 item's integer value if it is a small integer item.
	void put_IntValue(int newVal);

	// The number of sub-items contained within this ASN.1 item. Only constructed
	// items, such as Sequence and Set will contain sub-iitems. Primitive items such as
	// OIDs, octet strings, integers, etc. will never contain sub-items.
	int get_NumSubItems(void);

	// The ASN.1 item's tag as a descriptive string. Possible values are:
	// boolean
	// integer
	// bitString
	// octets
	// null
	// oid
	// utf8String
	// relativeOid
	// sequence
	// set
	// numericString
	// printableString
	// t61String
	// ia5String
	// utcTime
	// bmpString
	void get_Tag(CkString &str);
	// The ASN.1 item's tag as a descriptive string. Possible values are:
	// boolean
	// integer
	// bitString
	// octets
	// null
	// oid
	// utf8String
	// relativeOid
	// sequence
	// set
	// numericString
	// printableString
	// t61String
	// ia5String
	// utcTime
	// bmpString
	const wchar_t *tag(void);

	// The ASN.1 item's tag as a integer value. The integer values for possible tags
	// are as follows:
	// boolean (1)
	// integer (2)
	// bitString (3)
	// octets (4)
	// null (5)
	// oid (6)
	// utf8String (12)
	// relativeOid (13)
	// sequence (16)
	// set (17)
	// numericString (18)
	// printableString (19)
	// t61String (20)
	// ia5String (22)
	// utcTime (23)
	// bmpString (30)
	int get_TagValue(void);



	// ----------------------
	// Methods
	// ----------------------
	// Appends an ASN.1 integer, but one that is a big (huge) integer that is too large
	// to be represented by an integer variable. The bytes composing the integer are
	// passed in encoded string format (such as base64, hex, etc.). The byte order must
	// be big-endian. The encoding may be any of the following encodings: "Base64", "Hex",
	// "Base58", "modBase64", "Base32", "UU", "QP" (for quoted-printable), "URL" (for
	// url-encoding), "Q", "B", "url_oath", "url_rfc1738", "url_rfc2396", and
	// "url_rfc3986". The encoding name is case insensitive (for example, both "Base64" and
	// "base64" are treated the same).
	bool AppendBigInt(const wchar_t *encodedBytes, const wchar_t *encoding);

	// Appends an ASN.1 bit string to the caller's sub-items. The bytes containing the
	// bits are passed in encoded string format (such as base64, hex, etc.). The byte
	// order must be big-endian (MSB first). The encoding may be any of the following
	// encodings: "Base64", "Hex", "Base58", "modBase64", "Base32", "UU", "QP" (for
	// quoted-printable), "URL" (for url-encoding), "Q", "B", "url_oath",
	// "url_rfc1738", "url_rfc2396", and "url_rfc3986". The encoding name is case
	// insensitive (for example, both "Base64" and "base64" are treated the same).
	bool AppendBits(const wchar_t *encodedBytes, const wchar_t *encoding);

	// Appends an ASN.1 boolean item to the caller's sub-items. Items may only be
	// appended to constructed data types such as Sequence and Set.
	bool AppendBool(bool value);

	// Appends an ASN.1 context-specific constructed item to the caller's sub-items.
	bool AppendContextConstructed(int tag);

	// Appends an ASN.1 context-specific primitive item to the caller's sub-items. The
	// bytes are passed in encoded string format (such as base64, hex, etc.). The encoding
	// may be any of the following encodings: "Base64", "Hex", "Base58", "modBase64",
	// "Base32", "UU", "QP" (for quoted-printable), "URL" (for url-encoding), "Q", "B",
	// "url_oath", "url_rfc1738", "url_rfc2396", and "url_rfc3986". The encoding name is
	// case insensitive (for example, both "Base64" and "base64" are treated the same).
	bool AppendContextPrimitive(int tag, const wchar_t *encodedBytes, const wchar_t *encoding);

	// Appends an ASN.1 integer item to the caller's sub-items. Items may only be
	// appended to constructed data types such as Sequence and Set.
	bool AppendInt(int value);

	// Appends an ASN.1 null item to the caller's sub-items. Items may only be appended
	// to constructed data types such as Sequence and Set.
	bool AppendNull(void);

	// Appends an ASN.1 octet string to the caller's sub-items. The bytes are passed in
	// encoded string format (such as base64, hex, etc.). The encoding may be any of the
	// following encodings: "Base64", "Hex", "Base58", "modBase64", "Base32", "UU",
	// "QP" (for quoted-printable), "URL" (for url-encoding), "Q", "B", "url_oath",
	// "url_rfc1738", "url_rfc2396", and "url_rfc3986". The encoding name is case
	// insensitive (for example, both "Base64" and "base64" are treated the same).
	bool AppendOctets(const wchar_t *encodedBytes, const wchar_t *encoding);

	// Appends an ASN.1 OID (object identifier) to the caller's sub-items. The OID is
	// passed in string form, such as "1.2.840.113549.1.9.1".
	bool AppendOid(const wchar_t *oid);

	// Appends an ASN.1 sequence item to the caller's sub-items.
	bool AppendSequence(void);

	// Appends an ASN.1 sequence item to the caller's sub-items, and updates the
	// internal reference to point to the newly appended sequence item.
	bool AppendSequence2(void);

	// Appends an ASN.1 sequence item to the caller's sub-items, and returns the newly
	// appended sequence item.
	// The caller is responsible for deleting the object returned by this method.
	CkAsnW *AppendSequenceR(void);

	// Appends an ASN.1 set item to the caller's sub-items.
	bool AppendSet(void);

	// Appends an ASN.1 set item to the caller's sub-items, and updates the internal
	// reference to point to the newly appended set item.
	bool AppendSet2(void);

	// Appends an ASN.1 set item to the caller's sub-items, and returns the newly
	// appended set item.
	// The caller is responsible for deleting the object returned by this method.
	CkAsnW *AppendSetR(void);

	// Appends a string item to the caller's sub-items. The strType specifies the type of
	// string to be added. It may be "utf8", "ia5", "t61", "printable", "visible",
	// "numeric", "universal", or "bmp". The value must conform to the ASN.1
	// restrictions imposed for a given string type. The "utf8", "bmp", and "universal"
	// types have no restrictions on what characters are allowed. In general, unless a
	// specific type of string is required, choose the "utf8" type.
	bool AppendString(const wchar_t *strType, const wchar_t *value);

	// Appends a UTCTime item to the caller's sub-items. The timeFormat specifies the format
	// of the dateTimeStr. It should be set to "utc". (In the future, this method will be
	// expanded to append GeneralizedTime items by using "generalized" for timeFormat.) To
	// append the current date/time, set dateTimeStr equal to the empty string or the keyword
	// "now". Otherwise, the dateTimeStr should be in the UTC time format "YYMMDDhhmm[ss]Z" or
	// "YYMMDDhhmm[ss](+|-)hhmm".
	bool AppendTime(const wchar_t *timeFormat, const wchar_t *dateTimeStr);

	// Converts ASN.1 to XML and returns the XML string.
	bool AsnToXml(CkString &outStr);
	// Converts ASN.1 to XML and returns the XML string.
	const wchar_t *asnToXml(void);

	// Discards the Nth sub-item. (The 1st sub-item is at index 0.)
	bool DeleteSubItem(int index);

	// Returns the ASN.1 in binary DER form.
	bool GetBinaryDer(CkByteData &outBytes);

	// Returns the content of the ASN.1 item in encoded string form. The encoding may be
	// any of the following encodings: "Base64", "Hex", "Base58", "modBase64",
	// "Base32", "UU", "QP" (for quoted-printable), "URL" (for url-encoding), "Q", "B",
	// "url_oath", "url_rfc1738", "url_rfc2396", and "url_rfc3986". The encoding name is
	// case insensitive (for example, both "Base64" and "base64" are treated the same).
	bool GetEncodedContent(const wchar_t *encoding, CkString &outStr);
	// Returns the content of the ASN.1 item in encoded string form. The encoding may be
	// any of the following encodings: "Base64", "Hex", "Base58", "modBase64",
	// "Base32", "UU", "QP" (for quoted-printable), "URL" (for url-encoding), "Q", "B",
	// "url_oath", "url_rfc1738", "url_rfc2396", and "url_rfc3986". The encoding name is
	// case insensitive (for example, both "Base64" and "base64" are treated the same).
	const wchar_t *getEncodedContent(const wchar_t *encoding);
	// Returns the content of the ASN.1 item in encoded string form. The encoding may be
	// any of the following encodings: "Base64", "Hex", "Base58", "modBase64",
	// "Base32", "UU", "QP" (for quoted-printable), "URL" (for url-encoding), "Q", "B",
	// "url_oath", "url_rfc1738", "url_rfc2396", and "url_rfc3986". The encoding name is
	// case insensitive (for example, both "Base64" and "base64" are treated the same).
	const wchar_t *encodedContent(const wchar_t *encoding);

	// Returns the binary DER in encoded string form. The encoding indicates the encoding
	// and can be "base64", "hex", "uu", "quoted-printable", "base32", or "modbase64".
	bool GetEncodedDer(const wchar_t *encoding, CkString &outStr);
	// Returns the binary DER in encoded string form. The encoding indicates the encoding
	// and can be "base64", "hex", "uu", "quoted-printable", "base32", or "modbase64".
	const wchar_t *getEncodedDer(const wchar_t *encoding);
	// Returns the binary DER in encoded string form. The encoding indicates the encoding
	// and can be "base64", "hex", "uu", "quoted-printable", "base32", or "modbase64".
	const wchar_t *encodedDer(const wchar_t *encoding);

	// Returns the last ASN.1 sub-item. This method can be called immediately after any
	// Append* method to access the appended item.
	// The caller is responsible for deleting the object returned by this method.
	CkAsnW *GetLastSubItem(void);

	// Returns the Nth ASN.1 sub-item. The 1st sub-item is at index 0.
	// The caller is responsible for deleting the object returned by this method.
	CkAsnW *GetSubItem(int index);

	// Loads ASN.1 from the XML representation (such as that created by the AsnToXml
	// method).
	bool LoadAsnXml(const wchar_t *xmlStr);

	// Loads ASN.1 from the binary DER contained in bd.
	bool LoadBd(CkBinDataW &bd);

	// Loads ASN.1 from binary DER.
	bool LoadBinary(CkByteData &derBytes);

	// Loads ASN.1 from a binary DER file.
	bool LoadBinaryFile(const wchar_t *path);

	// Loads ASN.1 from an encoded string. The encoding can be "base64", "hex", "uu",
	// "quoted-printable", "base32", or "modbase64".
	bool LoadEncoded(const wchar_t *asnContent, const wchar_t *encoding);

	// Sets the content of this primitive ASN.1 item. The encoding may be any of the
	// following encodings: "Base64", "Hex", "Base58", "modBase64", "Base32", "UU",
	// "QP" (for quoted-printable), "URL" (for url-encoding), "Q", "B", "url_oath",
	// "url_rfc1738", "url_rfc2396", and "url_rfc3986". The encoding name is case
	// insensitive (for example, both "Base64" and "base64" are treated the same).
	bool SetEncodedContent(const wchar_t *encodedBytes, const wchar_t *encoding);

	// Appends the ASN.1 in binary DER format to bd.
	bool WriteBd(CkBinDataW &bd);

	// Writes the ASN.1 in binary DER form to a file.
	bool WriteBinaryDer(const wchar_t *path);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
