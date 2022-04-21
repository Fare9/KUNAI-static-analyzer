// CkStringBuilderW.h: interface for the CkStringBuilderW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkStringBuilderW_H
#define _CkStringBuilderW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkWideCharBase.h"

class CkBinDataW;
class CkByteData;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkStringBuilderW
class CK_VISIBLE_PUBLIC CkStringBuilderW  : public CkWideCharBase
{
	

	private:
	
	// Don't allow assignment or copying these objects.
	CkStringBuilderW(const CkStringBuilderW &);
	CkStringBuilderW &operator=(const CkStringBuilderW &);

    public:
	CkStringBuilderW(void);
	virtual ~CkStringBuilderW(void);

	

	static CkStringBuilderW *createNew(void);
	

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// Returns the content of the string converted to an integer.
	int get_IntValue(void);
	// Returns the content of the string converted to an integer.
	void put_IntValue(int newVal);

	// Returns true if the content contains only those characters allowed in the
	// base64 encoding. A base64 string is composed of characters 'A'..'Z', 'a'..'z',
	// '0'..'9', '+', '/' and it is often padded at the end with up to two '=', to make
	// the length a multiple of 4. Whitespace is ignored.
	bool get_IsBase64(void);

	// The number of characters of the string contained within this instance.
	int get_Length(void);



	// ----------------------
	// Methods
	// ----------------------
	// Appends a copy of the specified string to this instance.
	bool Append(const wchar_t *value);

	// Appends the contents of binData. The charset specifies the character encoding of the
	// bytes contained in binData. The charset can be any of the supported encodings listed
	// atChilkat Supported Character Encodings
	// <http://cknotes.com/chilkat-charsets-character-encodings-supported/>. To append
	// the entire contents of binData, set offset and numBytes equal to zero. To append a range
	// of binData, set the offset and numBytes to specify the range.
	bool AppendBd(CkBinDataW &binData, const wchar_t *charset, int offset, int numBytes);

	// Appends binary data using the encoding specified by encoding, such as "base64",
	// "hex", etc.
	bool AppendEncoded(CkByteData &binaryData, const wchar_t *encoding);

	// Appends the string representation of a specified 32-bit signed integer to this
	// instance.
	bool AppendInt(int value);

	// Appends the string representation of a specified 64-bit signed integer to this
	// instance.
	bool AppendInt64(__int64 value);

	// Appends the value followed by a CRLF or LF to the end of the curent StringBuilder
	// object. If crlf is true, then a CRLF line ending is used. Otherwise a LF line
	// ending is used.
	bool AppendLine(const wchar_t *value, bool crlf);

	// Appends the contents of another StringBuilder to this instance.
	bool AppendSb(CkStringBuilderW &sb);

	// Removes all characters from the current StringBuilder instance.
	void Clear(void);

	// Returns true if the str is contained within this object. For case sensitive
	// matching, set caseSensitive equal to true. For case-insensitive, set caseSensitive equal to
	// false.
	bool Contains(const wchar_t *str, bool caseSensitive);

	// Returns true if the word is contained within this object, but only if it is a
	// whole word. This method is limited to finding whole words in strings that only
	// contains characters in the Latin1 charset (i.e. iso-8859-1 or Windows-1252). A
	// whole word can only contain alphanumeric chars where the alpha chars are
	// restricted to those of the Latin1 alpha chars. (The underscore character is also
	// considered part of a word.)
	// 
	// For case sensitive matching, set caseSensitive equal to true. For case-insensitive, set
	// caseSensitive equal to false.
	// 
	bool ContainsWord(const wchar_t *word, bool caseSensitive);

	// Returns true if the contents of this object equals the str. Returns false
	// if unequal. For case insensitive equality, set caseSensitive equal to false.
	bool ContentsEqual(const wchar_t *str, bool caseSensitive);

	// Returns true if the contents of this object equals the sb. Returns false
	// if unequal. For case insensitive equality, set caseSensitive equal to false.
	bool ContentsEqualSb(CkStringBuilderW &sb, bool caseSensitive);

	// Decodes and replaces the contents with the decoded string. The encoding can be set
	// to any of the following strings: "base64", "hex", "quoted-printable" (or "qp"),
	// "url", "base32", "Q", "B", "url_rc1738", "url_rfc2396", "url_rfc3986",
	// "url_oauth", "uu", "modBase64", or "html" (for HTML entity encoding). The full
	// up-to-date list of supported binary encodings is available at the link entitled
	// "Supported Binary Encodings" below.
	// 
	// Note: This method can only be called if the encoded content decodes to a string.
	// The charset indicates the charset to be used in intepreting the decoded bytes. For
	// example, the charset can be "utf-8", "utf-16", "iso-8859-1", "shift_JIS", etc.
	// 
	bool Decode(const wchar_t *encoding, const wchar_t *charset);

	// Decodes a binary encoded string, where the binary encoding (such as "url",
	// "hex", "base64", etc.) is specified by encoding, and the underlying charset encoding
	// (such as "utf-8", "windows-1252", etc.) is specified by charset. The decoded string
	// is appended to this object.
	bool DecodeAndAppend(const wchar_t *value, const wchar_t *encoding, const wchar_t *charset);

	// Encodes to base64, hex, quoted-printable, URL encoding, etc. The encoding can be set
	// to any of the following strings: "base64", "hex", "quoted-printable" (or "qp"),
	// "url", "base32", "Q", "B", "url_rc1738", "url_rfc2396", "url_rfc3986",
	// "url_oauth", "uu", "modBase64", or "html" (for HTML entity encoding). The full
	// up-to-date list of supported binary encodings is available at the link entitled
	// "Supported Binary Encodings" below.
	bool Encode(const wchar_t *encoding, const wchar_t *charset);

	// Returns true if the string ends with substr. Otherwise returns false. The
	// comparison is case sensitive if caseSensitive is true, and case insensitive if caseSensitive is
	// false.
	bool EndsWith(const wchar_t *substr, bool caseSensitive);

	// Decodes HTML entities. SeeHTML entities
	// <https://duckduckgo.com/?q=html+entities&bext=wfp&ia=web> for more information
	// about HTML entities.
	bool EntityDecode(void);

	// Begin searching after the 1st occurrence of searchAfter is found, and then return the
	// substring found between the next occurrence of beginMark and the next occurrence of
	// endMark.
	bool GetAfterBetween(const wchar_t *searchAfter, const wchar_t *beginMark, const wchar_t *endMark, CkString &outStr);
	// Begin searching after the 1st occurrence of searchAfter is found, and then return the
	// substring found between the next occurrence of beginMark and the next occurrence of
	// endMark.
	const wchar_t *getAfterBetween(const wchar_t *searchAfter, const wchar_t *beginMark, const wchar_t *endMark);
	// Begin searching after the 1st occurrence of searchAfter is found, and then return the
	// substring found between the next occurrence of beginMark and the next occurrence of
	// endMark.
	const wchar_t *afterBetween(const wchar_t *searchAfter, const wchar_t *beginMark, const wchar_t *endMark);

	// Returns the substring found after the final occurrence of marker. If removeFlag is
	// true, the marker and the content that follows is removed from this content.
	// 
	// If the marker is not present, then the entire string is returned. In this case, if
	// removeFlag is true, this object is also cleared.
	// 
	bool GetAfterFinal(const wchar_t *marker, bool removeFlag, CkString &outStr);
	// Returns the substring found after the final occurrence of marker. If removeFlag is
	// true, the marker and the content that follows is removed from this content.
	// 
	// If the marker is not present, then the entire string is returned. In this case, if
	// removeFlag is true, this object is also cleared.
	// 
	const wchar_t *getAfterFinal(const wchar_t *marker, bool removeFlag);
	// Returns the substring found after the final occurrence of marker. If removeFlag is
	// true, the marker and the content that follows is removed from this content.
	// 
	// If the marker is not present, then the entire string is returned. In this case, if
	// removeFlag is true, this object is also cleared.
	// 
	const wchar_t *afterFinal(const wchar_t *marker, bool removeFlag);

	// Returns the contents as a string.
	bool GetAsString(CkString &outStr);
	// Returns the contents as a string.
	const wchar_t *getAsString(void);
	// Returns the contents as a string.
	const wchar_t *asString(void);

	// Returns the substring found before the 1st occurrence of marker. If removeFlag is
	// true, the content up to and including the marker is removed from this object's
	// contents.
	// 
	// If the marker is not present, then the entire string is returned. In this case, if
	// removeFlag is true, this object is also cleared.
	// 
	bool GetBefore(const wchar_t *marker, bool removeFlag, CkString &outStr);
	// Returns the substring found before the 1st occurrence of marker. If removeFlag is
	// true, the content up to and including the marker is removed from this object's
	// contents.
	// 
	// If the marker is not present, then the entire string is returned. In this case, if
	// removeFlag is true, this object is also cleared.
	// 
	const wchar_t *getBefore(const wchar_t *marker, bool removeFlag);
	// Returns the substring found before the 1st occurrence of marker. If removeFlag is
	// true, the content up to and including the marker is removed from this object's
	// contents.
	// 
	// If the marker is not present, then the entire string is returned. In this case, if
	// removeFlag is true, this object is also cleared.
	// 
	const wchar_t *before(const wchar_t *marker, bool removeFlag);

	// Returns the substring found between the 1st occurrence of beginMark and the next
	// occurrence of endMark.
	bool GetBetween(const wchar_t *beginMark, const wchar_t *endMark, CkString &outStr);
	// Returns the substring found between the 1st occurrence of beginMark and the next
	// occurrence of endMark.
	const wchar_t *getBetween(const wchar_t *beginMark, const wchar_t *endMark);
	// Returns the substring found between the 1st occurrence of beginMark and the next
	// occurrence of endMark.
	const wchar_t *between(const wchar_t *beginMark, const wchar_t *endMark);

	// Decodes and returns the decoded bytes. The encoding can be set to any of the
	// following strings: "base64", "hex", "quoted-printable" (or "qp"), "url",
	// "base32", "Q", "B", "url_rc1738", "url_rfc2396", "url_rfc3986", "url_oauth",
	// "uu", "modBase64", or "html" (for HTML entity encoding). The full up-to-date
	// list of supported binary encodings is available at the link entitled "Supported
	// Binary Encodings" below.
	bool GetDecoded(const wchar_t *encoding, CkByteData &outBytes);

	// Returns the string contents encoded in an encoding such as base64, hex,
	// quoted-printable, or URL-encoding. The encoding can be set to any of the following
	// strings: "base64", "hex", "quoted-printable" (or "qp"), "url", "base32", "Q",
	// "B", "url_rc1738", "url_rfc2396", "url_rfc3986", "url_oauth", "uu", "modBase64",
	// or "html" (for HTML entity encoding). The full up-to-date list of supported
	// binary encodings is available at the link entitled "Supported Binary Encodings"
	// below.
	// 
	// Note: The Encode method modifies the content of this object. The GetEncoded
	// method leaves this object's content unmodified.
	// 
	bool GetEncoded(const wchar_t *encoding, const wchar_t *charset, CkString &outStr);
	// Returns the string contents encoded in an encoding such as base64, hex,
	// quoted-printable, or URL-encoding. The encoding can be set to any of the following
	// strings: "base64", "hex", "quoted-printable" (or "qp"), "url", "base32", "Q",
	// "B", "url_rc1738", "url_rfc2396", "url_rfc3986", "url_oauth", "uu", "modBase64",
	// or "html" (for HTML entity encoding). The full up-to-date list of supported
	// binary encodings is available at the link entitled "Supported Binary Encodings"
	// below.
	// 
	// Note: The Encode method modifies the content of this object. The GetEncoded
	// method leaves this object's content unmodified.
	// 
	const wchar_t *getEncoded(const wchar_t *encoding, const wchar_t *charset);
	// Returns the string contents encoded in an encoding such as base64, hex,
	// quoted-printable, or URL-encoding. The encoding can be set to any of the following
	// strings: "base64", "hex", "quoted-printable" (or "qp"), "url", "base32", "Q",
	// "B", "url_rc1738", "url_rfc2396", "url_rfc3986", "url_oauth", "uu", "modBase64",
	// or "html" (for HTML entity encoding). The full up-to-date list of supported
	// binary encodings is available at the link entitled "Supported Binary Encodings"
	// below.
	// 
	// Note: The Encode method modifies the content of this object. The GetEncoded
	// method leaves this object's content unmodified.
	// 
	const wchar_t *encoded(const wchar_t *encoding, const wchar_t *charset);

	// Returns the Nth substring in string that is a list delimted by delimiterChar. The first
	// substring is at index 0. If exceptDoubleQuoted is true, then the delimiter char found
	// between double quotes is not treated as a delimiter. If exceptEscaped is true, then an
	// escaped (with a backslash) delimiter char is not treated as a delimiter.
	bool GetNth(int index, const wchar_t *delimiterChar, bool exceptDoubleQuoted, bool exceptEscaped, CkString &outStr);
	// Returns the Nth substring in string that is a list delimted by delimiterChar. The first
	// substring is at index 0. If exceptDoubleQuoted is true, then the delimiter char found
	// between double quotes is not treated as a delimiter. If exceptEscaped is true, then an
	// escaped (with a backslash) delimiter char is not treated as a delimiter.
	const wchar_t *getNth(int index, const wchar_t *delimiterChar, bool exceptDoubleQuoted, bool exceptEscaped);
	// Returns the Nth substring in string that is a list delimted by delimiterChar. The first
	// substring is at index 0. If exceptDoubleQuoted is true, then the delimiter char found
	// between double quotes is not treated as a delimiter. If exceptEscaped is true, then an
	// escaped (with a backslash) delimiter char is not treated as a delimiter.
	const wchar_t *nth(int index, const wchar_t *delimiterChar, bool exceptDoubleQuoted, bool exceptEscaped);

	// Returns a string containing the specified range of characters from this
	// instance. If removeFlag is true, then the range of chars is removed from this
	// instance.
	// 
	// Note: It was discovered that the range of chars was always removed regardless of
	// the value of removeFlag. This is fixed in v9.5.0.89.
	// 
	bool GetRange(int startIndex, int numChars, bool removeFlag, CkString &outStr);
	// Returns a string containing the specified range of characters from this
	// instance. If removeFlag is true, then the range of chars is removed from this
	// instance.
	// 
	// Note: It was discovered that the range of chars was always removed regardless of
	// the value of removeFlag. This is fixed in v9.5.0.89.
	// 
	const wchar_t *getRange(int startIndex, int numChars, bool removeFlag);
	// Returns a string containing the specified range of characters from this
	// instance. If removeFlag is true, then the range of chars is removed from this
	// instance.
	// 
	// Note: It was discovered that the range of chars was always removed regardless of
	// the value of removeFlag. This is fixed in v9.5.0.89.
	// 
	const wchar_t *range(int startIndex, int numChars, bool removeFlag);

	// Returns the last N lines of the text. If fewer than numLines lines exists, then all
	// of the text is returned. If bCrlf is true, then the line endings of the
	// returned string are converted to CRLF, otherwise the line endings are converted
	// to LF-only.
	bool LastNLines(int numLines, bool bCrlf, CkString &outStr);
	// Returns the last N lines of the text. If fewer than numLines lines exists, then all
	// of the text is returned. If bCrlf is true, then the line endings of the
	// returned string are converted to CRLF, otherwise the line endings are converted
	// to LF-only.
	const wchar_t *lastNLines(int numLines, bool bCrlf);

	// Loads the contents of a file.
	bool LoadFile(const wchar_t *path, const wchar_t *charset);

	// Obfuscates the string. (The Unobfuscate method can be called to reverse the
	// obfuscation to restore the original string.)
	// 
	// The Chilkat string obfuscation algorithm works by taking the utf-8 bytes of the
	// string, base64 encoding it, and then scrambling the letters of the base64
	// encoded string. It is deterministic in that the same string will always
	// obfuscate to the same result. It is NOT a secure way of encrypting a string. It
	// is only meant to be a simple means of transforming a string into something
	// unintelligible.
	// 
	void Obfuscate(void);

	// Prepends a copy of the specified string to this instance.
	bool Prepend(const wchar_t *value);

	// In-place decodes the string from punycode.
	bool PunyDecode(void);

	// In-place encodes the string to punycode.
	bool PunyEncode(void);

	// Removes the substring found after the final occurrence of the marker. Also removes
	// the marker. Returns true if the marker was found and content was removed.
	// Otherwise returns false.
	bool RemoveAfterFinal(const wchar_t *marker);

	// Removes the substring found before the 1st occurrence of the marker. Also removes
	// the marker. Returns true if the marker was found and content was removed.
	// Otherwise returns false.
	bool RemoveBefore(const wchar_t *marker);

	// Removes the specified range of characters from this instance.
	bool RemoveCharsAt(int startIndex, int numChars);

	// Replaces all occurrences of a specified string in this instance with another
	// specified string. Returns the number of replacements.
	int Replace(const wchar_t *value, const wchar_t *replacement);

	// Replaces the content found after the final occurrence of marker with replacement.
	bool ReplaceAfterFinal(const wchar_t *marker, const wchar_t *replacement);

	// Replaces the first occurrence of ALL the content found between beginMark and endMark
	// with replacement. The beginMark and endMark are included in what is replaced if replaceMarks is true.
	bool ReplaceAllBetween(const wchar_t *beginMark, const wchar_t *endMark, const wchar_t *replacement, bool replaceMarks);

	// Replaces all occurrences of value with replacement, but only where value is found between
	// beginMark and endMark. Returns the number of replacements made.
	int ReplaceBetween(const wchar_t *beginMark, const wchar_t *endMark, const wchar_t *value, const wchar_t *replacement);

	// Replaces the first occurrence of a specified string in this instance with
	// another string. Returns true if the value was found and replaced. Otherwise
	// returns false.
	bool ReplaceFirst(const wchar_t *value, const wchar_t *replacement);

	// Replaces all occurrences of value with the decimal integer replacement. Returns the
	// number of replacements.
	int ReplaceI(const wchar_t *value, int replacement);

	// Replaces all occurrences of value with replacement (case insensitive). Returns the
	// number of replacements.
	int ReplaceNoCase(const wchar_t *value, const wchar_t *replacement);

	// Replaces all word occurrences of a specified string in this instance with
	// another specified string. Returns the number of replacements made.
	// 
	// Important: This method is limited to replacing whole words in strings that only
	// contains characters in the Latin1 charset (i.e. iso-8859-1 or Windows-1252). A
	// whole word can only contain alphanumeric chars where the alpha chars are
	// restricted to those of the Latin1 alpha chars. (The underscore character is also
	// considered part of a word.)
	// 
	int ReplaceWord(const wchar_t *value, const wchar_t *replacement);

	// Removes all characters from the current StringBuilder instance, and write zero
	// bytes to the allocated memory before deallocating.
	void SecureClear(void);

	// Sets the Nth substring in string in a list delimted by delimiterChar. The first substring
	// is at index 0. If exceptDoubleQuoted is true, then the delimiter char found between double
	// quotes is not treated as a delimiter. If exceptEscaped is true, then an escaped (with a
	// backslash) delimiter char is not treated as a delimiter.
	bool SetNth(int index, const wchar_t *value, const wchar_t *delimiterChar, bool exceptDoubleQuoted, bool exceptEscaped);

	// Sets this instance to a copy of the specified string.
	bool SetString(const wchar_t *value);

	// Shortens the string by removing the last numChars chars.
	bool Shorten(int numChars);

	// Returns true if the string starts with substr. Otherwise returns false. The
	// comparison is case sensitive if caseSensitive is true, and case insensitive if caseSensitive is
	// false.
	bool StartsWith(const wchar_t *substr, bool caseSensitive);

	// Converts line endings to CRLF (Windows) format.
	bool ToCRLF(void);

	// Converts line endings to LF-only (UNIX) format.
	bool ToLF(void);

	// Converts the contents to lowercase.
	bool ToLowercase(void);

	// Converts the contents to uppercase.
	bool ToUppercase(void);

	// Trims whitespace from both ends of the string.
	bool Trim(void);

	// Replaces all tabs, CR's, and LF's, with SPACE chars, and removes extra SPACE's
	// so there are no occurances of more than one SPACE char in a row.
	bool TrimInsideSpaces(void);

	// Unobfuscates the string.
	// 
	// The Chilkat string obfuscation algorithm works by taking the utf-8 bytes of the
	// string, base64 encoding it, and then scrambling the letters of the base64
	// encoded string. It is deterministic in that the same string will always
	// obfuscate to the same result. It is not a secure way of encrypting a string. It
	// is only meant to be a simple means of transforming a string into something
	// unintelligible.
	// 
	void Unobfuscate(void);

	// Writes the contents to a file. If emitBom is true, then the BOM (also known as a
	// preamble), is emitted for charsets that define a BOM (such as utf-8, utf-16,
	// utf-32, etc.)
	bool WriteFile(const wchar_t *path, const wchar_t *charset, bool emitBom);

	// Writes the contents to a file, but only if it is a new file or if the contents
	// are different than the existing file. If emitBom is true, then the BOM (also
	// known as a preamble), is emitted for charsets that define a BOM (such as utf-8,
	// utf-16, utf-32, etc.)
	bool WriteFileIfModified(const wchar_t *path, const wchar_t *charset, bool emitBom);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
