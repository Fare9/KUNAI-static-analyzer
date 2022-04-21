// CkJsonArrayW.h: interface for the CkJsonArrayW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkJsonArrayW_H
#define _CkJsonArrayW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkWideCharBase.h"

class CkJsonObjectW;
class CkDateTimeW;
class CkDtObjW;
class CkStringBuilderW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkJsonArrayW
class CK_VISIBLE_PUBLIC CkJsonArrayW  : public CkWideCharBase
{
	

	private:
	
	// Don't allow assignment or copying these objects.
	CkJsonArrayW(const CkJsonArrayW &);
	CkJsonArrayW &operator=(const CkJsonArrayW &);

    public:
	CkJsonArrayW(void);
	virtual ~CkJsonArrayW(void);

	

	static CkJsonArrayW *createNew(void);
	

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// If true then the Emit method outputs in the most compact form possible (a
	// single-line with no extra whitespace). If false, then emits with whitespace
	// and indentation to make the JSON human-readable.
	// 
	// The default value is true.
	// 
	bool get_EmitCompact(void);
	// If true then the Emit method outputs in the most compact form possible (a
	// single-line with no extra whitespace). If false, then emits with whitespace
	// and indentation to make the JSON human-readable.
	// 
	// The default value is true.
	// 
	void put_EmitCompact(bool newVal);

	// If true then the Emit method uses CRLF line-endings when emitting the
	// non-compact (pretty-print) format. If false, then bare-LF's are emitted. (The
	// compact format emits to a single line with no end-of-line characters.) Windows
	// systems traditionally use CRLF line-endings, whereas Linux, Mac OS X, and other
	// systems traditionally use bare-LF line-endings.
	// 
	// The default value is true.
	// 
	bool get_EmitCrlf(void);
	// If true then the Emit method uses CRLF line-endings when emitting the
	// non-compact (pretty-print) format. If false, then bare-LF's are emitted. (The
	// compact format emits to a single line with no end-of-line characters.) Windows
	// systems traditionally use CRLF line-endings, whereas Linux, Mac OS X, and other
	// systems traditionally use bare-LF line-endings.
	// 
	// The default value is true.
	// 
	void put_EmitCrlf(bool newVal);

	// The number of JSON values in the array.
	int get_Size(void);



	// ----------------------
	// Methods
	// ----------------------
	// Inserts a new and empty JSON array member to the position indicated by index. To
	// prepend, pass an index of 0. To append, pass an index of -1. Indexing is 0-based
	// (the 1st member is at index 0).
	bool AddArrayAt(int index);

	// Inserts a new boolean member to the position indicated by index. To prepend, pass
	// an index of 0. To append, pass an index of -1. Indexing is 0-based (the 1st member
	// is at index 0).
	bool AddBoolAt(int index, bool value);

	// Inserts a new integer member to the position indicated by index. To prepend, pass
	// an index of 0. To append, pass an index of -1. Indexing is 0-based (the 1st member
	// is at index 0).
	bool AddIntAt(int index, int value);

	// Inserts a new null member to the position indicated by index. To prepend, pass an
	// index of 0. To append, pass an index of -1. Indexing is 0-based (the 1st member is
	// at index 0).
	bool AddNullAt(int index);

	// Inserts a new numeric member to the position indicated by index. The numericStr is an
	// integer, float, or double already converted to a string in the format desired by
	// the application. To prepend, pass an index of 0. To append, pass an index of -1.
	// Indexing is 0-based (the 1st member is at index 0).
	bool AddNumberAt(int index, const wchar_t *numericStr);

	// Inserts a new and empty JSON object member to the position indicated by index. To
	// prepend, pass an index of 0. To append, pass an index of -1. Indexing is 0-based
	// (the 1st member is at index 0).
	bool AddObjectAt(int index);

	// Inserts a copy of a JSON object to the position indicated by index. To prepend,
	// pass an index of 0. To append, pass an index of -1. Indexing is 0-based (the 1st
	// member is at index 0).
	bool AddObjectCopyAt(int index, CkJsonObjectW &jsonObj);

	// Inserts a new string at the position indicated by index. To prepend, pass an index
	// of 0. To append, pass an index of -1. Indexing is 0-based (the 1st member is at
	// index 0).
	bool AddStringAt(int index, const wchar_t *value);

	// Appends the array items contained in jarr.
	bool AppendArrayItems(CkJsonArrayW &jarr);

	// Returns the JSON array that is the value of the Nth array element. Indexing is
	// 0-based (the 1st member is at index 0).
	// The caller is responsible for deleting the object returned by this method.
	CkJsonArrayW *ArrayAt(int index);

	// Returns the boolean value of the Nth array element. Indexing is 0-based (the 1st
	// member is at index 0).
	bool BoolAt(int index);

	// Deletes all array elements.
	void Clear(void);

	// Fills the dateTime with the date/time string located in the Nth array element.
	// Indexing is 0-based (the 1st member is at index 0). Auto-recognizes the
	// following date/time string formats: ISO-8061 Timestamp (such as
	// "2009-11-04T19:55:41Z"), RFC822 date/time format (such as "Wed, 18 Apr 2018
	// 15:51:55 -0400"), or Unix timestamp integers.
	bool DateAt(int index, CkDateTimeW &dateTime);

	// Deletes the array element at the given index. Indexing is 0-based (the 1st member
	// is at index 0).
	bool DeleteAt(int index);

	// Fills the dt with the date/time string located in the Nth array element. If
	// bLocal is true, then dt is filled with the local date/time values, otherwise
	// it is filled with the UTC/GMT values. Indexing is 0-based (the 1st member is at
	// index 0). Auto-recognizes the following date/time string formats: ISO-8061
	// Timestamp (such as "2009-11-04T19:55:41Z"), RFC822 date/time format (such as
	// "Wed, 18 Apr 2018 15:51:55 -0400"), or Unix timestamp integers.
	bool DtAt(int index, bool bLocal, CkDtObjW &dt);

	// Writes the JSON array (rooted at the caller) and returns as a string.
	// 
	// Note: To control the compact/non-compact format, and to control the LF/CRLF
	// line-endings, set the EmitCompact and EmitCrlf properties.
	// 
	bool Emit(CkString &outStr);
	// Writes the JSON array (rooted at the caller) and returns as a string.
	// 
	// Note: To control the compact/non-compact format, and to control the LF/CRLF
	// line-endings, set the EmitCompact and EmitCrlf properties.
	// 
// QT defines the macro "emit" globally.  (Good grief!)
#if defined(QT_VERSION)
#pragma push_macro("emit")
#undef emit
const wchar_t *emit(void);
#pragma pop_macro("emit")
#else
const wchar_t *emit(void);
#endif



	// Writes the JSON array to the sb.
	// 
	// Note: To control the compact/non-compact format, and to control the LF/CRLF
	// line-endings, set the EmitCompact and EmitCrlf properties.
	// 
	bool EmitSb(CkStringBuilderW &sb);

	// Return the index of the first object in the array where value of the field at
	// name matches value. name is an object member name. The value is a value pattern
	// which can use "*" chars to indicate zero or more of any char. If caseSensitive is
	// false, then the matching is case insenstive, otherwise it is case sensitive.
	// Returns -1 if no matching string was found.
	int FindObject(const wchar_t *name, const wchar_t *value, bool caseSensitive);

	// Return the index of the first matching string in the array. The value is a value
	// pattern which can use "*" chars to indicate zero or more of any char. If caseSensitive is
	// false, then the matching is case insenstive, otherwise it is case sensitive.
	// Returns -1 if no matching string was found.
	int FindString(const wchar_t *value, bool caseSensitive);

	// Returns the integer value of the Nth array element. Indexing is 0-based (the 1st
	// member is at index 0).
	int IntAt(int index);

	// Returns the true if the Nth array element is null, otherwise returns false.
	// Indexing is 0-based (the 1st member is at index 0).
	bool IsNullAt(int index);

	// Loads a JSON array from a string. A JSON array must begin with a "[" and end
	// with a "]".
	// 
	// Note: The Load method causes the JsonArray to detach and become it's own JSON
	// document. It should only be called on new instances of the JsonArray. See the
	// example below.
	// 
	bool Load(const wchar_t *jsonArray);

	// Loads a JSON array from a StringBuilder. A JSON array must begin with a "[" and
	// end with a "]".
	// 
	// Note: The Load method causes the JsonArray to detach and become it's own JSON
	// document. It should only be called on new instances of the JsonArray. See the
	// example below.
	// 
	bool LoadSb(CkStringBuilderW &sb);

	// Returns the JSON object that is the value of the Nth array element. Indexing is
	// 0-based (the 1st member is at index 0).
	// The caller is responsible for deleting the object returned by this method.
	CkJsonObjectW *ObjectAt(int index);

	// Sets the boolean value of the Nth array element. Indexing is 0-based (the 1st
	// member is at index 0).
	bool SetBoolAt(int index, bool value);

	// Sets the integer value of the Nth array element. Indexing is 0-based (the 1st
	// member is at index 0).
	bool SetIntAt(int index, int value);

	// Sets the Nth array element to the value of null. Indexing is 0-based (the 1st
	// member is at index 0).
	bool SetNullAt(int index);

	// Sets the numeric value of the Nth array element. The value is an integer, float,
	// or double already converted to a string in the format desired by the
	// application. Indexing is 0-based (the 1st member is at index 0).
	bool SetNumberAt(int index, const wchar_t *value);

	// Sets the string value of the Nth array element. Indexing is 0-based (the 1st
	// member is at index 0).
	bool SetStringAt(int index, const wchar_t *value);

	// Returns the string value of the Nth array element. Indexing is 0-based (the 1st
	// member is at index 0).
	bool StringAt(int index, CkString &outStr);
	// Returns the string value of the Nth array element. Indexing is 0-based (the 1st
	// member is at index 0).
	const wchar_t *stringAt(int index);

	// Swaps the items at positions index1 and index2.
	bool Swap(int index1, int index2);

	// Returns the type of data at the given index. Possible return values are:
	//     string
	//     number
	//     object
	//     array
	//     boolean
	//     null
	// Returns -1 if no member exists at the given index.
	int TypeAt(int index);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
