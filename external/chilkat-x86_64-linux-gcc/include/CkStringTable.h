// CkStringTable.h: interface for the CkStringTable class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkStringTable_H
#define _CkStringTable_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkStringBuilder;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkStringTable
class CK_VISIBLE_PUBLIC CkStringTable  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkStringTable(const CkStringTable &);
	CkStringTable &operator=(const CkStringTable &);

    public:
	CkStringTable(void);
	virtual ~CkStringTable(void);

	static CkStringTable *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The number of strings in the table.
	int get_Count(void);



	// ----------------------
	// Methods
	// ----------------------
	// Appends a string to the table.
	bool Append(const char *value);


	// Appends strings, one per line, from a file. Each line in the path should be no
	// longer than the length specified in maxLineLen. The charset indicates the character
	// encoding of the contents of the file, such as "utf-8", "iso-8859-1",
	// "Shift_JIS", etc.
	bool AppendFromFile(int maxLineLen, const char *charset, const char *path);


	// Appends strings, one per line, from the contents of a StringBuilder object.
	bool AppendFromSb(CkStringBuilder &sb);


	// Removes all the strings from the table.
	void Clear(void);


	// Return the index of the first string in the table containing substr. Begins
	// searching strings starting at startIndex. If caseSensitive is true, then the search is case
	// sensitive. If caseSensitive is false then the search is case insensitive. Returns -1 if
	// the substr is not found.
	int FindSubstring(int startIndex, const char *substr, bool caseSensitive);


	// Return the number of strings specified by count, one per line, starting at startIdx.
	// To return the entire table, pass 0 values for both startIdx and count. Set crlf equal
	// to true to emit with CRLF line endings, or false to emit LF-only line
	// endings. The last string is emitted includes the line ending.
	bool GetStrings(int startIdx, int count, bool crlf, CkString &outStr);

	// Return the number of strings specified by count, one per line, starting at startIdx.
	// To return the entire table, pass 0 values for both startIdx and count. Set crlf equal
	// to true to emit with CRLF line endings, or false to emit LF-only line
	// endings. The last string is emitted includes the line ending.
	const char *getStrings(int startIdx, int count, bool crlf);
	// Return the number of strings specified by count, one per line, starting at startIdx.
	// To return the entire table, pass 0 values for both startIdx and count. Set crlf equal
	// to true to emit with CRLF line endings, or false to emit LF-only line
	// endings. The last string is emitted includes the line ending.
	const char *strings(int startIdx, int count, bool crlf);


	// Returns the Nth string in the table, converted to an integer value. The index is
	// 0-based. (The first string is at index 0.) Returns -1 if no string is found at
	// the specified index. Returns 0 if the string at the specified index exist, but
	// is not an integer.
	int IntAt(int index);


	// Saves the string table to a file. The charset is the character encoding to use,
	// such as "utf-8", "iso-8859-1", "windows-1252", "Shift_JIS", "gb2312", etc. If
	// bCrlf is true, then CRLF line endings are used, otherwise LF-only line endings
	// are used.
	bool SaveToFile(const char *charset, bool bCrlf, const char *path);


	// Sorts the strings in the collection in ascending or descending order. To sort in
	// ascending order, set ascending to true, otherwise set ascending equal to false.
	bool Sort(bool ascending, bool caseSensitive);


	// Splits a string into parts based on a single character delimiterChar. If exceptDoubleQuoted is true,
	// then the delimiter char found between double quotes is not treated as a
	// delimiter. If exceptEscaped is true, then an escaped (with a backslash) delimiter char
	// is not treated as a delimiter.
	bool SplitAndAppend(const char *inStr, const char *delimiterChar, bool exceptDoubleQuoted, bool exceptEscaped);


	// Returns the Nth string in the table. The index is 0-based. (The first string is
	// at index 0.)
	bool StringAt(int index, CkString &outStr);

	// Returns the Nth string in the table. The index is 0-based. (The first string is
	// at index 0.)
	const char *stringAt(int index);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
