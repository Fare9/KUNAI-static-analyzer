// CkHashtable.h: interface for the CkHashtable class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkHashtable_H
#define _CkHashtable_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkStringBuilder;
class CkStringTable;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkHashtable
class CK_VISIBLE_PUBLIC CkHashtable  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkHashtable(const CkHashtable &);
	CkHashtable &operator=(const CkHashtable &);

    public:
	CkHashtable(void);
	virtual ~CkHashtable(void);

	static CkHashtable *createNew(void);
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
	// Adds to the hash table from XML previously created by calling ToXmlSb.
	bool AddFromXmlSb(CkStringBuilder &sbXml);


	// Adds or replaces an entry with the given key and integer value to the hash
	// table. Returns true if a new hash entry was added or replaced. Returns false
	// if an out-of-memory condition occurred.
	bool AddInt(const char *key, int value);


	// Adds URL query parameters into the hashtable. The queryParams has the form:
	// "field1=value1&field2=value2&field3=value3...". It is assumed that the values
	// are URL encoded, and this method automatically URL decodes the values prior to
	// inserting into the hashtable.
	bool AddQueryParams(const char *queryParams);


	// Adds or replaces an entry with the given key and string value to the hash table.
	// Returns true if a new hash entry was added or replaced. Returns false if an
	// out-of-memory condition occurred.
	bool AddStr(const char *key, const char *value);


	// Removes all elements from the Hashtable.
	void Clear(void);


	// Removes all elements from the Hashtable and re-sizes with the specified capacity.
	// 
	// The capacity is the number of buckets in the hash table. In the case of a "hash
	// collision", a single bucket stores multiple entries, which must be searched
	// sequentially. One rule of thumb is to set the capacity to twice the number of
	// expected items to be hashed. It's also preferable to set the capacity to a prime
	// number. (The 1st 10,000 prime numbers are listed here:
	// https://primes.utm.edu/lists/small/10000.txt )
	// 
	// The initial default capacity of the hash table is 521.
	// 
	bool ClearWithNewCapacity(int capacity);


	// Determines if a given key is contained within the hash table. Returns true if
	// the key exists, otherwise returns false.
	bool Contains(const char *key);


	// Determines if a given key is contained within the hash table. Returns true if
	// the key exists, otherwise returns false.
	bool ContainsIntKey(int key);


	// Appends the complete set of hashtable key strings to strTable.
	bool GetKeys(CkStringTable &strTable);


	// Returns the integer value associated with the specified key. If the key is not
	// in the hash table, the return value is 0.
	int LookupInt(const char *key);


	// Returns the string value associated with the specified key.
	bool LookupStr(const char *key, CkString &outStr);

	// Returns the string value associated with the specified key.
	const char *lookupStr(const char *key);

	// Removes the entry with the specified key from the hash table. Returns true if
	// the key existed and was removed. Returns false if the key did not already
	// exist.
	bool Remove(const char *key);


	// Serializes the hash table to XML format. The XML is appended to sbXml.
	bool ToXmlSb(CkStringBuilder &sbXml);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
