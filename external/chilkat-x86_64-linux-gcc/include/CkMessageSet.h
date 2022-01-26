// CkMessageSet.h: interface for the CkMessageSet class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkMessageSet_H
#define _CkMessageSet_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkTask;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkMessageSet
class CK_VISIBLE_PUBLIC CkMessageSet  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkMessageSet(const CkMessageSet &);
	CkMessageSet &operator=(const CkMessageSet &);

    public:
	CkMessageSet(void);
	virtual ~CkMessageSet(void);

	static CkMessageSet *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The number of message UIDs (or sequence numbers) in this message set.
	int get_Count(void);

	// If true then the message set contains UIDs, otherwise it contains sequence
	// numbers.
	bool get_HasUids(void);
	// If true then the message set contains UIDs, otherwise it contains sequence
	// numbers.
	void put_HasUids(bool newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Returns true if the msgId is contained in the message set.
	bool ContainsId(int msgId);


	// Loads the message set from a compact-string representation. Here are some
	// examples:
	// 
	// Non-Compact String
	// 
	// Compact String
	// 
	// 1,2,3,4,5
	// 
	// 1:5
	// 
	// 1,2,3,4,5,8,9,10
	// 
	// 1:5,8:10
	// 
	// 1,3,4,5,8,9,10
	// 
	// 1,3:5,8:10
	// 
	bool FromCompactString(const char *str);


	// Returns the message ID of the Nth message in the set. (indexing begins at 0).
	// Returns -1 if the index is out of range.
	int GetId(int index);


	// Inserts a message ID into the set. If the ID already exists, a duplicate is not
	// inserted.
	void InsertId(int id);


	// Loads the message set from a completed asynchronous task.
	bool LoadTaskResult(CkTask &task);


	// Removes a message ID from the set.
	void RemoveId(int id);


	// Returns a string of comma-separated message IDs. (This is the non-compact string
	// format.)
	bool ToCommaSeparatedStr(CkString &outStr);

	// Returns a string of comma-separated message IDs. (This is the non-compact string
	// format.)
	const char *toCommaSeparatedStr(void);

	// Returns the set of message IDs represented as a compact string. Here are some
	// examples:
	// 
	// Non-Compact String
	// 
	// Compact String
	// 
	// 1,2,3,4,5
	// 
	// 1:5
	// 
	// 1,2,3,4,5,8,9,10
	// 
	// 1:5,8:10
	// 
	// 1,3,4,5,8,9,10
	// 
	// 1,3:5,8:10
	// 
	bool ToCompactString(CkString &outStr);

	// Returns the set of message IDs represented as a compact string. Here are some
	// examples:
	// 
	// Non-Compact String
	// 
	// Compact String
	// 
	// 1,2,3,4,5
	// 
	// 1:5
	// 
	// 1,2,3,4,5,8,9,10
	// 
	// 1:5,8:10
	// 
	// 1,3,4,5,8,9,10
	// 
	// 1,3:5,8:10
	// 
	const char *toCompactString(void);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
