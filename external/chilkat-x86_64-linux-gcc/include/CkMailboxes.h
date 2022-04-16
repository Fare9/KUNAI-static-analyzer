// CkMailboxes.h: interface for the CkMailboxes class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkMailboxes_H
#define _CkMailboxes_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkTask;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkMailboxes
class CK_VISIBLE_PUBLIC CkMailboxes  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkMailboxes(const CkMailboxes &);
	CkMailboxes &operator=(const CkMailboxes &);

    public:
	CkMailboxes(void);
	virtual ~CkMailboxes(void);

	static CkMailboxes *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The number of mailboxes in the collection.
	// 
	// Note: The Mailboxes class is for use with the Chilkat IMAP component.
	// 
	int get_Count(void);



	// ----------------------
	// Methods
	// ----------------------
	// Returns a comma-separated list of flags for the Nth mailbox. For example,
	// "\HasNoChildren,\Important".
	bool GetFlags(int index, CkString &outStr);

	// Returns a comma-separated list of flags for the Nth mailbox. For example,
	// "\HasNoChildren,\Important".
	const char *getFlags(int index);
	// Returns a comma-separated list of flags for the Nth mailbox. For example,
	// "\HasNoChildren,\Important".
	const char *flags(int index);


	// Returns the index of the mailbox having the specified name.
	int GetMailboxIndex(const char *mbxName);


	// The name of the Nth mailbox in this collection. Indexing begins at 0.
	bool GetName(int index, CkString &outStrName);

	// The name of the Nth mailbox in this collection. Indexing begins at 0.
	const char *getName(int index);
	// The name of the Nth mailbox in this collection. Indexing begins at 0.
	const char *name(int index);


	// Returns the name of the Nth flag for the Mth mailbox. The index is the index of
	// the mailbox. The flagIndex is the index of the flag.
	bool GetNthFlag(int index, int flagIndex, CkString &outStr);

	// Returns the name of the Nth flag for the Mth mailbox. The index is the index of
	// the mailbox. The flagIndex is the index of the flag.
	const char *getNthFlag(int index, int flagIndex);
	// Returns the name of the Nth flag for the Mth mailbox. The index is the index of
	// the mailbox. The flagIndex is the index of the flag.
	const char *nthFlag(int index, int flagIndex);


	// Returns the number of flags for the Nth mailbox. Returns -1 if the index is out
	// of range.
	int GetNumFlags(int index);


	// Returns true if the Nth mailbox has the specified flag set. The flag name is
	// case insensitive and should begin with a backslash character, such as
	// "\Flagged". The index is the index of the Nth mailbox.
	bool HasFlag(int index, const char *flagName);


	// Returns true if the Nth mailbox has inferiors (i.e. sub-mailboxes), or if it
	// is possible to create child mailboxes in the future.
	// 
	// Note: the HasNoChildren attribute/flag should not be confused with the IMAP4
	// [RFC-2060] defined attribute Noinferiors which indicates that no child mailboxes
	// exist now AND none can be created in the future.
	// 
	bool HasInferiors(int index);


	// Returns true if the Nth mailbox is marked.
	bool IsMarked(int index);


	// Returns true if the Nth mailbox is selectable.
	bool IsSelectable(int index);


	// Loads the mailboxes object from a completed asynchronous task.
	bool LoadTaskResult(CkTask &task);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
