// CkSFtpDir.h: interface for the CkSFtpDir class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkSFtpDir_H
#define _CkSFtpDir_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkSFtpFile;
class CkTask;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkSFtpDir
class CK_VISIBLE_PUBLIC CkSFtpDir  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkSFtpDir(const CkSFtpDir &);
	CkSFtpDir &operator=(const CkSFtpDir &);

    public:
	CkSFtpDir(void);
	virtual ~CkSFtpDir(void);

	static CkSFtpDir *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The number of entries in this directory listing.
	int get_NumFilesAndDirs(void);

	// The original path used to fetch this directory listing. This is the string that
	// was originally passed to the OpenDir method when the directory was read.
	void get_OriginalPath(CkString &str);
	// The original path used to fetch this directory listing. This is the string that
	// was originally passed to the OpenDir method when the directory was read.
	const char *originalPath(void);



	// ----------------------
	// Methods
	// ----------------------
	// Returns the Nth filename in the directory (indexing begins at 0).
	bool GetFilename(int index, CkString &outStr);

	// Returns the Nth filename in the directory (indexing begins at 0).
	const char *getFilename(int index);
	// Returns the Nth filename in the directory (indexing begins at 0).
	const char *filename(int index);


	// Returns the Nth entry in the directory. Indexing begins at 0.
	// The caller is responsible for deleting the object returned by this method.
	CkSFtpFile *GetFileObject(int index);


	// Loads the SFTP directory object from a completed asynchronous task.
	bool LoadTaskResult(CkTask &task);


	// Sorts the files and sub-directories in ascending or descending order based on
	// the field. Possible values for field are "filename", "filenameNoCase",
	// "lastModifiedTime", "lastAccessTime", "lastCreateTime", or "size". (For
	// case-insensitive filename sorting, use "filenameNoCase".)
	void Sort(const char *field, bool ascending);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
