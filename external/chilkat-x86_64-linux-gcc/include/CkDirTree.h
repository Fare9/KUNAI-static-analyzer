// CkDirTree.h: interface for the CkDirTree class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkDirTree_H
#define _CkDirTree_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"




#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkDirTree
class CK_VISIBLE_PUBLIC CkDirTree  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkDirTree(const CkDirTree &);
	CkDirTree &operator=(const CkDirTree &);

    public:
	CkDirTree(void);
	virtual ~CkDirTree(void);

	static CkDirTree *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// Begin iterating from this directory.
	void get_BaseDir(CkString &str);
	// Begin iterating from this directory.
	const char *baseDir(void);
	// Begin iterating from this directory.
	void put_BaseDir(const char *newVal);

	// Set to true when the last file or sub-directory has been iterated.
	bool get_DoneIterating(void);

	// The file size of the current file in the iteration. (0 if it is a directory.)
	int get_FileSize32(void);

	// The file size as a 64-bit integer of the current file in the iteration. (0 if it
	// is a directory.)
	__int64 get_FileSize64(void);

	// The file size of the current file in the iteration as a decimal string. (0 if it
	// is a directory.)
	void get_FileSizeStr(CkString &str);
	// The file size of the current file in the iteration as a decimal string. (0 if it
	// is a directory.)
	const char *fileSizeStr(void);

	// The absolute directory path of the current file or sub-directory.
	void get_FullPath(CkString &str);
	// The absolute directory path of the current file or sub-directory.
	const char *fullPath(void);

	// The absolute UNC directory path of the current file or sub-directory.
	void get_FullUncPath(CkString &str);
	// The absolute UNC directory path of the current file or sub-directory.
	const char *fullUncPath(void);

	// true if the current position is a sub-directory, false if it is a file.
	bool get_IsDirectory(void);

	// If true, the iteration will be recursive. If false the iteration is
	// non-recursive. The default value is true.
	bool get_Recurse(void);
	// If true, the iteration will be recursive. If false the iteration is
	// non-recursive. The default value is true.
	void put_Recurse(bool newVal);

	// The relative directory path of the current file or sub-directory. (Relative to
	// the BaseDir)
	void get_RelativePath(CkString &str);
	// The relative directory path of the current file or sub-directory. (Relative to
	// the BaseDir)
	const char *relativePath(void);



	// ----------------------
	// Methods
	// ----------------------
	// Advances the current position in the directory tree traversal to the next file
	// or sub-directory.
	// 
	// Important: If AdvancePosition returns false, it can be an error, or it could
	// be that there are no more files and directories. To distinguish between the two
	// cases, examine the DoneIterating property. If DoneIterating is true, then the
	// false return value is not an error, but instead indicates that the end has
	// been reached.
	// 
	bool AdvancePosition(void);


	// Begins a directory tree traversal. After calling this method, the various
	// property values such as Fullpath, FileSize32, etc. can be retrieved for the 1st
	// file / sub-directory in the traversal.
	// 
	// Important: If BeginIterate returns false, it can be an error, or it could be
	// that there are 0 files and directories. To distinguish between the two cases,
	// examine the DoneIterating property. If DoneIterating is true, then the false
	// return value is not an error, but instead indicates 0 files/directories.
	// 
	bool BeginIterate(void);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
