// CkBz2.h: interface for the CkBz2 class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkBz2_H
#define _CkBz2_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacks.h"

class CkTask;
class CkByteData;
class CkBaseProgress;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkBz2
class CK_VISIBLE_PUBLIC CkBz2  : public CkClassWithCallbacks
{
    private:

	// Don't allow assignment or copying these objects.
	CkBz2(const CkBz2 &);
	CkBz2 &operator=(const CkBz2 &);

    public:
	CkBz2(void);
	virtual ~CkBz2(void);

	static CkBz2 *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	CkBaseProgress *get_EventCallbackObject(void) const;
	void put_EventCallbackObject(CkBaseProgress *progress);


	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// When set to true, causes the currently running method to abort. Methods that
	// always finish quickly (i.e.have no length file operations or network
	// communications) are not affected. If no method is running, then this property is
	// automatically reset to false when the next method is called. When the abort
	// occurs, this property is reset to false. Both synchronous and asynchronous
	// method calls can be aborted. (A synchronous method call could be aborted by
	// setting this property from a separate thread.)
	bool get_AbortCurrent(void);
	// When set to true, causes the currently running method to abort. Methods that
	// always finish quickly (i.e.have no length file operations or network
	// communications) are not affected. If no method is running, then this property is
	// automatically reset to false when the next method is called. When the abort
	// occurs, this property is reset to false. Both synchronous and asynchronous
	// method calls can be aborted. (A synchronous method call could be aborted by
	// setting this property from a separate thread.)
	void put_AbortCurrent(bool newVal);

	// This is the number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any compress or decompress
	// operation prior to completion. If HeartbeatMs is 0, no AbortCheck event
	// callbacks will occur.
	int get_HeartbeatMs(void);
	// This is the number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any compress or decompress
	// operation prior to completion. If HeartbeatMs is 0, no AbortCheck event
	// callbacks will occur.
	void put_HeartbeatMs(int newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Compresses a file to create a BZip2 compressed file (.bz2).
	// 
	// Note: Both inFilename and toPath should be relative or absolute file paths (not just a
	// path to a directory). For example "someDir1/someDir2/myFile.txt" or
	// "c:/someDir1/myFile.bz2".
	// 
	bool CompressFile(const char *inFilename, const char *toPath);

	// Compresses a file to create a BZip2 compressed file (.bz2).
	// 
	// Note: Both inFilename and toPath should be relative or absolute file paths (not just a
	// path to a directory). For example "someDir1/someDir2/myFile.txt" or
	// "c:/someDir1/myFile.bz2".
	// 
	CkTask *CompressFileAsync(const char *inFilename, const char *toPath);


	// BZip2 compresses a file to an in-memory image of a .bz2 file.
	bool CompressFileToMem(const char *inFilename, CkByteData &outBytes);

	// BZip2 compresses a file to an in-memory image of a .bz2 file.
	CkTask *CompressFileToMemAsync(const char *inFilename);


	// Compresses in-memory data to an in-memory image of a .bz2 file.
	bool CompressMemory(CkByteData &inData, CkByteData &outBytes);

	// Compresses in-memory data to an in-memory image of a .bz2 file.
	CkTask *CompressMemoryAsync(CkByteData &inData);


	// BZip2 compresses and creates a .bz2 file from in-memory data.
	bool CompressMemToFile(CkByteData &inData, const char *toPath);

	// BZip2 compresses and creates a .bz2 file from in-memory data.
	CkTask *CompressMemToFileAsync(CkByteData &inData, const char *toPath);


	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTask &task);


	// Unzips a .bz2 file.
	bool UncompressFile(const char *inFilename, const char *toPath);

	// Unzips a .bz2 file.
	CkTask *UncompressFileAsync(const char *inFilename, const char *toPath);


	// Unzips a .bz2 file directly to memory.
	bool UncompressFileToMem(const char *inFilename, CkByteData &outBytes);

	// Unzips a .bz2 file directly to memory.
	CkTask *UncompressFileToMemAsync(const char *inFilename);


	// Unzips from an in-memory image of a .bz2 file directly into memory.
	bool UncompressMemory(CkByteData &inData, CkByteData &outBytes);

	// Unzips from an in-memory image of a .bz2 file directly into memory.
	CkTask *UncompressMemoryAsync(CkByteData &inData);


	// Unzips from an in-memory image of a .bz2 file to a file.
	bool UncompressMemToFile(CkByteData &inData, const char *toPath);

	// Unzips from an in-memory image of a .bz2 file to a file.
	CkTask *UncompressMemToFileAsync(CkByteData &inData, const char *toPath);


	// Unlocks the component allowing for the full functionality to be used. If a
	// purchased unlock code is passed, there is no expiration. Any other string
	// automatically begins a fully-functional 30-day trial the first time
	// UnlockComponent is called.
	bool UnlockComponent(const char *regCode);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
