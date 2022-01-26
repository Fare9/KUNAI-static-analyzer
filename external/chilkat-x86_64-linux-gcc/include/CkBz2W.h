// CkBz2W.h: interface for the CkBz2W class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkBz2W_H
#define _CkBz2W_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacksW.h"

class CkTaskW;
class CkByteData;
class CkBaseProgressW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkBz2W
class CK_VISIBLE_PUBLIC CkBz2W  : public CkClassWithCallbacksW
{
	private:
	bool m_cbOwned;

	private:
	
	// Don't allow assignment or copying these objects.
	CkBz2W(const CkBz2W &);
	CkBz2W &operator=(const CkBz2W &);

    public:
	CkBz2W(void);
	virtual ~CkBz2W(void);

	

	static CkBz2W *createNew(void);
	

	CkBz2W(bool bCallbackOwned);
	static CkBz2W *createNew(bool bCallbackOwned);

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	CkBaseProgressW *get_EventCallbackObject(void) const;
	void put_EventCallbackObject(CkBaseProgressW *progress);


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
	bool CompressFile(const wchar_t *inFilename, const wchar_t *toPath);

	// Creates an asynchronous task to call the CompressFile method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CompressFileAsync(const wchar_t *inFilename, const wchar_t *toPath);

	// BZip2 compresses a file to an in-memory image of a .bz2 file.
	bool CompressFileToMem(const wchar_t *inFilename, CkByteData &outBytes);

	// Creates an asynchronous task to call the CompressFileToMem method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CompressFileToMemAsync(const wchar_t *inFilename);

	// Compresses in-memory data to an in-memory image of a .bz2 file.
	bool CompressMemory(CkByteData &inData, CkByteData &outBytes);

	// Creates an asynchronous task to call the CompressMemory method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CompressMemoryAsync(CkByteData &inData);

	// BZip2 compresses and creates a .bz2 file from in-memory data.
	bool CompressMemToFile(CkByteData &inData, const wchar_t *toPath);

	// Creates an asynchronous task to call the CompressMemToFile method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CompressMemToFileAsync(CkByteData &inData, const wchar_t *toPath);

	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTaskW &task);

	// Unzips a .bz2 file.
	bool UncompressFile(const wchar_t *inFilename, const wchar_t *toPath);

	// Creates an asynchronous task to call the UncompressFile method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UncompressFileAsync(const wchar_t *inFilename, const wchar_t *toPath);

	// Unzips a .bz2 file directly to memory.
	bool UncompressFileToMem(const wchar_t *inFilename, CkByteData &outBytes);

	// Creates an asynchronous task to call the UncompressFileToMem method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UncompressFileToMemAsync(const wchar_t *inFilename);

	// Unzips from an in-memory image of a .bz2 file directly into memory.
	bool UncompressMemory(CkByteData &inData, CkByteData &outBytes);

	// Creates an asynchronous task to call the UncompressMemory method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UncompressMemoryAsync(CkByteData &inData);

	// Unzips from an in-memory image of a .bz2 file to a file.
	bool UncompressMemToFile(CkByteData &inData, const wchar_t *toPath);

	// Creates an asynchronous task to call the UncompressMemToFile method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UncompressMemToFileAsync(CkByteData &inData, const wchar_t *toPath);

	// Unlocks the component allowing for the full functionality to be used. If a
	// purchased unlock code is passed, there is no expiration. Any other string
	// automatically begins a fully-functional 30-day trial the first time
	// UnlockComponent is called.
	bool UnlockComponent(const wchar_t *regCode);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
