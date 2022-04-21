// CkCgiW.h: interface for the CkCgiW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkCgiW_H
#define _CkCgiW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkWideCharBase.h"

class CkByteData;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkCgiW
class CK_VISIBLE_PUBLIC CkCgiW  : public CkWideCharBase
{
	

	private:
	
	// Don't allow assignment or copying these objects.
	CkCgiW(const CkCgiW &);
	CkCgiW &operator=(const CkCgiW &);

    public:
	CkCgiW(void);
	virtual ~CkCgiW(void);

	

	static CkCgiW *createNew(void);
	

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------

	int get_AsyncBytesRead(void);


	bool get_AsyncInProgress(void);


	int get_AsyncPostSize(void);


	bool get_AsyncSuccess(void);


	int get_HeartbeatMs(void);

	void put_HeartbeatMs(int newVal);


	int get_IdleTimeoutMs(void);

	void put_IdleTimeoutMs(int newVal);


	int get_NumParams(void);


	int get_NumUploadFiles(void);


	int get_ReadChunkSize(void);

	void put_ReadChunkSize(int newVal);


	int get_SizeLimitKB(void);

	void put_SizeLimitKB(int newVal);


	bool get_StreamToUploadDir(void);

	void put_StreamToUploadDir(bool newVal);


	void get_UploadDir(CkString &str);

	const wchar_t *uploadDir(void);

	void put_UploadDir(const wchar_t *newVal);



	// ----------------------
	// Methods
	// ----------------------

	void AbortAsync(void);

#if defined(WIN32)

	bool AsyncReadRequest(void);
#endif


	bool GetEnv(const wchar_t *varName, CkString &outStr);

	const wchar_t *getEnv(const wchar_t *varName);

	const wchar_t *env(const wchar_t *varName);


	bool GetParam(const wchar_t *paramName, CkString &outStr);

	const wchar_t *getParam(const wchar_t *paramName);

	const wchar_t *param(const wchar_t *paramName);


	bool GetParamName(int index, CkString &outStr);

	const wchar_t *getParamName(int index);

	const wchar_t *paramName(int index);


	bool GetParamValue(int index, CkString &outStr);

	const wchar_t *getParamValue(int index);

	const wchar_t *paramValue(int index);


	bool GetRawPostData(CkByteData &outData);


	bool GetUploadData(int index, CkByteData &outData);


	bool GetUploadFilename(int index, CkString &outStr);

	const wchar_t *getUploadFilename(int index);

	const wchar_t *uploadFilename(int index);


	int GetUploadSize(int index);


	bool IsGet(void);


	bool IsHead(void);


	bool IsPost(void);


	bool IsUpload(void);


	bool ReadRequest(void);


	bool SaveNthToUploadDir(int index);


	void SleepMs(int millisec);


	bool TestConsumeAspUpload(const wchar_t *path);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
