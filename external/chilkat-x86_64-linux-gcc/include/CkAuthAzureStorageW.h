// CkAuthAzureStorageW.h: interface for the CkAuthAzureStorageW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkAuthAzureStorageW_H
#define _CkAuthAzureStorageW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkWideCharBase.h"




#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkAuthAzureStorageW
class CK_VISIBLE_PUBLIC CkAuthAzureStorageW  : public CkWideCharBase
{
	

	private:
	
	// Don't allow assignment or copying these objects.
	CkAuthAzureStorageW(const CkAuthAzureStorageW &);
	CkAuthAzureStorageW &operator=(const CkAuthAzureStorageW &);

    public:
	CkAuthAzureStorageW(void);
	virtual ~CkAuthAzureStorageW(void);

	

	static CkAuthAzureStorageW *createNew(void);
	

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// A valid base64 access key for the Azure storage account.
	void get_AccessKey(CkString &str);
	// A valid base64 access key for the Azure storage account.
	const wchar_t *accessKey(void);
	// A valid base64 access key for the Azure storage account.
	void put_AccessKey(const wchar_t *newVal);

	// The Azure storage account name. (A storage account can contain zero or more
	// containers. A container contains properties, metadata, and zero or more blobs. A
	// blob is any single entity comprised of binary data, properties, and metadata. )
	void get_Account(CkString &str);
	// The Azure storage account name. (A storage account can contain zero or more
	// containers. A container contains properties, metadata, and zero or more blobs. A
	// blob is any single entity comprised of binary data, properties, and metadata. )
	const wchar_t *account(void);
	// The Azure storage account name. (A storage account can contain zero or more
	// containers. A container contains properties, metadata, and zero or more blobs. A
	// blob is any single entity comprised of binary data, properties, and metadata. )
	void put_Account(const wchar_t *newVal);

	// Can be "SharedKey" or "SharedKeyLite". The default value is "SharedKey".
	void get_Scheme(CkString &str);
	// Can be "SharedKey" or "SharedKeyLite". The default value is "SharedKey".
	const wchar_t *scheme(void);
	// Can be "SharedKey" or "SharedKeyLite". The default value is "SharedKey".
	void put_Scheme(const wchar_t *newVal);

	// Can be "Blob", "Queue", "File", or "Table". The default is "Blob".
	// 
	// Note: Authentication for the "Table" service did not work in versions prior to
	// v9.5.0.83.
	// 
	void get_Service(CkString &str);
	// Can be "Blob", "Queue", "File", or "Table". The default is "Blob".
	// 
	// Note: Authentication for the "Table" service did not work in versions prior to
	// v9.5.0.83.
	// 
	const wchar_t *service(void);
	// Can be "Blob", "Queue", "File", or "Table". The default is "Blob".
	// 
	// Note: Authentication for the "Table" service did not work in versions prior to
	// v9.5.0.83.
	// 
	void put_Service(const wchar_t *newVal);

	// If set, automatically adds the "x-ms-version" HTTP request header to Azure
	// Storage requests. The default value is "2014-02-14".
	void get_XMsVersion(CkString &str);
	// If set, automatically adds the "x-ms-version" HTTP request header to Azure
	// Storage requests. The default value is "2014-02-14".
	const wchar_t *xMsVersion(void);
	// If set, automatically adds the "x-ms-version" HTTP request header to Azure
	// Storage requests. The default value is "2014-02-14".
	void put_XMsVersion(const wchar_t *newVal);



	// ----------------------
	// Methods
	// ----------------------




	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
