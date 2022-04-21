// CkAuthAzureStorage.h: interface for the CkAuthAzureStorage class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkAuthAzureStorage_H
#define _CkAuthAzureStorage_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"




#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkAuthAzureStorage
class CK_VISIBLE_PUBLIC CkAuthAzureStorage  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkAuthAzureStorage(const CkAuthAzureStorage &);
	CkAuthAzureStorage &operator=(const CkAuthAzureStorage &);

    public:
	CkAuthAzureStorage(void);
	virtual ~CkAuthAzureStorage(void);

	static CkAuthAzureStorage *createNew(void);
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
	const char *accessKey(void);
	// A valid base64 access key for the Azure storage account.
	void put_AccessKey(const char *newVal);

	// The Azure storage account name. (A storage account can contain zero or more
	// containers. A container contains properties, metadata, and zero or more blobs. A
	// blob is any single entity comprised of binary data, properties, and metadata. )
	void get_Account(CkString &str);
	// The Azure storage account name. (A storage account can contain zero or more
	// containers. A container contains properties, metadata, and zero or more blobs. A
	// blob is any single entity comprised of binary data, properties, and metadata. )
	const char *account(void);
	// The Azure storage account name. (A storage account can contain zero or more
	// containers. A container contains properties, metadata, and zero or more blobs. A
	// blob is any single entity comprised of binary data, properties, and metadata. )
	void put_Account(const char *newVal);

	// Can be "SharedKey" or "SharedKeyLite". The default value is "SharedKey".
	void get_Scheme(CkString &str);
	// Can be "SharedKey" or "SharedKeyLite". The default value is "SharedKey".
	const char *scheme(void);
	// Can be "SharedKey" or "SharedKeyLite". The default value is "SharedKey".
	void put_Scheme(const char *newVal);

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
	const char *service(void);
	// Can be "Blob", "Queue", "File", or "Table". The default is "Blob".
	// 
	// Note: Authentication for the "Table" service did not work in versions prior to
	// v9.5.0.83.
	// 
	void put_Service(const char *newVal);

	// If set, automatically adds the "x-ms-version" HTTP request header to Azure
	// Storage requests. The default value is "2014-02-14".
	void get_XMsVersion(CkString &str);
	// If set, automatically adds the "x-ms-version" HTTP request header to Azure
	// Storage requests. The default value is "2014-02-14".
	const char *xMsVersion(void);
	// If set, automatically adds the "x-ms-version" HTTP request header to Azure
	// Storage requests. The default value is "2014-02-14".
	void put_XMsVersion(const char *newVal);



	// ----------------------
	// Methods
	// ----------------------




	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
