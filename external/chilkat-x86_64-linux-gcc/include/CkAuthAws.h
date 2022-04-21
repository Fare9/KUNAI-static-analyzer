// CkAuthAws.h: interface for the CkAuthAws class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkAuthAws_H
#define _CkAuthAws_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"




#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkAuthAws
class CK_VISIBLE_PUBLIC CkAuthAws  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkAuthAws(const CkAuthAws &);
	CkAuthAws &operator=(const CkAuthAws &);

    public:
	CkAuthAws(void);
	virtual ~CkAuthAws(void);

	static CkAuthAws *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The AWS access key.
	void get_AccessKey(CkString &str);
	// The AWS access key.
	const char *accessKey(void);
	// The AWS access key.
	void put_AccessKey(const char *newVal);

	// If AWS Signature Version V2 is used, then this property must be set. The rules
	// for setting the canonicalized resource for the V2 signature method is described
	// here:Constructing the CanonicalizedResource Element
	// <http://docs.aws.amazon.com/AmazonS3/latest/dev/RESTAuthentication.html#Construct
	// ingTheCanonicalizedResourceElement>.
	void get_CanonicalizedResourceV2(CkString &str);
	// If AWS Signature Version V2 is used, then this property must be set. The rules
	// for setting the canonicalized resource for the V2 signature method is described
	// here:Constructing the CanonicalizedResource Element
	// <http://docs.aws.amazon.com/AmazonS3/latest/dev/RESTAuthentication.html#Construct
	// ingTheCanonicalizedResourceElement>.
	const char *canonicalizedResourceV2(void);
	// If AWS Signature Version V2 is used, then this property must be set. The rules
	// for setting the canonicalized resource for the V2 signature method is described
	// here:Constructing the CanonicalizedResource Element
	// <http://docs.aws.amazon.com/AmazonS3/latest/dev/RESTAuthentication.html#Construct
	// ingTheCanonicalizedResourceElement>.
	void put_CanonicalizedResourceV2(const char *newVal);

	// This property can optionally be set for AWS requests that have a non-empty
	// request body. This should be the base64 encoding of the 16 bytes of the MD5
	// hash. The most common need for this is if doing an S3 upload from a stream. (If
	// the pre-computed MD5 is not provided, then Chilkat is forced to stream the
	// entire file into memory so that it can calculate the MD5 for authentication.)
	// 
	// Note: AWS Signature Version 2 uses the MD5, whereas Signature Version 4 uses
	// SHA256.
	// 
	void get_PrecomputedMd5(CkString &str);
	// This property can optionally be set for AWS requests that have a non-empty
	// request body. This should be the base64 encoding of the 16 bytes of the MD5
	// hash. The most common need for this is if doing an S3 upload from a stream. (If
	// the pre-computed MD5 is not provided, then Chilkat is forced to stream the
	// entire file into memory so that it can calculate the MD5 for authentication.)
	// 
	// Note: AWS Signature Version 2 uses the MD5, whereas Signature Version 4 uses
	// SHA256.
	// 
	const char *precomputedMd5(void);
	// This property can optionally be set for AWS requests that have a non-empty
	// request body. This should be the base64 encoding of the 16 bytes of the MD5
	// hash. The most common need for this is if doing an S3 upload from a stream. (If
	// the pre-computed MD5 is not provided, then Chilkat is forced to stream the
	// entire file into memory so that it can calculate the MD5 for authentication.)
	// 
	// Note: AWS Signature Version 2 uses the MD5, whereas Signature Version 4 uses
	// SHA256.
	// 
	void put_PrecomputedMd5(const char *newVal);

	// This property can optionally be set for AWS requests that have a non-empty
	// request body. This should be the lowercase hex encoding of the 32-bytes of the
	// SHA256 hash. The most common need for this is if doing an S3 upload from a
	// stream. (If the pre-computed SHA-256 is not provided, then Chilkat is forced to
	// stream the entire file into memory so that it can calculate the SHA-256 for
	// authentication.)
	// 
	// Note: AWS Signature Version 4 uses the SHA256 hash. (AWS Signature Version 2
	// uses MD5)
	// 
	void get_PrecomputedSha256(CkString &str);
	// This property can optionally be set for AWS requests that have a non-empty
	// request body. This should be the lowercase hex encoding of the 32-bytes of the
	// SHA256 hash. The most common need for this is if doing an S3 upload from a
	// stream. (If the pre-computed SHA-256 is not provided, then Chilkat is forced to
	// stream the entire file into memory so that it can calculate the SHA-256 for
	// authentication.)
	// 
	// Note: AWS Signature Version 4 uses the SHA256 hash. (AWS Signature Version 2
	// uses MD5)
	// 
	const char *precomputedSha256(void);
	// This property can optionally be set for AWS requests that have a non-empty
	// request body. This should be the lowercase hex encoding of the 32-bytes of the
	// SHA256 hash. The most common need for this is if doing an S3 upload from a
	// stream. (If the pre-computed SHA-256 is not provided, then Chilkat is forced to
	// stream the entire file into memory so that it can calculate the SHA-256 for
	// authentication.)
	// 
	// Note: AWS Signature Version 4 uses the SHA256 hash. (AWS Signature Version 2
	// uses MD5)
	// 
	void put_PrecomputedSha256(const char *newVal);

	// The AWS region, such as "us-east-1", "us-west-2", "eu-west-1", "eu-central-1",
	// etc. The default is "us-east-1". It is only used when the SignatureVersion
	// property is set to 4. This property is unused when the SignatureVersion property
	// is set to 2.
	void get_Region(CkString &str);
	// The AWS region, such as "us-east-1", "us-west-2", "eu-west-1", "eu-central-1",
	// etc. The default is "us-east-1". It is only used when the SignatureVersion
	// property is set to 4. This property is unused when the SignatureVersion property
	// is set to 2.
	const char *region(void);
	// The AWS region, such as "us-east-1", "us-west-2", "eu-west-1", "eu-central-1",
	// etc. The default is "us-east-1". It is only used when the SignatureVersion
	// property is set to 4. This property is unused when the SignatureVersion property
	// is set to 2.
	void put_Region(const char *newVal);

	// The AWS secret key.
	void get_SecretKey(CkString &str);
	// The AWS secret key.
	const char *secretKey(void);
	// The AWS secret key.
	void put_SecretKey(const char *newVal);

	// The AWS service namespace, such as "s3", "ses", etc. See
	// http://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html#genref-
	// aws-service-namespaces
	// 
	// This property is unused when the SignatureVersion property is set to 2.
	// 
	void get_ServiceName(CkString &str);
	// The AWS service namespace, such as "s3", "ses", etc. See
	// http://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html#genref-
	// aws-service-namespaces
	// 
	// This property is unused when the SignatureVersion property is set to 2.
	// 
	const char *serviceName(void);
	// The AWS service namespace, such as "s3", "ses", etc. See
	// http://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html#genref-
	// aws-service-namespaces
	// 
	// This property is unused when the SignatureVersion property is set to 2.
	// 
	void put_ServiceName(const char *newVal);

	// The AWS Signature Version to be used in authentication. The default value is 4.
	// This can optionally be set to the value 2 to use the older V2 signature version.
	int get_SignatureVersion(void);
	// The AWS Signature Version to be used in authentication. The default value is 4.
	// This can optionally be set to the value 2 to use the older V2 signature version.
	void put_SignatureVersion(int newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Generates a temporary pre-signed URL for an Amazon AWS service using AWS
	// Signature V4. Requires that the SecretKey, AccessKey, and Region properties be
	// set to valid values prior to calling this method.
	// 
	// The URL that is generated has this format:
	// https:///?X-Amz-Algorithm=AWS4-HMAC-SHA256
	// &X-Amz-Credential=////aws4_request
	// &X-Amz-Date=&X-Amz-Expires=&X-Amz-SignedHeaders=host
	// &X-Amz-Signature=
	// 
	// The httpVerb is the HTTP verb such as "GET", "PUT", "POST", "DELETE", etc.  The awsService is a string naming the AWS service, such as "s3", "execute-api", etc.   If useHttps is true, then the returned URL begins with "https://", otherwise it begins with "http://".
	// 
	bool GenPresignedUrl(const char *httpVerb, bool useHttps, const char *domain, const char *path, int numSecondsValid, const char *awsService, CkString &outStr);

	// Generates a temporary pre-signed URL for an Amazon AWS service using AWS
	// Signature V4. Requires that the SecretKey, AccessKey, and Region properties be
	// set to valid values prior to calling this method.
	// 
	// The URL that is generated has this format:
	// https:///?X-Amz-Algorithm=AWS4-HMAC-SHA256
	// &X-Amz-Credential=////aws4_request
	// &X-Amz-Date=&X-Amz-Expires=&X-Amz-SignedHeaders=host
	// &X-Amz-Signature=
	// 
	// The httpVerb is the HTTP verb such as "GET", "PUT", "POST", "DELETE", etc.  The awsService is a string naming the AWS service, such as "s3", "execute-api", etc.   If useHttps is true, then the returned URL begins with "https://", otherwise it begins with "http://".
	// 
	const char *genPresignedUrl(const char *httpVerb, bool useHttps, const char *domain, const char *path, int numSecondsValid, const char *awsService);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
