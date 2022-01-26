// CkHttp.h: interface for the CkHttp class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkHttp_H
#define _CkHttp_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacks.h"

class CkTask;
class CkJsonObject;
class CkBinData;
class CkStringBuilder;
class CkCert;
class CkHashtable;
class CkHttpResponse;
class CkByteData;
class CkHttpRequest;
class CkStringArray;
class CkDateTime;
class CkPrivateKey;
class CkSecureString;
class CkHttpProgress;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkHttp
class CK_VISIBLE_PUBLIC CkHttp  : public CkClassWithCallbacks
{
    private:

	// Don't allow assignment or copying these objects.
	CkHttp(const CkHttp &);
	CkHttp &operator=(const CkHttp &);

    public:
	CkHttp(void);
	virtual ~CkHttp(void);

	static CkHttp *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	CkHttpProgress *get_EventCallbackObject(void) const;
	void put_EventCallbackObject(CkHttpProgress *progress);


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

	// The Accept header field to be automatically included with GET requests issued by
	// QuickGet or QuickGetStr. The default value is "*/*".
	void get_Accept(CkString &str);
	// The Accept header field to be automatically included with GET requests issued by
	// QuickGet or QuickGetStr. The default value is "*/*".
	const char *ck_accept(void);
	// The Accept header field to be automatically included with GET requests issued by
	// QuickGet or QuickGetStr. The default value is "*/*".
	void put_Accept(const char *newVal);

	// The Accept-Charset header field to be automatically included with GET requests
	// issued by QuickGet or QuickGetStr. The default value is
	// "ISO-8859-1,utf-8;q=0.7,*;q=0.7".
	void get_AcceptCharset(CkString &str);
	// The Accept-Charset header field to be automatically included with GET requests
	// issued by QuickGet or QuickGetStr. The default value is
	// "ISO-8859-1,utf-8;q=0.7,*;q=0.7".
	const char *acceptCharset(void);
	// The Accept-Charset header field to be automatically included with GET requests
	// issued by QuickGet or QuickGetStr. The default value is
	// "ISO-8859-1,utf-8;q=0.7,*;q=0.7".
	void put_AcceptCharset(const char *newVal);

	// The Accept-Language header field to be automatically included with GET requests
	// issued by QuickGet or QuickGetStr. The default value is "en-us,en;q=0.5".
	void get_AcceptLanguage(CkString &str);
	// The Accept-Language header field to be automatically included with GET requests
	// issued by QuickGet or QuickGetStr. The default value is "en-us,en;q=0.5".
	const char *acceptLanguage(void);
	// The Accept-Language header field to be automatically included with GET requests
	// issued by QuickGet or QuickGetStr. The default value is "en-us,en;q=0.5".
	void put_AcceptLanguage(const char *newVal);

	// Controls whether the "Accept-Encoding: gzip" header is added to HTTP requests
	// sent via any method that sends an HTTP request without using the HttpRequest
	// object (such as QuickGetStr). If false, then the empty Accept-Encoding header
	// is added which means the server response should contain the uncompressed
	// content. The default value is true, which means the server, if it chooses, may
	// send a gzipped response.
	bool get_AllowGzip(void);
	// Controls whether the "Accept-Encoding: gzip" header is added to HTTP requests
	// sent via any method that sends an HTTP request without using the HttpRequest
	// object (such as QuickGetStr). If false, then the empty Accept-Encoding header
	// is added which means the server response should contain the uncompressed
	// content. The default value is true, which means the server, if it chooses, may
	// send a gzipped response.
	void put_AllowGzip(bool newVal);

	// If this property is set to false, then no MIME header folding will be
	// automatically applied to any request header. The default is true.
	bool get_AllowHeaderFolding(void);
	// If this property is set to false, then no MIME header folding will be
	// automatically applied to any request header. The default is true.
	void put_AllowHeaderFolding(bool newVal);

	// If set, then automatically adds the Authorization: Signature keyId="...",
	// algorithm="...", headers="...", signature="..." header to all requests. This
	// property should be set to a JSON string containing the information needed to
	// compute and automatically add the HTTP Signature. See the example(s) below.
	void get_AuthSignature(CkString &str);
	// If set, then automatically adds the Authorization: Signature keyId="...",
	// algorithm="...", headers="...", signature="..." header to all requests. This
	// property should be set to a JSON string containing the information needed to
	// compute and automatically add the HTTP Signature. See the example(s) below.
	const char *authSignature(void);
	// If set, then automatically adds the Authorization: Signature keyId="...",
	// algorithm="...", headers="...", signature="..." header to all requests. This
	// property should be set to a JSON string containing the information needed to
	// compute and automatically add the HTTP Signature. See the example(s) below.
	void put_AuthSignature(const char *newVal);

	// If set, then automatically adds the "Authorization: Bearer " header to all
	// requests. (If you have an OAuth2 access token, set this property equal to the
	// OAuth2 access token string. Note: For OAuth1 (older) tokens, use the OAuthTOken
	// property. )
	void get_AuthToken(CkString &str);
	// If set, then automatically adds the "Authorization: Bearer " header to all
	// requests. (If you have an OAuth2 access token, set this property equal to the
	// OAuth2 access token string. Note: For OAuth1 (older) tokens, use the OAuthTOken
	// property. )
	const char *authToken(void);
	// If set, then automatically adds the "Authorization: Bearer " header to all
	// requests. (If you have an OAuth2 access token, set this property equal to the
	// OAuth2 access token string. Note: For OAuth1 (older) tokens, use the OAuthTOken
	// property. )
	void put_AuthToken(const char *newVal);

	// If set to true, the "Host" header field will automatically be added to the
	// request header for any QuickGet or QuickGetStr method calls. The value of the
	// Host header field is taken from the hostname part of the URL passed to
	// QuickGet/QuickGetStr.
	bool get_AutoAddHostHeader(void);
	// If set to true, the "Host" header field will automatically be added to the
	// request header for any QuickGet or QuickGetStr method calls. The value of the
	// Host header field is taken from the hostname part of the URL passed to
	// QuickGet/QuickGetStr.
	void put_AutoAddHostHeader(bool newVal);

	// The AWS Access Key to be used with the Amazon S3 methods listed below.
	void get_AwsAccessKey(CkString &str);
	// The AWS Access Key to be used with the Amazon S3 methods listed below.
	const char *awsAccessKey(void);
	// The AWS Access Key to be used with the Amazon S3 methods listed below.
	void put_AwsAccessKey(const char *newVal);

	// The regional endpoint (domain) to be used for Amazon S3 method calls. The
	// default value is "s3.amazonaws.com". This can be set to any valid Amazon S3
	// endpoint, such as "s3-eu-west-1.amazonaws.com", or the endpoints for S3-API
	// compatible services from other different providers.
	void get_AwsEndpoint(CkString &str);
	// The regional endpoint (domain) to be used for Amazon S3 method calls. The
	// default value is "s3.amazonaws.com". This can be set to any valid Amazon S3
	// endpoint, such as "s3-eu-west-1.amazonaws.com", or the endpoints for S3-API
	// compatible services from other different providers.
	const char *awsEndpoint(void);
	// The regional endpoint (domain) to be used for Amazon S3 method calls. The
	// default value is "s3.amazonaws.com". This can be set to any valid Amazon S3
	// endpoint, such as "s3-eu-west-1.amazonaws.com", or the endpoints for S3-API
	// compatible services from other different providers.
	void put_AwsEndpoint(const char *newVal);

	// The AWS (S3) region, such as "us-east-1", "us-west-2", "eu-west-1",
	// "eu-central-1", etc. This propery defaults to "us-east-1". It is only used when
	// the AwsSignatureVersion property is set to 4. When the AwsSignatureVersion
	// property is set to 2, then this property is unused.
	void get_AwsRegion(CkString &str);
	// The AWS (S3) region, such as "us-east-1", "us-west-2", "eu-west-1",
	// "eu-central-1", etc. This propery defaults to "us-east-1". It is only used when
	// the AwsSignatureVersion property is set to 4. When the AwsSignatureVersion
	// property is set to 2, then this property is unused.
	const char *awsRegion(void);
	// The AWS (S3) region, such as "us-east-1", "us-west-2", "eu-west-1",
	// "eu-central-1", etc. This propery defaults to "us-east-1". It is only used when
	// the AwsSignatureVersion property is set to 4. When the AwsSignatureVersion
	// property is set to 2, then this property is unused.
	void put_AwsRegion(const char *newVal);

	// The AWS Secret Key to be used with the Amazon S3 methods listed below.
	void get_AwsSecretKey(CkString &str);
	// The AWS Secret Key to be used with the Amazon S3 methods listed below.
	const char *awsSecretKey(void);
	// The AWS Secret Key to be used with the Amazon S3 methods listed below.
	void put_AwsSecretKey(const char *newVal);

	// Selects the AWS Signature Version algorithm. The default value is 4. May be set
	// to 2 to select AWS Signature Version 2. (The only valid choices are 2 and 4.)
	int get_AwsSignatureVersion(void);
	// Selects the AWS Signature Version algorithm. The default value is 4. May be set
	// to 2 to select AWS Signature Version 2. (The only valid choices are 2 and 4.)
	void put_AwsSignatureVersion(int newVal);

	// The AWS sub-resources to be used with the Amazon S3 methods listed below.
	// 
	// If the S3 request needs to address a sub-resource, like ?versioning, ?policy,
	// ?location, ?acl, or ?torrent, or ?versionid append the sub-resource and its
	// value if it has one. Note that in case of multiple sub-resources, sub-resources
	// must be lexicographically sorted by sub-resource name and separated by '&'. e.g.
	// "acl&versionId=value"
	// 
	// The list of sub-resources that can be included are: acl, location, logging,
	// notification, partNumber, policy, requestPayment, torrent, uploadId, uploads,
	// versionId, versioning, versions and website.
	// 
	void get_AwsSubResources(CkString &str);
	// The AWS sub-resources to be used with the Amazon S3 methods listed below.
	// 
	// If the S3 request needs to address a sub-resource, like ?versioning, ?policy,
	// ?location, ?acl, or ?torrent, or ?versionid append the sub-resource and its
	// value if it has one. Note that in case of multiple sub-resources, sub-resources
	// must be lexicographically sorted by sub-resource name and separated by '&'. e.g.
	// "acl&versionId=value"
	// 
	// The list of sub-resources that can be included are: acl, location, logging,
	// notification, partNumber, policy, requestPayment, torrent, uploadId, uploads,
	// versionId, versioning, versions and website.
	// 
	const char *awsSubResources(void);
	// The AWS sub-resources to be used with the Amazon S3 methods listed below.
	// 
	// If the S3 request needs to address a sub-resource, like ?versioning, ?policy,
	// ?location, ?acl, or ?torrent, or ?versionid append the sub-resource and its
	// value if it has one. Note that in case of multiple sub-resources, sub-resources
	// must be lexicographically sorted by sub-resource name and separated by '&'. e.g.
	// "acl&versionId=value"
	// 
	// The list of sub-resources that can be included are: acl, location, logging,
	// notification, partNumber, policy, requestPayment, torrent, uploadId, uploads,
	// versionId, versioning, versions and website.
	// 
	void put_AwsSubResources(const char *newVal);

	// If non-zero, limits (throttles) the download bandwidth to approximately this
	// maximum number of bytes per second. The default value of this property is 0.
	int get_BandwidthThrottleDown(void);
	// If non-zero, limits (throttles) the download bandwidth to approximately this
	// maximum number of bytes per second. The default value of this property is 0.
	void put_BandwidthThrottleDown(int newVal);

	// If non-zero, limits (throttles) the upload bandwidth to approximately this
	// maximum number of bytes per second. The default value of this property is 0.
	int get_BandwidthThrottleUp(void);
	// If non-zero, limits (throttles) the upload bandwidth to approximately this
	// maximum number of bytes per second. The default value of this property is 0.
	void put_BandwidthThrottleUp(int newVal);

	// If HTTP basic authentication is needed, this property must be set to true. The
	// HTTP protocol allows for several different types of authentication schemes, such
	// as NTLM, Digest, OAuth1, etc. A given server will support (or allow) certain
	// authentication schemes (also known as authentication methods). Except for the
	// "Basic" authentication method, the other forms of authentication do not involve
	// sending the login and password in plain unencrypted text over the connection.
	// The Basic authentication method is insecure in that it sends the login/password
	// for all to see. If the connection is SSL/TLS, then this might be considered OK.
	// Chilkat takes the safe approach and will not allow Basic authentication unless
	// this property has been explicitly set to true. The default value of this
	// property is false.
	// 
	// Note: It is not required to know the authentication methods accepted by the
	// server beforehand (except for the case of Basic authentication). When
	// authentication is required, Chilkat will first send the request without the
	// Authorization header, receive back the 401 Authorization Required response along
	// with information about what authentication methods are accepted, and then
	// re-send with an accepted authentication method. If the authentication method is
	// known in advance, then an application may set the appropriate property, such as
	// NtlmAuth to true so that the extra (internal) round-trip is not required.
	// 
	// Note: The login and password is sent using the utf-8 byte representation. Some
	// servers expect the ANSI byte representation (typically Windows-1252). Starting
	// in Chilkat v9.5.0.87, to send the login/password using ANSI, add the "AnsiLogin"
	// keyword to the UncommonOptions property.
	// 
	bool get_BasicAuth(void);
	// If HTTP basic authentication is needed, this property must be set to true. The
	// HTTP protocol allows for several different types of authentication schemes, such
	// as NTLM, Digest, OAuth1, etc. A given server will support (or allow) certain
	// authentication schemes (also known as authentication methods). Except for the
	// "Basic" authentication method, the other forms of authentication do not involve
	// sending the login and password in plain unencrypted text over the connection.
	// The Basic authentication method is insecure in that it sends the login/password
	// for all to see. If the connection is SSL/TLS, then this might be considered OK.
	// Chilkat takes the safe approach and will not allow Basic authentication unless
	// this property has been explicitly set to true. The default value of this
	// property is false.
	// 
	// Note: It is not required to know the authentication methods accepted by the
	// server beforehand (except for the case of Basic authentication). When
	// authentication is required, Chilkat will first send the request without the
	// Authorization header, receive back the 401 Authorization Required response along
	// with information about what authentication methods are accepted, and then
	// re-send with an accepted authentication method. If the authentication method is
	// known in advance, then an application may set the appropriate property, such as
	// NtlmAuth to true so that the extra (internal) round-trip is not required.
	// 
	// Note: The login and password is sent using the utf-8 byte representation. Some
	// servers expect the ANSI byte representation (typically Windows-1252). Starting
	// in Chilkat v9.5.0.87, to send the login/password using ANSI, add the "AnsiLogin"
	// keyword to the UncommonOptions property.
	// 
	void put_BasicAuth(bool newVal);

	// The IP address to use for computers with multiple network interfaces or IP
	// addresses. For computers with a single network interface (i.e. most computers),
	// this property should not be set. For multihoming computers, the default IP
	// address is automatically used if this property is not set.
	// 
	// The IP address is a string such as in dotted notation using numbers, not domain
	// names, such as "165.164.55.124".
	// 
	void get_ClientIpAddress(CkString &str);
	// The IP address to use for computers with multiple network interfaces or IP
	// addresses. For computers with a single network interface (i.e. most computers),
	// this property should not be set. For multihoming computers, the default IP
	// address is automatically used if this property is not set.
	// 
	// The IP address is a string such as in dotted notation using numbers, not domain
	// names, such as "165.164.55.124".
	// 
	const char *clientIpAddress(void);
	// The IP address to use for computers with multiple network interfaces or IP
	// addresses. For computers with a single network interface (i.e. most computers),
	// this property should not be set. For multihoming computers, the default IP
	// address is automatically used if this property is not set.
	// 
	// The IP address is a string such as in dotted notation using numbers, not domain
	// names, such as "165.164.55.124".
	// 
	void put_ClientIpAddress(const char *newVal);

	// This property will be set to the status of the last HTTP connection made (or
	// failed to be made) by any HTTP method.
	// 
	// Possible values are:
	// 0 = success
	// 
	// Normal (non-TLS) sockets:
	// 1 = empty hostname
	// 2 = DNS lookup failed
	// 3 = DNS timeout
	// 4 = Aborted by application.
	// 5 = Internal failure.
	// 6 = Connect Timed Out
	// 7 = Connect Rejected (or failed for some other reason)
	// 50 = HTTP proxy authentication failure.
	// 98 = Async operation in progress.
	// 99 = Product is not unlocked.
	// 
	// SSL/TLS:
	// 100 = TLS internal error.
	// 101 = Failed to send client hello.
	// 102 = Unexpected handshake message.
	// 103 = Failed to read server hello.
	// 104 = No server certificate.
	// 105 = Unexpected TLS protocol version.
	// 106 = Server certificate verify failed (the server certificate is expired or the cert's signature verification failed).
	// 107 = Unacceptable TLS protocol version.
	// 108 = App-defined server certificate requirements failure.
	// 109 = Failed to read handshake messages.
	// 110 = Failed to send client certificate handshake message.
	// 111 = Failed to send client key exchange handshake message.
	// 112 = Client certificate's private key not accessible.
	// 113 = Failed to send client cert verify handshake message.
	// 114 = Failed to send change cipher spec handshake message.
	// 115 = Failed to send finished handshake message.
	// 116 = Server's Finished message is invalid.
	// 125 = TLS Pin Set Mismatch.
	// 
	int get_ConnectFailReason(void);

	// The Connection header field to be automatically included with GET requests
	// issued by QuickGet or QuickGetStr. The default value is "Keep-Alive". To prevent
	// the Connection header from being added to the HTTP header, set this property to
	// the empty string.
	void get_Connection(CkString &str);
	// The Connection header field to be automatically included with GET requests
	// issued by QuickGet or QuickGetStr. The default value is "Keep-Alive". To prevent
	// the Connection header from being added to the HTTP header, set this property to
	// the empty string.
	const char *connection(void);
	// The Connection header field to be automatically included with GET requests
	// issued by QuickGet or QuickGetStr. The default value is "Keep-Alive". To prevent
	// the Connection header from being added to the HTTP header, set this property to
	// the empty string.
	void put_Connection(const char *newVal);

	// The amount of time in seconds to wait before timing out when connecting to an
	// HTTP server. The default ConnectTimeout is 30 seconds.
	// 
	// Note: This is the maximum number of seconds to wait for a server to accept a TCP
	// connection. Once the connection is accepted, and bytes begin flowing
	// back-and-forth, then it is the ReadTimeout property that applies. It is the
	// ReadTimeout that applies when receiving data, which includes the reads that
	// occur during a TLS handshake.
	// 
	int get_ConnectTimeout(void);
	// The amount of time in seconds to wait before timing out when connecting to an
	// HTTP server. The default ConnectTimeout is 30 seconds.
	// 
	// Note: This is the maximum number of seconds to wait for a server to accept a TCP
	// connection. Once the connection is accepted, and bytes begin flowing
	// back-and-forth, then it is the ReadTimeout property that applies. It is the
	// ReadTimeout that applies when receiving data, which includes the reads that
	// occur during a TLS handshake.
	// 
	void put_ConnectTimeout(int newVal);

	// Specifies a directory where cookies are automatically persisted if the
	// Http.SaveCookies property is turned on. Cookies are stored in XML formatted
	// files, one per domain, to main it easy for other programs to understand and
	// parse. May be set to the string "memory" to cache cookies in memory.
	void get_CookieDir(CkString &str);
	// Specifies a directory where cookies are automatically persisted if the
	// Http.SaveCookies property is turned on. Cookies are stored in XML formatted
	// files, one per domain, to main it easy for other programs to understand and
	// parse. May be set to the string "memory" to cache cookies in memory.
	const char *cookieDir(void);
	// Specifies a directory where cookies are automatically persisted if the
	// Http.SaveCookies property is turned on. Cookies are stored in XML formatted
	// files, one per domain, to main it easy for other programs to understand and
	// parse. May be set to the string "memory" to cache cookies in memory.
	void put_CookieDir(const char *newVal);

	// The default freshness period (in minutes) for cached documents when the
	// FreshnessAlgorithm property is set to 0. The default value is 10080 (1 week).
	int get_DefaultFreshPeriod(void);
	// The default freshness period (in minutes) for cached documents when the
	// FreshnessAlgorithm property is set to 0. The default value is 10080 (1 week).
	void put_DefaultFreshPeriod(int newVal);

	// Setting this property to true causes the HTTP component to use digest
	// authentication. The default value is false.
	bool get_DigestAuth(void);
	// Setting this property to true causes the HTTP component to use digest
	// authentication. The default value is false.
	void put_DigestAuth(bool newVal);

	// Set to true if pages should be fetched from cache when possible. Only HTTP GET
	// requests are cached. HTTP responses that include Set-Cookie headers are not
	// cached. A page is fetched from the disk cache if it is present and it is "fresh"
	// according to the FreshnessAlgorithm property. If a page exists in cache but is
	// not fresh, the HTTP component will issue a revalidate request and update the
	// cache appropriately according to the response.
	bool get_FetchFromCache(void);
	// Set to true if pages should be fetched from cache when possible. Only HTTP GET
	// requests are cached. HTTP responses that include Set-Cookie headers are not
	// cached. A page is fetched from the disk cache if it is present and it is "fresh"
	// according to the FreshnessAlgorithm property. If a page exists in cache but is
	// not fresh, the HTTP component will issue a revalidate request and update the
	// cache appropriately according to the response.
	void put_FetchFromCache(bool newVal);

	// If an HTTP GET was redirected (as indicated by the WasRedirected property), this
	// property will contain the final redirect URL, assuming the FollowRedirects
	// property is true.
	// 
	// Note: Starting in v9.5.0.49, this property will contain the redirect URL for
	// 301/302 responses even if FollowRedirects is not set to true.
	// 
	void get_FinalRedirectUrl(CkString &str);
	// If an HTTP GET was redirected (as indicated by the WasRedirected property), this
	// property will contain the final redirect URL, assuming the FollowRedirects
	// property is true.
	// 
	// Note: Starting in v9.5.0.49, this property will contain the redirect URL for
	// 301/302 responses even if FollowRedirects is not set to true.
	// 
	const char *finalRedirectUrl(void);

	// If true, then 301, 302, 303, 307, and 308 redirects are automatically followed
	// when calling QuickGet and QuickGetStr. FollowRedirects is true by default.
	bool get_FollowRedirects(void);
	// If true, then 301, 302, 303, 307, and 308 redirects are automatically followed
	// when calling QuickGet and QuickGetStr. FollowRedirects is true by default.
	void put_FollowRedirects(bool newVal);

	// The freshness algorithm to use when determining the freshness of a cached HTTP
	// GET response. A value of 1 causes an LM-factor algorithm to be used. This is the
	// default. The LMFactor property is a value between 1 and 100 indicating the
	// percentage of time based on the last-modified date of the HTML page. For
	// example, if the LMFactor is 50, and an HTML page was modified 10 days ago, then
	// the page will expire (i.e. no longer be fresh) in 5 days (50% of 10 days). This
	// only applies to HTTP responses that do not have page expiration information. If
	// the FreshnessAlgorithm = 0, then a constant expire time period determined by the
	// DefaultFreshPeriod property is used.
	int get_FreshnessAlgorithm(void);
	// The freshness algorithm to use when determining the freshness of a cached HTTP
	// GET response. A value of 1 causes an LM-factor algorithm to be used. This is the
	// default. The LMFactor property is a value between 1 and 100 indicating the
	// percentage of time based on the last-modified date of the HTML page. For
	// example, if the LMFactor is 50, and an HTML page was modified 10 days ago, then
	// the page will expire (i.e. no longer be fresh) in 5 days (50% of 10 days). This
	// only applies to HTTP responses that do not have page expiration information. If
	// the FreshnessAlgorithm = 0, then a constant expire time period determined by the
	// DefaultFreshPeriod property is used.
	void put_FreshnessAlgorithm(int newVal);

	// This property is only valid in programming environment and languages that allow
	// for event callbacks.
	// 
	// Specifies the time interval in milliseconds between AbortCheck events. A value
	// of 0 (the default) indicate that no AbortCheck events will fire. Any HTTP
	// operation can be aborted via the AbortCheck event.
	// 
	int get_HeartbeatMs(void);
	// This property is only valid in programming environment and languages that allow
	// for event callbacks.
	// 
	// Specifies the time interval in milliseconds between AbortCheck events. A value
	// of 0 (the default) indicate that no AbortCheck events will fire. Any HTTP
	// operation can be aborted via the AbortCheck event.
	// 
	void put_HeartbeatMs(int newVal);

	// Some HTTP responses contain a "Cache-Control: must-revalidate" header. If this
	// is present, the server is requesting that the client always issue a revalidate
	// HTTP request instead of serving the page directly from cache. If
	// IgnoreMustRevalidate is set to true, then Chilkat HTTP will serve the page
	// directly from cache without revalidating until the page is no longer fresh.
	// 
	// The default value of this property is false.
	// 
	bool get_IgnoreMustRevalidate(void);
	// Some HTTP responses contain a "Cache-Control: must-revalidate" header. If this
	// is present, the server is requesting that the client always issue a revalidate
	// HTTP request instead of serving the page directly from cache. If
	// IgnoreMustRevalidate is set to true, then Chilkat HTTP will serve the page
	// directly from cache without revalidating until the page is no longer fresh.
	// 
	// The default value of this property is false.
	// 
	void put_IgnoreMustRevalidate(bool newVal);

	// Some HTTP responses contain headers of various types that indicate that the page
	// should not be cached. Chilkat HTTP will adhere to this unless this property is
	// set to true.
	// 
	// The default value of this property is false.
	// 
	bool get_IgnoreNoCache(void);
	// Some HTTP responses contain headers of various types that indicate that the page
	// should not be cached. Chilkat HTTP will adhere to this unless this property is
	// set to true.
	// 
	// The default value of this property is false.
	// 
	void put_IgnoreNoCache(bool newVal);

	// If true, then the response body, if text, is saved to the LastResponseBody
	// property for all methods that do not return an HttpResponse object. The default
	// value of this property is false.
	bool get_KeepResponseBody(void);
	// If true, then the response body, if text, is saved to the LastResponseBody
	// property for all methods that do not return an HttpResponse object. The default
	// value of this property is false.
	void put_KeepResponseBody(bool newVal);

	// The content-type of the last HTTP response received by the HTTP component.
	void get_LastContentType(CkString &str);
	// The content-type of the last HTTP response received by the HTTP component.
	const char *lastContentType(void);

	// The text of the last HTTP header sent by any of this class's methods. The
	// purpose of this property is to allow the developer to examine the exact HTTP
	// header for debugging purposes.
	void get_LastHeader(CkString &str);
	// The text of the last HTTP header sent by any of this class's methods. The
	// purpose of this property is to allow the developer to examine the exact HTTP
	// header for debugging purposes.
	const char *lastHeader(void);

	// The value of the Last-Modified header in the last HTTP response received by the
	// HTTP component.
	void get_LastModDate(CkString &str);
	// The value of the Last-Modified header in the last HTTP response received by the
	// HTTP component.
	const char *lastModDate(void);

	// The response body of the last HTTP response received by the HTTP component (for
	// methods that do not return an HttpResponse object). The last response body is
	// only saved to this property IF the KeepResponseBody property is set to true.
	void get_LastResponseBody(CkString &str);
	// The response body of the last HTTP response received by the HTTP component (for
	// methods that do not return an HttpResponse object). The last response body is
	// only saved to this property IF the KeepResponseBody property is set to true.
	const char *lastResponseBody(void);

	// The entire response header for the last HTTP response received by the HTTP
	// component (for methods that do not return an HttpResponse object).
	void get_LastResponseHeader(CkString &str);
	// The entire response header for the last HTTP response received by the HTTP
	// component (for methods that do not return an HttpResponse object).
	const char *lastResponseHeader(void);

	// The last HTTP status value received by the HTTP component. This only applies to
	// methods that do not return an HTTP response object. For methods that return an
	// HTTP response object, such as SynchronousRequest, the status code is found in
	// the StatusCode property of the response object.
	int get_LastStatus(void);

	// The last HTTP status text received by the HTTP component. This only applies to
	// methods that do not return an HTTP response object. For methods that return an
	// HTTP response object, such as SynchronousRequest, the status text is found in
	// the StatusText property of the response object.
	void get_LastStatusText(CkString &str);
	// The last HTTP status text received by the HTTP component. This only applies to
	// methods that do not return an HTTP response object. For methods that return an
	// HTTP response object, such as SynchronousRequest, the status text is found in
	// the StatusText property of the response object.
	const char *lastStatusText(void);

	// An integer between 1 and 100 that indicates the percentage of time from the HTTP
	// page's last-modified date that will be used for the freshness period. The
	// default value is 25. For example, if a page is fetched with a last-modified date
	// of 4 weeks ago, and the LMFactor = 25, then the page will be considered fresh in
	// the cache for 1 week (25% of 4 weeks).
	int get_LMFactor(void);
	// An integer between 1 and 100 that indicates the percentage of time from the HTTP
	// page's last-modified date that will be used for the freshness period. The
	// default value is 25. For example, if a page is fetched with a last-modified date
	// of 4 weeks ago, and the LMFactor = 25, then the page will be considered fresh in
	// the cache for 1 week (25% of 4 weeks).
	void put_LMFactor(int newVal);

	// The HTTP login for pages requiring a login/password. Chilkat HTTP can do Basic,
	// Digest, and NTLM HTTP authentication. (NTLM is also known as SPA (or Windows
	// Integrated Authentication). To use Basic authentication, the BasicAuth property
	// must be set equal to true. It is not necessary to set the NtlmAuth or
	// DigestAuth properties beforehand if NTLM or Digest authentication is needed.
	// However, it is most efficient to pre-set these properties when the type of
	// authentication is known in advance.
	// 
	// Important: If NTLM authentication is used, it may be incorrect to set the Login
	// property equal to "mydomain\mylogin". Instead, set the LoginDomain property
	// equal to "mydomain", and set this property equal to "mylogin".
	// 
	// Note: When the Login and Password properties are set, and the type of
	// authentication is specified by setting one of the following properties equal to
	// true (BasicAuth, DigestAuth, NtlmAuth), Chilkat will automatically add the
	// "Authorization: ..." header in the correct format.
	// 
	void get_Login(CkString &str);
	// The HTTP login for pages requiring a login/password. Chilkat HTTP can do Basic,
	// Digest, and NTLM HTTP authentication. (NTLM is also known as SPA (or Windows
	// Integrated Authentication). To use Basic authentication, the BasicAuth property
	// must be set equal to true. It is not necessary to set the NtlmAuth or
	// DigestAuth properties beforehand if NTLM or Digest authentication is needed.
	// However, it is most efficient to pre-set these properties when the type of
	// authentication is known in advance.
	// 
	// Important: If NTLM authentication is used, it may be incorrect to set the Login
	// property equal to "mydomain\mylogin". Instead, set the LoginDomain property
	// equal to "mydomain", and set this property equal to "mylogin".
	// 
	// Note: When the Login and Password properties are set, and the type of
	// authentication is specified by setting one of the following properties equal to
	// true (BasicAuth, DigestAuth, NtlmAuth), Chilkat will automatically add the
	// "Authorization: ..." header in the correct format.
	// 
	const char *login(void);
	// The HTTP login for pages requiring a login/password. Chilkat HTTP can do Basic,
	// Digest, and NTLM HTTP authentication. (NTLM is also known as SPA (or Windows
	// Integrated Authentication). To use Basic authentication, the BasicAuth property
	// must be set equal to true. It is not necessary to set the NtlmAuth or
	// DigestAuth properties beforehand if NTLM or Digest authentication is needed.
	// However, it is most efficient to pre-set these properties when the type of
	// authentication is known in advance.
	// 
	// Important: If NTLM authentication is used, it may be incorrect to set the Login
	// property equal to "mydomain\mylogin". Instead, set the LoginDomain property
	// equal to "mydomain", and set this property equal to "mylogin".
	// 
	// Note: When the Login and Password properties are set, and the type of
	// authentication is specified by setting one of the following properties equal to
	// true (BasicAuth, DigestAuth, NtlmAuth), Chilkat will automatically add the
	// "Authorization: ..." header in the correct format.
	// 
	void put_Login(const char *newVal);

	// The optional domain name to be used with NTLM / Kerberos / Negotiate
	// authentication.
	void get_LoginDomain(CkString &str);
	// The optional domain name to be used with NTLM / Kerberos / Negotiate
	// authentication.
	const char *loginDomain(void);
	// The optional domain name to be used with NTLM / Kerberos / Negotiate
	// authentication.
	void put_LoginDomain(const char *newVal);

	// The maximum number of simultaneous open HTTP connections managed by the HTTP
	// component. The Chilkat HTTP component automatically manages HTTP connections. If
	// the number of open HTTP connections is about to be exceeded, the connection with
	// the least recent activity is automatically closed.
	int get_MaxConnections(void);
	// The maximum number of simultaneous open HTTP connections managed by the HTTP
	// component. The Chilkat HTTP component automatically manages HTTP connections. If
	// the number of open HTTP connections is about to be exceeded, the connection with
	// the least recent activity is automatically closed.
	void put_MaxConnections(int newVal);

	// Limits the amount of time a document can be kept "fresh" in the cache. The
	// MaxFreshPeriod is specified in minutes, and the default value is 525600 which is
	// equal to 1 year.
	int get_MaxFreshPeriod(void);
	// Limits the amount of time a document can be kept "fresh" in the cache. The
	// MaxFreshPeriod is specified in minutes, and the default value is 525600 which is
	// equal to 1 year.
	void put_MaxFreshPeriod(int newVal);

	// The maximum HTTP response size to be accepted by the calling program. A value of
	// 0 (the default) indicates that there is no maximum value.
	unsigned long get_MaxResponseSize(void);
	// The maximum HTTP response size to be accepted by the calling program. A value of
	// 0 (the default) indicates that there is no maximum value.
	void put_MaxResponseSize(unsigned long newVal);

	// The Http class will automatically fail any URL longer than this length. The
	// default MaxUrlLen is 2000 characters.
	int get_MaxUrlLen(void);
	// The Http class will automatically fail any URL longer than this length. The
	// default MaxUrlLen is 2000 characters.
	void put_MaxUrlLen(int newVal);

	// If set to true, then the appropriate headers to mimic Mozilla/FireFox are
	// automatically added to requests sent via the QuickGet and QuickGetStr methods.
	bool get_MimicFireFox(void);
	// If set to true, then the appropriate headers to mimic Mozilla/FireFox are
	// automatically added to requests sent via the QuickGet and QuickGetStr methods.
	void put_MimicFireFox(bool newVal);

	// If set to true, then the appropriate headers to mimic Internet Explorer are
	// automatically added to requests sent via the QuickGet and QuickGetStr methods.
	bool get_MimicIE(void);
	// If set to true, then the appropriate headers to mimic Internet Explorer are
	// automatically added to requests sent via the QuickGet and QuickGetStr methods.
	void put_MimicIE(bool newVal);

	// The freshness period for a document in cache will not be less than this value
	// (in minutes). The default value is 30.
	int get_MinFreshPeriod(void);
	// The freshness period for a document in cache will not be less than this value
	// (in minutes). The default value is 30.
	void put_MinFreshPeriod(int newVal);

	// Set this property equal to true for Negotiate authentication. Negotiate
	// authentication will dynamically select Kerberos or NTLM authentication depending
	// on what the server requires.
	// 
	// Note: The NegotiateAuth property is only available for the Microsoft Windows
	// operating system.
	// 
	bool get_NegotiateAuth(void);
	// Set this property equal to true for Negotiate authentication. Negotiate
	// authentication will dynamically select Kerberos or NTLM authentication depending
	// on what the server requires.
	// 
	// Note: The NegotiateAuth property is only available for the Microsoft Windows
	// operating system.
	// 
	void put_NegotiateAuth(bool newVal);

	// Setting this property to true causes the HTTP component to use NTLM
	// authentication (also known as IWA -- or Integrated Windows Authentication) when
	// authentication with an HTTP server. The default value is false.
	bool get_NtlmAuth(void);
	// Setting this property to true causes the HTTP component to use NTLM
	// authentication (also known as IWA -- or Integrated Windows Authentication) when
	// authentication with an HTTP server. The default value is false.
	void put_NtlmAuth(bool newVal);

	// The number of directory levels to be used under each cache root. The default is
	// 0, meaning that each cached HTML page is stored in a cache root directory. A
	// value of 1 causes each cached page to be stored in one of 255 subdirectories
	// named "0","1", "2", ..."255" under a cache root. A value of 2 causes two levels
	// of subdirectories ("0..255/0..255") under each cache root. The HTTP control
	// automatically creates subdirectories as needed. The reason for mutliple levels
	// is to alleviate problems that may arise with unrelated software when huge
	// numbers of files are stored in a single directory. For example, Windows Explorer
	// does not behave well when trying to display the contents of directories with
	// thousands of files.
	int get_NumCacheLevels(void);
	// The number of directory levels to be used under each cache root. The default is
	// 0, meaning that each cached HTML page is stored in a cache root directory. A
	// value of 1 causes each cached page to be stored in one of 255 subdirectories
	// named "0","1", "2", ..."255" under a cache root. A value of 2 causes two levels
	// of subdirectories ("0..255/0..255") under each cache root. The HTTP control
	// automatically creates subdirectories as needed. The reason for mutliple levels
	// is to alleviate problems that may arise with unrelated software when huge
	// numbers of files are stored in a single directory. For example, Windows Explorer
	// does not behave well when trying to display the contents of directories with
	// thousands of files.
	void put_NumCacheLevels(int newVal);

	// The number of cache roots to be used for the HTTP cache. This allows the disk
	// cache spread out over multiple disk drives. Each cache root is a string
	// indicating the drive letter and directory path. For example, "E:\Cache". An
	// example of a very large low-cost cache might be four USB external drives. To
	// create a cache with four roots, call AddCacheRoot once for each directory root.
	int get_NumCacheRoots(void);

	// If true then causes an OAuth Authorization header to be added to any request
	// sent by the HTTP object. For example:
	// Authorization: OAuth realm="http://sp.example.com/",
	//                 oauth_consumer_key="0685bd9184jfhq22",
	//                 oauth_token="ad180jjd733klru7",
	//                 oauth_signature_method="HMAC-SHA1",
	//                 oauth_signature="wOJIO9A2W5mFwDgiDvZbTSMK%2FPY%3D",
	//                 oauth_timestamp="137131200",
	//                 oauth_nonce="4572616e48616d6d65724c61686176",
	//                 oauth_version="1.0"
	// The information used to compute the OAuth Authorization header is obtained from
	// the other OAuth* properties, such as OAuthConsumerKey, OAuthConsumerSecret,
	// OAuthRealm, etc.
	bool get_OAuth1(void);
	// If true then causes an OAuth Authorization header to be added to any request
	// sent by the HTTP object. For example:
	// Authorization: OAuth realm="http://sp.example.com/",
	//                 oauth_consumer_key="0685bd9184jfhq22",
	//                 oauth_token="ad180jjd733klru7",
	//                 oauth_signature_method="HMAC-SHA1",
	//                 oauth_signature="wOJIO9A2W5mFwDgiDvZbTSMK%2FPY%3D",
	//                 oauth_timestamp="137131200",
	//                 oauth_nonce="4572616e48616d6d65724c61686176",
	//                 oauth_version="1.0"
	// The information used to compute the OAuth Authorization header is obtained from
	// the other OAuth* properties, such as OAuthConsumerKey, OAuthConsumerSecret,
	// OAuthRealm, etc.
	void put_OAuth1(bool newVal);

	// The OAuth 1.0 callback URL. Defaults to "oob".
	void get_OAuthCallback(CkString &str);
	// The OAuth 1.0 callback URL. Defaults to "oob".
	const char *oAuthCallback(void);
	// The OAuth 1.0 callback URL. Defaults to "oob".
	void put_OAuthCallback(const char *newVal);

	// The OAuth consumer key to be used in the Authorization header.
	void get_OAuthConsumerKey(CkString &str);
	// The OAuth consumer key to be used in the Authorization header.
	const char *oAuthConsumerKey(void);
	// The OAuth consumer key to be used in the Authorization header.
	void put_OAuthConsumerKey(const char *newVal);

	// The OAuth consumer secret to be used in computing the contents of the
	// Authorization header.
	void get_OAuthConsumerSecret(CkString &str);
	// The OAuth consumer secret to be used in computing the contents of the
	// Authorization header.
	const char *oAuthConsumerSecret(void);
	// The OAuth consumer secret to be used in computing the contents of the
	// Authorization header.
	void put_OAuthConsumerSecret(const char *newVal);

	// The OAuth realm to be used in the Authorization header.
	void get_OAuthRealm(CkString &str);
	// The OAuth realm to be used in the Authorization header.
	const char *oAuthRealm(void);
	// The OAuth realm to be used in the Authorization header.
	void put_OAuthRealm(const char *newVal);

	// The OAuth signature method, such as "HMAC-SHA1" to be used in the Authorization
	// header. The default is "HMAC-SHA1". It is also possible to choose "HMAC-SHA256",
	// "RSA-SHA1" or "RSA-SHA2". For RSA algorithms, an RSA private key would need to
	// be provided via the SetOAuthRsaKey method.
	// 
	// Note: RSA-SHA2 is supported starting in Chilkat v9.5.0.56
	// 
	void get_OAuthSigMethod(CkString &str);
	// The OAuth signature method, such as "HMAC-SHA1" to be used in the Authorization
	// header. The default is "HMAC-SHA1". It is also possible to choose "HMAC-SHA256",
	// "RSA-SHA1" or "RSA-SHA2". For RSA algorithms, an RSA private key would need to
	// be provided via the SetOAuthRsaKey method.
	// 
	// Note: RSA-SHA2 is supported starting in Chilkat v9.5.0.56
	// 
	const char *oAuthSigMethod(void);
	// The OAuth signature method, such as "HMAC-SHA1" to be used in the Authorization
	// header. The default is "HMAC-SHA1". It is also possible to choose "HMAC-SHA256",
	// "RSA-SHA1" or "RSA-SHA2". For RSA algorithms, an RSA private key would need to
	// be provided via the SetOAuthRsaKey method.
	// 
	// Note: RSA-SHA2 is supported starting in Chilkat v9.5.0.56
	// 
	void put_OAuthSigMethod(const char *newVal);

	// The OAuth1 token to be used in the Authorization header. Note: This is for
	// OAuth1. Use the AuthToken property for OAuth2.
	void get_OAuthToken(CkString &str);
	// The OAuth1 token to be used in the Authorization header. Note: This is for
	// OAuth1. Use the AuthToken property for OAuth2.
	const char *oAuthToken(void);
	// The OAuth1 token to be used in the Authorization header. Note: This is for
	// OAuth1. Use the AuthToken property for OAuth2.
	void put_OAuthToken(const char *newVal);

	// The OAuth token secret to be used in computing the Authorization header.
	void get_OAuthTokenSecret(CkString &str);
	// The OAuth token secret to be used in computing the Authorization header.
	const char *oAuthTokenSecret(void);
	// The OAuth token secret to be used in computing the Authorization header.
	void put_OAuthTokenSecret(const char *newVal);

	// The OAuth verifier to be used in the Authorization header.
	void get_OAuthVerifier(CkString &str);
	// The OAuth verifier to be used in the Authorization header.
	const char *oAuthVerifier(void);
	// The OAuth verifier to be used in the Authorization header.
	void put_OAuthVerifier(const char *newVal);

	// The HTTP password for pages requiring a login/password. Chilkat HTTP can do
	// Basic, Digest, and NTLM HTTP authentication. (NTLM is also known as SPA (or
	// Windows Integrated Authentication). To use Basic authentication, the BasicAuth
	// property must be set equal to true. It is not necessary to set the NtlmAuth or
	// DigestAuth properties beforehand if NTLM or Digest authentication is needed.
	// However, it is most efficient to pre-set these properties when the type of
	// authentication is known in advance.
	// 
	// Note: When the Login and Password properties are set, and the type of
	// authentication is specified by setting one of the following properties equal to
	// true (BasicAuth, DigestAuth, NtlmAuth), Chilkat will automatically add the
	// "Authorization: ..." header in the correct format.
	// 
	void get_Password(CkString &str);
	// The HTTP password for pages requiring a login/password. Chilkat HTTP can do
	// Basic, Digest, and NTLM HTTP authentication. (NTLM is also known as SPA (or
	// Windows Integrated Authentication). To use Basic authentication, the BasicAuth
	// property must be set equal to true. It is not necessary to set the NtlmAuth or
	// DigestAuth properties beforehand if NTLM or Digest authentication is needed.
	// However, it is most efficient to pre-set these properties when the type of
	// authentication is known in advance.
	// 
	// Note: When the Login and Password properties are set, and the type of
	// authentication is specified by setting one of the following properties equal to
	// true (BasicAuth, DigestAuth, NtlmAuth), Chilkat will automatically add the
	// "Authorization: ..." header in the correct format.
	// 
	const char *password(void);
	// The HTTP password for pages requiring a login/password. Chilkat HTTP can do
	// Basic, Digest, and NTLM HTTP authentication. (NTLM is also known as SPA (or
	// Windows Integrated Authentication). To use Basic authentication, the BasicAuth
	// property must be set equal to true. It is not necessary to set the NtlmAuth or
	// DigestAuth properties beforehand if NTLM or Digest authentication is needed.
	// However, it is most efficient to pre-set these properties when the type of
	// authentication is known in advance.
	// 
	// Note: When the Login and Password properties are set, and the type of
	// authentication is specified by setting one of the following properties equal to
	// true (BasicAuth, DigestAuth, NtlmAuth), Chilkat will automatically add the
	// "Authorization: ..." header in the correct format.
	// 
	void put_Password(const char *newVal);

	// This property is only valid in programming environment and languages that allow
	// for event callbacks.
	// 
	// Sets the value to be defined as 100% complete for the purpose of PercentDone
	// event callbacks. The defaut value of 100 means that at most 100 event
	// PercentDone callbacks will occur in a method that (1) is event enabled and (2)
	// is such that it is possible to measure progress as a percentage completed. This
	// property may be set to larger numbers to get more fine-grained PercentDone
	// callbacks. For example, setting this property equal to 1000 will provide
	// callbacks with .1 percent granularity. For example, a value of 453 would
	// indicate 45.3% competed. This property is clamped to a minimum value of 10, and
	// a maximum value of 100000.
	// 
	int get_PercentDoneScale(void);
	// This property is only valid in programming environment and languages that allow
	// for event callbacks.
	// 
	// Sets the value to be defined as 100% complete for the purpose of PercentDone
	// event callbacks. The defaut value of 100 means that at most 100 event
	// PercentDone callbacks will occur in a method that (1) is event enabled and (2)
	// is such that it is possible to measure progress as a percentage completed. This
	// property may be set to larger numbers to get more fine-grained PercentDone
	// callbacks. For example, setting this property equal to 1000 will provide
	// callbacks with .1 percent granularity. For example, a value of 453 would
	// indicate 45.3% competed. This property is clamped to a minimum value of 10, and
	// a maximum value of 100000.
	// 
	void put_PercentDoneScale(int newVal);

	// If true, then use IPv6 over IPv4 when both are supported for a particular
	// domain. The default value of this property is false, which will choose IPv4
	// over IPv6.
	bool get_PreferIpv6(void);
	// If true, then use IPv6 over IPv4 when both are supported for a particular
	// domain. The default value of this property is false, which will choose IPv4
	// over IPv6.
	void put_PreferIpv6(bool newVal);

	// Set this to "basic" if you know in advance that Basic authentication is to be
	// used for the HTTP proxy. Otherwise leave this property unset. Note: It is not
	// necessary to set this property. The HTTP component will automatically handle
	// proxy authentication for any of the supported authentication methods: NTLM,
	// Digest, or Basic. Setting this property equal to "basic" prevents the 407
	// response which is automatically handled internal to Chilkat and never seen by
	// your application.
	// 
	// Note: If NTLM authentication does not succeed, set the Global.DefaultNtlmVersion
	// property equal to 1 and then retry.
	// 
	void get_ProxyAuthMethod(CkString &str);
	// Set this to "basic" if you know in advance that Basic authentication is to be
	// used for the HTTP proxy. Otherwise leave this property unset. Note: It is not
	// necessary to set this property. The HTTP component will automatically handle
	// proxy authentication for any of the supported authentication methods: NTLM,
	// Digest, or Basic. Setting this property equal to "basic" prevents the 407
	// response which is automatically handled internal to Chilkat and never seen by
	// your application.
	// 
	// Note: If NTLM authentication does not succeed, set the Global.DefaultNtlmVersion
	// property equal to 1 and then retry.
	// 
	const char *proxyAuthMethod(void);
	// Set this to "basic" if you know in advance that Basic authentication is to be
	// used for the HTTP proxy. Otherwise leave this property unset. Note: It is not
	// necessary to set this property. The HTTP component will automatically handle
	// proxy authentication for any of the supported authentication methods: NTLM,
	// Digest, or Basic. Setting this property equal to "basic" prevents the 407
	// response which is automatically handled internal to Chilkat and never seen by
	// your application.
	// 
	// Note: If NTLM authentication does not succeed, set the Global.DefaultNtlmVersion
	// property equal to 1 and then retry.
	// 
	void put_ProxyAuthMethod(const char *newVal);

	// Set to true if the proxy server expects a direct TLS connection. (This is
	// where the initial connection to the HTTP proxy server is TLS. SeeSquid Direct
	// TLS Connection
	// <https://wiki.squid-cache.org/Features/HTTPS>. The default value of this
	// property is false.
	bool get_ProxyDirectTls(void);
	// Set to true if the proxy server expects a direct TLS connection. (This is
	// where the initial connection to the HTTP proxy server is TLS. SeeSquid Direct
	// TLS Connection
	// <https://wiki.squid-cache.org/Features/HTTPS>. The default value of this
	// property is false.
	void put_ProxyDirectTls(bool newVal);

	// The domain name of a proxy host if an HTTP proxy is used. This can also be set
	// to an IP address.
	void get_ProxyDomain(CkString &str);
	// The domain name of a proxy host if an HTTP proxy is used. This can also be set
	// to an IP address.
	const char *proxyDomain(void);
	// The domain name of a proxy host if an HTTP proxy is used. This can also be set
	// to an IP address.
	void put_ProxyDomain(const char *newVal);

	// If an HTTP proxy is used and it requires authentication, this property specifies
	// the HTTP proxy login.
	void get_ProxyLogin(CkString &str);
	// If an HTTP proxy is used and it requires authentication, this property specifies
	// the HTTP proxy login.
	const char *proxyLogin(void);
	// If an HTTP proxy is used and it requires authentication, this property specifies
	// the HTTP proxy login.
	void put_ProxyLogin(const char *newVal);

	// The NTLM authentication domain (optional) if NTLM authentication is used.
	void get_ProxyLoginDomain(CkString &str);
	// The NTLM authentication domain (optional) if NTLM authentication is used.
	const char *proxyLoginDomain(void);
	// The NTLM authentication domain (optional) if NTLM authentication is used.
	void put_ProxyLoginDomain(const char *newVal);

	// If an HTTP proxy is used and it requires authentication, this property specifies
	// the HTTP proxy password.
	void get_ProxyPassword(CkString &str);
	// If an HTTP proxy is used and it requires authentication, this property specifies
	// the HTTP proxy password.
	const char *proxyPassword(void);
	// If an HTTP proxy is used and it requires authentication, this property specifies
	// the HTTP proxy password.
	void put_ProxyPassword(const char *newVal);

	// The port number of a proxy server if an HTTP proxy is used.
	int get_ProxyPort(void);
	// The port number of a proxy server if an HTTP proxy is used.
	void put_ProxyPort(int newVal);

	// The amount of time in seconds to wait before timing out when reading from an
	// HTTP server. The ReadTimeout is the amount of time that needs to elapse while no
	// additional data is forthcoming. During a long download, if the data stream halts
	// for more than this amount, it will timeout. Otherwise, there is no limit on the
	// length of time for the entire download.
	// 
	// The default value is 60 seconds. Note: Prior to v9.5.0.76, the default was 20
	// seconds.
	// 
	int get_ReadTimeout(void);
	// The amount of time in seconds to wait before timing out when reading from an
	// HTTP server. The ReadTimeout is the amount of time that needs to elapse while no
	// additional data is forthcoming. During a long download, if the data stream halts
	// for more than this amount, it will timeout. Otherwise, there is no limit on the
	// length of time for the entire download.
	// 
	// The default value is 60 seconds. Note: Prior to v9.5.0.76, the default was 20
	// seconds.
	// 
	void put_ReadTimeout(int newVal);

	// Indicates the HTTP verb, such as GET, POST, PUT, etc. to be used for a redirect
	// when the FollowRedirects property is set to true. The default value of this
	// property is "GET". This will produce the same behavior as a web browser (such as
	// FireFox). If this property is set to the empty string, then it will cause the
	// same verb as the original HTTP request to be used.
	// 
	// Note: Prior to version 9.5.0.44, the default value of this property was the
	// empty string.
	// 
	void get_RedirectVerb(CkString &str);
	// Indicates the HTTP verb, such as GET, POST, PUT, etc. to be used for a redirect
	// when the FollowRedirects property is set to true. The default value of this
	// property is "GET". This will produce the same behavior as a web browser (such as
	// FireFox). If this property is set to the empty string, then it will cause the
	// same verb as the original HTTP request to be used.
	// 
	// Note: Prior to version 9.5.0.44, the default value of this property was the
	// empty string.
	// 
	const char *redirectVerb(void);
	// Indicates the HTTP verb, such as GET, POST, PUT, etc. to be used for a redirect
	// when the FollowRedirects property is set to true. The default value of this
	// property is "GET". This will produce the same behavior as a web browser (such as
	// FireFox). If this property is set to the empty string, then it will cause the
	// same verb as the original HTTP request to be used.
	// 
	// Note: Prior to version 9.5.0.44, the default value of this property was the
	// empty string.
	// 
	void put_RedirectVerb(const char *newVal);

	// The Referer header field to be automatically included with GET requests issued
	// by QuickGet or QuickGetStr. The default value is the empty string which causes
	// the Referer field to be omitted from the request header.
	void get_Referer(CkString &str);
	// The Referer header field to be automatically included with GET requests issued
	// by QuickGet or QuickGetStr. The default value is the empty string which causes
	// the Referer field to be omitted from the request header.
	const char *referer(void);
	// The Referer header field to be automatically included with GET requests issued
	// by QuickGet or QuickGetStr. The default value is the empty string which causes
	// the Referer field to be omitted from the request header.
	void put_Referer(const char *newVal);

	// If set, then any HTTP response to any POST or GET, including downloads, will be
	// rejected if the content-type in the response header does not match this setting.
	// If the content-type does not match, only the header of the HTTP response is
	// read, the connection to the HTTP server is closed, and the remainder of the
	// response is never read.
	// 
	// This property is empty (zero-length string) by default.
	// 
	// Some typical content-types are "text/html", "text/xml", "image/gif",
	// "image/jpeg", "application/zip", "application/msword", "application/pdf", etc.
	// 
	void get_RequiredContentType(CkString &str);
	// If set, then any HTTP response to any POST or GET, including downloads, will be
	// rejected if the content-type in the response header does not match this setting.
	// If the content-type does not match, only the header of the HTTP response is
	// read, the connection to the HTTP server is closed, and the remainder of the
	// response is never read.
	// 
	// This property is empty (zero-length string) by default.
	// 
	// Some typical content-types are "text/html", "text/xml", "image/gif",
	// "image/jpeg", "application/zip", "application/msword", "application/pdf", etc.
	// 
	const char *requiredContentType(void);
	// If set, then any HTTP response to any POST or GET, including downloads, will be
	// rejected if the content-type in the response header does not match this setting.
	// If the content-type does not match, only the header of the HTTP response is
	// read, the connection to the HTTP server is closed, and the remainder of the
	// response is never read.
	// 
	// This property is empty (zero-length string) by default.
	// 
	// Some typical content-types are "text/html", "text/xml", "image/gif",
	// "image/jpeg", "application/zip", "application/msword", "application/pdf", etc.
	// 
	void put_RequiredContentType(const char *newVal);

	// If true, then the HTTP client will verify the server's SSL certificate. The
	// certificate is expired, or if the cert's signature is invalid, the connection is
	// not allowed. The default value of this property is false.
	bool get_RequireSslCertVerify(void);
	// If true, then the HTTP client will verify the server's SSL certificate. The
	// certificate is expired, or if the cert's signature is invalid, the connection is
	// not allowed. The default value of this property is false.
	void put_RequireSslCertVerify(bool newVal);

	// If true, then all S3_* methods will use a secure SSL/TLS connection for
	// communications. The default value is true.
	bool get_S3Ssl(void);
	// If true, then all S3_* methods will use a secure SSL/TLS connection for
	// communications. The default value is true.
	void put_S3Ssl(bool newVal);

	// If this property is true, cookies are automatically persisted to XML files in
	// the directory specified by the CookiesDir property (or in memory if CookieDir =
	// "memory"). Both CookiesDir and SaveCookies must be set for cookies to be
	// persisted.
	bool get_SaveCookies(void);
	// If this property is true, cookies are automatically persisted to XML files in
	// the directory specified by the CookiesDir property (or in memory if CookieDir =
	// "memory"). Both CookiesDir and SaveCookies must be set for cookies to be
	// persisted.
	void put_SaveCookies(bool newVal);

	// The buffer size to be used with the underlying TCP/IP socket for sending. The
	// default value is 65535.
	int get_SendBufferSize(void);
	// The buffer size to be used with the underlying TCP/IP socket for sending. The
	// default value is 65535.
	void put_SendBufferSize(int newVal);

	// If true, then cookies previously persisted to the CookiesDir are automatically
	// added to all HTTP requests. Only cookies matching the domain and path are added.
	bool get_SendCookies(void);
	// If true, then cookies previously persisted to the CookiesDir are automatically
	// added to all HTTP requests. Only cookies matching the domain and path are added.
	void put_SendCookies(bool newVal);

	// Enables file-based session logging. If set to a filename (or relative/absolute
	// filepath), then the exact HTTP requests and responses are logged to a file. The
	// file is created if it does not already exist, otherwise it is appended.
	void get_SessionLogFilename(CkString &str);
	// Enables file-based session logging. If set to a filename (or relative/absolute
	// filepath), then the exact HTTP requests and responses are logged to a file. The
	// file is created if it does not already exist, otherwise it is appended.
	const char *sessionLogFilename(void);
	// Enables file-based session logging. If set to a filename (or relative/absolute
	// filepath), then the exact HTTP requests and responses are logged to a file. The
	// file is created if it does not already exist, otherwise it is appended.
	void put_SessionLogFilename(const char *newVal);

	// Specifies the SNI hostname to be used in the TLS ClientHello. This property is
	// only needed when the domain is specified via a dotted IP address and an SNI
	// hostname is desired. (Normally, Chilkat automatically uses the domain name in
	// the SNI hostname TLS ClientHello extension.)
	void get_SniHostname(CkString &str);
	// Specifies the SNI hostname to be used in the TLS ClientHello. This property is
	// only needed when the domain is specified via a dotted IP address and an SNI
	// hostname is desired. (Normally, Chilkat automatically uses the domain name in
	// the SNI hostname TLS ClientHello extension.)
	const char *sniHostname(void);
	// Specifies the SNI hostname to be used in the TLS ClientHello. This property is
	// only needed when the domain is specified via a dotted IP address and an SNI
	// hostname is desired. (Normally, Chilkat automatically uses the domain name in
	// the SNI hostname TLS ClientHello extension.)
	void put_SniHostname(const char *newVal);

	// The SOCKS4/SOCKS5 hostname or IPv4 address (in dotted decimal notation). This
	// property is only used if the SocksVersion property is set to 4 or 5).
	void get_SocksHostname(CkString &str);
	// The SOCKS4/SOCKS5 hostname or IPv4 address (in dotted decimal notation). This
	// property is only used if the SocksVersion property is set to 4 or 5).
	const char *socksHostname(void);
	// The SOCKS4/SOCKS5 hostname or IPv4 address (in dotted decimal notation). This
	// property is only used if the SocksVersion property is set to 4 or 5).
	void put_SocksHostname(const char *newVal);

	// The SOCKS5 password (if required). The SOCKS4 protocol does not include the use
	// of a password, so this does not apply to SOCKS4.
	void get_SocksPassword(CkString &str);
	// The SOCKS5 password (if required). The SOCKS4 protocol does not include the use
	// of a password, so this does not apply to SOCKS4.
	const char *socksPassword(void);
	// The SOCKS5 password (if required). The SOCKS4 protocol does not include the use
	// of a password, so this does not apply to SOCKS4.
	void put_SocksPassword(const char *newVal);

	// The SOCKS4/SOCKS5 proxy port. The default value is 1080. This property only
	// applies if a SOCKS proxy is used (if the SocksVersion property is set to 4 or
	// 5).
	int get_SocksPort(void);
	// The SOCKS4/SOCKS5 proxy port. The default value is 1080. This property only
	// applies if a SOCKS proxy is used (if the SocksVersion property is set to 4 or
	// 5).
	void put_SocksPort(int newVal);

	// The SOCKS4/SOCKS5 proxy username. This property is only used if the SocksVersion
	// property is set to 4 or 5).
	void get_SocksUsername(CkString &str);
	// The SOCKS4/SOCKS5 proxy username. This property is only used if the SocksVersion
	// property is set to 4 or 5).
	const char *socksUsername(void);
	// The SOCKS4/SOCKS5 proxy username. This property is only used if the SocksVersion
	// property is set to 4 or 5).
	void put_SocksUsername(const char *newVal);

	// SocksVersion May be set to one of the following integer values:
	// 
	// 0 - No SOCKS proxy is used. This is the default.
	// 4 - Connect via a SOCKS4 proxy.
	// 5 - Connect via a SOCKS5 proxy.
	// 
	int get_SocksVersion(void);
	// SocksVersion May be set to one of the following integer values:
	// 
	// 0 - No SOCKS proxy is used. This is the default.
	// 4 - Connect via a SOCKS4 proxy.
	// 5 - Connect via a SOCKS5 proxy.
	// 
	void put_SocksVersion(int newVal);

	// Sets the receive buffer size socket option. Normally, this property should be
	// left unchanged. The default value is 4194304.
	// 
	// This property can be increased if download performance seems slow. It is
	// recommended to be a multiple of 4096.
	// 
	int get_SoRcvBuf(void);
	// Sets the receive buffer size socket option. Normally, this property should be
	// left unchanged. The default value is 4194304.
	// 
	// This property can be increased if download performance seems slow. It is
	// recommended to be a multiple of 4096.
	// 
	void put_SoRcvBuf(int newVal);

	// Sets the send buffer size socket option. Normally, this property should be left
	// unchanged. The default value is 262144.
	// 
	// This property can be increased if upload performance seems slow. It is
	// recommended to be a multiple of 4096. Testing with sizes such as 512K and 1MB is
	// reasonable.
	// 
	int get_SoSndBuf(void);
	// Sets the send buffer size socket option. Normally, this property should be left
	// unchanged. The default value is 262144.
	// 
	// This property can be increased if upload performance seems slow. It is
	// recommended to be a multiple of 4096. Testing with sizes such as 512K and 1MB is
	// reasonable.
	// 
	void put_SoSndBuf(int newVal);

	// Provides a means for setting a list of ciphers that are allowed for SSL/TLS
	// connections. The default (empty string) indicates that all implemented ciphers
	// are possible. The TLS ciphers supported in Chilkat v9.5.0.55 and later are:
	// TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA
	// TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256
	// TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA
	// TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA
	// TLS_DHE_RSA_WITH_AES_256_CBC_SHA256
	// TLS_DHE_RSA_WITH_AES_256_GCM_SHA384
	// TLS_DHE_RSA_WITH_AES_256_CBC_SHA
	// TLS_RSA_WITH_AES_256_CBC_SHA256
	// TLS_RSA_WITH_AES_256_GCM_SHA384
	// TLS_RSA_WITH_AES_256_CBC_SHA
	// TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256
	// TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
	// TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA
	// TLS_DHE_RSA_WITH_AES_128_CBC_SHA256
	// TLS_DHE_RSA_WITH_AES_128_GCM_SHA256
	// TLS_DHE_RSA_WITH_AES_128_CBC_SHA
	// TLS_RSA_WITH_AES_128_CBC_SHA256
	// TLS_RSA_WITH_AES_128_GCM_SHA256
	// TLS_RSA_WITH_AES_128_CBC_SHA
	// TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_DHE_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_ECDHE_RSA_WITH_RC4_128_SHA
	// TLS_RSA_WITH_RC4_128_SHA
	// TLS_RSA_WITH_RC4_128_MD5
	// TLS_DHE_RSA_WITH_DES_CBC_SHA
	// TLS_RSA_WITH_DES_CBC_SHA
	// To restrict SSL/TLS connections to one or more specific ciphers, set this
	// property to a comma-separated list of ciphers such as
	// "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384, TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384".
	// The order should be in terms of preference, with the preferred algorithms listed
	// first. (Note that the client cannot specifically choose the algorithm is picked
	// because it is the server that chooses. The client simply provides the server
	// with a list from which to choose.)
	// 
	// The property can also disallow connections with servers having certificates with
	// RSA keys less than a certain size. By default, server certificates having RSA
	// keys of 512 bits or greater are allowed. Add the keyword "rsa1024" to disallow
	// connections with servers having keys smaller than 1024 bits. Add the keyword
	// "rsa2048" to disallow connections with servers having keys smaller than 2048
	// bits.
	// 
	// Note: Prior to Chilkat v9.5.0.55, it was not possible to explicitly list allowed
	// cipher suites. The deprecated means for indicating allowed ciphers was both
	// incomplete and unprecise. For example, the following keywords could be listed to
	// allow matching ciphers: "aes256-cbc", "aes128-cbc", "3des-cbc", and "rc4". These
	// keywords will still be recognized, but programs should be updated to explicitly
	// list the allowed ciphers.
	// 
	// secure-renegotiation: Starting in Chilkat v9.5.0.55, the keyword
	// "secure-renegotiation" may be added to require that all renegotions be done
	// securely (as per RFC 5746).
	// 
	// best-practices: Starting in Chilkat v9.5.0.55, this property may be set to the
	// single keyword "best-practices". This will allow ciphers based on the current
	// best practices. As new versions of Chilkat are released, the best practices may
	// change. Changes will be noted here. The current best practices are:
	// 
	//     If the server uses an RSA key, it must be 1024 bits or greater.
	//     All renegotations must be secure renegotiations.
	//     All ciphers using RC4, DES, or 3DES are disallowed.
	// 
	// Example: The following string would restrict to 2 specific cipher suites,
	// require RSA keys to be 1024 bits or greater, and require secure renegotiations:
	// "TLS_DHE_RSA_WITH_AES_256_CBC_SHA256, TLS_RSA_WITH_AES_256_CBC_SHA, rsa1024,
	// secure-renegotiation"
	// 
	void get_SslAllowedCiphers(CkString &str);
	// Provides a means for setting a list of ciphers that are allowed for SSL/TLS
	// connections. The default (empty string) indicates that all implemented ciphers
	// are possible. The TLS ciphers supported in Chilkat v9.5.0.55 and later are:
	// TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA
	// TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256
	// TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA
	// TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA
	// TLS_DHE_RSA_WITH_AES_256_CBC_SHA256
	// TLS_DHE_RSA_WITH_AES_256_GCM_SHA384
	// TLS_DHE_RSA_WITH_AES_256_CBC_SHA
	// TLS_RSA_WITH_AES_256_CBC_SHA256
	// TLS_RSA_WITH_AES_256_GCM_SHA384
	// TLS_RSA_WITH_AES_256_CBC_SHA
	// TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256
	// TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
	// TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA
	// TLS_DHE_RSA_WITH_AES_128_CBC_SHA256
	// TLS_DHE_RSA_WITH_AES_128_GCM_SHA256
	// TLS_DHE_RSA_WITH_AES_128_CBC_SHA
	// TLS_RSA_WITH_AES_128_CBC_SHA256
	// TLS_RSA_WITH_AES_128_GCM_SHA256
	// TLS_RSA_WITH_AES_128_CBC_SHA
	// TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_DHE_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_ECDHE_RSA_WITH_RC4_128_SHA
	// TLS_RSA_WITH_RC4_128_SHA
	// TLS_RSA_WITH_RC4_128_MD5
	// TLS_DHE_RSA_WITH_DES_CBC_SHA
	// TLS_RSA_WITH_DES_CBC_SHA
	// To restrict SSL/TLS connections to one or more specific ciphers, set this
	// property to a comma-separated list of ciphers such as
	// "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384, TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384".
	// The order should be in terms of preference, with the preferred algorithms listed
	// first. (Note that the client cannot specifically choose the algorithm is picked
	// because it is the server that chooses. The client simply provides the server
	// with a list from which to choose.)
	// 
	// The property can also disallow connections with servers having certificates with
	// RSA keys less than a certain size. By default, server certificates having RSA
	// keys of 512 bits or greater are allowed. Add the keyword "rsa1024" to disallow
	// connections with servers having keys smaller than 1024 bits. Add the keyword
	// "rsa2048" to disallow connections with servers having keys smaller than 2048
	// bits.
	// 
	// Note: Prior to Chilkat v9.5.0.55, it was not possible to explicitly list allowed
	// cipher suites. The deprecated means for indicating allowed ciphers was both
	// incomplete and unprecise. For example, the following keywords could be listed to
	// allow matching ciphers: "aes256-cbc", "aes128-cbc", "3des-cbc", and "rc4". These
	// keywords will still be recognized, but programs should be updated to explicitly
	// list the allowed ciphers.
	// 
	// secure-renegotiation: Starting in Chilkat v9.5.0.55, the keyword
	// "secure-renegotiation" may be added to require that all renegotions be done
	// securely (as per RFC 5746).
	// 
	// best-practices: Starting in Chilkat v9.5.0.55, this property may be set to the
	// single keyword "best-practices". This will allow ciphers based on the current
	// best practices. As new versions of Chilkat are released, the best practices may
	// change. Changes will be noted here. The current best practices are:
	// 
	//     If the server uses an RSA key, it must be 1024 bits or greater.
	//     All renegotations must be secure renegotiations.
	//     All ciphers using RC4, DES, or 3DES are disallowed.
	// 
	// Example: The following string would restrict to 2 specific cipher suites,
	// require RSA keys to be 1024 bits or greater, and require secure renegotiations:
	// "TLS_DHE_RSA_WITH_AES_256_CBC_SHA256, TLS_RSA_WITH_AES_256_CBC_SHA, rsa1024,
	// secure-renegotiation"
	// 
	const char *sslAllowedCiphers(void);
	// Provides a means for setting a list of ciphers that are allowed for SSL/TLS
	// connections. The default (empty string) indicates that all implemented ciphers
	// are possible. The TLS ciphers supported in Chilkat v9.5.0.55 and later are:
	// TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA
	// TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256
	// TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA
	// TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA
	// TLS_DHE_RSA_WITH_AES_256_CBC_SHA256
	// TLS_DHE_RSA_WITH_AES_256_GCM_SHA384
	// TLS_DHE_RSA_WITH_AES_256_CBC_SHA
	// TLS_RSA_WITH_AES_256_CBC_SHA256
	// TLS_RSA_WITH_AES_256_GCM_SHA384
	// TLS_RSA_WITH_AES_256_CBC_SHA
	// TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256
	// TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
	// TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA
	// TLS_DHE_RSA_WITH_AES_128_CBC_SHA256
	// TLS_DHE_RSA_WITH_AES_128_GCM_SHA256
	// TLS_DHE_RSA_WITH_AES_128_CBC_SHA
	// TLS_RSA_WITH_AES_128_CBC_SHA256
	// TLS_RSA_WITH_AES_128_GCM_SHA256
	// TLS_RSA_WITH_AES_128_CBC_SHA
	// TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_DHE_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_ECDHE_RSA_WITH_RC4_128_SHA
	// TLS_RSA_WITH_RC4_128_SHA
	// TLS_RSA_WITH_RC4_128_MD5
	// TLS_DHE_RSA_WITH_DES_CBC_SHA
	// TLS_RSA_WITH_DES_CBC_SHA
	// To restrict SSL/TLS connections to one or more specific ciphers, set this
	// property to a comma-separated list of ciphers such as
	// "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384, TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384".
	// The order should be in terms of preference, with the preferred algorithms listed
	// first. (Note that the client cannot specifically choose the algorithm is picked
	// because it is the server that chooses. The client simply provides the server
	// with a list from which to choose.)
	// 
	// The property can also disallow connections with servers having certificates with
	// RSA keys less than a certain size. By default, server certificates having RSA
	// keys of 512 bits or greater are allowed. Add the keyword "rsa1024" to disallow
	// connections with servers having keys smaller than 1024 bits. Add the keyword
	// "rsa2048" to disallow connections with servers having keys smaller than 2048
	// bits.
	// 
	// Note: Prior to Chilkat v9.5.0.55, it was not possible to explicitly list allowed
	// cipher suites. The deprecated means for indicating allowed ciphers was both
	// incomplete and unprecise. For example, the following keywords could be listed to
	// allow matching ciphers: "aes256-cbc", "aes128-cbc", "3des-cbc", and "rc4". These
	// keywords will still be recognized, but programs should be updated to explicitly
	// list the allowed ciphers.
	// 
	// secure-renegotiation: Starting in Chilkat v9.5.0.55, the keyword
	// "secure-renegotiation" may be added to require that all renegotions be done
	// securely (as per RFC 5746).
	// 
	// best-practices: Starting in Chilkat v9.5.0.55, this property may be set to the
	// single keyword "best-practices". This will allow ciphers based on the current
	// best practices. As new versions of Chilkat are released, the best practices may
	// change. Changes will be noted here. The current best practices are:
	// 
	//     If the server uses an RSA key, it must be 1024 bits or greater.
	//     All renegotations must be secure renegotiations.
	//     All ciphers using RC4, DES, or 3DES are disallowed.
	// 
	// Example: The following string would restrict to 2 specific cipher suites,
	// require RSA keys to be 1024 bits or greater, and require secure renegotiations:
	// "TLS_DHE_RSA_WITH_AES_256_CBC_SHA256, TLS_RSA_WITH_AES_256_CBC_SHA, rsa1024,
	// secure-renegotiation"
	// 
	void put_SslAllowedCiphers(const char *newVal);

	// Note: Chilkat v9.5.0.84 supports TLS 1.3. In any TLS connection, during the
	// initial TLS handshake, the client (such as Chilkat) indicates which versions of
	// TLS it supports, and the server selects the version of TLS to be used for the
	// connection. Most applications should NOT explictly set this value.
	// 
	// Selects the secure protocol to be used for secure (SSL/TLS) connections.
	// Possible values are:
	// 
	//     default
	//     TLS 1.3
	//     TLS 1.2
	//     TLS 1.1
	//     TLS 1.0
	//     SSL 3.0
	//     TLS 1.3 or higher
	//     TLS 1.2 or higher
	//     TLS 1.1 or higher
	//     TLS 1.0 or higher
	//     
	// 
	// The default value is "default" which will choose the, which allows for the
	// protocol to be selected dynamically at runtime based on the requirements of the
	// server. Choosing an exact protocol will cause the connection to fail unless that
	// exact protocol is negotiated. It is better to choose "X or higher" than an exact
	// protocol. The "default" is effectively "SSL 3.0 or higher".
	// 
	void get_SslProtocol(CkString &str);
	// Note: Chilkat v9.5.0.84 supports TLS 1.3. In any TLS connection, during the
	// initial TLS handshake, the client (such as Chilkat) indicates which versions of
	// TLS it supports, and the server selects the version of TLS to be used for the
	// connection. Most applications should NOT explictly set this value.
	// 
	// Selects the secure protocol to be used for secure (SSL/TLS) connections.
	// Possible values are:
	// 
	//     default
	//     TLS 1.3
	//     TLS 1.2
	//     TLS 1.1
	//     TLS 1.0
	//     SSL 3.0
	//     TLS 1.3 or higher
	//     TLS 1.2 or higher
	//     TLS 1.1 or higher
	//     TLS 1.0 or higher
	//     
	// 
	// The default value is "default" which will choose the, which allows for the
	// protocol to be selected dynamically at runtime based on the requirements of the
	// server. Choosing an exact protocol will cause the connection to fail unless that
	// exact protocol is negotiated. It is better to choose "X or higher" than an exact
	// protocol. The "default" is effectively "SSL 3.0 or higher".
	// 
	const char *sslProtocol(void);
	// Note: Chilkat v9.5.0.84 supports TLS 1.3. In any TLS connection, during the
	// initial TLS handshake, the client (such as Chilkat) indicates which versions of
	// TLS it supports, and the server selects the version of TLS to be used for the
	// connection. Most applications should NOT explictly set this value.
	// 
	// Selects the secure protocol to be used for secure (SSL/TLS) connections.
	// Possible values are:
	// 
	//     default
	//     TLS 1.3
	//     TLS 1.2
	//     TLS 1.1
	//     TLS 1.0
	//     SSL 3.0
	//     TLS 1.3 or higher
	//     TLS 1.2 or higher
	//     TLS 1.1 or higher
	//     TLS 1.0 or higher
	//     
	// 
	// The default value is "default" which will choose the, which allows for the
	// protocol to be selected dynamically at runtime based on the requirements of the
	// server. Choosing an exact protocol will cause the connection to fail unless that
	// exact protocol is negotiated. It is better to choose "X or higher" than an exact
	// protocol. The "default" is effectively "SSL 3.0 or higher".
	// 
	void put_SslProtocol(const char *newVal);

	// Allows for the HTTP response body to be streamed directly into a file. If this
	// property is set, then any method returning an HTTP response object will stream
	// the response body directly to the file path specified. The HTTP response object
	// will still contain the response header. (This property is useful when the HTTP
	// response is too large to fit into memory.)
	void get_StreamResponseBodyPath(CkString &str);
	// Allows for the HTTP response body to be streamed directly into a file. If this
	// property is set, then any method returning an HTTP response object will stream
	// the response body directly to the file path specified. The HTTP response object
	// will still contain the response header. (This property is useful when the HTTP
	// response is too large to fit into memory.)
	const char *streamResponseBodyPath(void);
	// Allows for the HTTP response body to be streamed directly into a file. If this
	// property is set, then any method returning an HTTP response object will stream
	// the response body directly to the file path specified. The HTTP response object
	// will still contain the response header. (This property is useful when the HTTP
	// response is too large to fit into memory.)
	void put_StreamResponseBodyPath(const char *newVal);

	// Contains the current or last negotiated TLS cipher suite. If no TLS connection
	// has yet to be established, or if a connection as attempted and failed, then this
	// will be empty. A sample cipher suite string looks like this:
	// TLS_DHE_RSA_WITH_AES_256_CBC_SHA256.
	void get_TlsCipherSuite(CkString &str);
	// Contains the current or last negotiated TLS cipher suite. If no TLS connection
	// has yet to be established, or if a connection as attempted and failed, then this
	// will be empty. A sample cipher suite string looks like this:
	// TLS_DHE_RSA_WITH_AES_256_CBC_SHA256.
	const char *tlsCipherSuite(void);

	// Specifies a set of pins for Public Key Pinning for TLS connections. This
	// property lists the expected SPKI fingerprints for the server certificates. If
	// the server's certificate (sent during the TLS handshake) does not match any of
	// the SPKI fingerprints, then the TLS handshake is aborted and the connection
	// fails. The format of this string property is as follows:
	// hash_algorithm, encoding, SPKI_fingerprint_1, SPKI_fingerprint_2, ...
	// For example, the following string specifies a single sha256 base64-encoded SPKI
	// fingerprint:
	// "sha256, base64, lKg1SIqyhPSK19tlPbjl8s02yChsVTDklQpkMCHvsTE="
	// This example specifies two SPKI fingerprints:
	// "sha256, base64, 4t37LpnGmrMEAG8HEz9yIrnvJV2euVRwCLb9EH5WZyI=, 68b0G5iqMvWVWvUCjMuhLEyekM5729PadtnU5tdXZKs="
	// Any of the following hash algorithms are allowed:.sha1, sha256, sha384, sha512,
	// md2, md5, haval, ripemd128, ripemd160,ripemd256, or ripemd320.
	// 
	// The following encodings are allowed: base64, hex, and any of the encodings
	// indicated in the link below.
	// 
	void get_TlsPinSet(CkString &str);
	// Specifies a set of pins for Public Key Pinning for TLS connections. This
	// property lists the expected SPKI fingerprints for the server certificates. If
	// the server's certificate (sent during the TLS handshake) does not match any of
	// the SPKI fingerprints, then the TLS handshake is aborted and the connection
	// fails. The format of this string property is as follows:
	// hash_algorithm, encoding, SPKI_fingerprint_1, SPKI_fingerprint_2, ...
	// For example, the following string specifies a single sha256 base64-encoded SPKI
	// fingerprint:
	// "sha256, base64, lKg1SIqyhPSK19tlPbjl8s02yChsVTDklQpkMCHvsTE="
	// This example specifies two SPKI fingerprints:
	// "sha256, base64, 4t37LpnGmrMEAG8HEz9yIrnvJV2euVRwCLb9EH5WZyI=, 68b0G5iqMvWVWvUCjMuhLEyekM5729PadtnU5tdXZKs="
	// Any of the following hash algorithms are allowed:.sha1, sha256, sha384, sha512,
	// md2, md5, haval, ripemd128, ripemd160,ripemd256, or ripemd320.
	// 
	// The following encodings are allowed: base64, hex, and any of the encodings
	// indicated in the link below.
	// 
	const char *tlsPinSet(void);
	// Specifies a set of pins for Public Key Pinning for TLS connections. This
	// property lists the expected SPKI fingerprints for the server certificates. If
	// the server's certificate (sent during the TLS handshake) does not match any of
	// the SPKI fingerprints, then the TLS handshake is aborted and the connection
	// fails. The format of this string property is as follows:
	// hash_algorithm, encoding, SPKI_fingerprint_1, SPKI_fingerprint_2, ...
	// For example, the following string specifies a single sha256 base64-encoded SPKI
	// fingerprint:
	// "sha256, base64, lKg1SIqyhPSK19tlPbjl8s02yChsVTDklQpkMCHvsTE="
	// This example specifies two SPKI fingerprints:
	// "sha256, base64, 4t37LpnGmrMEAG8HEz9yIrnvJV2euVRwCLb9EH5WZyI=, 68b0G5iqMvWVWvUCjMuhLEyekM5729PadtnU5tdXZKs="
	// Any of the following hash algorithms are allowed:.sha1, sha256, sha384, sha512,
	// md2, md5, haval, ripemd128, ripemd160,ripemd256, or ripemd320.
	// 
	// The following encodings are allowed: base64, hex, and any of the encodings
	// indicated in the link below.
	// 
	void put_TlsPinSet(const char *newVal);

	// Contains the current or last negotiated TLS protocol version. If no TLS
	// connection has yet to be established, or if a connection as attempted and
	// failed, then this will be empty. Possible values are "SSL 3.0", "TLS 1.0", "TLS
	// 1.1", "TLS 1.2", and "TLS 1.3".
	void get_TlsVersion(CkString &str);
	// Contains the current or last negotiated TLS protocol version. If no TLS
	// connection has yet to be established, or if a connection as attempted and
	// failed, then this will be empty. Possible values are "SSL 3.0", "TLS 1.0", "TLS
	// 1.1", "TLS 1.2", and "TLS 1.3".
	const char *tlsVersion(void);

	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty. Can be set to a
	// list of the following comma separated keywords:
	//     "QuickDisconnect" - Introduced in v9.5.0.77. In the call to
	//     CloseAllConnections, do not disconnect cleanly. Instead just disconnect as
	//     quickly as possible.
	//     "ProtectFromVpn" - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	//     "TlsNoClientRootCert" - Introduced in v9.5.0.82. Will exclude root CA certs
	//     from being included in the client certificate chain that is sent to the server
	//     for client-side authentication. This must be set prior to calling
	//     SetSslClientCert.
	//     "AllowEmptyHeaders" - Introduced in v9.5.0.82. If present, an empty value
	//     string passed to SetHeaderField will cause the header to be added with an empty
	//     value. Otherwise, for historical purposes and backward compatibility, the header
	//     field is removed when an empty value string is passed.
	//     "AnsiLogin" - Introduced in v9.5.0.87. For HTTP basic authentication, the
	//     login and password is sent using the utf-8 byte representation. Some servers
	//     expect the ANSI byte representation (typically Windows-1252). Use this keyword
	//     to send the login/password using ANSI.
	void get_UncommonOptions(CkString &str);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty. Can be set to a
	// list of the following comma separated keywords:
	//     "QuickDisconnect" - Introduced in v9.5.0.77. In the call to
	//     CloseAllConnections, do not disconnect cleanly. Instead just disconnect as
	//     quickly as possible.
	//     "ProtectFromVpn" - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	//     "TlsNoClientRootCert" - Introduced in v9.5.0.82. Will exclude root CA certs
	//     from being included in the client certificate chain that is sent to the server
	//     for client-side authentication. This must be set prior to calling
	//     SetSslClientCert.
	//     "AllowEmptyHeaders" - Introduced in v9.5.0.82. If present, an empty value
	//     string passed to SetHeaderField will cause the header to be added with an empty
	//     value. Otherwise, for historical purposes and backward compatibility, the header
	//     field is removed when an empty value string is passed.
	//     "AnsiLogin" - Introduced in v9.5.0.87. For HTTP basic authentication, the
	//     login and password is sent using the utf-8 byte representation. Some servers
	//     expect the ANSI byte representation (typically Windows-1252). Use this keyword
	//     to send the login/password using ANSI.
	const char *uncommonOptions(void);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty. Can be set to a
	// list of the following comma separated keywords:
	//     "QuickDisconnect" - Introduced in v9.5.0.77. In the call to
	//     CloseAllConnections, do not disconnect cleanly. Instead just disconnect as
	//     quickly as possible.
	//     "ProtectFromVpn" - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	//     "TlsNoClientRootCert" - Introduced in v9.5.0.82. Will exclude root CA certs
	//     from being included in the client certificate chain that is sent to the server
	//     for client-side authentication. This must be set prior to calling
	//     SetSslClientCert.
	//     "AllowEmptyHeaders" - Introduced in v9.5.0.82. If present, an empty value
	//     string passed to SetHeaderField will cause the header to be added with an empty
	//     value. Otherwise, for historical purposes and backward compatibility, the header
	//     field is removed when an empty value string is passed.
	//     "AnsiLogin" - Introduced in v9.5.0.87. For HTTP basic authentication, the
	//     login and password is sent using the utf-8 byte representation. Some servers
	//     expect the ANSI byte representation (typically Windows-1252). Use this keyword
	//     to send the login/password using ANSI.
	void put_UncommonOptions(const char *newVal);

	// Controls whether the cache is automatically updated with the responses from HTTP
	// GET requests.
	bool get_UpdateCache(void);
	// Controls whether the cache is automatically updated with the responses from HTTP
	// GET requests.
	void put_UpdateCache(bool newVal);

	// If true, the proxy address/port used by Internet Explorer will also be used by
	// Chilkat HTTP. Note: This property only pays attention to the proxy address and
	// port, and does not pay attention to additional information such as IE proxy
	// server exceptions.
	bool get_UseIEProxy(void);
	// If true, the proxy address/port used by Internet Explorer will also be used by
	// Chilkat HTTP. Note: This property only pays attention to the proxy address and
	// port, and does not pay attention to additional information such as IE proxy
	// server exceptions.
	void put_UseIEProxy(bool newVal);

	// The UserAgent header field to be automatically included with GET requests issued
	// by QuickGet or QuickGetStr. The default value is "Mozilla/5.0 (Windows NT 6.3;
	// WOW64; rv:49.0) Gecko/20100101 Firefox/49.0". The reason for this default is to
	// get the same server behavior for a recent version of a typical and popular
	// browser. Some sites may respond differently depending on the User-Agent, and the
	// goal is to avoid strange responses that are different than what a typical
	// browser would receive.
	void get_UserAgent(CkString &str);
	// The UserAgent header field to be automatically included with GET requests issued
	// by QuickGet or QuickGetStr. The default value is "Mozilla/5.0 (Windows NT 6.3;
	// WOW64; rv:49.0) Gecko/20100101 Firefox/49.0". The reason for this default is to
	// get the same server behavior for a recent version of a typical and popular
	// browser. Some sites may respond differently depending on the User-Agent, and the
	// goal is to avoid strange responses that are different than what a typical
	// browser would receive.
	const char *userAgent(void);
	// The UserAgent header field to be automatically included with GET requests issued
	// by QuickGet or QuickGetStr. The default value is "Mozilla/5.0 (Windows NT 6.3;
	// WOW64; rv:49.0) Gecko/20100101 Firefox/49.0". The reason for this default is to
	// get the same server behavior for a recent version of a typical and popular
	// browser. Some sites may respond differently depending on the User-Agent, and the
	// goal is to avoid strange responses that are different than what a typical
	// browser would receive.
	void put_UserAgent(const char *newVal);

	// Indicates whether the last HTTP request was redirected (i.e. the redirect was
	// automatically followed). If FollowRedirects is false and the response status
	// code is 301, 302, 303, 307, or 308, then this property will remain false but
	// the FinalRedirectUrl will contain the redirect URL that was not automatically
	// followed.
	bool get_WasRedirected(void);



	// ----------------------
	// Methods
	// ----------------------
	// This method must be called at least once if disk caching is to be used. The file
	// path (including drive letter) such as "E:\MyHttpCache\" is passed to
	// AddCacheRoot to specify the root directory. The cache can be spread across
	// multiple disk drives by calling AddCacheRoot multiple times, each with a
	// directory path on a separate disk drive.
	void AddCacheRoot(const char *dir);


	// Removes all headers added via the SetRequestHeader method.
	void ClearHeaders(void);


	// Clears all cookies cached in memory. Calling this only makes sense if the
	// CookieDir property is set to the string "memory".
	void ClearInMemoryCookies(void);


	// Clears all URL variable values previously set by one or more calls to SetUrlVar.
	void ClearUrlVars(void);


	// Closes all connections still open from previous HTTP requests.
	// 
	// An HTTP object instance will maintain up to 10 connections. If the HTTP server's
	// response does not include a "Connection: Close" header, the connection will
	// remain open and will be re-used if possible for the next HTTP request to the
	// same hostname:port. (It uses the IP address (in string form) or the domain name,
	// whichever is used in the URL provided by the application.) If 10 connections are
	// already open and another is needed, the object will close the least recently
	// used connection.
	// 
	bool CloseAllConnections(void);

	// Closes all connections still open from previous HTTP requests.
	// 
	// An HTTP object instance will maintain up to 10 connections. If the HTTP server's
	// response does not include a "Connection: Close" header, the connection will
	// remain open and will be re-used if possible for the next HTTP request to the
	// same hostname:port. (It uses the IP address (in string form) or the domain name,
	// whichever is used in the URL provided by the application.) If 10 connections are
	// already open and another is needed, the object will close the least recently
	// used connection.
	// 
	CkTask *CloseAllConnectionsAsync(void);


	// Creates an OCSP request for one or more certificates. The requestDetails is a JSON
	// document describing the content of the OCSP request to be created. The examples
	// in the links below show how to build the JSON request details.
	// 
	// Note: This method only creates an OCSP request. After creating, it may be sent
	// to a server to get the OCSP response.
	// 
	bool CreateOcspRequest(CkJsonObject &requestDetails, CkBinData &ocspRequest);


	// Creates an RFC 3161 time-stamp request and returns the binary request token in
	// timestampToken. The hashAlg can be "sha1", "sha256", "sha384", "sha512", or "md5", The hashVal
	// is the base64 hash of the data to be timestamped. The optional reqPolicyOid is the
	// requested policy OID in a format such as "1.3.6.1.4.1.47272.1.2". The addNonce
	// indicates whether to auto-generate and include a nonce in the request. It may be
	// true or false. The reqTsaCert determines whether or not to request the TSA's
	// certificate (true = Yes, false = No).
	// 
	// Note: This method only creates a timestamp request. After creating, it may be
	// sent to a server to get the binary timestamp token.
	// 
	bool CreateTimestampRequest(const char *hashAlg, const char *hashVal, const char *reqPolicyOid, bool addNonce, bool reqTsaCert, CkBinData &timestampToken);


	// Clears the Chilkat-wide in-memory hostname-to-IP address DNS cache. Chilkat
	// automatically maintains this in-memory cache to prevent redundant DNS lookups.
	// If the TTL on the DNS A records being accessed are short and/or these DNS
	// records change frequently, then this method can be called clear the internal
	// cache. Note: The DNS cache is used/shared among all Chilkat objects in a
	// program, and clearing the cache affects all Chilkat objects.
	void DnsCacheClear(void);


	// Retrieves the content at a URL and saves to a file. All content is saved in
	// streaming mode such that the memory footprint is small and steady. HTTPS is
	// fully supported, as it is with all the methods of this class.
	bool Download(const char *url, const char *localFilePath);

	// Retrieves the content at a URL and saves to a file. All content is saved in
	// streaming mode such that the memory footprint is small and steady. HTTPS is
	// fully supported, as it is with all the methods of this class.
	CkTask *DownloadAsync(const char *url, const char *localFilePath);


	// Same as the Download method, but the output file is open for append.
	bool DownloadAppend(const char *url, const char *filename);

	// Same as the Download method, but the output file is open for append.
	CkTask *DownloadAppendAsync(const char *url, const char *filename);


	// Downloads the content at the url into a BinData object.
	bool DownloadBd(const char *url, CkBinData &binData);

	// Downloads the content at the url into a BinData object.
	CkTask *DownloadBdAsync(const char *url, CkBinData &binData);


	// Retrieves the content at a URL and computes and returns a hash of the content.
	// The hash is returned as an encoded string according to the encoding, which may be
	// "Base64", "modBase64", "Base32", "UU", "QP" (for quoted-printable), "URL" (for
	// url-encoding), "Hex", "Q", "B", "url_oath", "url_rfc1738", "url_rfc2396", and
	// "url_rfc3986". The hashAlgorithm may be "sha1", "sha256", "sha384", "sha512", "md2",
	// "md5", "haval", "ripemd128", "ripemd160","ripemd256", or "ripemd320".
	bool DownloadHash(const char *url, const char *hashAlgorithm, const char *encoding, CkString &outStr);

	// Retrieves the content at a URL and computes and returns a hash of the content.
	// The hash is returned as an encoded string according to the encoding, which may be
	// "Base64", "modBase64", "Base32", "UU", "QP" (for quoted-printable), "URL" (for
	// url-encoding), "Hex", "Q", "B", "url_oath", "url_rfc1738", "url_rfc2396", and
	// "url_rfc3986". The hashAlgorithm may be "sha1", "sha256", "sha384", "sha512", "md2",
	// "md5", "haval", "ripemd128", "ripemd160","ripemd256", or "ripemd320".
	const char *downloadHash(const char *url, const char *hashAlgorithm, const char *encoding);
	// Retrieves the content at a URL and computes and returns a hash of the content.
	// The hash is returned as an encoded string according to the encoding, which may be
	// "Base64", "modBase64", "Base32", "UU", "QP" (for quoted-printable), "URL" (for
	// url-encoding), "Hex", "Q", "B", "url_oath", "url_rfc1738", "url_rfc2396", and
	// "url_rfc3986". The hashAlgorithm may be "sha1", "sha256", "sha384", "sha512", "md2",
	// "md5", "haval", "ripemd128", "ripemd160","ripemd256", or "ripemd320".
	CkTask *DownloadHashAsync(const char *url, const char *hashAlgorithm, const char *encoding);


	// Downloads the content at the url into a Chilkat StringBuilder object. The charset
	// tells Chilkat how to interpret the bytes received. The sb is appended with the
	// downloaded text data.
	bool DownloadSb(const char *url, const char *charset, CkStringBuilder &sb);

	// Downloads the content at the url into a Chilkat StringBuilder object. The charset
	// tells Chilkat how to interpret the bytes received. The sb is appended with the
	// downloaded text data.
	CkTask *DownloadSbAsync(const char *url, const char *charset, CkStringBuilder &sb);


	// Convenience method for extracting the META refresh URL from HTML. For example,
	// if the htmlContent contains a META refresh tag, such as:
	// _LT_meta http-equiv="refresh" content="5;URL='http://example.com/'"_GT_
	// Then the return value of this method would be "http://example.com/".
	bool ExtractMetaRefreshUrl(const char *htmlContent, CkString &outStr);

	// Convenience method for extracting the META refresh URL from HTML. For example,
	// if the htmlContent contains a META refresh tag, such as:
	// _LT_meta http-equiv="refresh" content="5;URL='http://example.com/'"_GT_
	// Then the return value of this method would be "http://example.com/".
	const char *extractMetaRefreshUrl(const char *htmlContent);

	// Makes an access token request to obtain a Google API OAuth2 access token for a
	// service account. Access tokens issued by the Google OAuth 2.0 Authorization
	// Server expire one hour after they are issued. When an access token expires, then
	// the application should generate another JWT, sign it, and request another access
	// token. The iss is the service account email address of the application making
	// the access token request. The scope is a space-delimited list of the permissions
	// that the application requests. (See
	// https://developers.google.com/accounts/docs/OAuth2ServiceAccount )
	// 
	// The subEmail is the email address of the user for which the application is
	// requesting delegated access. The subEmail may be left empty if there is no such
	// email address.
	// 
	bool G_SvcOauthAccessToken(const char *iss, const char *scope, const char *subEmail, int numSec, CkCert &cert, CkString &outStr);

	// Makes an access token request to obtain a Google API OAuth2 access token for a
	// service account. Access tokens issued by the Google OAuth 2.0 Authorization
	// Server expire one hour after they are issued. When an access token expires, then
	// the application should generate another JWT, sign it, and request another access
	// token. The iss is the service account email address of the application making
	// the access token request. The scope is a space-delimited list of the permissions
	// that the application requests. (See
	// https://developers.google.com/accounts/docs/OAuth2ServiceAccount )
	// 
	// The subEmail is the email address of the user for which the application is
	// requesting delegated access. The subEmail may be left empty if there is no such
	// email address.
	// 
	const char *g_SvcOauthAccessToken(const char *iss, const char *scope, const char *subEmail, int numSec, CkCert &cert);
	// Makes an access token request to obtain a Google API OAuth2 access token for a
	// service account. Access tokens issued by the Google OAuth 2.0 Authorization
	// Server expire one hour after they are issued. When an access token expires, then
	// the application should generate another JWT, sign it, and request another access
	// token. The iss is the service account email address of the application making
	// the access token request. The scope is a space-delimited list of the permissions
	// that the application requests. (See
	// https://developers.google.com/accounts/docs/OAuth2ServiceAccount )
	// 
	// The subEmail is the email address of the user for which the application is
	// requesting delegated access. The subEmail may be left empty if there is no such
	// email address.
	// 
	CkTask *G_SvcOauthAccessTokenAsync(const char *iss, const char *scope, const char *subEmail, int numSec, CkCert &cert);


	// The same as the G_SvcOauthAccessToken method, but with added flexibility for
	// more customization. The 1st three args of the G_SvcOauthAccessToken are replaced
	// with claimParams allowing for future expansion of name-value params. See the example
	// below.
	bool G_SvcOauthAccessToken2(CkHashtable &claimParams, int numSec, CkCert &cert, CkString &outStr);

	// The same as the G_SvcOauthAccessToken method, but with added flexibility for
	// more customization. The 1st three args of the G_SvcOauthAccessToken are replaced
	// with claimParams allowing for future expansion of name-value params. See the example
	// below.
	const char *g_SvcOauthAccessToken2(CkHashtable &claimParams, int numSec, CkCert &cert);
	// The same as the G_SvcOauthAccessToken method, but with added flexibility for
	// more customization. The 1st three args of the G_SvcOauthAccessToken are replaced
	// with claimParams allowing for future expansion of name-value params. See the example
	// below.
	CkTask *G_SvcOauthAccessToken2Async(CkHashtable &claimParams, int numSec, CkCert &cert);


	// Returns the current GMT (also known as UTC) date/time in a string that is
	// compliant with RFC 2616 format.
	bool GenTimeStamp(CkString &outStr);

	// Returns the current GMT (also known as UTC) date/time in a string that is
	// compliant with RFC 2616 format.
	const char *genTimeStamp(void);

	// Returns the Nth cache root (indexing begins at 0). Cache roots are set by
	// calling AddCacheRoot one or more times.
	bool GetCacheRoot(int index, CkString &outStr);

	// Returns the Nth cache root (indexing begins at 0). Cache roots are set by
	// calling AddCacheRoot one or more times.
	const char *getCacheRoot(int index);
	// Returns the Nth cache root (indexing begins at 0). Cache roots are set by
	// calling AddCacheRoot one or more times.
	const char *cacheRoot(int index);


	// Returns the cookies in XML format for a specific domain. Cookies are only
	// persisted if the SaveCookies property is set to true. If the CookieDir
	// property is set to the keyword "memory", then cookies are saved in-memory.
	bool GetCookieXml(const char *domain, CkString &outStr);

	// Returns the cookies in XML format for a specific domain. Cookies are only
	// persisted if the SaveCookies property is set to true. If the CookieDir
	// property is set to the keyword "memory", then cookies are saved in-memory.
	const char *getCookieXml(const char *domain);
	// Returns the cookies in XML format for a specific domain. Cookies are only
	// persisted if the SaveCookies property is set to true. If the CookieDir
	// property is set to the keyword "memory", then cookies are saved in-memory.
	const char *cookieXml(const char *domain);


	// Utility method for extracting the domain name from a full URL. For example, if
	// "http://www.chilkatsoft.com/default.asp" is the URL passed in, then
	// "www.chilkatsoft.com" is returned.
	bool GetDomain(const char *url, CkString &outStr);

	// Utility method for extracting the domain name from a full URL. For example, if
	// "http://www.chilkatsoft.com/default.asp" is the URL passed in, then
	// "www.chilkatsoft.com" is returned.
	const char *getDomain(const char *url);
	// Utility method for extracting the domain name from a full URL. For example, if
	// "http://www.chilkatsoft.com/default.asp" is the URL passed in, then
	// "www.chilkatsoft.com" is returned.
	const char *domain(const char *url);


	// Sends an HTTP HEAD request for a URL and returns a response object. (Note: HEAD
	// requests will never automatically follow redirects.)
	// The caller is responsible for deleting the object returned by this method.
	CkHttpResponse *GetHead(const char *url);

	// Sends an HTTP HEAD request for a URL and returns a response object. (Note: HEAD
	// requests will never automatically follow redirects.)
	CkTask *GetHeadAsync(const char *url);


	// Returns the value of a header field that has been pre-defined to be sent with
	// all HTTP GET requests issued by the QuickGet and QuickGetStr methods. By
	// default, this includes header fields such as Accept, AcceptCharset,
	// AcceptLanguage, Connection, UserAgent, etc.
	bool GetRequestHeader(const char *name, CkString &outStr);

	// Returns the value of a header field that has been pre-defined to be sent with
	// all HTTP GET requests issued by the QuickGet and QuickGetStr methods. By
	// default, this includes header fields such as Accept, AcceptCharset,
	// AcceptLanguage, Connection, UserAgent, etc.
	const char *getRequestHeader(const char *name);
	// Returns the value of a header field that has been pre-defined to be sent with
	// all HTTP GET requests issued by the QuickGet and QuickGetStr methods. By
	// default, this includes header fields such as Accept, AcceptCharset,
	// AcceptLanguage, Connection, UserAgent, etc.
	const char *requestHeader(const char *name);


	// Establishes an SSL/TLS connection with a web server for the purpose of
	// retrieving the server's SSL certificate (public-key only of course...). Nothing
	// is retrieved from the web server. This method simply makes a connection, gets
	// the certificate information, and closes the connection.
	// The caller is responsible for deleting the object returned by this method.
	CkCert *GetServerSslCert(const char *domain, int port);

	// Establishes an SSL/TLS connection with a web server for the purpose of
	// retrieving the server's SSL certificate (public-key only of course...). Nothing
	// is retrieved from the web server. This method simply makes a connection, gets
	// the certificate information, and closes the connection.
	CkTask *GetServerSslCertAsync(const char *domain, int port);


	// Returns the path part of a URL. The syntax of a URL is
	// _LT_scheme>://_LT_user>:_LT_password>@_LT_host>:_LT_port>/_LT_path>;_LT_params>?_
	// LT_query>#_LT_frag>. This method returns the "path" part.
	bool GetUrlPath(const char *url, CkString &outStr);

	// Returns the path part of a URL. The syntax of a URL is
	// _LT_scheme>://_LT_user>:_LT_password>@_LT_host>:_LT_port>/_LT_path>;_LT_params>?_
	// LT_query>#_LT_frag>. This method returns the "path" part.
	const char *getUrlPath(const char *url);
	// Returns the path part of a URL. The syntax of a URL is
	// _LT_scheme>://_LT_user>:_LT_password>@_LT_host>:_LT_port>/_LT_path>;_LT_params>?_
	// LT_query>#_LT_frag>. This method returns the "path" part.
	const char *urlPath(const char *url);


	// Returns true if the specified header field is defined such that it will be sent
	// with all GET requests issued by the QuickGet and QuickGetStr methods.
	bool HasRequestHeader(const char *name);


	// Returns true if the Http class has been unlocked. It is necessary to call
	// Http.UnlockComponent before calling any other methods. Passing any string to
	// UnlockComponent will automatically activate a 30-day trial period.
	bool IsUnlocked(void);


	// Provides information about what transpired in the last method called on this
	// object instance. For many methods, there is no information. However, for some
	// methods, details about what occurred can be obtained by getting the LastJsonData
	// right after the method call returns.
	// The caller is responsible for deleting the object returned by this method.
	CkJsonObject *LastJsonData(void);


	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTask &task);


	// Gets the server certificate at a domain:port and then sends an OCSP request to
	// the certificate's OCSP URL to determine if the certificate has been revoked.
	// Returns the OCSP status, which has one of the following values:
	//     -1: Unable to check. See the contents of the LastErrorText property for more
	//     informaiton.
	//     0: Good
	//     1: Revoked
	//     2: Unknown
	int OcspCheck(const char *domain, int port);

	// Gets the server certificate at a domain:port and then sends an OCSP request to
	// the certificate's OCSP URL to determine if the certificate has been revoked.
	// Returns the OCSP status, which has one of the following values:
	//     -1: Unable to check. See the contents of the LastErrorText property for more
	//     informaiton.
	//     0: Good
	//     1: Revoked
	//     2: Unknown
	CkTask *OcspCheckAsync(const char *domain, int port);


	// Parses an OCSP reply. Returns the following possible integer values:
	//     -1: The ocspReply does not contain a valid OCSP reply.
	//     0: Successful - Response has valid confirmations..
	//     1: Malformed request - Illegal confirmation request.
	//     2: Internal error - Internal error in issuer.
	//     3: Try later - Try again later.
	//     4: Not used - This value is never returned.
	//     5: Sig required - Must sign the request.
	//     6: Unauthorized - Request unauthorized.
	// 
	// The binaryOCSP reply is provided in ocspReply. The replyData is populated with data parsed
	// from ocspReply.
	// 
	int ParseOcspReply(CkBinData &ocspReply, CkJsonObject &replyData);


	// Sends an HTTP request to the url. The verb can be "POST", "PUT", "PATCH", etc.
	// The body of the HTTP request contains the bytes passed in byteData. The contentType is a
	// content type such as "image/gif", "application/pdf", etc. If md5 is true,
	// then a Content-MD5 header is added with the base64 MD5 hash of the byteData. Servers
	// aware of the Content-MD5 header will perform a message integrity check to ensure
	// that the data has not been corrupted. If gzip is true, the byteData is compressed
	// using the gzip algorithm. The HTTP request body will contain the GZIP compressed
	// data, and a "Content-Encoding: gzip" header is automatically added to indicate
	// that the request data needs to be ungzipped when received (at the server).
	// The caller is responsible for deleting the object returned by this method.
	CkHttpResponse *PBinary(const char *verb, const char *url, CkByteData &byteData, const char *contentType, bool md5, bool gzip);

	// Sends an HTTP request to the url. The verb can be "POST", "PUT", "PATCH", etc.
	// The body of the HTTP request contains the bytes passed in byteData. The contentType is a
	// content type such as "image/gif", "application/pdf", etc. If md5 is true,
	// then a Content-MD5 header is added with the base64 MD5 hash of the byteData. Servers
	// aware of the Content-MD5 header will perform a message integrity check to ensure
	// that the data has not been corrupted. If gzip is true, the byteData is compressed
	// using the gzip algorithm. The HTTP request body will contain the GZIP compressed
	// data, and a "Content-Encoding: gzip" header is automatically added to indicate
	// that the request data needs to be ungzipped when received (at the server).
	CkTask *PBinaryAsync(const char *verb, const char *url, CkByteData &byteData, const char *contentType, bool md5, bool gzip);


	// The same as PBinary, but the data to be uploaded is passed in data.
	// The caller is responsible for deleting the object returned by this method.
	CkHttpResponse *PBinaryBd(const char *verb, const char *url, CkBinData &data, const char *contentType, bool md5, bool gzip);

	// The same as PBinary, but the data to be uploaded is passed in data.
	CkTask *PBinaryBdAsync(const char *verb, const char *url, CkBinData &data, const char *contentType, bool md5, bool gzip);


	// Sends an HTTP/HTTPS request to the url. The verb can be "POST", "PUT", "PATCH",
	// etc. The url can begin with "http://" or "https://" depending if TLS is
	// desired. The body of the HTTP request is streamed directly from the localFilePath. The
	// contentType is a content type such as "image/gif", "application/pdf", "text/xml", etc.
	// If md5 is true, then a Content-MD5 header is added with the base64 MD5 hash
	// of the localFilePath. Servers aware of the Content-MD5 header will perform a message
	// integrity check to ensure that the data has not been corrupted. If gzip is
	// true, the localFilePath is compressed using the gzip algorithm. The HTTP request body
	// will contain the GZIP compressed data, and a "Content-Encoding: gzip" header is
	// automatically added to indicate that the request data needs to be ungzipped when
	// received (at the server).
	// The caller is responsible for deleting the object returned by this method.
	CkHttpResponse *PFile(const char *verb, const char *url, const char *localFilePath, const char *contentType, bool md5, bool gzip);

	// Sends an HTTP/HTTPS request to the url. The verb can be "POST", "PUT", "PATCH",
	// etc. The url can begin with "http://" or "https://" depending if TLS is
	// desired. The body of the HTTP request is streamed directly from the localFilePath. The
	// contentType is a content type such as "image/gif", "application/pdf", "text/xml", etc.
	// If md5 is true, then a Content-MD5 header is added with the base64 MD5 hash
	// of the localFilePath. Servers aware of the Content-MD5 header will perform a message
	// integrity check to ensure that the data has not been corrupted. If gzip is
	// true, the localFilePath is compressed using the gzip algorithm. The HTTP request body
	// will contain the GZIP compressed data, and a "Content-Encoding: gzip" header is
	// automatically added to indicate that the request data needs to be ungzipped when
	// received (at the server).
	CkTask *PFileAsync(const char *verb, const char *url, const char *localFilePath, const char *contentType, bool md5, bool gzip);


	// Sends an HTTP POST request to the url. The body of the HTTP request contains
	// the bytes passed in byteData. The contentType is a content type such as "image/gif",
	// "application/pdf", etc. If md5 is true, then a Content-MD5 header is added
	// with the base64 MD5 hash of the byteData. Servers aware of the Content-MD5 header
	// will perform a message integrity check to ensure that the data has not been
	// corrupted. If gzip is true, the byteData is compressed using the gzip algorithm.
	// The HTTP request body will contain the GZIP compressed data, and a
	// "Content-Encoding: gzip" header is automatically added to indicate that the
	// request data needs to be ungzipped when received (at the server).
	// 
	// Returns the text body of the HTTP response if the HTTP response has a success
	// status code. Otherwise the method is considered to have failed. If more details
	// of the HTTP response are required, call PBinary instead (which returns the HTTP
	// response object).
	// 
	// Note: The HTTP response code is available in the LastStatus property. Other
	// properties having information include LastResponseHeader, LastResponseBody,
	// LastModDate, LastContentType, etc.
	// 
	bool PostBinary(const char *url, CkByteData &byteData, const char *contentType, bool md5, bool gzip, CkString &outStr);

	// Sends an HTTP POST request to the url. The body of the HTTP request contains
	// the bytes passed in byteData. The contentType is a content type such as "image/gif",
	// "application/pdf", etc. If md5 is true, then a Content-MD5 header is added
	// with the base64 MD5 hash of the byteData. Servers aware of the Content-MD5 header
	// will perform a message integrity check to ensure that the data has not been
	// corrupted. If gzip is true, the byteData is compressed using the gzip algorithm.
	// The HTTP request body will contain the GZIP compressed data, and a
	// "Content-Encoding: gzip" header is automatically added to indicate that the
	// request data needs to be ungzipped when received (at the server).
	// 
	// Returns the text body of the HTTP response if the HTTP response has a success
	// status code. Otherwise the method is considered to have failed. If more details
	// of the HTTP response are required, call PBinary instead (which returns the HTTP
	// response object).
	// 
	// Note: The HTTP response code is available in the LastStatus property. Other
	// properties having information include LastResponseHeader, LastResponseBody,
	// LastModDate, LastContentType, etc.
	// 
	const char *postBinary(const char *url, CkByteData &byteData, const char *contentType, bool md5, bool gzip);
	// Sends an HTTP POST request to the url. The body of the HTTP request contains
	// the bytes passed in byteData. The contentType is a content type such as "image/gif",
	// "application/pdf", etc. If md5 is true, then a Content-MD5 header is added
	// with the base64 MD5 hash of the byteData. Servers aware of the Content-MD5 header
	// will perform a message integrity check to ensure that the data has not been
	// corrupted. If gzip is true, the byteData is compressed using the gzip algorithm.
	// The HTTP request body will contain the GZIP compressed data, and a
	// "Content-Encoding: gzip" header is automatically added to indicate that the
	// request data needs to be ungzipped when received (at the server).
	// 
	// Returns the text body of the HTTP response if the HTTP response has a success
	// status code. Otherwise the method is considered to have failed. If more details
	// of the HTTP response are required, call PBinary instead (which returns the HTTP
	// response object).
	// 
	// Note: The HTTP response code is available in the LastStatus property. Other
	// properties having information include LastResponseHeader, LastResponseBody,
	// LastModDate, LastContentType, etc.
	// 
	CkTask *PostBinaryAsync(const char *url, CkByteData &byteData, const char *contentType, bool md5, bool gzip);


	// A simplified way of sending a JSON POST and receiving the JSON response. The
	// HTTP response is returned in an HTTP response object. The content type of the
	// HTTP request is "application/json". To send a JSON POST using a different
	// content-type, call the PostJson2 method where the content type can be explicitly
	// specified.
	// The caller is responsible for deleting the object returned by this method.
	CkHttpResponse *PostJson(const char *url, const char *jsonText);

	// A simplified way of sending a JSON POST and receiving the JSON response. The
	// HTTP response is returned in an HTTP response object. The content type of the
	// HTTP request is "application/json". To send a JSON POST using a different
	// content-type, call the PostJson2 method where the content type can be explicitly
	// specified.
	CkTask *PostJsonAsync(const char *url, const char *jsonText);


	// The same as PostJson,except it allows for the content type to be explicitly
	// provided. The PostJson method automatically uses "application/jsonrequest". If
	// the application needs for the content type to be "application/json", or some
	// other content type, then PostJson2 provides the means.
	// The caller is responsible for deleting the object returned by this method.
	CkHttpResponse *PostJson2(const char *url, const char *contentType, const char *jsonText);

	// The same as PostJson,except it allows for the content type to be explicitly
	// provided. The PostJson method automatically uses "application/jsonrequest". If
	// the application needs for the content type to be "application/json", or some
	// other content type, then PostJson2 provides the means.
	CkTask *PostJson2Async(const char *url, const char *contentType, const char *jsonText);


	// The same as PostJson2,except a JSON object is passed in for the request body.
	// The caller is responsible for deleting the object returned by this method.
	CkHttpResponse *PostJson3(const char *url, const char *contentType, CkJsonObject &json);

	// The same as PostJson2,except a JSON object is passed in for the request body.
	CkTask *PostJson3Async(const char *url, const char *contentType, CkJsonObject &json);


	// Sends a simple URL encoded POST. The form parameters are sent in the body of the
	// HTTP request in x-www-form-urlencoded format. The content-type is
	// "application/x-www-form-urlencoded".
	// The caller is responsible for deleting the object returned by this method.
	CkHttpResponse *PostUrlEncoded(const char *url, CkHttpRequest &req);

	// Sends a simple URL encoded POST. The form parameters are sent in the body of the
	// HTTP request in x-www-form-urlencoded format. The content-type is
	// "application/x-www-form-urlencoded".
	CkTask *PostUrlEncodedAsync(const char *url, CkHttpRequest &req);


	// A simplified way of posting XML content to a web server. This method is good for
	// making SOAP calls using HTTP POST. The xmlCharset should match the character encoding
	// used in the xmlContent, which is typically "utf-8". The HTTP response is returned in
	// an HTTP response object.
	// 
	// Important: This method sends the POST with a "Content-Type" header value of
	// "application/xml". In rare cases, a server might require the Content-Type header
	// to be "text/xml". To use "text/xml" instead of the default "application/xml",
	// call SetRequestHeader("Content-Type","text/xml") prior to calling this method.
	// 
	// To use HTTPS simply pass an endpointUrl beginning with "https://" instead of "http://".
	// This applies to any Chilkat method where a URL is passed as an argument.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkHttpResponse *PostXml(const char *endpointUrl, const char *xmlContent, const char *xmlCharset);

	// A simplified way of posting XML content to a web server. This method is good for
	// making SOAP calls using HTTP POST. The xmlCharset should match the character encoding
	// used in the xmlContent, which is typically "utf-8". The HTTP response is returned in
	// an HTTP response object.
	// 
	// Important: This method sends the POST with a "Content-Type" header value of
	// "application/xml". In rare cases, a server might require the Content-Type header
	// to be "text/xml". To use "text/xml" instead of the default "application/xml",
	// call SetRequestHeader("Content-Type","text/xml") prior to calling this method.
	// 
	// To use HTTPS simply pass an endpointUrl beginning with "https://" instead of "http://".
	// This applies to any Chilkat method where a URL is passed as an argument.
	// 
	CkTask *PostXmlAsync(const char *endpointUrl, const char *xmlContent, const char *xmlCharset);


	// Sends an HTTP request to the url. The verb can be "POST", "PUT", "PATCH", etc.
	// The body of the HTTP request contains the text passed in textData. The contentType is a
	// content type such as "text/xml", "application/json", etc. If md5 is true,
	// then a Content-MD5 header is added with the base64 MD5 hash of the textData. Servers
	// aware of the Content-MD5 header will perform a message integrity check to ensure
	// that the data has not been corrupted. If gzip is true, the contentType is compressed
	// using the gzip algorithm. The HTTP request body will contain the GZIP compressed
	// data, and a "Content-Encoding: gzip" header is automatically added to indicate
	// that the request data needs to be ungzipped when received (at the server).
	// The caller is responsible for deleting the object returned by this method.
	CkHttpResponse *PText(const char *verb, const char *url, const char *textData, const char *charset, const char *contentType, bool md5, bool gzip);

	// Sends an HTTP request to the url. The verb can be "POST", "PUT", "PATCH", etc.
	// The body of the HTTP request contains the text passed in textData. The contentType is a
	// content type such as "text/xml", "application/json", etc. If md5 is true,
	// then a Content-MD5 header is added with the base64 MD5 hash of the textData. Servers
	// aware of the Content-MD5 header will perform a message integrity check to ensure
	// that the data has not been corrupted. If gzip is true, the contentType is compressed
	// using the gzip algorithm. The HTTP request body will contain the GZIP compressed
	// data, and a "Content-Encoding: gzip" header is automatically added to indicate
	// that the request data needs to be ungzipped when received (at the server).
	CkTask *PTextAsync(const char *verb, const char *url, const char *textData, const char *charset, const char *contentType, bool md5, bool gzip);


	// The same as PText, but the data to be uploaded is passed in textData.
	// The caller is responsible for deleting the object returned by this method.
	CkHttpResponse *PTextSb(const char *verb, const char *url, CkStringBuilder &textData, const char *charset, const char *contentType, bool md5, bool gzip);

	// The same as PText, but the data to be uploaded is passed in textData.
	CkTask *PTextSbAsync(const char *verb, const char *url, CkStringBuilder &textData, const char *charset, const char *contentType, bool md5, bool gzip);


	// Sends an HTTP PUT request to the url. The body of the HTTP request is byteData. The
	// contentType is a content type such as "image/gif", "application/pdf", etc. If md5 is
	// true, then a Content-MD5 header is added with the base64 MD5 hash of the byteData.
	// Servers aware of the Content-MD5 header will perform a message integrity check
	// to ensure that the data has not been corrupted. If gzip is true, the byteData is
	// compressed using the gzip algorithm. The HTTP request body will contain the GZIP
	// compressed data, and a "Content-Encoding: gzip" header is automatically added to
	// indicate that the request data needs to be ungzipped when received (at the
	// server).
	// 
	// Returns the text body of the HTTP response if the HTTP response has a success
	// status code. Otherwise the method is considered to have failed. If more details
	// of the HTTP response are required, call PBinary instead (which returns the HTTP
	// response object).
	// 
	bool PutBinary(const char *url, CkByteData &byteData, const char *contentType, bool md5, bool gzip, CkString &outStr);

	// Sends an HTTP PUT request to the url. The body of the HTTP request is byteData. The
	// contentType is a content type such as "image/gif", "application/pdf", etc. If md5 is
	// true, then a Content-MD5 header is added with the base64 MD5 hash of the byteData.
	// Servers aware of the Content-MD5 header will perform a message integrity check
	// to ensure that the data has not been corrupted. If gzip is true, the byteData is
	// compressed using the gzip algorithm. The HTTP request body will contain the GZIP
	// compressed data, and a "Content-Encoding: gzip" header is automatically added to
	// indicate that the request data needs to be ungzipped when received (at the
	// server).
	// 
	// Returns the text body of the HTTP response if the HTTP response has a success
	// status code. Otherwise the method is considered to have failed. If more details
	// of the HTTP response are required, call PBinary instead (which returns the HTTP
	// response object).
	// 
	const char *putBinary(const char *url, CkByteData &byteData, const char *contentType, bool md5, bool gzip);
	// Sends an HTTP PUT request to the url. The body of the HTTP request is byteData. The
	// contentType is a content type such as "image/gif", "application/pdf", etc. If md5 is
	// true, then a Content-MD5 header is added with the base64 MD5 hash of the byteData.
	// Servers aware of the Content-MD5 header will perform a message integrity check
	// to ensure that the data has not been corrupted. If gzip is true, the byteData is
	// compressed using the gzip algorithm. The HTTP request body will contain the GZIP
	// compressed data, and a "Content-Encoding: gzip" header is automatically added to
	// indicate that the request data needs to be ungzipped when received (at the
	// server).
	// 
	// Returns the text body of the HTTP response if the HTTP response has a success
	// status code. Otherwise the method is considered to have failed. If more details
	// of the HTTP response are required, call PBinary instead (which returns the HTTP
	// response object).
	// 
	CkTask *PutBinaryAsync(const char *url, CkByteData &byteData, const char *contentType, bool md5, bool gzip);


	// Sends an HTTP PUT request to the url. The body of the HTTP request is textData. The
	// charset should be set to a charset name such as "iso-8859-1", "windows-1252",
	// "Shift_JIS", "utf-8", etc. The string "ansi" may also be used as a charset name.
	// The contentType is a content type such as "text/plain", "text/xml", etc. If md5 is
	// true, then a Content-MD5 header is added with the base64 MD5 hash of the textData.
	// Servers aware of the Content-MD5 header will perform a message integrity check
	// to ensure that the data has not been corrupted. If gzip is true, the textData is
	// compressed using the gzip algorithm. The HTTP request body will contain the GZIP
	// compressed data, and a "Content-Encoding: gzip" header is automatically added to
	// indicate that the request data needs to be ungzipped when received (at the
	// server).
	// 
	// Returns the text body of the HTTP response if the HTTP response has a success
	// status code. Otherwise the method is considered to have failed. If more details
	// of the HTTP response are required, call PText instead (which returns the HTTP
	// response object).
	// 
	bool PutText(const char *url, const char *textData, const char *charset, const char *contentType, bool md5, bool gzip, CkString &outStr);

	// Sends an HTTP PUT request to the url. The body of the HTTP request is textData. The
	// charset should be set to a charset name such as "iso-8859-1", "windows-1252",
	// "Shift_JIS", "utf-8", etc. The string "ansi" may also be used as a charset name.
	// The contentType is a content type such as "text/plain", "text/xml", etc. If md5 is
	// true, then a Content-MD5 header is added with the base64 MD5 hash of the textData.
	// Servers aware of the Content-MD5 header will perform a message integrity check
	// to ensure that the data has not been corrupted. If gzip is true, the textData is
	// compressed using the gzip algorithm. The HTTP request body will contain the GZIP
	// compressed data, and a "Content-Encoding: gzip" header is automatically added to
	// indicate that the request data needs to be ungzipped when received (at the
	// server).
	// 
	// Returns the text body of the HTTP response if the HTTP response has a success
	// status code. Otherwise the method is considered to have failed. If more details
	// of the HTTP response are required, call PText instead (which returns the HTTP
	// response object).
	// 
	const char *putText(const char *url, const char *textData, const char *charset, const char *contentType, bool md5, bool gzip);
	// Sends an HTTP PUT request to the url. The body of the HTTP request is textData. The
	// charset should be set to a charset name such as "iso-8859-1", "windows-1252",
	// "Shift_JIS", "utf-8", etc. The string "ansi" may also be used as a charset name.
	// The contentType is a content type such as "text/plain", "text/xml", etc. If md5 is
	// true, then a Content-MD5 header is added with the base64 MD5 hash of the textData.
	// Servers aware of the Content-MD5 header will perform a message integrity check
	// to ensure that the data has not been corrupted. If gzip is true, the textData is
	// compressed using the gzip algorithm. The HTTP request body will contain the GZIP
	// compressed data, and a "Content-Encoding: gzip" header is automatically added to
	// indicate that the request data needs to be ungzipped when received (at the
	// server).
	// 
	// Returns the text body of the HTTP response if the HTTP response has a success
	// status code. Otherwise the method is considered to have failed. If more details
	// of the HTTP response are required, call PText instead (which returns the HTTP
	// response object).
	// 
	CkTask *PutTextAsync(const char *url, const char *textData, const char *charset, const char *contentType, bool md5, bool gzip);


	// Same as QuickGetStr, but uses the HTTP DELETE method instead of the GET method.
	// 
	// Note: The HTTP response code is available in the LastStatus property. Other
	// properties having information include LastResponseHeader, LastResponseBody,
	// LastModDate, LastContentType, etc.
	// 
	bool QuickDeleteStr(const char *url, CkString &outStr);

	// Same as QuickGetStr, but uses the HTTP DELETE method instead of the GET method.
	// 
	// Note: The HTTP response code is available in the LastStatus property. Other
	// properties having information include LastResponseHeader, LastResponseBody,
	// LastModDate, LastContentType, etc.
	// 
	const char *quickDeleteStr(const char *url);
	// Same as QuickGetStr, but uses the HTTP DELETE method instead of the GET method.
	// 
	// Note: The HTTP response code is available in the LastStatus property. Other
	// properties having information include LastResponseHeader, LastResponseBody,
	// LastModDate, LastContentType, etc.
	// 
	CkTask *QuickDeleteStrAsync(const char *url);


	// Sends an HTTP GET request for a URL and returns the binary response body. The
	// URL may contain query parameters. If the SendCookies property is true,
	// matching cookies previously persisted to the CookiesDir are automatically
	// included in the request. If the FetchFromCache property is true, the resource
	// may be fetched directly from cache.
	// 
	// Note: The HTTP response code is available in the LastStatus property. Other
	// properties having information include LastResponseHeader, LastResponseBody,
	// LastModDate, LastContentType, etc.
	// 
	// A response code >= 400 is considered a failure. If the error response was
	// textual in nature, then it will also be available in the LastResponseBody
	// property.
	// 
	bool QuickGet(const char *url, CkByteData &outData);

	// Sends an HTTP GET request for a URL and returns the binary response body. The
	// URL may contain query parameters. If the SendCookies property is true,
	// matching cookies previously persisted to the CookiesDir are automatically
	// included in the request. If the FetchFromCache property is true, the resource
	// may be fetched directly from cache.
	// 
	// Note: The HTTP response code is available in the LastStatus property. Other
	// properties having information include LastResponseHeader, LastResponseBody,
	// LastModDate, LastContentType, etc.
	// 
	// A response code >= 400 is considered a failure. If the error response was
	// textual in nature, then it will also be available in the LastResponseBody
	// property.
	// 
	CkTask *QuickGetAsync(const char *url);


	// The same as QuickGet, but returns the content in a Chilkat BinData object. The
	// existing content of binData, if any, is cleared and replaced with the downloaded
	// content.
	// 
	// A response code >= 400 is considered a failure and the method will return
	// false. However, the error response will still be returned in binData. If the
	// error response was textual in nature, then it will also be available in the
	// LastResponseBody property.
	// 
	bool QuickGetBd(const char *url, CkBinData &binData);

	// The same as QuickGet, but returns the content in a Chilkat BinData object. The
	// existing content of binData, if any, is cleared and replaced with the downloaded
	// content.
	// 
	// A response code >= 400 is considered a failure and the method will return
	// false. However, the error response will still be returned in binData. If the
	// error response was textual in nature, then it will also be available in the
	// LastResponseBody property.
	// 
	CkTask *QuickGetBdAsync(const char *url, CkBinData &binData);


	// Sends an HTTP GET request for a URL and returns the response object. If the
	// SendCookies property is true, matching cookies previously persisted to the
	// CookiesDir are automatically included in the request. If the FetchFromCache
	// property is true, the page could be fetched directly from cache.
	// The caller is responsible for deleting the object returned by this method.
	CkHttpResponse *QuickGetObj(const char *url);

	// Sends an HTTP GET request for a URL and returns the response object. If the
	// SendCookies property is true, matching cookies previously persisted to the
	// CookiesDir are automatically included in the request. If the FetchFromCache
	// property is true, the page could be fetched directly from cache.
	CkTask *QuickGetObjAsync(const char *url);


	// The same as QuickGetStr, but returns the content in a Chilkat StringBuilder
	// object. The existing content of sbContent, if any, is cleared and replaced with the
	// downloaded content.
	// 
	// If the response status code is >= 400, then this method returns false, but the
	// body of the HTTP response is still returned in sbContent. This allows for the
	// application to examine the response body for cases where an error is returned,
	// but the expected content is not received.
	// 
	bool QuickGetSb(const char *url, CkStringBuilder &sbContent);

	// The same as QuickGetStr, but returns the content in a Chilkat StringBuilder
	// object. The existing content of sbContent, if any, is cleared and replaced with the
	// downloaded content.
	// 
	// If the response status code is >= 400, then this method returns false, but the
	// body of the HTTP response is still returned in sbContent. This allows for the
	// application to examine the response body for cases where an error is returned,
	// but the expected content is not received.
	// 
	CkTask *QuickGetSbAsync(const char *url, CkStringBuilder &sbContent);


	// Sends an HTTP GET request for a URL and returns the response body as a string.
	// The URL may contain query parameters. If the SendCookies property is true,
	// matching cookies previously persisted to the CookiesDir are automatically
	// included in the request. If the FetchFromCache property is true, the page
	// could be fetched directly from cache. If the HTTP request fails, a _NULL_ value
	// is returned and error information can be found in the LastErrorText,
	// LastErrorXml, or LastErrorHtml properties.
	// 
	// Note: The HTTP response code is available in the LastStatus property. Other
	// properties having information include LastResponseHeader, LastResponseBody,
	// LastModDate, LastContentType, etc.
	// 
	bool QuickGetStr(const char *url, CkString &outStr);

	// Sends an HTTP GET request for a URL and returns the response body as a string.
	// The URL may contain query parameters. If the SendCookies property is true,
	// matching cookies previously persisted to the CookiesDir are automatically
	// included in the request. If the FetchFromCache property is true, the page
	// could be fetched directly from cache. If the HTTP request fails, a _NULL_ value
	// is returned and error information can be found in the LastErrorText,
	// LastErrorXml, or LastErrorHtml properties.
	// 
	// Note: The HTTP response code is available in the LastStatus property. Other
	// properties having information include LastResponseHeader, LastResponseBody,
	// LastModDate, LastContentType, etc.
	// 
	const char *quickGetStr(const char *url);
	// Sends an HTTP GET request for a URL and returns the response body as a string.
	// The URL may contain query parameters. If the SendCookies property is true,
	// matching cookies previously persisted to the CookiesDir are automatically
	// included in the request. If the FetchFromCache property is true, the page
	// could be fetched directly from cache. If the HTTP request fails, a _NULL_ value
	// is returned and error information can be found in the LastErrorText,
	// LastErrorXml, or LastErrorHtml properties.
	// 
	// Note: The HTTP response code is available in the LastStatus property. Other
	// properties having information include LastResponseHeader, LastResponseBody,
	// LastModDate, LastContentType, etc.
	// 
	CkTask *QuickGetStrAsync(const char *url);


	// Same as QuickGetStr, but uses the HTTP PUT method instead of the GET method.
	// 
	// Note: The HTTP response code is available in the LastStatus property. Other
	// properties having information include LastResponseHeader, LastResponseBody,
	// LastModDate, LastContentType, etc.
	// 
	bool QuickPutStr(const char *url, CkString &outStr);

	// Same as QuickGetStr, but uses the HTTP PUT method instead of the GET method.
	// 
	// Note: The HTTP response code is available in the LastStatus property. Other
	// properties having information include LastResponseHeader, LastResponseBody,
	// LastModDate, LastContentType, etc.
	// 
	const char *quickPutStr(const char *url);
	// Same as QuickGetStr, but uses the HTTP PUT method instead of the GET method.
	// 
	// Note: The HTTP response code is available in the LastStatus property. Other
	// properties having information include LastResponseHeader, LastResponseBody,
	// LastModDate, LastContentType, etc.
	// 
	CkTask *QuickPutStrAsync(const char *url);


	// Sends an HTTP request with no body for a URL and returns the response object.
	// The verb is typically "GET" or "DELETE".
	// 
	// If the SendCookies property is true, matching cookies previously persisted to
	// the CookiesDir are automatically included in the request. If the FetchFromCache
	// property is true, the page could be fetched directly from cache.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkHttpResponse *QuickRequest(const char *verb, const char *url);

	// Sends an HTTP request with no body for a URL and returns the response object.
	// The verb is typically "GET" or "DELETE".
	// 
	// If the SendCookies property is true, matching cookies previously persisted to
	// the CookiesDir are automatically included in the request. If the FetchFromCache
	// property is true, the page could be fetched directly from cache.
	// 
	CkTask *QuickRequestAsync(const char *verb, const char *url);


	// This is the same as the QuickRequest method, except query params can be provided
	// as non-URL-encoded name/value members in the json.
	// The caller is responsible for deleting the object returned by this method.
	CkHttpResponse *QuickRequestParams(const char *verb, const char *url, CkJsonObject &json);

	// This is the same as the QuickRequest method, except query params can be provided
	// as non-URL-encoded name/value members in the json.
	CkTask *QuickRequestParamsAsync(const char *verb, const char *url, CkJsonObject &json);


	// Removes a header from the internal list of custom header field name/value pairs
	// to be automatically added when HTTP requests are sent via methods that do not
	// use the HTTP request object. (The SetRequestHeader method is called to add
	// custom header fields.)
	void RemoveRequestHeader(const char *name);


	// Same as QuickGet, but does not send the HTTP GET. Instead, it builds the HTTP
	// request that would've been sent and returns it.
	bool RenderGet(const char *url, CkString &outStr);

	// Same as QuickGet, but does not send the HTTP GET. Instead, it builds the HTTP
	// request that would've been sent and returns it.
	const char *renderGet(const char *url);

	// Same as the Download method, except a failed download may be resumed. The targetFilename
	// is automatically checked and if it exists, the download will resume at the point
	// where it previously failed. ResumeDownload may be called any number of times
	// until the full download is complete.
	bool ResumeDownload(const char *url, const char *targetFilename);

	// Same as the Download method, except a failed download may be resumed. The targetFilename
	// is automatically checked and if it exists, the download will resume at the point
	// where it previously failed. ResumeDownload may be called any number of times
	// until the full download is complete.
	CkTask *ResumeDownloadAsync(const char *url, const char *targetFilename);


	// Same as the DownloadBd method, except a failed download may be resumed. The
	// download will resume at a point based on the number of bytes already contained
	// in binData. ResumeDownloadBd may be called any number of times until the full
	// download is complete.
	// 
	// The incoming data is appended to binData.
	// 
	bool ResumeDownloadBd(const char *url, CkBinData &binData);

	// Same as the DownloadBd method, except a failed download may be resumed. The
	// download will resume at a point based on the number of bytes already contained
	// in binData. ResumeDownloadBd may be called any number of times until the full
	// download is complete.
	// 
	// The incoming data is appended to binData.
	// 
	CkTask *ResumeDownloadBdAsync(const char *url, CkBinData &binData);


	// Creates a new Amazon S3 bucket.
	// 
	// Note: x-amz-* headers, including metadata, can be added to any S3 request by
	// adding each header with a call to SetRequestHeader. This applies to all S3
	// methods, even if not explicitly stated.
	// 
	bool S3_CreateBucket(const char *bucketPath);

	// Creates a new Amazon S3 bucket.
	// 
	// Note: x-amz-* headers, including metadata, can be added to any S3 request by
	// adding each header with a call to SetRequestHeader. This applies to all S3
	// methods, even if not explicitly stated.
	// 
	CkTask *S3_CreateBucketAsync(const char *bucketPath);


	// Deletes an Amazon S3 bucket.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	bool S3_DeleteBucket(const char *bucketPath);

	// Deletes an Amazon S3 bucket.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	CkTask *S3_DeleteBucketAsync(const char *bucketPath);


	// Deletes multiple objects from a bucket using a single HTTP request. The bucketName
	// contains the names (also known as "keys") of the objects to be deleted. To
	// delete a specific version of an object, append a versionId attribute to the
	// object name. For example: "SampleDocument.txt;
	// VersionId="OYcLXagmS.WaD..oyH4KRguB95_YhLs7""
	// The caller is responsible for deleting the object returned by this method.
	CkHttpResponse *S3_DeleteMultipleObjects(const char *bucketName, CkStringArray &objectNames);

	// Deletes multiple objects from a bucket using a single HTTP request. The bucketName
	// contains the names (also known as "keys") of the objects to be deleted. To
	// delete a specific version of an object, append a versionId attribute to the
	// object name. For example: "SampleDocument.txt;
	// VersionId="OYcLXagmS.WaD..oyH4KRguB95_YhLs7""
	CkTask *S3_DeleteMultipleObjectsAsync(const char *bucketName, CkStringArray &objectNames);


	// Deletes a remote file (object) on the Amazon S3 service.
	bool S3_DeleteObject(const char *bucketPath, const char *objectName);

	// Deletes a remote file (object) on the Amazon S3 service.
	CkTask *S3_DeleteObjectAsync(const char *bucketPath, const char *objectName);


	// The same as DownloadFile, except the downloaded file data is appended to bd.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	bool S3_DownloadBd(const char *bucketPath, const char *objectName, CkBinData &bd);

	// The same as DownloadFile, except the downloaded file data is appended to bd.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	CkTask *S3_DownloadBdAsync(const char *bucketPath, const char *objectName, CkBinData &bd);


	// The same as DownloadFile, except the file data is returned directly in-memory
	// instead of being written to a local file.
	bool S3_DownloadBytes(const char *bucketPath, const char *objectName, CkByteData &outBytes);

	// The same as DownloadFile, except the file data is returned directly in-memory
	// instead of being written to a local file.
	CkTask *S3_DownloadBytesAsync(const char *bucketPath, const char *objectName);


	// Downloads a file from the Amazon S3 service.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	bool S3_DownloadFile(const char *bucketPath, const char *objectName, const char *localFilePath);

	// Downloads a file from the Amazon S3 service.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	CkTask *S3_DownloadFileAsync(const char *bucketPath, const char *objectName, const char *localFilePath);


	// Downloads a text file (object) from the Amazon S3 service directly into a string
	// variable. The charset specifies the character encoding, such as "utf-8", of the
	// remote text object.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	bool S3_DownloadString(const char *bucketPath, const char *objectName, const char *charset, CkString &outStr);

	// Downloads a text file (object) from the Amazon S3 service directly into a string
	// variable. The charset specifies the character encoding, such as "utf-8", of the
	// remote text object.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	const char *s3_DownloadString(const char *bucketPath, const char *objectName, const char *charset);
	// Downloads a text file (object) from the Amazon S3 service directly into a string
	// variable. The charset specifies the character encoding, such as "utf-8", of the
	// remote text object.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	CkTask *S3_DownloadStringAsync(const char *bucketPath, const char *objectName, const char *charset);


	// Determines if a remote object (file) exists. Returns 1 if the file exists, 0 if
	// it does not exist, -1 if there was a failure in checking, or 2 if using in
	// asynchronous mode to indicate that the background task was successfully started.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	int S3_FileExists(const char *bucketPath, const char *objectName);

	// Determines if a remote object (file) exists. Returns 1 if the file exists, 0 if
	// it does not exist, -1 if there was a failure in checking, or 2 if using in
	// asynchronous mode to indicate that the background task was successfully started.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	CkTask *S3_FileExistsAsync(const char *bucketPath, const char *objectName);


	// Generates a temporary pre-signed URL for Amazon S3 using AWS Signature V2. (Call
	// S3_GenerateUrlV4 to generate AWS Signature V4 pre-signed URLs.) Requires that
	// the AwsSecretKey and AwsAccessKey be set to valid values prior to calling this
	// method.
	// 
	// Note: This method can only generate URLs that are for HTTP GET requests (i.e.
	// URLs you can paste into a browser address bar). This method does not generate
	// URLs for POST, PUT, DELETE, etc.
	// 
	bool S3_GenerateUrl(const char *bucket, const char *path, CkDateTime &expire, CkString &outStr);

	// Generates a temporary pre-signed URL for Amazon S3 using AWS Signature V2. (Call
	// S3_GenerateUrlV4 to generate AWS Signature V4 pre-signed URLs.) Requires that
	// the AwsSecretKey and AwsAccessKey be set to valid values prior to calling this
	// method.
	// 
	// Note: This method can only generate URLs that are for HTTP GET requests (i.e.
	// URLs you can paste into a browser address bar). This method does not generate
	// URLs for POST, PUT, DELETE, etc.
	// 
	const char *s3_GenerateUrl(const char *bucket, const char *path, CkDateTime &expire);

	// Note: The S3_GenPresignedUrl method is new in Chilkat v9.5.0.83 and can create
	// S3 pre-signed URLs for GET, POST, PUT, DELETE, or any other HTTP verb.
	// 
	// Generates a temporary pre-signed URL for Amazon S3 using AWS Signature V4. (Call
	// S3_GenerateUrl to generate AWS Signature V2 pre-signed URLs.) Requires that the
	// AwsSecretKey, AwsAccessKey, and AwsRegion properties be set to valid values
	// prior to calling this method. Also requires the AwsEndpoint property to be set
	// if the endpoint is different than "s3.amazonaws.com".
	// 
	// The URL that is generated has this format:
	// https:////?X-Amz-Algorithm=AWS4-HMAC-SHA256
	// &X-Amz-Credential=////aws4_request
	// &X-Amz-Date=&X-Amz-Expires=&X-Amz-SignedHeaders=host
	// &X-Amz-Signature=
	// 
	// The awsService is a string naming the AWS service, such as "s3".   If useHttps is true, then the URL begins with "https://", otherwise it begins with "http://".
	// 
	// Note: This method can only generate URLs that are for HTTP GET requests (i.e. URLs you can paste into a browser address bar).  This method does not generate URLs for POST, PUT, DELETE, etc.
	// 
	bool S3_GenerateUrlV4(bool useHttps, const char *bucketName, const char *path, int numSecondsValid, const char *awsService, CkString &outStr);

	// Note: The S3_GenPresignedUrl method is new in Chilkat v9.5.0.83 and can create
	// S3 pre-signed URLs for GET, POST, PUT, DELETE, or any other HTTP verb.
	// 
	// Generates a temporary pre-signed URL for Amazon S3 using AWS Signature V4. (Call
	// S3_GenerateUrl to generate AWS Signature V2 pre-signed URLs.) Requires that the
	// AwsSecretKey, AwsAccessKey, and AwsRegion properties be set to valid values
	// prior to calling this method. Also requires the AwsEndpoint property to be set
	// if the endpoint is different than "s3.amazonaws.com".
	// 
	// The URL that is generated has this format:
	// https:////?X-Amz-Algorithm=AWS4-HMAC-SHA256
	// &X-Amz-Credential=////aws4_request
	// &X-Amz-Date=&X-Amz-Expires=&X-Amz-SignedHeaders=host
	// &X-Amz-Signature=
	// 
	// The awsService is a string naming the AWS service, such as "s3".   If useHttps is true, then the URL begins with "https://", otherwise it begins with "http://".
	// 
	// Note: This method can only generate URLs that are for HTTP GET requests (i.e. URLs you can paste into a browser address bar).  This method does not generate URLs for POST, PUT, DELETE, etc.
	// 
	const char *s3_GenerateUrlV4(bool useHttps, const char *bucketName, const char *path, int numSecondsValid, const char *awsService);

	// Generates a temporary pre-signed URL for Amazon S3 using AWS Signature V4.
	// Requires that the AwsSecretKey, AwsAccessKey, and AwsRegion properties be set to
	// valid values prior to calling this method. Also requires the AwsEndpoint
	// property to be set if the endpoint is different than "s3.amazonaws.com".
	// 
	// The URL that is generated has this format:
	// https:////?X-Amz-Algorithm=AWS4-HMAC-SHA256
	// &X-Amz-Credential=////aws4_request
	// &X-Amz-Date=&X-Amz-Expires=&X-Amz-SignedHeaders=host
	// &X-Amz-Signature=
	// 
	// The httpVerb is the HTTP verb such as "GET", "PUT", "POST", "DELETE", etc.  The awsService is a string naming the AWS service, such as "s3" or "s3-accelerate".   If useHttps is true, then the URL begins with "https://", otherwise it begins with "http://".
	// 
	bool S3_GenPresignedUrl(const char *httpVerb, bool useHttps, const char *bucketName, const char *path, int numSecondsValid, const char *awsService, CkString &outStr);

	// Generates a temporary pre-signed URL for Amazon S3 using AWS Signature V4.
	// Requires that the AwsSecretKey, AwsAccessKey, and AwsRegion properties be set to
	// valid values prior to calling this method. Also requires the AwsEndpoint
	// property to be set if the endpoint is different than "s3.amazonaws.com".
	// 
	// The URL that is generated has this format:
	// https:////?X-Amz-Algorithm=AWS4-HMAC-SHA256
	// &X-Amz-Credential=////aws4_request
	// &X-Amz-Date=&X-Amz-Expires=&X-Amz-SignedHeaders=host
	// &X-Amz-Signature=
	// 
	// The httpVerb is the HTTP verb such as "GET", "PUT", "POST", "DELETE", etc.  The awsService is a string naming the AWS service, such as "s3" or "s3-accelerate".   If useHttps is true, then the URL begins with "https://", otherwise it begins with "http://".
	// 
	const char *s3_GenPresignedUrl(const char *httpVerb, bool useHttps, const char *bucketName, const char *path, int numSecondsValid, const char *awsService);

	// Retrieves the XML listing of the objects contained within an Amazon S3 bucket.
	// (This is like a directory listing, but in XML format.)
	// 
	// The bucketPath name may be qualified with URL-encoded params. For example, to list the
	// objects in a bucket named "ChilkatABC" with max-keys = 2000 and marker = "xyz",
	// call S3_ListBucketObject passing the following string for bucketPath:
	// "ChilkatABC?max-keys=2000&marker=xyz"
	// 
	// The S3_ListBucketObjects method recognized all params listed in the AWS
	// documentation for listing objects in a bucket: delimiter, marker, max-keys, and
	// prefix. See Amazon's AWS online documentation for more information.
	// 
	bool S3_ListBucketObjects(const char *bucketPath, CkString &outStr);

	// Retrieves the XML listing of the objects contained within an Amazon S3 bucket.
	// (This is like a directory listing, but in XML format.)
	// 
	// The bucketPath name may be qualified with URL-encoded params. For example, to list the
	// objects in a bucket named "ChilkatABC" with max-keys = 2000 and marker = "xyz",
	// call S3_ListBucketObject passing the following string for bucketPath:
	// "ChilkatABC?max-keys=2000&marker=xyz"
	// 
	// The S3_ListBucketObjects method recognized all params listed in the AWS
	// documentation for listing objects in a bucket: delimiter, marker, max-keys, and
	// prefix. See Amazon's AWS online documentation for more information.
	// 
	const char *s3_ListBucketObjects(const char *bucketPath);
	// Retrieves the XML listing of the objects contained within an Amazon S3 bucket.
	// (This is like a directory listing, but in XML format.)
	// 
	// The bucketPath name may be qualified with URL-encoded params. For example, to list the
	// objects in a bucket named "ChilkatABC" with max-keys = 2000 and marker = "xyz",
	// call S3_ListBucketObject passing the following string for bucketPath:
	// "ChilkatABC?max-keys=2000&marker=xyz"
	// 
	// The S3_ListBucketObjects method recognized all params listed in the AWS
	// documentation for listing objects in a bucket: delimiter, marker, max-keys, and
	// prefix. See Amazon's AWS online documentation for more information.
	// 
	CkTask *S3_ListBucketObjectsAsync(const char *bucketPath);


	// Retrieves the XML listing of the buckets for an Amazon S3 account.
	bool S3_ListBuckets(CkString &outStr);

	// Retrieves the XML listing of the buckets for an Amazon S3 account.
	const char *s3_ListBuckets(void);
	// Retrieves the XML listing of the buckets for an Amazon S3 account.
	CkTask *S3_ListBucketsAsync(void);


	// The same as S3_UploadFile, except the contents of the file come from bd
	// instead of a local file.
	// 
	// Note: x-amz-* headers, including metadata, can be added to any S3 request by
	// adding each header with a call to SetRequestHeader. This applies to all S3
	// methods, even if not explicitly stated.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	bool S3_UploadBd(CkBinData &bd, const char *contentType, const char *bucketPath, const char *objectName);

	// The same as S3_UploadFile, except the contents of the file come from bd
	// instead of a local file.
	// 
	// Note: x-amz-* headers, including metadata, can be added to any S3 request by
	// adding each header with a call to SetRequestHeader. This applies to all S3
	// methods, even if not explicitly stated.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	CkTask *S3_UploadBdAsync(CkBinData &bd, const char *contentType, const char *bucketPath, const char *objectName);


	// The same as S3_UploadFile, except the contents of the file come from contentBytes
	// instead of a local file.
	// 
	// Note: x-amz-* headers, including metadata, can be added to any S3 request by
	// adding each header with a call to SetRequestHeader. This applies to all S3
	// methods, even if not explicitly stated.
	// 
	bool S3_UploadBytes(CkByteData &contentBytes, const char *contentType, const char *bucketPath, const char *objectName);

	// The same as S3_UploadFile, except the contents of the file come from contentBytes
	// instead of a local file.
	// 
	// Note: x-amz-* headers, including metadata, can be added to any S3 request by
	// adding each header with a call to SetRequestHeader. This applies to all S3
	// methods, even if not explicitly stated.
	// 
	CkTask *S3_UploadBytesAsync(CkByteData &contentBytes, const char *contentType, const char *bucketPath, const char *objectName);


	// Uploads a file to the Amazon S3 service.
	// 
	// Note: x-amz-* headers, including metadata, can be added to any S3 request by
	// adding each header with a call to SetRequestHeader. This applies to all S3
	// methods, even if not explicitly stated.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	bool S3_UploadFile(const char *localFilePath, const char *contentType, const char *bucketPath, const char *objectName);

	// Uploads a file to the Amazon S3 service.
	// 
	// Note: x-amz-* headers, including metadata, can be added to any S3 request by
	// adding each header with a call to SetRequestHeader. This applies to all S3
	// methods, even if not explicitly stated.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	CkTask *S3_UploadFileAsync(const char *localFilePath, const char *contentType, const char *bucketPath, const char *objectName);


	// Uploads an in-memory string to the Amazon S3 service. This is the same as
	// UploadFile, except that the file contents are from an in-memory string instead
	// of a local file. Internal to this method, the objectContent is converted to the character
	// encoding specified by charset prior to uploading.
	// 
	// Note: x-amz-* headers, including metadata, can be added to any S3 request by
	// adding each header with a call to SetRequestHeader. This applies to all S3
	// methods, even if not explicitly stated.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	bool S3_UploadString(const char *objectContent, const char *charset, const char *contentType, const char *bucketPath, const char *objectName);

	// Uploads an in-memory string to the Amazon S3 service. This is the same as
	// UploadFile, except that the file contents are from an in-memory string instead
	// of a local file. Internal to this method, the objectContent is converted to the character
	// encoding specified by charset prior to uploading.
	// 
	// Note: x-amz-* headers, including metadata, can be added to any S3 request by
	// adding each header with a call to SetRequestHeader. This applies to all S3
	// methods, even if not explicitly stated.
	// 
	// Note: If the bucket is in a region different than us-east-1, makes sure to set
	// the AwsEndpoint property to the correct region, such as "eu-central-1". Also, if
	// using an S3 compatible service such as Wasabi, always set the AwsEndpoint
	// property. For example: "s3.wasabisys.com", "s3.eu-central-1.wasabisys.com".
	// 
	CkTask *S3_UploadStringAsync(const char *objectContent, const char *charset, const char *contentType, const char *bucketPath, const char *objectName);


	// Sets the private key to be used with some forms of authentication. Starting in
	// v9.5.0.89, this is used automatically add the Authorization header (Signature)
	// for Amazon Pay requests.
	bool SetAuthPrivateKey(const char *publicKeyId, CkPrivateKey &privKey);


	// Restores cookies for a particular domain. It is assumed that the cookie XML was
	// previously retrieved via the GetCookieXml method, and saved to some sort of
	// persistent storage, such as within a database table. It is then possible for an
	// application to restore the cookies by calling this method.
	bool SetCookieXml(const char *domain, const char *cookieXml);


	// Sets the RSA key to be used with OAuth authentication when the RSA-SHA1 OAuth
	// signature method is used (see the OAuthSigMethod property).
	bool SetOAuthRsaKey(CkPrivateKey &privKey);


	// Equivalent to setting the Password property, but provides for a more secure way
	// of passing the password in a secure string object.
	// 
	// Note: Starting in v9.5.0.76, this method has been copied to SetSecurePassword.
	// Applications should call SetSecurePassword instead because this method is now
	// deprecated.
	// 
	bool SetPassword(CkSecureString &password);


	// Adds a custom header field to any HTTP request sent by a method that does not
	// use the HTTP request object. These methods include Download, DownloadAppend,
	// GetHead, PostBinary, PostXml, PutBinary, PutText, QuickDeleteStr, QuickGet,
	// QuickGetObj, QuickGetStr, QuickPutStr, XmlRpc, and XmlRpcPut.
	// 
	// Cookies may be explictly added by calling this method passing "Cookie" for the
	// headerFieldName.
	// 
	// The RemoveRequestHeader method can be called to remove a custom header.
	// 
	// Note1: Do not call this method to add Basic authentication, i.e. "Authorization:
	// Basic ....". Instead, set the Login and Password properties, and also set the
	// BasicAuth property = true. Chilkat will add the "Authorization: Basic ..."
	// header using the correct format automatically.
	// 
	// Note 2: Never explicitly set the Content-Length header field. Chilkat will
	// automatically compute the correct length and add the Content-Length header to
	// all POST, PUT, or any other request where the Content-Length needs to be
	// specified. (GET requests always have a 0 length body, and therefore never need a
	// Content-Length header field.)
	// 
	// Note 3: To add more than one header, call this method once per header field.
	// 
	// Note 4:: Passing an empty value string causes the header to be removed. This was
	// the unfortunate behavior of the method from the beginning, and cannot be changed
	// for backward compatibility reasons. A workaround exists starting in v9.5.0.82 by
	// adding the "AllowEmptyHeaders" keyword to the UncommonOptions property.
	// 
	void SetRequestHeader(const char *headerFieldName, const char *headerFieldValue);


	// Equivalent to setting the Password property, but provides for a more secure way
	// of passing the password in a secure string object.
	bool SetSecurePassword(CkSecureString &password);


	// Enforces a requirement on the server's certificate. The reqName can be one of the
	// following:
	//     SubjectDN
	//     SubjectCN
	//     IssuerDN
	//     IssuerCN
	//     SAN (added in v9.5.0.84)
	// 
	// The reqName specifies the part of the certificate, and the reqValue is the value that
	// it must match exactly or with a wildcard (*). If the server's certificate does
	// not match, the SSL / TLS connection is aborted.
	// 
	void SetSslCertRequirement(const char *reqName, const char *reqValue);


	// Allows for a client-side certificate to be used for an SSL connection.
	bool SetSslClientCert(CkCert &cert);


	// Allows for a client-side certificate + private key to be used for the SSL / TLS
	// connection (often called 2-way SSL).
	bool SetSslClientCertPem(const char *pemDataOrPath, const char *pemPassword);


	// Allows for a client-side certificate + private key to be used for the SSL / TLS
	// connection (often called 2-way SSL).
	bool SetSslClientCertPfx(const char *pfxPath, const char *pfxPassword);


	// Sets the value of a variable for substitutions in URLs passed to any method.
	// Variables can appear in URLs in the following format: {$varName}. For example:
	// https://graph.microsoft.com/v1.0/users/{$id}
	bool SetUrlVar(const char *name, const char *value);


	// Authenticates with SharePoint Online, resulting in a cookie being set and used
	// for subsequent SharePoint HTTP requests. Prior to calling this method, an
	// application should set the CookieDir property to either "memory" or a directory
	// path to persist the SharePoint authentication cookie to be automatically used in
	// subsequent runs.
	// 
	// This method has the side-effect of setting the SaveCookies and SendCookies
	// properties to true, because these settings are required for SharePoint Online
	// authentication.
	// 
	// The siteUrl is a URL such as "https://yourdomain.sharepoint.com/". The username is an
	// email address such as "username@yourdomain.com". The extraInfo is reserved for
	// providing additional information as needed in the future.
	// 
	bool SharePointOnlineAuth(const char *siteUrl, const char *username, CkSecureString &password, CkJsonObject &extraInfo);

	// Authenticates with SharePoint Online, resulting in a cookie being set and used
	// for subsequent SharePoint HTTP requests. Prior to calling this method, an
	// application should set the CookieDir property to either "memory" or a directory
	// path to persist the SharePoint authentication cookie to be automatically used in
	// subsequent runs.
	// 
	// This method has the side-effect of setting the SaveCookies and SendCookies
	// properties to true, because these settings are required for SharePoint Online
	// authentication.
	// 
	// The siteUrl is a URL such as "https://yourdomain.sharepoint.com/". The username is an
	// email address such as "username@yourdomain.com". The extraInfo is reserved for
	// providing additional information as needed in the future.
	// 
	CkTask *SharePointOnlineAuthAsync(const char *siteUrl, const char *username, CkSecureString &password, CkJsonObject &extraInfo);


	// Convenience method to force the calling process to sleep for a number of
	// milliseconds.
	void SleepMs(int millisec);


	// Sends an explicit HttpRequest to an HTTP server and returns an HttpResponse
	// object. The HttpResponse object provides full access to the response including
	// all headers and the response body. This method may be used to send POST
	// requests, as well as GET, HEAD, file uploads, and XMLHTTP. To send via HTTPS
	// (i.e. TLS), set the ssl property = true. Otherwise set it to false.
	// 
	// NOTE: The 1st argument of this method is a domain, not a URL. For example, DO
	// NOT pass "https://www.somedomain.com/" in the 1st argument. Instead, pass
	// "www.somedomain.com".
	// 
	// The Parts of a URL
	// 
	// http://example.com:8042/over/there?name=ferret#nose
	// \__/   \______________/\_________/ \________/ \__/
	//  |           |            |            |        |
	// scheme   domain+port     path        query   fragment
	// The caller is responsible for deleting the object returned by this method.
	CkHttpResponse *SynchronousRequest(const char *domain, int port, bool ssl, CkHttpRequest &req);

	// Sends an explicit HttpRequest to an HTTP server and returns an HttpResponse
	// object. The HttpResponse object provides full access to the response including
	// all headers and the response body. This method may be used to send POST
	// requests, as well as GET, HEAD, file uploads, and XMLHTTP. To send via HTTPS
	// (i.e. TLS), set the ssl property = true. Otherwise set it to false.
	// 
	// NOTE: The 1st argument of this method is a domain, not a URL. For example, DO
	// NOT pass "https://www.somedomain.com/" in the 1st argument. Instead, pass
	// "www.somedomain.com".
	// 
	// The Parts of a URL
	// 
	// http://example.com:8042/over/there?name=ferret#nose
	// \__/   \______________/\_________/ \________/ \__/
	//  |           |            |            |        |
	// scheme   domain+port     path        query   fragment
	CkTask *SynchronousRequestAsync(const char *domain, int port, bool ssl, CkHttpRequest &req);


	// Unlocks the Http class/component. It is necessary to call Http.UnlockComponent
	// before calling any other methods. Passing any string to UnlockComponent will
	// automatically activate a 30-day trial period.
	bool UnlockComponent(const char *unlockCode);


	// URL decodes a string.
	bool UrlDecode(const char *str, CkString &outStr);

	// URL decodes a string.
	const char *urlDecode(const char *str);

	// URL encodes a string.
	bool UrlEncode(const char *str, CkString &outStr);

	// URL encodes a string.
	const char *urlEncode(const char *str);

	// Verifies a timestamp reply received from a timestamp authority (TSA). Returns
	// the following possible integer values:
	//     -1: The timestampReply does not contain a valid timestamp reply.
	//     -2: The timestampReply is a valid timestamp reply, but failed verification using the
	//     public key of the tsaCert.
	//     0: Granted and verified.
	//     1: Granted and verified, with mods (see RFC 3161)
	//     2: Rejected.
	//     3: Waiting.
	//     4: Revocation Warning
	//     5: Revocation Notification
	// 
	// If the timestamp reply (timestampReply) is known to be from a trusted source, then the
	// tsaCert may be empty. If tsaCert is empty (never loaded with a certificate), then the
	// verification will use the certificate embedded in the timestamp reply.
	// 
	int VerifyTimestampReply(CkBinData &timestampReply, CkCert &tsaCert);


	// Makes an XML RPC call to a URL endpoint. The XML string is passed in an HTTP
	// POST, and the XML response is returned.
	bool XmlRpc(const char *urlEndpoint, const char *xmlIn, CkString &outStr);

	// Makes an XML RPC call to a URL endpoint. The XML string is passed in an HTTP
	// POST, and the XML response is returned.
	const char *xmlRpc(const char *urlEndpoint, const char *xmlIn);
	// Makes an XML RPC call to a URL endpoint. The XML string is passed in an HTTP
	// POST, and the XML response is returned.
	CkTask *XmlRpcAsync(const char *urlEndpoint, const char *xmlIn);


	// Same as XmlRpc, but uses the HTTP PUT method instead of the POST method.
	bool XmlRpcPut(const char *urlEndpoint, const char *xmlIn, CkString &outStr);

	// Same as XmlRpc, but uses the HTTP PUT method instead of the POST method.
	const char *xmlRpcPut(const char *urlEndpoint, const char *xmlIn);
	// Same as XmlRpc, but uses the HTTP PUT method instead of the POST method.
	CkTask *XmlRpcPutAsync(const char *urlEndpoint, const char *xmlIn);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
